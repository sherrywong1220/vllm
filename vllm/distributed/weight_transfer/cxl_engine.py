# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CXL / shared-canonical-store weight transfer engine.

P2 of the wt_cxl canonical-store curriculum. Each vLLM TP worker reads the
committed model version DIRECTLY from a shared canonical weight store (an
mmap-backed `shared_weight_store` library, on `/tmp` today, a CXL DAX device
later) and feeds the tensors into ``model.load_weights`` itself — eliminating
veRL's ``BucketedWeightSender`` per-GPU IPC hop. (That hop is NOT a single leader
funnel: ``update_weights`` is ``Dispatch.ONE_TO_ALL`` so every rank sends the full
model to its co-located vLLM worker over its OWN per-GPU ZMQ/IPC socket, serialized
one bucket at a time — N parallel per-GPU transfers, each with per-bucket ack + a
CE→IPC→vLLM double-hop. It was the entire 7B read cost: G2=3.04s of D4d=3.12s, while
the load itself is 0.1s.)

P2 uses ``is_checkpoint_format=True``: each worker reads the full canonical
tensor and vLLM's existing per-parameter ``weight_loader``s slice it for this
TP rank. P3 will add a loader-aware copy plan so each worker reads only its
~1/TP slice (kills the over-read); that lives behind this same interface.

The store is the persistent shared buffer; the trainer writes it via the
``shared_weight_store`` FSDP writer (driven by veRL's cxl checkpoint engine),
so the ``trainer_send_weights`` static hook here is intentionally unused.
"""

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_OPEN_TIMEOUT_S = 120.0
_WAIT_TIMEOUT_S = 600.0


@dataclass
class _EPEntry:
    """P7 plan entry for one fused 3D expert store tensor (``mlp.experts.gate_up_proj``
    ``[E, 2I, H]`` / ``down_proj`` ``[E, H, I]``). Ownership is the engine's live
    ``expert_map`` turned into contiguous read runs (``shared_weight_store.expert_plan``),
    never recomputed from config (design §3.2)."""

    kind: str  # "w13" | "w2"
    runs: list  # [ExpertRun]: (global_start, local_start, count)
    param: torch.nn.Parameter  # experts.w13_weight / w2_weight — this rank's [E_local, ...]
    module: Any  # the FusedMoE layer (owns weight_loader + _expert_map)
    param_name: str
    n_local: int


def ep_bridge_loader(entry: _EPEntry, run, block: torch.Tensor) -> int:
    """Bridge A (reuse): feed one contiguous expert block (``[count, ...]``, GLOBAL ids
    ``run.global_start + i``) through vLLM's own per-expert ``weight_loader``. The loader
    re-derives the local slot from the engine's ``_expert_map``, so a disagreement
    between our runs and the engine's ownership surfaces as ``False`` (raised here)
    instead of a silent mis-place. Cost: 2 (w13) / 1 (w2) loader calls per expert, each
    a d2d ``copy_`` plus an ``_expert_map[...].item()`` device sync — the §3.4
    per-tensor overhead this phase must attribute, which is why bridge B exists.

    ``weight_name`` handed to the loader is the PARAM name (``…experts.w13_weight``),
    never the store name: the loader dispatches on substrings of that string
    (``"weight" in weight_name`` selects the model-weight path; a name without it
    returns False for every expert). C1 (test_ep_bridge) caught this."""
    n = 0
    name = entry.param_name
    if entry.kind == "w13":
        inter = block.shape[1] // 2
        for i in range(run.count):
            gid = run.global_start + i
            ok1 = entry.module.weight_loader(
                entry.param, block[i, :inter], name, shard_id="w1", expert_id=gid, return_success=True
            )
            ok3 = entry.module.weight_loader(
                entry.param, block[i, inter:], name, shard_id="w3", expert_id=gid, return_success=True
            )
            if not (ok1 and ok3):
                raise RuntimeError(
                    f"P7 ep-slice: engine refused expert {gid} of {name} (gate={ok1}, up={ok3}) "
                    "— read plan and engine expert_map disagree"
                )
            n += 1
    else:
        for i in range(run.count):
            gid = run.global_start + i
            ok = entry.module.weight_loader(
                entry.param, block[i], name, shard_id="w2", expert_id=gid, return_success=True
            )
            if not ok:
                raise RuntimeError(
                    f"P7 ep-slice: engine refused expert {gid} of {name} — read plan and "
                    "engine expert_map disagree"
                )
            n += 1
    return n


def ep_bridge_direct(entry: _EPEntry, run, block: torch.Tensor) -> int:
    """Bridge B (direct): ONE d2d copy of the whole run into the param's local slots.
    Valid for unquantized experts because the fused ``w13`` layout is ``[gate; up]`` on
    dim 1 — the same order HF's ``Concatenate(dim=1)`` produces and ``_load_w13`` writes
    (C1b: bitwise-equal to bridge A). Casts if the store dtype (fp32 under P6-M2)
    differs from the param dtype."""
    dst = entry.param.data.narrow(0, run.local_start, run.count)
    if tuple(dst.shape) != tuple(block.shape):
        raise RuntimeError(
            f"P7 ep-slice direct: local slot shape {tuple(dst.shape)} != block "
            f"{tuple(block.shape)} for {entry.param_name}"
        )
    dst.copy_(block)
    return run.count


@dataclass
class CXLWeightTransferInitInfo(WeightTransferInitInfo):
    """One field: where the shared canonical store lives."""

    store_uri: str = ""


@dataclass
class CXLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """The committed weight version this worker should consume. Tensor names and
    shapes come from the store's own manifest, so they are not repeated here."""

    version: int = 0


class CXLWeightTransferEngine(
    WeightTransferEngine[CXLWeightTransferInitInfo, CXLWeightTransferUpdateInfo]
):
    init_info_cls = CXLWeightTransferInitInfo
    update_info_cls = CXLWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)
        self._store_uri: str | None = None
        self._store = None  # CanonicalWeightStore, opened lazily on first receive
        self._owner_names: list[str] = []
        self._cuda_direct = False  # True once the store is cudaHostRegister'd (P2 one-hop read)
        self._pinned: dict[str, torch.Tensor] = {}  # two-hop fallback buffers
        self._specs: dict = {}  # name -> TensorSpec (full shape) for P3A slice geometry
        # Attribution/toggle delivery — env OR a filesystem sentinel on shared /grand.
        # This engine only runs when the cxl checkpoint engine is configured with
        # reader=vllm_native (reader=vllm_checkpoint reads via veRL's own bucket generator
        # and never enters this file — the cause of an earlier round of "no reader output").
        # The sentinel is a delivery channel that does not depend on the ray-launched rollout
        # worker inheriting the launcher's exported env: the profiling dir is derived from
        # this module's own path (…/RL_post_training/vllm/vllm/distributed/weight_transfer/
        # cxl_engine.py → 4 parents up = project root), and a sentinel file in that dir enables
        # each toggle. Env still works and takes precedence. Default (no sentinel, no env) =
        # OFF, so production behaviour is unchanged.
        _proj = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, os.pardir, os.pardir)
        )
        self._read_prof_dir = (
            os.environ.get("WT_CXL_READ_PROF_DIR")
            or os.path.join(_proj, "wt_cxl_runs", "read_prof_dump")
        )

        def _on(env_key, env_true, sentinel):
            env_val = os.environ.get(env_key, "0")
            if env_true(env_val):
                return True
            try:
                return os.path.exists(os.path.join(self._read_prof_dir, sentinel))
            except OSError:
                return False

        # P3A TP-local slice read (kill the over-read); see p3a_reuse_loader_decision.md.
        self._tp_slice = _on("WT_CXL_TP_SLICE", lambda v: v == "1", "ENABLE_TP_SLICE")
        self._plan: dict | None = None  # name -> (dim, start, size); built once on first receive
        # P7 EP-aware expert slice read: off (full read of the fused 3D experts; the
        # loader filters to local experts — the P3D-style correctness baseline),
        # loader (read only this rank's expert run(s), bridge A), direct (bridge B).
        # See p7_ep_aware_slice_read_design.md §1.
        self._ep_slice = os.environ.get("WT_CXL_EP_SLICE", "off").strip().lower() or "off"
        if self._ep_slice not in ("off", "loader", "direct"):
            raise ValueError(f"WT_CXL_EP_SLICE must be off|loader|direct, got {self._ep_slice!r}")
        self._ep_plan: dict[str, _EPEntry] = {}
        self._classes: dict[str, str] = {}  # owner name -> tp | expert | repl (§1.1 accounting)
        self._class_bytes: dict[str, int] = {"tp": 0, "expert": 0, "repl": 0}
        self._param_names: set[str] | None = None  # model params, for the coverage oracle
        self._world: dict = {"tp": 1, "ep": 1, "dp": 1, "ep_rank": 0}
        self._coaxial: str = "na"
        # D4d attribution: split residual read cost F into GPU-copy vs Python-dispatch to
        # decide P3B vs batching the loads. Instrumented → attribution-only, never a verdict.
        self._read_prof = _on("WT_CXL_READ_CPROFILE", lambda v: v not in ("0", ""),
                              "ENABLE_READ_PROF")
        self._recv_idx = 0  # receive_weights call counter (0 = warmup, excluded like WSPHASE)
        # P5 sparse change set (design claude_log/weight_sync/p5_sparse_change_set_design.md §1.4.3):
        # the store carries per-trainer-rank change sets next to the dense anchor; version 1 is the
        # dense read above, every later version (is_checkpoint_format=False) patches the resident
        # bf16 weights in place.
        self._cs_world: int | None = None  # trainer world encoded in the manifest; None = no change set
        self._model = None  # the live vLLM model, cached on the dense call (load_weights.__self__)
        self._last_version = 0  # last version consumed (dense or sparse); sparse needs +1 continuity
        self._cs_reader = None  # ChangeSetReader
        self._cs_plans: list | None = None  # DstPlan per owner, built once from the live model

    def _prof_emit(self, line: str) -> None:
        """Emit an attribution line to BOTH stdout (ray, if it forwards) and a file on
        shared /grand (robust to ray's stdout-forwarding drop). Best-effort file write."""
        print(line, flush=True)
        d = self._read_prof_dir
        if not d:
            return
        try:
            import socket
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"readsplit_{socket.gethostname()}_pid{os.getpid()}.log")
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def init_transfer_engine(self, init_info: CXLWeightTransferInitInfo) -> None:
        if not init_info.store_uri:
            raise ValueError("CXL weight transfer needs a non-empty store_uri")
        self._store_uri = init_info.store_uri
        logger.info("CXL weight transfer engine: store_uri=%s", self._store_uri)

    def _ensure_open(self):
        if self._store is not None:
            return
        # Imported lazily so a non-RL vLLM install doesn't need the store library.
        from shared_weight_store import CanonicalWeightStore
        from shared_weight_store.gpu_h2d import slice_h2d

        self._slice_h2d = slice_h2d  # guarded pitched-DMA / copy_ for slice reads (P3A)

        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        while True:
            try:
                # readonly=False so the mmap is writable → cudaHostRegister-able for
                # the one-hop read DMA. We never write through it (region_view reads
                # only); RW is solely to permit host-registration.
                self._store = CanonicalWeightStore.open(self._store_uri, readonly=False)
                break
            except RuntimeError as e:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"CXL store {self._store_uri} not openable after "
                        f"{_OPEN_TIMEOUT_S}s: {e}"
                    ) from e
                time.sleep(0.2)
        from shared_weight_store.change_set import cs_world, is_cs_name

        self._owner_names = [
            t.name for t in self._store.manifest.tensors if t.alias_of is None and not is_cs_name(t.name)
        ]
        self._specs = {t.name: t for t in self._store.manifest.tensors}  # full shapes (P3A)
        self._cs_world = cs_world(self._store.manifest)  # P5: None unless the trainer publishes change sets
        # P2 one-hop read: page-lock the store payload span so each tensor can DMA
        # straight store->cuda (no pinned-CPU stage), reusing P1B's mechanism on the
        # read side. Best-effort: if registration fails (e.g. the gpudirect WRITER
        # already pinned the same shared /tmp pages, or a readonly mapping slipped
        # through), fall back to the staged two-hop read — never crash the sync.
        if torch.cuda.is_available():
            try:
                ok = self._store.register_host_for_cuda()
                probe_pinned = (
                    ok
                    and self._owner_names
                    and self._store.region_view(
                        self._store.active_slot, self._owner_names[0]
                    ).is_pinned()
                )
                self._cuda_direct = bool(probe_pinned)
            except RuntimeError as e:
                print(f"[WT-CXL] GPU-direct read unavailable ({e}) — staged two-hop",
                      flush=True)
                self._cuda_direct = False
        if self._is_tp_rank0():
            print(f"[WT-CXL] reader gpu_direct={self._cuda_direct} "
                  f"({len(self._owner_names)} owner tensors)", flush=True)
            # Unconditional "reader ran" marker to the prof dir (bypasses ray fwd): confirms
            # cxl_engine.receive_weights actually executed on this rollout worker, and records
            # whether the slice + profiling toggles resolved ON (via env or /grand sentinel).
            if self._read_prof:
                self._prof_emit(
                    f"[WT-CXL-READ-OPEN] gpu_direct={self._cuda_direct} "
                    f"tp_slice={self._tp_slice} read_prof={self._read_prof} "
                    f"owner_tensors={len(self._owner_names)} prof_dir={self._read_prof_dir}")

    def receive_weights(
        self,
        update_info: CXLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Read this version's canonical tensors from the shared store and load
        them incrementally. Each worker reads independently (the store mmap pages
        are shared → page-cache hits, not N physical reads); no per-GPU ZMQ/IPC hop.

        ``load_weights`` is vLLM's ``model.load_weights`` (checkpoint format): it
        applies fused-QKV/gate-up remapping and narrows each tensor to this TP
        rank, so a full canonical tensor in → this rank's shard loaded."""
        if self._store_uri is None:
            raise RuntimeError("init_transfer_engine was not called before receive")
        self._ensure_open()

        version = update_info.version
        if not update_info.is_checkpoint_format:
            self._receive_sparse(version)  # P5: in-place patch, no layerwise reload, no model.load_weights
            return
        model_obj = getattr(load_weights, "__self__", None)
        if model_obj is not None:
            self._model = model_obj  # P5 anchor: the sparse path patches this model's live parameters
        # P6 attribution (ALWAYS-ON, WSPHASE-class): D4d decomposes into exactly two
        # parts — the reader blocked on the trainer's write+commit, and the read loop
        # itself. Archived logs put the former at ~70% of D4d (and the veRL server
        # adapter at ~0.00 s), but only by subtraction; this times it directly. Cost is
        # two perf_counter reads + one rank-0 line per call = the same perturbation
        # class as the D4b/D4d markers, so it stays valid on an un-instrumented verdict
        # run. The wait is what P6 removes; see p6_offload_fused_write_design.md.
        _t_wait = time.perf_counter()
        self._store.wait_committed(min_version=version, timeout_s=_WAIT_TIMEOUT_S)
        _wait_s = time.perf_counter() - _t_wait

        # P3A: build the TP-slice copy plan once. load_weights is model.load_weights
        # (a bound method) on the checkpoint-format path, so __self__ is the live
        # vLLM model we introspect for shard geometry + set is_sharded_weight on.
        if self._plan is None:  # classification (accounting) runs even with every slice off
            self._build_plan(getattr(load_weights, "__self__", None))
        plan = self._plan
        ep_plan = self._ep_plan

        use_cuda = torch.cuda.is_available()
        c2 = os.environ.get("WT_CXL_C2", "0") == "1"  # anti-cheat digest oracle
        digest = 0
        total_bytes = 0
        # The committed slot (1 - writing_slot); stable for a reader after
        # wait_committed (flips only at the next commit, which the stop-the-world
        # sync defers). region_view is a zero-copy CPU view into the store mmap.
        slot = self._store.active_slot
        if c2:
            from shared_weight_store.digest import xor_digest_update

        # --- D4d attribution (WT_CXL_READ_CPROFILE; rank-0; quarantined, never a verdict) ---
        # Split the residual read cost F into GPU-copy (the 339 param_data.copy_) vs
        # Python-dispatch (per-tensor weight_loader/_layerwise_process under the GIL), to
        # decide P3B (copy-stream overlap) vs batching the loads (see p3b_read_pipeline_design.md).
        # Method A (steady calls 1,3): insert per-tensor cuda.synchronize() to time copy_gpu
        # (F.gpu_copy) vs load_dispatch (F.dispatch) directly — immune to cProfile's blindness
        # inside aten::copy_. Method B (steady call 2): cProfile the natural (pipelined) path to
        # attribute the Python dispatch to specific frames (= the batching target if dispatch-bound).
        _idx = self._recv_idx
        self._recv_idx += 1
        _instr = self._read_prof and use_cuda and self._is_tp_rank0()
        _do_split = _instr and _idx in (1, 3)  # per-tensor sync: copy_gpu vs load_dispatch
        _do_cprof = _instr and _idx == 2       # cProfile natural path: which Python frames
        if _instr and _idx == 0:
            self._prof_emit(
                f"[WT-CXL-READ-ENV] WT_CXL_READ_CPROFILE={self._read_prof} "
                f"gpu_direct={self._cuda_direct} tp_slice={self._tp_slice} "
                f"owner_tensors={len(self._owner_names)} "
                f"sliced={len(plan) if plan else 0}")
        _pr = None
        if _do_cprof:
            import cProfile
            _pr = cProfile.Profile()
            _pr.enable()
        _s_stage = _s_ldisp = _s_copygpu = 0.0  # split accumulators (s): H2D, load-dispatch, copy-gpu
        read_by_class = {"tp": 0, "expert": 0, "repl": 0}  # §1.1 three-way accounting
        loaded: set[str] = set()  # absolute load-coverage oracle (§3.6.1)
        ep_written: dict[str, int] = {}  # param_name -> local expert slots written this call
        n_ep_runs = n_ep_calls = 0

        t0 = time.perf_counter()
        for name in self._owner_names:
            _ts = time.perf_counter() if _do_split else 0.0
            cls = self._classes.get(name, "repl")
            ep = ep_plan.get(name)
            if ep is not None:
                # P7: read ONLY this rank's expert run(s) of the fused [E, ...] canonical
                # (contiguous dim-0 narrows -> plain copy_ DMA), then bridge them into the
                # local [E_local, ...] param. No model.load_weights on this path: the
                # fused name would be re-routed per expert anyway (shared-downstream fix),
                # and the bridge is where bytes-vs-per-tensor cost is attributed.
                for run in ep.runs:
                    sl = self._store.region_view(slot, name).narrow(0, run.global_start, run.count)
                    if c2:
                        digest = xor_digest_update(digest, name, sl.contiguous())
                    if use_cuda and self._cuda_direct:
                        block = torch.empty(tuple(sl.shape), dtype=sl.dtype, device="cuda")
                        self._slice_h2d(sl, block)
                    elif use_cuda:
                        staged = torch.empty(tuple(sl.shape), dtype=sl.dtype, device="cpu", pin_memory=True)
                        staged.copy_(sl)
                        block = staged.to("cuda", non_blocking=True)
                    else:
                        block = sl.contiguous()
                    nb = sl.numel() * sl.element_size()
                    total_bytes += nb
                    read_by_class[cls] += nb
                    n_ep_runs += 1
                    if _do_split:
                        torch.cuda.synchronize()
                        _t1 = time.perf_counter()
                    if self._ep_slice == "direct":
                        n = ep_bridge_direct(ep, run, block)
                        n_ep_calls += 1
                    else:
                        n = ep_bridge_loader(ep, run, block)
                        n_ep_calls += n * (2 if ep.kind == "w13" else 1)
                    ep_written[ep.param_name] = ep_written.get(ep.param_name, 0) + n
                    if _do_split:
                        _t2 = time.perf_counter()
                        torch.cuda.synchronize()
                        _t3 = time.perf_counter()
                        _s_stage += _t1 - _ts
                        _s_ldisp += _t2 - _t1
                        _s_copygpu += _t3 - _t2
                        _ts = _t3
                continue
            entry = plan.get(name) if plan else None
            if entry is not None:
                # P3A reuse-path: read ONLY this rank's slice from the store. The
                # loader skips its narrow (is_sharded_weight, set in _build_plan) but
                # still does fused placement + quant repack — so we move full/TP bytes.
                dim, start, size = entry
                sl = self._store.region_view(slot, name).narrow(dim, start, size)
                if c2:
                    digest = xor_digest_update(digest, name, sl.contiguous())
                if use_cuda and self._cuda_direct:
                    # Fast path: the store region is host-REGISTERED (pinned), so DMA
                    # straight from it. Guarded H2D: strided (row) slice ->
                    # cudaMemcpy2DAsync (pitched DMA, 5-20x over torch's strided copy_);
                    # contiguous (column/vocab/qkv) slice or dtype-mismatch -> copy_
                    # (keeps auto-cast). See gpu_h2d.slice_h2d.
                    tensor = torch.empty(tuple(sl.shape), dtype=sl.dtype, device="cuda")
                    self._slice_h2d(sl, tensor)
                elif use_cuda:
                    # Store NOT host-registered (cudaHostRegister unavailable on this
                    # node — gpu_direct=False). A DIRECT H2D from the non-pinned,
                    # file-backed mmap region can fail (cudaErrorInvalidValue), so stage
                    # the slice through a normal pinned CPU buffer first (copy_ gathers a
                    # strided row slice on the CPU), then H2D — the same two-hop
                    # robustness the full-read path uses below. Slower, but only on an
                    # already-degraded no-GPU-direct node; the perf verdict runs pinned.
                    staged = torch.empty(
                        tuple(sl.shape), dtype=sl.dtype, device="cpu", pin_memory=True
                    )
                    staged.copy_(sl)
                    tensor = staged.to("cuda", non_blocking=True)
                else:
                    tensor = sl.contiguous()
                total_bytes += sl.numel() * sl.element_size()
                read_by_class[cls] += sl.numel() * sl.element_size()
            elif self._cuda_direct:
                # One-hop DMA: registered (pinned) store region -> a fresh cuda
                # tensor, no pinned-CPU stage. Same stream as load_weights below, so
                # the non_blocking copy is ordered before the load reads it.
                region = self._store.region_view(slot, name)
                if c2:
                    digest = xor_digest_update(digest, name, region)
                tensor = torch.empty(region.shape, dtype=region.dtype, device="cuda")
                tensor.copy_(region, non_blocking=True)
                total_bytes += region.nbytes
                read_by_class[cls] += region.nbytes
            elif use_cuda:
                # Two-hop fallback (iter-7): store -> reused pinned CPU buffer -> H2D.
                buf = self._pinned.get(name)
                if buf is None:
                    spec = next(t for t in self._store.manifest.tensors if t.name == name)
                    buf = torch.empty(spec.shape, dtype=spec.dtype, device="cpu", pin_memory=True)
                    self._pinned[name] = buf
                self._store.read_tensor_into(version, name, buf)
                if c2:
                    digest = xor_digest_update(digest, name, buf)
                tensor = buf.to("cuda", non_blocking=True)
                total_bytes += buf.nbytes
                read_by_class[cls] += buf.nbytes
            else:  # CPU-only (C1 / no CUDA): hand the store view straight to load
                region = self._store.region_view(slot, name)
                if c2:
                    digest = xor_digest_update(digest, name, region)
                tensor = region
                total_bytes += region.nbytes
                read_by_class[cls] += region.nbytes
            # Incremental load (one tensor) keeps peak host/device memory bounded
            # and matches the layerwise-reload contract (tensors arrive in manifest
            # = checkpoint order, so each layer's set completes in order).
            if _do_split:
                torch.cuda.synchronize()  # H2D (staging) GPU done
                _t1 = time.perf_counter()
            ret = load_weights([(name, tensor)])
            if ret:
                loaded.update(ret)  # model.load_weights returns the param names it touched
            if _do_split:
                _t2 = time.perf_counter()  # load DISPATCH done (Python; async copy_ launched, not yet run)
                torch.cuda.synchronize()   # copy_ GPU execution done
                _t3 = time.perf_counter()
                _s_stage += _t1 - _ts      # H2D dispatch + GPU (≈ B, per-tensor)
                _s_ldisp += _t2 - _t1      # load_weights Python dispatch (narrow skip + fused placement + copy_ launch)
                _s_copygpu += _t3 - _t2    # param_data.copy_ GPU execution (the d2d placement copy)
        if _instr:
            _t_pre = time.perf_counter()
        if use_cuda:
            torch.cuda.synchronize()
        _t_loop_end = time.perf_counter()
        if _instr:
            _t_post = _t_loop_end
            if _do_split:
                _F = _s_ldisp + _s_copygpu
                self._prof_emit(
                    f"[WT-CXL-READ-SPLIT idx={_idx}] owner={len(self._owner_names)} "
                    f"sliced={len(plan) if plan else 0} | "
                    f"F.gpu_copy(copy_)={_s_copygpu:.3f}s  F.dispatch(load)={_s_ldisp:.3f}s  "
                    f"B.H2D(stage)={_s_stage:.3f}s | F={_F:.3f}s "
                    f"gpu_copy_frac={(_s_copygpu / _F if _F else 0):.2f}")
            # Coarse cross-check (esp. the cProfile/natural call, no per-tensor syncs):
            # loop_dispatch_wall = Python racing through all launches; final_sync_gpu_tail =
            # total GPU work still outstanding at loop end (= the writer's 92/8 method, D4d side).
            self._prof_emit(
                f"[WT-CXL-READ-COARSE idx={_idx}] loop_dispatch_wall={_t_pre - t0:.3f}s "
                f"final_sync_gpu_tail={_t_post - _t_pre:.3f}s total={_t_post - t0:.3f}s "
                f"split={'Y' if _do_split else 'N'} cprof={'Y' if _do_cprof else 'N'}")
        # --- absolute load-coverage oracle (§3.6.1; both slice toggles, every call) ---
        # C2 is RELATIVE (cxl vs nccl through one shared loader), so a name silently
        # dropped by the loader is invisible to it. This is the absolute check: every
        # model parameter was written this call, either by model.load_weights (names it
        # returns) or by the EP bridge (every local expert slot of every MoE param).
        if self._param_names is not None:
            covered = set(loaded)
            for pn, n in ep_written.items():
                ep_entry = next((e for e in ep_plan.values() if e.param_name == pn), None)
                if ep_entry is not None and n == ep_entry.n_local:
                    covered.add(pn)
            missing = self._param_names - covered
            if self._is_tp_rank0():
                print(f"[WT-CXL-COVERAGE idx={_idx}] params={len(self._param_names)} "
                      f"loaded={len(loaded)} ep_sliced={len(ep_written)} missing={len(missing)} "
                      f"sample={sorted(missing)[:5]}", flush=True)
            if missing and os.environ.get("WT_CXL_COVERAGE", "strict") != "warn":
                raise RuntimeError(
                    f"P7 coverage oracle: {len(missing)} model params not loaded this sync "
                    f"(e.g. {sorted(missing)[:8]}) — a silently dropped name is weight "
                    "corruption, not a warning (WT_CXL_COVERAGE=warn to downgrade)"
                )

        if self._is_tp_rank0():
            # §1.1 three-way accounting: per-class bytes read vs the theoretical floor
            # (tp class / tp_size, expert class / ep_size, replicated class unchanged),
            # plus the EP call counts that carry the §3.4 granularity story. Always-on,
            # one rank-0 line = same perturbation class as the WSPHASE markers.
            from shared_weight_store.expert_plan import lower_bounds
            lb = lower_bounds(self._class_bytes, self._world["tp"], self._world["ep"])
            self._prof_emit(
                f"[WSPHASE-CXL-READ-ACCT idx={_idx}] tp_size={self._world['tp']} "
                f"ep_size={self._world['ep']} dp_size={self._world['dp']} "
                f"tp_slice={int(self._tp_slice)} ep_slice={self._ep_slice} "
                f"tp_read_gb={read_by_class['tp'] / 1e9:.3f} tp_lb_gb={lb['tp'] / 1e9:.3f} "
                f"expert_read_gb={read_by_class['expert'] / 1e9:.3f} expert_lb_gb={lb['expert'] / 1e9:.3f} "
                f"repl_read_gb={read_by_class['repl'] / 1e9:.3f} repl_lb_gb={lb['repl'] / 1e9:.3f} "
                f"expert_tensors={sum(1 for c in self._classes.values() if c == 'expert')} "
                f"ep_sliced_tensors={len(ep_plan)} ep_runs={n_ep_runs} ep_bridge_calls={n_ep_calls} "
                f"coaxial={self._coaxial}")
            # The always-on D4d split (see the note at wait_committed above).
            # read_gb is here because it is the NODE-HEALTH CANARY: Sophia shows ~8x
            # node-to-node PCIe variance, and A/Bs used to gate comparability on D4b
            # (the trainer write). P6-M2 deletes the write, so a zero D4b now means
            # "the mechanism worked", not "the node is fast" — the canary had to move
            # to the read side. The read loop is H2D-bandwidth bound (P3B/P3E), so
            # read_gb / read_loop is a direct PCIe reading, and it stays comparable
            # across arms whose store dtype makes them move different bytes by design.
            # One extra number in an existing rank-0 line = same perturbation class,
            # so it remains valid on an un-instrumented verdict run.
            self._prof_emit(
                f"[WSPHASE-CXL-READ idx={_idx}] D4d_wait_committed={_wait_s:.3f}s "
                f"D4d_read_loop={_t_loop_end - t0:.3f}s "
                f"read_gb={total_bytes / 1e9:.3f}")
        if _do_cprof:
            _pr.disable()
            import io as _io
            import pstats as _pstats
            _b1 = _io.StringIO()
            _pstats.Stats(_pr, stream=_b1).sort_stats("cumulative").print_stats(45)
            self._prof_emit(f"[WT-CXL-READ-CPROFILE-CUMULATIVE idx={_idx}]\n{_b1.getvalue()}")
            _b2 = _io.StringIO()
            _pstats.Stats(_pr, stream=_b2).sort_stats("tottime").print_stats(35)
            self._prof_emit(f"[WT-CXL-READ-CPROFILE-TOTTIME idx={_idx}]\n{_b2.getvalue()}")
        if c2 and self._is_tp_rank0():
            # Full-read path: digest == the trainer's send digest (over full_tensor())
            # proves the per-worker store read reconstructed the canonical weights
            # bitwise. Slice path: this rank only read its slice, so the digest is NOT
            # comparable to send-over-full — labelled role=recv-sliced and excluded
            # from the equality oracle (its bitwise gate is C1b test_slice_plan; its
            # runtime oracle is generation-match). One print from TP rank 0.
            role = "recv-sliced" if (plan or ep_plan) else "recv"
            print(
                f"[WT-CXL-C2] role={role} version={version} "
                f"count={len(self._owner_names)} digest={digest:08x}",
                flush=True,
            )
        self._last_version = version
        logger.info(
            "CXL receive_weights v%d: %.2f GB from %d tensors in %.2fs",
            version,
            total_bytes / 1e9,
            len(self._owner_names),
            time.perf_counter() - t0,
        )

    # -- P5 sparse path ---------------------------------------------------------------------

    def _receive_sparse(self, version: int) -> None:
        """Apply version ``version``'s change sets (every trainer rank's) to the resident weights.

        Precondition: this worker holds the bf16 rendition of ``version - 1`` (it read version 1
        densely and applied every version since). The change set is the elements whose bf16 bits
        changed; applying it reproduces a dense reload bitwise (the in-situ C2 below checks it).
        """
        from shared_weight_store.change_set import ChangeSetReader, owner_specs

        _t_wait = time.perf_counter()
        self._store.wait_committed(min_version=version, timeout_s=_WAIT_TIMEOUT_S)
        _wait_s = time.perf_counter() - _t_wait
        if self._cs_world is None:
            raise RuntimeError("P5: a sparse version was requested but the store has no change-set regions")
        if self._model is None:
            raise RuntimeError("P5: sparse version before any dense version (no anchor on this worker)")
        if version != self._last_version + 1:
            raise RuntimeError(
                f"P5: version gap {self._last_version} -> {version}: a change set applies only to the "
                "immediately preceding version (no ring, by design)"
            )
        _idx = self._recv_idx
        self._recv_idx += 1
        use_cuda = torch.cuda.is_available()
        dev = torch.device("cuda") if use_cuda else torch.device("cpu")
        if self._cs_reader is None:
            owners = owner_specs(self._store.manifest)
            assert [o.name for o in owners] == self._owner_names
            self._cs_reader = ChangeSetReader(owners, self._cs_world)
            self._cs_plans = self._build_sparse_plan(self._model, owners)
        slot = self._store.active_slot
        t0 = time.perf_counter()
        entries = self._cs_reader.read(self._store, slot, dev)
        if use_cuda:
            torch.cuda.synchronize()
        t_read = time.perf_counter()
        res = self._cs_reader.apply(entries, self._cs_plans)
        if use_cuda:
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        self._last_version = version
        if self._is_tp_rank0():
            # Same keys as the dense path so run_bench's collector and the node-health canary keep working.
            self._prof_emit(
                f"[WSPHASE-CXL-READ idx={_idx}] D4d_wait_committed={_wait_s:.3f}s "
                f"D4d_read_loop={t_end - t0:.3f}s "
                f"read_gb={entries.bytes_read / 1e9:.6f}")
            n_el = sum(self._cs_reader.n_elems)
            self._prof_emit(
                f"[WSPHASE-CXL-SPARSE idx={_idx}] version={version} trainer_world={self._cs_world} "
                f"nnz={sum(entries.nnz_per_rank)} nnz_per_rank={entries.nnz_per_rank} "
                f"density={sum(entries.nnz_per_rank) / max(n_el, 1):.5f} applied={res['applied']} "
                f"dropped={res['dropped']} owners_touched={res['owners']} "
                f"read_s={t_read - t0:.4f} apply_s={t_end - t_read:.4f} bytes={entries.bytes_read}")
        if os.environ.get("WT_CXL_C2", "0") == "1":
            self._c2_sparse_check(slot, version, dev)

    def _c2_sparse_check(self, slot: int, version: int, dev) -> None:
        """Tier-S oracle for P5: after the patch, re-read every owner densely (the fp32 anchor, cast
        on this GPU exactly as the dense path casts) and compare this rank's slice with the resident
        parameter, bitwise. A mismatch is weight corruption and raises."""
        from vllm.distributed.weight_transfer.sparse_plan import c2_compare, plan_slice

        bad, checked, max_abs = [], 0, 0.0
        for name, plan in zip(self._owner_names, self._cs_plans):
            if plan.dst is None:
                continue
            want = plan_slice(plan, self._store.region_view(slot, name)).to(dev, non_blocking=False).to(torch.bfloat16)
            checked += 1
            n_bad, mx = c2_compare(plan, want)
            if n_bad:
                max_abs = max(max_abs, mx)
                bad.append((name, n_bad))
        if self._is_tp_rank0() or bad:
            print(f"[WT-CXL-P5-C2] version={version} checked={checked} mismatched_params={len(bad)} "
                  f"max_abs_diff={max_abs:.3e} sample={bad[:4]}", flush=True)
        if bad:
            raise RuntimeError(f"P5 C2: sparse apply != dense reload for {len(bad)} params at v{version}: {bad[:4]}")

    def _build_sparse_plan(self, model, owners) -> list:
        """One ``DstPlan`` per store owner (``sparse_plan.build_sparse_plan``: shared with the NCCL
        transport ablation so both arms apply the same change set under the same placement)."""
        from vllm.distributed.weight_transfer.sparse_plan import build_sparse_plan

        return build_sparse_plan(model, owners)

    def _build_plan(self, model) -> None:
        """Classify every store owner tensor (§1.1 accounting) and build the per-rank
        slice plans: the P3A TP plan (``self._plan``) and the P7 EP plan
        (``self._ep_plan``). Classification always runs; slicing is applied only when
        the corresponding toggle is on, so an OFF arm reports the same class bytes.

        Classes / plans:

        * **expert (P7)** — fused 3D routed-expert params (``mlp.experts.gate_up_proj``
          ``[E, 2I, H]`` / ``down_proj`` ``[E, H, I]``) whose FusedMoE module is found
          on the model. With EP on and ``WT_CXL_EP_SLICE != off``, ownership comes from
          the module's live ``_expert_map`` (never recomputed from config — §3.2) as
          contiguous runs; refused (full read, loader filters) on: no EP, EPLB on,
          quantized param, per-expert shape mismatch, or a map the runs cannot cover.
        * **tp (P3A)** — P3A-1 row / P3A-2 merged-column / P3A-3 fused-QKV /
          P3A-4 vocab (each ``is_sharded_weight``-marked so the loader skips only its
          per-rank narrow; C1b-verified) — unchanged from P3A.
        * **repl** — everything else (norms, biases, router gate, quantized/packed,
          unclassifiable): full read, never silently mis-sliced (P3D coverage).
        """
        from shared_weight_store.expert_plan import coaxial, expert_runs
        from shared_weight_store.slice_plan import (
            column_slice_plan,
            qkv_slice_plan,
            row_slice_plan,
            vocab_slice_plan,
        )

        self._plan = {}
        self._ep_plan = {}
        if model is None:
            # Non-checkpoint-format caller (no bound model) — never slice, no oracle.
            return
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank, tp_size = 0, 1
        self._world["tp"] = tp_size

        params = dict(model.named_parameters())
        self._param_names = set(params)
        self._model_for_mapping = model  # packed_modules_mapping lookups (P3A-2/3)
        modules = dict(model.named_modules())
        plan: dict = {}
        n_row = n_col = n_qkv = n_vocab = 0
        n_expert = 0
        ep_refused: dict[str, int] = {}
        ep_info = None
        trainer_world = os.environ.get("WT_CXL_TRAINER_WORLD")

        def _nbytes(spec) -> int:
            n = 1
            for d in spec.shape:
                n *= int(d)
            return n * torch.empty((), dtype=spec.dtype).element_size()

        for name in self._owner_names:
            spec = self._specs.get(name)
            if spec is None:
                self._classes[name] = "repl"
                continue
            nb = _nbytes(spec)

            # --- P7: fused 3D expert params -----------------------------------------
            ep = self._ep_target(name, spec, params, modules)
            if ep is not None:
                kind, module, param, pname = ep
                self._classes[name] = "expert"
                self._class_bytes["expert"] += nb
                n_expert += 1
                if ep_info is None:
                    ep_info = module  # world facts come from the FIRST MoE layer
                    self._world["ep"] = int(getattr(module, "ep_size", 1) or 1)
                    self._world["dp"] = int(getattr(module, "dp_size", 1) or 1)
                    self._world["ep_rank"] = int(getattr(module, "ep_rank", 0) or 0)
                if self._ep_slice == "off":
                    continue
                reason = None
                emap = getattr(module, "_expert_map", None)
                if not getattr(module, "use_ep", False) or emap is None:
                    reason = "no-ep"
                elif getattr(module, "enable_eplb", False):
                    reason = "eplb"  # plan would change at runtime (§3.8) — not this phase
                elif not self._sliceable(param):
                    reason = "quantized"
                elif tuple(param.shape[1:]) != tuple(spec.shape[1:]):
                    reason = "per-expert-shape"  # TP inside EP (moe_tensor_parallel) — loader narrows
                else:
                    try:
                        runs = expert_runs(emap.tolist())
                    except ValueError:
                        runs, reason = [], "bad-expert-map"
                    if reason is None:
                        n_local = sum(r.count for r in runs)
                        if n_local != int(getattr(module, "local_num_experts", -1)):
                            reason = "local-count-mismatch"
                        elif tuple(param.shape) != (n_local,) + tuple(spec.shape[1:]):
                            reason = "local-param-shape"
                if reason is not None:
                    ep_refused[reason] = ep_refused.get(reason, 0) + 1
                    continue
                self._ep_plan[name] = _EPEntry(kind, runs, param, module, pname, n_local)
                if self._coaxial == "na" and trainer_world:
                    self._coaxial = "yes" if coaxial(
                        emap.tolist(), self._world["ep_rank"], int(trainer_world)) else "no"
                continue

            # --- P3A: TP classes -------------------------------------------------------
            tp = self._tp_target(name, spec, params, modules, tp_rank, tp_size,
                                 row_slice_plan, column_slice_plan, qkv_slice_plan,
                                 vocab_slice_plan)
            if tp is None:
                self._classes[name] = "repl"
                self._class_bytes["repl"] += nb
                continue
            entry, mark, kind = tp
            self._classes[name] = "tp"
            self._class_bytes["tp"] += nb
            if not self._tp_slice:
                continue
            plan[name] = entry
            setattr(mark, "is_sharded_weight", True)  # loader skips its narrow (idempotent)
            if kind == "row":
                n_row += 1
            elif kind == "col":
                n_col += 1
            elif kind == "qkv":
                n_qkv += 1
            else:
                n_vocab += 1

        self._plan = plan
        if self._is_tp_rank0() or self._ep_plan:
            # Rank-0 prints the TP plan; EVERY rank prints its EP plan line, because
            # co-axiality (§3.7) is per-rank (60 experts / 8 ranks: ranks 0-3 coincide,
            # 4-7 do not) and only the rank itself knows its expert_map.
            if self._is_tp_rank0():
                print(
                    f"[WT-CXL] P3A tp-slice plan: {len(plan)}/{len(self._owner_names)} "
                    f"tensors sliced ({n_row} row-parallel + {n_col} merged-column + "
                    f"{n_qkv} qkv + {n_vocab} vocab), rest full-read "
                    f"(rank {tp_rank}/{tp_size}); classes: "
                    f"tp={sum(1 for c in self._classes.values() if c == 'tp')} "
                    f"expert={n_expert} repl={sum(1 for c in self._classes.values() if c == 'repl')}",
                    flush=True,
                )
            if n_expert:
                runs0 = next(iter(self._ep_plan.values())).runs if self._ep_plan else []
                print(
                    f"[WT-CXL] P7 ep-slice plan (mode={self._ep_slice}) rank tp{tp_rank} "
                    f"ep{self._world['ep_rank']}/{self._world['ep']} dp={self._world['dp']}: "
                    f"{len(self._ep_plan)}/{n_expert} expert tensors sliced, refused={ep_refused or '-'}, "
                    f"local_experts={next(iter(self._ep_plan.values())).n_local if self._ep_plan else '-'} "
                    f"runs={[(r.global_start, r.local_start, r.count) for r in runs0]} "
                    f"coaxial(trainer_world={trainer_world})={self._coaxial}",
                    flush=True,
                )

    @staticmethod
    def _ep_target(name, spec, params, modules):
        """If store owner ``name`` is a fused 3D routed-expert param of a FusedMoE
        layer on this model, return ``(kind, module, param, param_name)`` with
        ``kind`` ∈ {``w13``, ``w2``}; else ``None``. Duck-typed on the module having a
        ``weight_loader`` and an ``_expert_map`` attribute (FusedMoE); no
        ``isinstance`` coupling."""
        if len(spec.shape) != 3:
            return None
        if name.endswith(".mlp.experts.gate_up_proj"):
            kind, pfx = "w13", "w13_weight"
        elif name.endswith(".mlp.experts.down_proj"):
            kind, pfx = "w2", "w2_weight"
        else:
            return None
        mod_name = name.rsplit(".", 1)[0]
        module = modules.get(mod_name)
        if module is None or not hasattr(module, "weight_loader") or not hasattr(module, "_expert_map"):
            return None
        pname = f"{mod_name}.{pfx}"
        param = params.get(pname)
        if param is None:
            return None
        return kind, module, param, pname

    def _tp_target(self, name, spec, params, modules, tp_rank, tp_size,
                   row_slice_plan, column_slice_plan, qkv_slice_plan, vocab_slice_plan):
        """P3A classification, unchanged in logic from iter 9-12: return
        ``(plan_entry, param_to_mark, kind)`` for a TP-sliceable owner, else ``None``.
        ``kind`` ∈ {row, col, qkv, vocab}. Pure classification — the caller decides
        whether to apply it (``is_sharded_weight`` is set only when TP slicing is on)."""
        if tp_size <= 1:
            return None  # TP=1: no over-read, nothing to slice
        p = params.get(name)
        if p is not None:
            # P3A-4: vocab-parallel (embed_tokens/lm_head) — intercepted BEFORE the row
            # branch; slice = [org_vocab_start_index, org_vocab_end_index) off the live
            # module's shard_indices, NOT an even shard (p3a4 §3.2).
            if self._sliceable(p):
                vgeom = self._vocab_target(name, p, modules)
                if vgeom is not None:
                    ventry = vocab_slice_plan(*vgeom)
                    if ventry is not None:
                        return ventry, p, "vocab"
            # P3A-1: direct model param → pure row-parallel (o_proj/down_proj) by shape
            # delta; must be a parallel linear weight the loader narrows (input_dim).
            if not self._sliceable(p) or getattr(p, "input_dim", None) is None:
                return None
            entry = row_slice_plan(tuple(spec.shape), tuple(p.shape), tp_rank, tp_size)
            if entry is None:
                return None  # not an evenly-sharded 2D row weight → full read
            return entry, p, "row"

        # P3A-2: unfused merged-column constituent (gate_proj/up_proj) → fused gate_up_proj.
        fused = self._merged_column_target(self._model_for_mapping, name, params)
        if fused is not None:
            if not self._sliceable(fused) or getattr(fused, "output_dim", None) is None:
                return None
            entry = column_slice_plan(tuple(spec.shape), tp_rank, tp_size)
            if entry is None:
                return None
            return entry, fused, "col"

        # P3A-3: unfused q/k/v → fused qkv_proj + its QKVParallelLinear module (GQA rule).
        qkv = self._qkv_target(self._model_for_mapping, name, params, modules)
        if qkv is None:
            return None
        fused_p, shard_id, qkv_mod = qkv
        if not self._sliceable(fused_p) or getattr(fused_p, "output_dim", None) is None:
            return None
        if not hasattr(qkv_mod, "num_kv_head_replicas"):
            return None  # not a QKVParallelLinear-like module → full read (P3D)
        entry = qkv_slice_plan(
            shard_id, tp_rank, tp_size,
            qkv_mod.num_heads, qkv_mod.num_kv_heads, qkv_mod.head_size,
            getattr(qkv_mod, "v_head_size", qkv_mod.head_size),
            qkv_mod.num_kv_head_replicas,
        )
        if entry is None:
            return None
        return entry, fused_p, "qkv"

    @staticmethod
    def _sliceable(p) -> bool:
        from vllm.distributed.weight_transfer.sparse_plan import sliceable

        return sliceable(p)

    @staticmethod
    def _merged_column_target(model, name, params):
        """If store owner ``name`` is an unfused constituent of the merged-COLUMN fused
        param ``gate_up_proj`` (2 plain column shards, no GQA replicas), return the
        fused model param; else ``None``.

        ``qkv_proj`` is deliberately EXCLUDED — its k/v shards use the GQA
        ``num_kv_head_replicas`` rule (P3A-3), not an even column split. Keyed on the
        literal ``gate_up_proj`` mapping (the campaign Qwen2/Llama name); a model that
        does not expose it simply returns ``None`` → unchanged full read."""
        mapping = getattr(model, "packed_modules_mapping", None) or {}
        constituents = mapping.get("gate_up_proj")
        if not constituents:
            return None
        for c in constituents:
            needle = f".{c}."
            if needle in name:
                return params.get(name.replace(needle, ".gate_up_proj."))
        return None

    @staticmethod
    def _qkv_target(model, name, params, modules):
        """If store owner ``name`` is an unfused q/k/v constituent of a fused
        ``qkv_proj``, return ``(fused_param, shard_id, qkv_module)``; else ``None``.

        ``shard_id`` ∈ {``q``,``k``,``v``} by position in
        ``packed_modules_mapping['qkv_proj']``. The **module** is returned (not just the
        param) because the GQA geometry (``num_kv_head_replicas``, per-rank
        ``num_heads``/``num_kv_heads``, ``head_size``) that P3A-3 needs lives on the
        ``QKVParallelLinear`` module, not on the param."""
        mapping = getattr(model, "packed_modules_mapping", None) or {}
        constituents = mapping.get("qkv_proj")
        if not constituents:
            return None
        shard_ids = ("q", "k", "v")
        for i, c in enumerate(constituents):
            needle = f".{c}."
            if needle in name and i < len(shard_ids):
                fused_name = name.replace(needle, ".qkv_proj.")
                fused_p = params.get(fused_name)
                if fused_p is None or not fused_name.endswith(".weight"):
                    return None
                module = modules.get(fused_name[: -len(".weight")])
                if module is None:
                    return None
                return fused_p, shard_ids[i], module
        return None

    @staticmethod
    def _vocab_target(name, param, modules):
        from vllm.distributed.weight_transfer.sparse_plan import vocab_target

        return vocab_target(name, param, modules)

    @staticmethod
    def _is_tp_rank0() -> bool:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank() == 0
        except Exception:
            return True

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        self._pinned.clear()

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError(
            "CXL trainer writes go through the shared_weight_store FSDP writer "
            "(driven by veRL's cxl checkpoint engine), not this vLLM hook."
        )
