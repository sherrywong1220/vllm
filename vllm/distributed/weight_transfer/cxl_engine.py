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
        # P3E: batch the 339 per-tensor load_weights into ONE call over a lazy generator, so
        # qwen2.load_weights builds `params_dict = dict(named_parameters())` once, not 339x
        # (kills the O(N_tensors x N_modules) module-tree re-walk = 92% of the residual read
        # dispatch, per the P3B F-split). Orthogonal to the slice. See p3e_batch_load_design.md.
        self._batch_load = _on("WT_CXL_BATCH_LOAD", lambda v: v == "1", "ENABLE_BATCH_LOAD")
        # D4d attribution: split residual read cost F into GPU-copy vs Python-dispatch to
        # decide P3B vs batching the loads. Instrumented → attribution-only, never a verdict.
        self._read_prof = _on("WT_CXL_READ_CPROFILE", lambda v: v not in ("0", ""),
                              "ENABLE_READ_PROF")
        self._recv_idx = 0  # receive_weights call counter (0 = warmup, excluded like WSPHASE)

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
        self._owner_names = [
            t.name for t in self._store.manifest.tensors if t.alias_of is None
        ]
        self._specs = {t.name: t for t in self._store.manifest.tensors}  # full shapes (P3A)
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
                    f"tp_slice={self._tp_slice} batch_load={self._batch_load} "
                    f"read_prof={self._read_prof} "
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
        self._store.wait_committed(min_version=version, timeout_s=_WAIT_TIMEOUT_S)

        # P3A: build the TP-slice copy plan once. load_weights is model.load_weights
        # (a bound method) on the checkpoint-format path, so __self__ is the live
        # vLLM model we introspect for shard geometry + set is_sharded_weight on.
        if self._tp_slice and self._plan is None:
            self._build_plan(getattr(load_weights, "__self__", None))
        plan = self._plan

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

        t0 = time.perf_counter()
        if self._batch_load:
            # P3E: ONE load_weights call over a lazy generator. qwen2.load_weights builds
            # params_dict = dict(named_parameters()) ONCE (not per tensor), killing the
            # O(N_tensors x N_modules) re-walk. The generator yields one tensor at a time and
            # its local ref drops at each yield, so peak alive H2D memory stays one layer's
            # worth (the layerwise loader frees each layer on completion) — same bound the
            # per-tensor path kept, WITHOUT holding all 339 tensors. Split instrumentation is
            # inherently per-tensor (skipped here); the coarse timing + cProfile still apply.
            _stats = {"bytes": 0, "digest": 0}

            def _weight_gen():
                for name in self._owner_names:
                    tensor, nbytes, dsrc = self._read_one(
                        name, plan, slot, use_cuda, version, c2
                    )
                    _stats["bytes"] += nbytes
                    if c2:
                        _stats["digest"] = xor_digest_update(
                            _stats["digest"], name, dsrc
                        )
                    yield (name, tensor)

            load_weights(_weight_gen())
            total_bytes = _stats["bytes"]
            digest = _stats["digest"]
        else:
            # Incremental per-tensor load (default). Tensors arrive in manifest = checkpoint
            # order, so each layer's set completes in order (layerwise-reload contract) and
            # peak host/device memory stays bounded to one layer.
            for name in self._owner_names:
                _ts = time.perf_counter() if _do_split else 0.0
                tensor, nbytes, dsrc = self._read_one(
                    name, plan, slot, use_cuda, version, c2
                )
                if c2:
                    digest = xor_digest_update(digest, name, dsrc)
                total_bytes += nbytes
                if _do_split:
                    torch.cuda.synchronize()  # H2D (staging) GPU done
                    _t1 = time.perf_counter()
                load_weights([(name, tensor)])
                if _do_split:
                    _t2 = time.perf_counter()  # load DISPATCH done (Python; async copy_ launched)
                    torch.cuda.synchronize()   # copy_ GPU execution done
                    _t3 = time.perf_counter()
                    _s_stage += _t1 - _ts      # H2D dispatch + GPU (≈ B, per-tensor)
                    _s_ldisp += _t2 - _t1      # load_weights Python dispatch
                    _s_copygpu += _t3 - _t2    # param_data.copy_ GPU execution (d2d placement)
        if _instr:
            _t_pre = time.perf_counter()
        if use_cuda:
            torch.cuda.synchronize()
        if _instr:
            _t_post = time.perf_counter()
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
                f"batch={'Y' if self._batch_load else 'N'} "
                f"split={'Y' if _do_split else 'N'} cprof={'Y' if _do_cprof else 'N'}")
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
            role = "recv-sliced" if plan else "recv"
            print(
                f"[WT-CXL-C2] role={role} version={version} "
                f"count={len(self._owner_names)} digest={digest:08x}",
                flush=True,
            )
        logger.info(
            "CXL receive_weights v%d: %.2f GB from %d tensors in %.2fs%s",
            version,
            total_bytes / 1e9,
            len(self._owner_names),
            time.perf_counter() - t0,
            " [batched]" if self._batch_load else "",
        )

    def _read_one(self, name, plan, slot, use_cuda, version, c2):
        """Prepare one store tensor for ``load_weights``: the P3A slice, full one-hop
        GPU-direct, two-hop pinned fallback, or CPU path — whichever applies. Returns
        ``(tensor, nbytes, digest_src)`` where ``digest_src`` is the CPU-side bytes to fold
        into the C2 digest (``None`` when ``c2`` is off, to skip the extra contiguous copy).

        Shared verbatim by the per-tensor loop and the P3E batched generator so the read
        semantics (bytes, H2D method, ordering) are one source of truth — batching changes
        only how many times ``load_weights`` is entered, never what a tensor is read as."""
        entry = plan.get(name) if plan else None
        if entry is not None:
            # P3A reuse-path: read ONLY this rank's slice. The loader skips its narrow
            # (is_sharded_weight) but keeps fused placement + quant repack.
            dim, start, size = entry
            sl = self._store.region_view(slot, name).narrow(dim, start, size)
            dsrc = sl.contiguous() if c2 else None
            if use_cuda and self._cuda_direct:
                # Registered (pinned) store region -> DMA straight in. Guarded H2D: strided
                # (row) slice -> cudaMemcpy2DAsync (pitched, 5-20x over torch strided copy_);
                # contiguous (column/vocab/qkv) or dtype-mismatch -> copy_. See slice_h2d.
                tensor = torch.empty(tuple(sl.shape), dtype=sl.dtype, device="cuda")
                self._slice_h2d(sl, tensor)
            elif use_cuda:
                # gpu_direct=False node: a direct H2D from the non-pinned file-backed mmap can
                # fail, so stage through a pinned CPU buffer first, then H2D (two-hop).
                staged = torch.empty(
                    tuple(sl.shape), dtype=sl.dtype, device="cpu", pin_memory=True
                )
                staged.copy_(sl)
                tensor = staged.to("cuda", non_blocking=True)
            else:
                tensor = sl.contiguous()
            return tensor, sl.numel() * sl.element_size(), dsrc
        if self._cuda_direct:
            # One-hop DMA: registered store region -> fresh cuda tensor, same (default)
            # stream as load_weights so the non_blocking copy is ordered before the load reads.
            region = self._store.region_view(slot, name)
            tensor = torch.empty(region.shape, dtype=region.dtype, device="cuda")
            tensor.copy_(region, non_blocking=True)
            return tensor, region.nbytes, (region if c2 else None)
        if use_cuda:
            # Two-hop fallback (iter-7): store -> reused pinned CPU buffer -> H2D.
            buf = self._pinned.get(name)
            if buf is None:
                spec = next(t for t in self._store.manifest.tensors if t.name == name)
                buf = torch.empty(spec.shape, dtype=spec.dtype, device="cpu", pin_memory=True)
                self._pinned[name] = buf
            self._store.read_tensor_into(version, name, buf)
            tensor = buf.to("cuda", non_blocking=True)
            return tensor, buf.nbytes, (buf if c2 else None)
        # CPU-only (C1 / no CUDA): hand the store view straight to load.
        region = self._store.region_view(slot, name)
        return region, region.nbytes, (region if c2 else None)

    def _build_plan(self, model) -> None:
        """Classify each store owner tensor and build the per-rank slice plan.

        Two classes are sliced (each ``is_sharded_weight``-marked so the loader skips
        only its per-rank narrow but keeps fused placement + quant repack — C1b-verified):

        * **P3A-1 pure row-parallel** (``o_proj``/``down_proj``): a direct model param,
          sharded on ``input_dim``, classified by shape delta → strided dim-1 slice.
        * **P3A-2 merged-column** (``gate_up_proj``): the store owner is the *unfused*
          constituent (``gate_proj``/``up_proj``), NOT a model param, so it is mapped to
          the fused param via ``packed_modules_mapping`` → contiguous dim-0 column slice.
        * **P3A-3 fused-QKV** (``qkv_proj``): unfused ``q_proj``/``k_proj``/``v_proj``
          mapped to the fused param + its ``QKVParallelLinear`` module → contiguous dim-0
          slice with the **GQA ``num_kv_head_replicas``** rule for k/v (read off the module).
        * **P3A-4 vocab-parallel** (``embed_tokens``/``lm_head``): a direct
          ``VocabParallelEmbedding`` param → contiguous dim-0 slice on the **padded** vocab
          dim, using ``[org_vocab_start_index, org_vocab_end_index)`` read off the module's
          ``shard_indices`` (NOT an even shard — vocab padding; the extended loader keeps
          the partition zero-fill). C1b-verified incl. a padded / zero-org-row rank.

        Everything else — quantized/packed, or any param we cannot classify — gets NO plan
        entry and rides the unchanged P2 full read (P3D coverage). So the reader never
        silently mis-slices.
        """
        from shared_weight_store.slice_plan import (
            column_slice_plan,
            qkv_slice_plan,
            row_slice_plan,
            vocab_slice_plan,
        )

        if model is None:
            # Non-checkpoint-format caller (no bound model) — never slice.
            self._plan = {}
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

        params = dict(model.named_parameters())
        # P3A-3 needs the QKVParallelLinear module (GQA geometry lives on the module,
        # not the param); built once, plan is cached after.
        modules = dict(model.named_modules())
        plan: dict = {}
        n_row = n_col = n_qkv = n_vocab = 0
        for name in self._owner_names:
            if tp_size <= 1:
                continue  # TP=1: no over-read, nothing to slice
            spec = self._specs.get(name)
            if spec is None:
                continue

            p = params.get(name)
            if p is not None:
                # P3A-4: vocab-parallel (embed_tokens/lm_head) — a direct model param
                # sharded on the padded vocab dim by VocabParallelEmbedding. Intercepted
                # BEFORE the row branch: the slice is [org_vocab_start_index,
                # org_vocab_end_index) read off the live module's shard_indices, NOT an
                # even shard (vocab padding; see p3a4_vocab_slice_design.md §3.2). The
                # loader (extended to honour is_sharded_weight) keeps the padding
                # zero-fill; is_sharded_weight only skips its per-rank narrow.
                if self._sliceable(p):
                    vgeom = self._vocab_target(name, p, modules)
                    if vgeom is not None:
                        ventry = vocab_slice_plan(*vgeom)
                        if ventry is not None:
                            plan[name] = ventry
                            setattr(p, "is_sharded_weight", True)
                            n_vocab += 1
                            continue
                # P3A-1: direct model param → pure row-parallel (o_proj/down_proj),
                # classified by shape delta. Must be a parallel linear weight the
                # loader narrows + honours is_sharded_weight on (input_dim present on
                # every ModelWeightParameter); column/replicated → None → full.
                if not self._sliceable(p) or getattr(p, "input_dim", None) is None:
                    continue
                entry = row_slice_plan(
                    tuple(spec.shape), tuple(p.shape), tp_rank, tp_size
                )
                if entry is None:
                    continue  # not an evenly-sharded 2D row weight → full read
                plan[name] = entry
                setattr(p, "is_sharded_weight", True)  # loader skips its narrow
                n_row += 1
                continue

            # P3A-2: fused checkpoint name — the store owner is an UNFUSED constituent
            # (gate_proj/up_proj), not a model param. Map it to the fused merged-column
            # param (gate_up_proj) via packed_modules_mapping and take its CONTIGUOUS
            # dim-0 column slice. The loader still does fused PLACEMENT; is_sharded_weight
            # only skips its per-rank narrow (v2 load_merged_column_weight; C1b-verified).
            fused = self._merged_column_target(model, name, params)
            if fused is not None:
                if not self._sliceable(fused) or getattr(fused, "output_dim", None) is None:
                    continue
                entry = column_slice_plan(tuple(spec.shape), tp_rank, tp_size)
                if entry is None:
                    continue
                plan[name] = entry
                setattr(fused, "is_sharded_weight", True)  # on the fused param (idempotent)
                n_col += 1
                continue

            # P3A-3: fused-QKV — store owner is unfused q/k/v; map to the fused qkv_proj
            # param AND its QKVParallelLinear module (the GQA num_kv_head_replicas geometry
            # is on the module, not the param). Contiguous dim-0 slice with the GQA replica
            # rule (v2 load_qkv_weight keeps fused placement; C1b-verified incl. replicas=2).
            qkv = self._qkv_target(model, name, params, modules)
            if qkv is None:
                continue
            fused_p, shard_id, qkv_mod = qkv
            if not self._sliceable(fused_p) or getattr(fused_p, "output_dim", None) is None:
                continue
            if not hasattr(qkv_mod, "num_kv_head_replicas"):
                continue  # not a QKVParallelLinear-like module → full read (P3D)
            entry = qkv_slice_plan(
                shard_id, tp_rank, tp_size,
                qkv_mod.num_heads, qkv_mod.num_kv_heads, qkv_mod.head_size,
                getattr(qkv_mod, "v_head_size", qkv_mod.head_size),
                qkv_mod.num_kv_head_replicas,
            )
            if entry is None:
                continue
            plan[name] = entry
            setattr(fused_p, "is_sharded_weight", True)  # on the fused param (idempotent)
            n_qkv += 1

        self._plan = plan
        if self._is_tp_rank0():
            print(
                f"[WT-CXL] P3A tp-slice plan: {len(plan)}/{len(self._owner_names)} "
                f"tensors sliced ({n_row} row-parallel + {n_col} merged-column + "
                f"{n_qkv} qkv + {n_vocab} vocab), rest full-read "
                f"(rank {tp_rank}/{tp_size})",
                flush=True,
            )

    @staticmethod
    def _sliceable(p) -> bool:
        """A param is TP-sliceable only if unquantized/unpacked — quantized (GPTQ/AWQ/
        Marlin/FP8) and bitsandbytes params carry a kernel-tiled/packed layout whose
        shape differs from the logical one for reasons unrelated to TP, so they must go
        through the framework loader + ``process_weights_after_loading`` (P3D)."""
        return (
            not getattr(p, "use_bitsandbytes_4bit", False)
            and getattr(p, "packed_dim", None) is None
        )

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
        """If store owner ``name`` is a vocab-parallel weight (``embed_tokens``/
        ``lm_head`` on a ``VocabParallelEmbedding`` module), return the geometry
        ``(org_vocab_start_index, org_vocab_end_index, num_embeddings_per_partition)``
        for :func:`vocab_slice_plan`; else ``None`` (→ unchanged full read, P3D).

        The boundaries are read off the live module's ``shard_indices`` — the single
        source of truth for the **padded** vocab layout. They are NOT ``even_shard`` of
        the vocab: vLLM shards the padded dim then min's the org boundaries down, so a
        rank near the padding boundary owns fewer (or zero) org rows (see
        p3a4_vocab_slice_design.md §3.2). Duck-typed on ``shard_indices`` +
        ``num_embeddings_per_partition`` (no ``isinstance`` coupling); a module lacking
        them (e.g. a row/column linear) returns ``None`` and falls through."""
        if getattr(param, "output_dim", None) is None:
            return None  # replicated / no shardable output dim → full read
        if not name.endswith(".weight"):
            return None
        module = modules.get(name[: -len(".weight")])
        si = getattr(module, "shard_indices", None)
        part = getattr(module, "num_embeddings_per_partition", None)
        if si is None or part is None:
            return None  # not a VocabParallelEmbedding → full read (P3D)
        try:
            return (
                int(si.org_vocab_start_index),
                int(si.org_vocab_end_index),
                int(part),
            )
        except AttributeError:
            return None

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
