# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL transport for the P5 change set — the transport ablation of the CXL sparse path.

Same change set (``shared_weight_store.change_set.ChangeSetProducer``), same placement
(``sparse_plan.build_sparse_plan``) and same in-place apply (``ChangeSetReader.apply``) as the
CXL engine; only the wire differs. Here the trainer's rank 0 collects every trainer rank's
compacted change set (count exchange + padded gather, as upstream veRL's ``delta_sharded``
does) and broadcasts it to this replica's TP workers over the vLLM-side NCCL group
(``NCCLWeightTransferEngine``: trainer rank 0 + every worker, no checkpoint-engine-worker /
CUDA-IPC hop — i.e. the most direct NCCL route available, deliberately more favourable to
NCCL than veRL's shipped dense path).

Wire protocol (src = trainer rank 0, every message a broadcast on the group):

* version 1 (``is_checkpoint_format=True``, the dense anchor):
  ``int64[1]`` = length of the pickled manifest ``{"world": W, "tensors": [(name, shape, dtype)]}``;
  ``uint8[len]`` = that manifest; then one bf16 broadcast per tensor, loaded through
  ``model.load_weights`` (the ordinary vLLM NCCL dense path).
* version v > 1 (``is_checkpoint_format=False``):
  ``int64[2 + W]`` = ``[version, c2, nnz_0 … nnz_{W-1}]``;
  ``uint8[Σ_r (4·T_r + 4·nnz_r)]`` = the rank-major concatenation of ``pack_change_set`` (T_r is static
  geometry, so the receiver sizes the buffer from the header alone); if ``c2``: one dense bf16
  broadcast per owner afterwards, for the Tier-S bitwise oracle.
"""

import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.distributed.weight_transfer.base import WeightTransferUpdateInfo
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class NCCLSparseUpdateInfo(WeightTransferUpdateInfo):
    """The version this worker should consume; names/shapes travel on the wire (v1) or are
    static geometry (v > 1)."""

    version: int = 0


class NCCLSparseWeightTransferEngine(NCCLWeightTransferEngine):
    update_info_cls = NCCLSparseUpdateInfo

    def __init__(self, config, parallel_config) -> None:
        super().__init__(config, parallel_config)
        self._world: int | None = None  # trainer world (from the v1 manifest)
        self._owners: list | None = None  # TensorSpec per store owner, manifest order
        self._model = None  # the live vLLM model, cached on the dense call (load_weights.__self__)
        self._reader = None  # ChangeSetReader
        self._plans: list | None = None  # DstPlan per owner, built once from the live model
        self._last_version = 0
        self._recv_idx = 0  # sparse receive counter (0 = warmup, excluded like WSPHASE)

    # -- helpers ------------------------------------------------------------------------------

    def _dev(self) -> torch.device:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _bcast(self, t: torch.Tensor) -> torch.Tensor:
        self.model_update_group.broadcast(t, src=0, stream=torch.cuda.current_stream() if t.is_cuda else None)
        return t

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

    @staticmethod
    def _is_tp_rank0() -> bool:
        from vllm.distributed.weight_transfer.sparse_plan import is_tp_rank0

        return is_tp_rank0()

    # -- receive --------------------------------------------------------------------------------

    def receive_weights(
        self,
        update_info: NCCLSparseUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self.model_update_group is None:
            raise RuntimeError("NCCL sparse weight transfer not initialized. Call init_transfer_engine() first.")
        version = update_info.version
        if update_info.is_checkpoint_format:
            self._receive_dense(version, load_weights)
        else:
            self._receive_sparse(version)

    def _receive_dense(self, version: int, load_weights) -> None:
        """The anchor: manifest handshake, then the ordinary per-tensor NCCL dense load."""
        from shared_weight_store import TensorSpec

        model_obj = getattr(load_weights, "__self__", None)
        if model_obj is not None:
            self._model = model_obj
        dev = self._dev()
        n = self._bcast(torch.empty(1, dtype=torch.int64, device=dev))
        self._sync()
        buf = self._bcast(torch.empty(int(n[0]), dtype=torch.uint8, device=dev))
        self._sync()
        man = pickle.loads(buf.cpu().numpy().tobytes())
        self._world = int(man["world"])
        self._owners = [TensorSpec(name, tuple(shape), getattr(torch, dt)) for name, shape, dt in man["tensors"]]
        # the parent's one-by-one path, with the device taken from this engine (the parent hardcodes
        # "cuda", which keeps the protocol out of the CPU-only C1 harness)
        for o in self._owners:
            weight = torch.empty(tuple(o.shape), dtype=o.dtype, device=dev)
            self._bcast(weight)
            load_weights([(o.name, weight)])
            del weight
        self._last_version = version
        if self._is_tp_rank0():
            print(f"[WT-NCCLS] dense anchor v{version}: {len(self._owners)} tensors, trainer_world={self._world}",
                  flush=True)

    def _receive_sparse(self, version: int) -> None:
        """Apply version ``version``'s change sets (every trainer rank's) to the resident weights.
        Precondition: this worker holds the bf16 rendition of ``version - 1``."""
        from shared_weight_store.change_set import ChangeSetReader

        if self._owners is None or self._model is None:
            raise RuntimeError("NCCL sparse: sparse version before the dense anchor (no manifest on this worker)")
        if version != self._last_version + 1:
            raise RuntimeError(
                f"NCCL sparse: version gap {self._last_version} -> {version}: a change set applies only to the "
                "immediately preceding version (no ring, by design)"
            )
        _idx = self._recv_idx
        self._recv_idx += 1
        dev = self._dev()
        if self._reader is None:
            from vllm.distributed.weight_transfer.sparse_plan import build_sparse_plan

            self._reader = ChangeSetReader(self._owners, self._world)
            if self._plans is None:
                self._plans = build_sparse_plan(self._model, self._owners, tag="P5/nccl")
        W = self._world
        # 1. header: blocks until the trainer has compacted + gathered (the NCCL analogue of wait_committed)
        t0 = time.perf_counter()
        header = self._bcast(torch.empty(2 + W, dtype=torch.int64, device=dev))
        self._sync()
        t_hdr = time.perf_counter()
        hv = header.tolist()
        if hv[0] != version:
            raise RuntimeError(f"NCCL sparse: header version {hv[0]} != expected {version}")
        c2, ns = bool(hv[1]), hv[2:]
        tiles = self._reader.tiles_per_rank()
        nbytes = sum(4 * t + 4 * n for t, n in zip(tiles, ns))
        # 2. payload
        blob = self._bcast(torch.empty(nbytes, dtype=torch.uint8, device=dev))
        self._sync()
        t_blob = time.perf_counter()
        entries = self._reader.entries_from_blob(blob, ns, dev)
        self._sync()
        t_read = time.perf_counter()
        res = self._reader.apply(entries, self._plans)
        self._sync()
        t_end = time.perf_counter()
        self._last_version = version
        if self._is_tp_rank0():
            # Same keys as the cxl reader so run_bench's collector reports both arms alike:
            # wait = header broadcast (trainer-side compaction + gather), read_loop = payload + build + apply.
            print(f"[WSPHASE-NCCLS-READ idx={_idx}] D4d_wait_committed={t_hdr - t0:.3f}s "
                  f"D4d_read_loop={t_end - t_hdr:.3f}s read_gb={nbytes / 1e9:.6f}", flush=True)
            n_el = sum(self._reader.n_elems)
            print(f"[WSPHASE-NCCLS-SPARSE idx={_idx}] version={version} trainer_world={W} "
                  f"nnz={sum(ns)} nnz_per_rank={ns} density={sum(ns) / max(n_el, 1):.5f} applied={res['applied']} "
                  f"dropped={res['dropped']} owners_touched={res['owners']} "
                  f"read_s={t_read - t_hdr:.4f} apply_s={t_end - t_read:.4f} bytes={nbytes} "
                  f"bcast_payload_s={t_blob - t_hdr:.4f}", flush=True)
        if c2:
            self._c2_dense_compare(version, dev)

    def _c2_dense_compare(self, version: int, dev) -> None:
        """Tier-S oracle: the trainer re-sends every owner densely (bf16, the exact rendition a dense
        sync would ship); this rank's slice must equal the patched parameter, bitwise."""
        from vllm.distributed.weight_transfer.sparse_plan import c2_compare, plan_slice

        bad, checked, max_abs = [], 0, 0.0
        for o, plan in zip(self._owners, self._plans):
            full = self._bcast(torch.empty(tuple(o.shape), dtype=torch.bfloat16, device=dev))
            self._sync()
            if plan.dst is None:
                continue
            checked += 1
            n_bad, mx = c2_compare(plan, plan_slice(plan, full))
            if n_bad:
                max_abs = max(max_abs, mx)
                bad.append((o.name, n_bad))
        if self._is_tp_rank0() or bad:
            print(f"[WT-CXL-P5-C2] version={version} checked={checked} mismatched_params={len(bad)} "
                  f"max_abs_diff={max_abs:.3e} sample={bad[:4]} (nccl transport)", flush=True)
        if bad:
            raise RuntimeError(f"P5/nccl C2: sparse apply != dense send for {len(bad)} params at v{version}: {bad[:4]}")
