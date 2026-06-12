# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CXL / shared-canonical-store weight transfer engine.

P2 of the wt_cxl canonical-store curriculum. Each vLLM TP worker reads the
committed model version DIRECTLY from a shared canonical weight store (an
mmap-backed `shared_weight_store` library, on `/tmp` today, a CXL DAX device
later) and feeds the tensors into ``model.load_weights`` itself — eliminating
veRL's ``BucketedWeightSender`` leader funnel (the leader reading the full model
and re-distributing it to TP workers over ZMQ/IPC, serialized one bucket at a
time, was the entire 7B read cost: G2=3.04s of D4d=3.12s, while the load itself
is 0.1s).

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
        self._pinned: dict[str, torch.Tensor] = {}

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

        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        while True:
            try:
                self._store = CanonicalWeightStore.open(self._store_uri, readonly=True)
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

    def receive_weights(
        self,
        update_info: CXLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Read this version's canonical tensors from the shared store and load
        them incrementally. Each worker reads independently (the store mmap pages
        are shared → page-cache hits, not N physical reads); no leader funnel.

        ``load_weights`` is vLLM's ``model.load_weights`` (checkpoint format): it
        applies fused-QKV/gate-up remapping and narrows each tensor to this TP
        rank, so a full canonical tensor in → this rank's shard loaded."""
        if self._store_uri is None:
            raise RuntimeError("init_transfer_engine was not called before receive")
        self._ensure_open()

        version = update_info.version
        self._store.wait_committed(min_version=version, timeout_s=_WAIT_TIMEOUT_S)

        use_cuda = torch.cuda.is_available()
        c2 = os.environ.get("WT_CXL_C2", "0") == "1"  # anti-cheat digest oracle
        digest = 0
        total_bytes = 0
        t0 = time.perf_counter()
        for name in self._owner_names:
            buf = self._pinned.get(name)
            if buf is None:
                spec = next(t for t in self._store.manifest.tensors if t.name == name)
                buf = torch.empty(spec.shape, dtype=spec.dtype, pin_memory=use_cuda)
                self._pinned[name] = buf
            self._store.read_tensor_into(version, name, buf)
            if c2:
                from shared_weight_store.digest import xor_digest_update
                digest = xor_digest_update(digest, name, buf)
            tensor = buf.to("cuda", non_blocking=True) if use_cuda else buf
            # Incremental load (one tensor) keeps peak host/device memory bounded
            # and matches the layerwise-reload contract (tensors arrive in manifest
            # = checkpoint order, so each layer's set completes in order).
            load_weights([(name, tensor)])
            total_bytes += buf.nbytes
        if use_cuda:
            torch.cuda.synchronize()
        if c2 and self._is_tp_rank0():
            # Mirrors the trainer's send digest (over full_tensor()); equal digests
            # prove the per-worker store read reconstructed the canonical weights
            # bitwise. One print from TP rank 0 (every worker reads the full model).
            print(
                f"[WT-CXL-C2] role=recv version={version} "
                f"count={len(self._owner_names)} digest={digest:08x}",
                flush=True,
            )
        logger.info(
            "CXL receive_weights v%d: %.2f GB from %d tensors in %.2fs",
            version,
            total_bytes / 1e9,
            len(self._owner_names),
            time.perf_counter() - t0,
        )

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
