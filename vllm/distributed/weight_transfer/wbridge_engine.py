# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reader of Meta's WeightBridge baseline (veRL backend ``wbridge``): a ``ReceiverAdapter`` on the live model.

The first receive builds the adapter, which probes this worker's own ``model.load_weights`` for the plan (WeightBridge
learns the TP layout from the loader), then each receive polls until the trainer's update has landed in the parameters.
Under ``WT_WB_C2`` (the ``v`` arm, never timed) the base reader's resident check compares those parameters with the
trainer's fp32 weights in the pool store, bitwise, on this worker's slice.
"""

import os
import time
from dataclasses import dataclass, field

import torch

from vllm.distributed.weight_transfer.base import WeightTransferInitInfo
from vllm.distributed.weight_transfer.cxl_engine import (
    CXLWeightTransferEngine,
    CXLWeightTransferUpdateInfo,
)


@dataclass
class WBridgeWeightTransferInitInfo(WeightTransferInitInfo):
    wire_shapes: dict = field(default_factory=dict)  # the shared wire format (shared_weight_store.wb_adapter)
    ipc: str = ""  # this replica's coordinator
    ip: str = ""  # this host's fabric address
    oracle_uri: str | None = None  # WT_WB_C2 only: the pool store the check reads


class WBridgeWeightTransferEngine(CXLWeightTransferEngine):
    init_info_cls = WBridgeWeightTransferInitInfo
    update_info_cls = CXLWeightTransferUpdateInfo

    def init_transfer_engine(self, init_info: WBridgeWeightTransferInitInfo) -> None:
        self._wb_info = init_info
        self._store_uri = init_info.oracle_uri or "wbridge"  # the base reader opens it only for the check
        self._receiver = None

    def _build_receiver(self, model) -> None:
        from shared_weight_store import wb_adapter
        from wbridge.frontend.adapters import AdapterContext, ReceiverAdapter

        info = self._wb_info
        wb_adapter.pin(info.ip)
        shapes = {name: tuple(shape) for name, shape in info.wire_shapes.items()}
        ctx = AdapterContext(
            hf_weights=wb_adapter.sources(shapes),
            hf_shapes=shapes,
            wksd_factory=lambda: {name: p.data for name, p in model.named_parameters()},
            # A generator: the loader takes one tensor at a time, and a whole model would not fit in one list.
            load_weights=lambda fetcher: model.load_weights((n, fetch()) for n, fetch in fetcher.items()),
            rank=torch.distributed.get_rank(),
        )
        self._receiver = ReceiverAdapter(ctx, info.ipc, num_workers=torch.distributed.get_world_size())
        if self._is_tp_rank0():
            print(f"[WT-WB] receiver up: {len(shapes)} wire tensors, ipc={info.ipc}", flush=True)

    def receive_weights(self, update_info: CXLWeightTransferUpdateInfo, load_weights) -> None:
        model = load_weights.model  # gpu_worker: the kernel-format path gives the loader this handle
        if self._receiver is None:
            self._build_receiver(model)
        while not self._receiver.poll_requests():  # connect() and empty polls return False
            time.sleep(1e-4)
        self._receiver.flush_profile_outputs()
        if os.environ.get("WT_WB_C2") == "1":
            self._model = model
            self._ensure_open()
            self._store.wait_committed(min_version=update_info.version, timeout_s=600.0)
            self._c2_resident_check(self._store.active_slot, update_info.version, torch.device("cuda"), "C2-WB")
