# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AReaL-DTE receiver in the vLLM worker: the delta-transfer baseline behind veRL's ``dte`` engine.

DTE (areal-project/AReaL-DTE, arXiv 2608.00455) finds the changed bf16 elements on the trainer after
the optimizer step, remaps them on the SENDER into each receiver's own parameter layout, and delivers
them point-to-point from every trainer rank to every inference rank in two rounds: one int32 count per
operation, then one coalesced (int32 index, bf16 value) pair per peer. This engine is the receiving
end inside each vLLM TP worker; the sender is veRL's ``verl/checkpoint_engine/dte_checkpoint_engine.py``.

Placement is ``sparse_plan.build_sparse_plan``, the rule the P5 engines use (fused qkv / gate_up
offsets, TP row and column slices, padded vocab rows). Its ``DstPlan`` per tensor is sent to the
trainers once; each trainer rank turns it into one DTE operation per tensor whose rows it holds.

Group: trainer ranks [0, T) and every vLLM worker of every replica [T, T + R) in a torch
ProcessGroupNCCL over a TCPStore hosted by trainer rank 0; worker rank = ``rank_offset`` + TP rank.
Every message is point-to-point and each side walks the other in increasing rank order.

* version 1 (``is_checkpoint_format=True``, the dense anchor): from trainer 0, int64 length + the
  pickled manifest ``[(name, shape, [(row0, rows) per trainer rank])]``; then, per tensor, its rows
  from each trainer rank, loaded through ``model.load_weights``.
* first delta: this worker sends its geometry, ``(r0, r1, c0, c1, cols, doff, dcols, drows)`` per
  tensor, to every trainer rank (int64 length + pickled list).
* version v > 1 (``is_checkpoint_format=False``): from each trainer rank, int32 counts (one per
  operation), then, if any is nonzero, the int32 indices and the bf16 values, scattered in place
  with ``dte.core.apply_sparse_patch_``.
* ``VERL_DTE_VERIFY=1``, v > 1: the trainers then send the rows of the replica they expect, and this
  worker compares its live parameters with them bitwise under the plan.
"""

import os
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DTEInitInfo(WeightTransferInitInfo):
    master_address: str
    master_port: int
    trainer_world: int
    world_size: int
    rank_offset: int  # group rank of this replica's TP rank 0


@dataclass
class DTEUpdateInfo(WeightTransferUpdateInfo):
    version: int = 0


class DTEWeightTransferEngine(WeightTransferEngine[DTEInitInfo, DTEUpdateInfo]):
    init_info_cls = DTEInitInfo
    update_info_cls = DTEUpdateInfo

    def __init__(self, config, parallel_config) -> None:
        super().__init__(config, parallel_config)
        self._pg = None
        self.rank = self.world = self.trainer_world = None
        self._tp_rank = 0
        self._manifest = None  # [(name, shape, [(row0, rows) per trainer rank])]
        self._model = None
        self._plans = None  # DstPlan per tensor, manifest order, built at the first delta
        self._ops = None  # per trainer rank: tensor indices it sends to this worker, in its order
        self._last_version = 0
        self._verify = os.getenv("VERL_DTE_VERIFY", "0") == "1"

    # -- helpers ------------------------------------------------------------------------------

    @staticmethod
    def _dev() -> torch.device:
        return torch.device("cuda", torch.cuda.current_device())

    def _send(self, t: torch.Tensor, dst: int) -> None:
        self._pg.send([t], dst, 0).wait()

    def _recv(self, t: torch.Tensor, src: int) -> None:
        self._pg.recv([t], src, 0).wait()

    def _recv_rows(self, shape, parts) -> torch.Tensor:
        full = torch.empty(tuple(shape), dtype=torch.bfloat16, device=self._dev())
        for r, (row0, rows) in enumerate(parts):
            if rows:
                self._recv(full[row0:row0 + rows], r)
        return full

    # -- engine API -----------------------------------------------------------------------------

    def init_transfer_engine(self, init_info: DTEInitInfo) -> None:
        from vllm.distributed.weight_transfer.sparse_plan import tp_rank_size

        self._tp_rank, _ = tp_rank_size()
        self.rank = init_info.rank_offset + self._tp_rank
        self.world, self.trainer_world = init_info.world_size, init_info.trainer_world
        t0 = time.perf_counter()
        store = dist.TCPStore(init_info.master_address, init_info.master_port, self.world, is_master=False,
                              timeout=timedelta(seconds=900), wait_for_workers=False)
        self._pg = dist.ProcessGroupNCCL(store, self.rank, self.world)
        probe = torch.zeros(1, dtype=torch.int32, device=self._dev())
        for src in range(self.trainer_world):  # the trainers' probe: creates every pair communicator now
            self._recv(probe, src)
        torch.cuda.synchronize()
        print(f"[WT-DTE-VLLM] rank={self.rank} (tp {self._tp_rank}) joined {self.trainer_world} trainers, "
              f"world {self.world}, in {time.perf_counter() - t0:.2f}s", flush=True)

    def receive_weights(
        self,
        update_info: DTEUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self._pg is None:
            raise RuntimeError("dte: init_transfer_engine() was not called")
        model = getattr(load_weights, "model", None) or getattr(load_weights, "__self__", None)
        if model is not None:
            self._model = model
        version = update_info.version
        t0 = time.perf_counter()
        if update_info.is_checkpoint_format:
            self._receive_dense(load_weights)
            torch.cuda.synchronize()
            print(f"[WT-DTE-VLLM] rank={self.rank} v{version} dense anchor: {len(self._manifest)} tensors "
                  f"loaded in {time.perf_counter() - t0:.3f}s", flush=True)
            self._last_version = version
            return
        if version != self._last_version + 1:
            raise RuntimeError(f"dte: version gap {self._last_version} -> {version}; a delta applies only to "
                               "the version right before it")
        t_geo = 0.0
        if self._ops is None:
            self._send_geometry()
            t_geo = time.perf_counter() - t0
        t1 = time.perf_counter()
        applied = self._receive_delta()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self._last_version = version
        msg = (f"[WT-DTE-VLLM] rank={self.rank} v{version} applied={applied} recv_apply_s={t2 - t1:.4f}"
               + (f" geometry_s={t_geo:.3f}" if t_geo else ""))
        if self._verify:
            bad, owners_bad = self._verify_expected()
            msg += f" | verify: mismatched={bad} tensors_bad={owners_bad}"
        print(msg, flush=True)

    def shutdown(self) -> None:
        self._pg = None

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        raise NotImplementedError("dte: the sender is veRL's dte checkpoint engine (it needs the optimizer state)")

    # -- protocol -------------------------------------------------------------------------------

    def _receive_dense(self, load_weights) -> None:
        dev = self._dev()
        size = torch.empty(1, dtype=torch.int64, device=dev)
        self._recv(size, 0)
        blob = torch.empty(int(size.item()), dtype=torch.uint8, device=dev)
        self._recv(blob, 0)
        self._manifest = pickle.loads(blob.cpu().numpy().tobytes())
        for name, shape, parts in self._manifest:
            full = self._recv_rows(shape, parts)
            load_weights([(name, full)])
            del full

    def _send_geometry(self) -> None:
        """Build the placement from the live model (after the anchor's reload) and give it to every trainer."""
        from vllm.distributed.weight_transfer.sparse_plan import build_sparse_plan

        if self._manifest is None or self._model is None:
            raise RuntimeError("dte: a delta before the dense anchor (no manifest or model on this worker)")
        owners = [SimpleNamespace(name=name, shape=tuple(shape)) for name, shape, _ in self._manifest]
        self._plans = build_sparse_plan(self._model, owners, tag="DTE")
        geometry = [(p.r0, p.r1, p.c0, p.c1, p.cols, p.doff, p.dcols, p.dst.numel() // p.dcols) for p in self._plans]
        dev = self._dev()
        blob = torch.frombuffer(bytearray(pickle.dumps(geometry)), dtype=torch.uint8).to(dev)
        size = torch.tensor([blob.numel()], dtype=torch.int64, device=dev)
        for t in range(self.trainer_world):
            self._send(size, t)
            self._send(blob, t)
        # the trainer rank r sends one operation per tensor whose rows it holds inside this worker's slice
        self._ops = []
        for r in range(self.trainer_world):
            ops = []
            for k, (_, _, parts) in enumerate(self._manifest):
                row0, rows = parts[r]
                p = self._plans[k]
                if max(p.r0, row0) < min(p.r1, row0 + rows) and p.c0 < p.c1:
                    ops.append(k)
            self._ops.append(ops)

    def _receive_delta(self) -> int:
        dev = self._dev()
        applied = 0
        for r in range(self.trainer_world):
            ops = self._ops[r]
            if not ops:
                continue
            counts = torch.empty(len(ops), dtype=torch.int32, device=dev)
            self._recv(counts, r)  # round 1
            sizes = counts.tolist()
            total = sum(sizes)
            if not total:
                continue
            idx = torch.empty(total, dtype=torch.int32, device=dev)
            vals = torch.empty(total, dtype=torch.bfloat16, device=dev)
            self._recv(idx, r)  # round 2
            self._recv(vals, r)
            from dte.core import apply_sparse_patch_

            for k, i_s, v_s, n in zip(ops, idx.split(sizes), vals.split(sizes), sizes):
                if n:
                    apply_sparse_patch_(self._plans[k].dst.view(-1), i_s, v_s)
            applied += total
        return applied

    def _verify_expected(self) -> tuple[int, int]:
        from vllm.distributed.weight_transfer.sparse_plan import c2_compare, plan_slice

        bad = tensors_bad = 0
        for k, (_, shape, parts) in enumerate(self._manifest):
            full = self._recv_rows(shape, parts)
            p = self._plans[k]
            if p.r0 < p.r1 and p.c0 < p.c1:
                n_bad, _ = c2_compare(p, plan_slice(p, full))
                bad += n_bad
                tensors_bad += bool(n_bad)
            del full
        return bad, tensors_bad
