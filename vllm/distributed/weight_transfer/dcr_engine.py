# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NeMo-DCR receiver in the vLLM worker: the delta-compressed refit baseline behind veRL's ``dcr`` engine.

NeMo-DCR (arXiv 2610.08430, NVIDIA-NeMo/RL PR #2444) describes every update in Hugging Face canonical coordinates and
leaves placement to vLLM's own loader: a body's items are scattered into a full-size scratch and passed through
``model.load_weights`` under a dispatch mode that turns the loader's final copies into XOR or overwrite writes. This
engine runs NeMo's applier unchanged (``nemo_dcr.vllm_sparse_delta``, vendored from bbc5808); the sender is veRL's
``verl/checkpoint_engine/dcr_checkpoint_engine.py``.

Group: trainer ranks [0, T) and every vLLM worker of every replica [T, T + R) in a torch ProcessGroupNCCL over a
TCPStore hosted by trainer rank 0, as for the ``dte`` engine; worker rank = ``rank_offset`` + TP rank.

* version 1 (``is_checkpoint_format=True``, the dense anchor): the manifest ``[(name, shape, dtype, owner)]`` from
  trainer 0, then every tensor whole from its owner, loaded through ``load_weights``.
* first delta: NeMo's loader dry run (``discover_native_skips``) on the live model; the names whose loader path cannot
  take XOR go to every trainer rank, which encodes them as overwrites.
* version v > 1: from each trainer rank in turn, bodies framed as int64 length + 16-byte blake2b digest + body (zstd of
  torch.save), ended by a zero length; each body is checked, decompressed and applied as it arrives.
* ``VERL_DCR_VERIFY=1``: the trainers then send every tensor whole; this worker snapshots its parameters, loads them
  densely with ``model.load_weights`` and counts parameters whose bits differ (NeMo's dense-refit equivalence).
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
class DCRInitInfo(WeightTransferInitInfo):
    master_address: str
    master_port: int
    trainer_world: int
    world_size: int
    rank_offset: int  # group rank of this replica's TP rank 0


@dataclass
class DCRUpdateInfo(WeightTransferUpdateInfo):
    version: int = 0


class DCRWeightTransferEngine(WeightTransferEngine[DCRInitInfo, DCRUpdateInfo]):
    init_info_cls = DCRInitInfo
    update_info_cls = DCRUpdateInfo

    def __init__(self, config, parallel_config) -> None:
        super().__init__(config, parallel_config)
        self._pg = None
        self.rank = self.world = self.trainer_world = None
        self._tp_rank = 0
        self._manifest = None  # [(name, shape, dtype name, owner trainer rank)] in export order
        self._applier = None  # nemo_dcr.vllm_sparse_delta.VllmSparseDeltaApplier, built at the first delta
        self._last_version = 0
        self._verify = os.getenv("VERL_DCR_VERIFY", "0") == "1"

    # -- helpers ------------------------------------------------------------------------------

    @staticmethod
    def _dev() -> torch.device:
        return torch.device("cuda", torch.cuda.current_device())

    def _send(self, t: torch.Tensor, dst: int) -> None:
        self._pg.send([t], dst, 0).wait()

    def _recv(self, t: torch.Tensor, src: int) -> None:
        self._pg.recv([t], src, 0).wait()

    # -- engine API -----------------------------------------------------------------------------

    def init_transfer_engine(self, init_info: DCRInitInfo) -> None:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        self._tp_rank = get_tensor_model_parallel_rank()
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
        print(f"[WT-DCR-VLLM] rank={self.rank} (tp {self._tp_rank}) joined {self.trainer_world} trainers, "
              f"world {self.world}, in {time.perf_counter() - t0:.2f}s", flush=True)

    def receive_weights(
        self,
        update_info: DCRUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self._pg is None:
            raise RuntimeError("dcr: init_transfer_engine() was not called")
        model = getattr(load_weights, "model", None) or getattr(load_weights, "__self__", None)
        version = update_info.version
        t0 = time.perf_counter()
        if update_info.is_checkpoint_format:
            n = self._receive_dense(load_weights)
            torch.cuda.synchronize()
            print(f"[WT-DCR-VLLM] rank={self.rank} v{version} dense anchor: {n} tensors loaded in "
                  f"{time.perf_counter() - t0:.3f}s", flush=True)
            self._last_version = version
            return
        if version != self._last_version + 1:
            raise RuntimeError(f"dcr: version gap {self._last_version} -> {version}; a delta applies only to the "
                               "version right before it")
        t_dry = 0.0
        if self._applier is None:
            self._start_applier(model)
            t_dry = time.perf_counter() - t0
        t1 = time.perf_counter()
        stats = self._receive_delta()
        self._applier.finish_sparse_delta_refit()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self._last_version = version
        msg = (f"[WT-DCR-VLLM] rank={self.rank} v{version} bodies={stats['bodies']} wire_MB={stats['bytes'] / 1e6:.2f} "
               f"recv_s={stats['recv_s']:.4f} decompress_s={stats['decompress_s']:.4f} apply_s={stats['apply_s']:.4f} "
               f"recv_apply_s={t2 - t1:.4f}" + (f" dry_run_s={t_dry:.3f}" if t_dry else ""))
        if self._verify:
            bad, params_bad = self._verify_dense(self._applier.model_runner.model)
            msg += f" | verify: mismatched={bad} params_bad={params_bad}"
        print(msg, flush=True)

    def shutdown(self) -> None:
        self._pg = None

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        raise NotImplementedError("dcr: the sender is veRL's dcr checkpoint engine")

    # -- protocol -------------------------------------------------------------------------------

    def _receive_dense(self, load_weights) -> int:
        dev = self._dev()
        size = torch.empty(1, dtype=torch.int64, device=dev)
        self._recv(size, 0)
        blob = torch.empty(int(size.item()), dtype=torch.uint8, device=dev)
        self._recv(blob, 0)
        self._manifest = pickle.loads(blob.cpu().numpy().tobytes())
        for name, shape, dtype, owner in self._manifest:
            t = torch.empty(tuple(shape), dtype=getattr(torch, dtype), device=dev)
            self._recv(t, owner)
            load_weights([(name, t)])
            del t
        return len(self._manifest)

    def _start_applier(self, model) -> None:
        """NeMo's applier and loader dry run on the live model (after the anchor's reload); the names that cannot
        take XOR go to every trainer rank."""
        from nemo_dcr.vllm_sparse_delta import VllmSparseDeltaApplier

        if self._manifest is None or model is None:
            raise RuntimeError("dcr: a delta before the dense anchor (no manifest or model on this worker)")
        self._applier = VllmSparseDeltaApplier(SimpleNamespace(model=model), self._dev())
        state = {name: (tuple(shape), getattr(torch, dtype)) for name, shape, dtype, _ in self._manifest}
        overwrite = sorted(self._applier.discover_native_skips(state))
        dev = self._dev()
        blob = torch.frombuffer(bytearray(pickle.dumps(overwrite)), dtype=torch.uint8).to(dev)
        size = torch.tensor([blob.numel()], dtype=torch.int64, device=dev)
        for t in range(self.trainer_world):
            self._send(size, t)
            self._send(blob, t)
        print(f"[WT-DCR-VLLM] rank={self.rank} loader dry run: {len(self._applier._skipped_names)} tensors skipped on "
              f"this rank, {len(overwrite)} take overwrite", flush=True)

    def _receive_delta(self) -> dict:
        from nemo_dcr.stream import decode_sparse_payload

        dev = self._dev()
        stats = {"bodies": 0, "bytes": 0, "recv_s": 0.0, "decompress_s": 0.0, "apply_s": 0.0}
        length = torch.empty(1, dtype=torch.int64, device=dev)
        digest = torch.empty(16, dtype=torch.uint8, device=dev)
        for r in range(self.trainer_world):
            while True:
                ta = time.perf_counter()
                self._recv(length, r)
                n = int(length.item())
                if n == 0:
                    stats["recv_s"] += time.perf_counter() - ta
                    break
                self._recv(digest, r)
                body = torch.empty(n, dtype=torch.uint8, device=dev)
                self._recv(body, r)
                body_bytes = body.cpu().numpy().tobytes()
                checksum = digest.cpu().numpy().tobytes().hex()
                tb = time.perf_counter()
                raw = decode_sparse_payload(body_bytes, checksum)
                tc = time.perf_counter()
                self._applier.update_weights_from_decoded_sparse_payload(raw)
                torch.cuda.synchronize()
                td = time.perf_counter()
                stats["bodies"] += 1
                stats["bytes"] += n
                stats["recv_s"] += tb - ta
                stats["decompress_s"] += tc - tb
                stats["apply_s"] += td - tc
        return stats

    def _verify_dense(self, model) -> tuple[int, int]:
        from nemo_dcr.sparse_codec import integer_view

        dev = self._dev()
        snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}
        for name, shape, dtype, owner in self._manifest:
            t = torch.empty(tuple(shape), dtype=getattr(torch, dtype), device=dev)
            self._recv(t, owner)
            model.load_weights([(name, t)])
            del t
        torch.cuda.synchronize()
        bad = params_bad = 0
        for name, p in model.named_parameters():
            diff = int(integer_view(p.detach()).ne(integer_view(snapshot[name])).sum())
            bad += diff
            params_bad += bool(diff)
        return bad, params_bad
