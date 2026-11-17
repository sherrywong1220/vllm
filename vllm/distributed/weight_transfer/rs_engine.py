# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reader of the store-over-RDMA baseline (veRL backend ``rs_store``).

It is the cxl reader with one substitution: where that reader DMAs from the mapped pool, this one issues a one-sided
READ of the same bytes from the trainer rank that owns those rows. The slice plan, the loader bridge, the change-set
apply and the C2 checks are the cxl engine's own code, so the two arms differ only in the medium.

The base code needs a store object for geometry and views. Here that is a local shadow: an unfaulted memfd with the
trainer's manifest, so a view's address gives the region offset, which is the same on every trainer rank.
"""

import base64
import os
import pickle
import time
from bisect import bisect_right
from dataclasses import dataclass, field

import torch

from vllm.distributed.weight_transfer.cxl_engine import (
    _WAIT_TIMEOUT_S,
    CXLWeightTransferEngine,
    CXLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.base import WeightTransferInitInfo

_STAGE_BYTES = 2 << 30  # the registered HBM window a READ lands in, before the GPU copies it to its tensor


@dataclass
class RSWeightTransferInitInfo(WeightTransferInitInfo):
    ranks: list = field(default_factory=list)  # shared_weight_store.rs_store records, in trainer rank order
    ip: str = ""  # this host's fabric address: a vLLM worker is not a Ray process, and its default route may differ


class _ShadowStore:
    """The base reader's store: views, manifest and geometry from a local shadow, change sets from a small registered
    one (the base reader DMAs them to the GPU), and the commit doorbell from the trainer ranks."""

    active_slot = 0  # one slot per trainer rank, as the pool arm's offload store
    is_pool = False

    def __init__(self, model, cs, wait_committed):
        self._model, self._cs, self.wait_committed = model, cs, wait_committed

    def __getattr__(self, name):
        return getattr(self._model, name)

    def region_view(self, slot, name):
        from shared_weight_store.change_set import is_cs_name

        return self._cs.region_view(0, name) if is_cs_name(name) else self._model.region_view(0, name)


class RSStoreWeightTransferEngine(CXLWeightTransferEngine):
    init_info_cls = RSWeightTransferInitInfo
    update_info_cls = CXLWeightTransferUpdateInfo

    def init_transfer_engine(self, init_info: RSWeightTransferInitInfo) -> None:
        if not init_info.ranks:
            raise ValueError("rs_store: no trainer ranks in the init info")
        self._ranks, self._ip = init_info.ranks, init_info.ip
        self._store_uri = "rs_store"

    def _memfd_store(self, name, manifest):
        from shared_weight_store import CanonicalWeightStore, numa

        fd = numa.memfd_create(name)
        return CanonicalWeightStore.create(f"mmap:///proc/{os.getpid()}/fd/{fd}", manifest, n_slots=1)

    def _ensure_open(self):
        if self._store is not None:
            return
        from shared_weight_store import rs_store
        from shared_weight_store.change_set import cs_world, is_cs_name
        from shared_weight_store.manifest import ModelManifest

        manifest = pickle.loads(base64.b64decode(self._ranks[0]["manifest"]))
        model = self._memfd_store("rs_shadow", manifest)
        span = model._native.payload_span_view()
        self._shadow_base = span.data_ptr()
        self._shadow_nbytes = span.numel() * span.element_size()
        self._cs_specs = [t for t in manifest.tensors if is_cs_name(t.name)]
        cs = None
        self._client = rs_store.Mooncake(self._ip)
        if self._cs_specs:
            cs = self._memfd_store("rs_cs", ModelManifest(self._cs_specs))
            cs.register_host_for_cuda()
            sp = cs._native.payload_span_view()
            self._client.register(sp.data_ptr(), sp.numel() * sp.element_size())
        self._cs_store = cs
        self._store = _ShadowStore(model, cs, self._wait_committed)
        self._owner_names = [t.name for t in manifest.tensors if t.alias_of is None and not is_cs_name(t.name)]
        self._specs = {t.name: t for t in manifest.tensors}
        self._cs_world = cs_world(manifest)
        # address -> (name, spec): the base reader hands _slice_h2d a view, and the view's address names the region
        regions = sorted((model.region_view(0, n).data_ptr(), n) for n in self._owner_names)
        self._region_starts = [a for a, _ in regions]
        self._region_names = [n for _, n in regions]
        # The window a READ lands in: HBM (GDR, the path being measured), or pinned host memory for a TCP emulation
        dev = "cpu" if os.environ.get("RS_STAGE") == "cpu" else "cuda"
        self._stage = torch.empty(_STAGE_BYTES, dtype=torch.uint8, device=dev, pin_memory=dev == "cpu")
        self._stage_ptr = self._stage.data_ptr()
        self._client.register(self._stage_ptr, _STAGE_BYTES)
        self._stage_off = 0
        self._doorbell = torch.zeros(1, dtype=torch.int64)
        self._client.register(self._doorbell.data_ptr(), 8)
        self._slice_h2d = self._read_view
        self._full_h2d = self._read_view
        self._cuda_direct = True  # the base reader's "no staging, call the hook" branch
        if self._is_tp_rank0():
            print(f"[WT-RS] reader ({len(self._owner_names)} owner tensors) pool=0 device={self._client.device} "
                  f"protocol={self._client.protocol} stage={dev}", flush=True)

    def _wait_committed(self, min_version=None, timeout_s=None):
        """Every trainer rank's doorbell has reached the version, read with one-sided READs (no control plane)."""
        t0 = time.monotonic()
        ptr = self._doorbell.data_ptr()
        for r in self._ranks:
            while True:
                self._client.read(r["session"], ptr, r["doorbell"], 8)
                if int(self._doorbell[0]) >= (min_version or 0):
                    break
                if time.monotonic() - t0 > (timeout_s or _WAIT_TIMEOUT_S):
                    raise TimeoutError(f"rs_store: rank doorbell {int(self._doorbell[0])} < v{min_version}")
        return min_version

    def _read_range(self, dst: torch.Tensor, region_off: int, shape, itemsize: int, lo: int, hi: int) -> None:
        """READ bytes ``[lo, hi)`` of a tensor into the uint8 tensor ``dst`` through the HBM window. The GPU copies out
        of the window on its stream, so the window is reused only after a synchronize; it runs when the window wraps,
        which keeps the copies pipelined behind the READs."""
        from shared_weight_store.rs_store import row_chunks

        for rank, off, n in row_chunks(tuple(shape), itemsize, len(self._ranks), lo, hi):
            r = self._ranks[rank]
            done = 0
            while done < n:
                m = min(n - done, _STAGE_BYTES)
                if self._stage_off + m > _STAGE_BYTES:
                    torch.cuda.synchronize()
                    self._stage_off = 0
                s = self._stage_off
                self._client.read(r["session"], self._stage_ptr + s, r["base"] + region_off + off + done, m)
                dst[off - lo + done:off - lo + done + m].copy_(self._stage[s:s + m], non_blocking=True)
                self._stage_off += m
                done += m

    def _read_view(self, sl: torch.Tensor, tensor: torch.Tensor) -> None:
        """READ what the view ``sl`` of a shadow region holds on the trainer into ``tensor``. A contiguous slice is
        read exactly. A strided one (TP column slice) cannot be expressed as one READ per row at any useful message
        rate, so it reads the whole tensor and narrows on the GPU: the bytes-per-link columns show this over-read."""
        ptr = sl.data_ptr()
        i = bisect_right(self._region_starts, ptr) - 1
        name = self._region_names[i]
        spec = self._specs[name]
        base = self._region_starts[i]
        region_off = base - self._shadow_base
        size = sl.element_size()
        if sl.is_contiguous():
            lo = ptr - base
            self._read_range(tensor.view(-1).view(torch.uint8), region_off, spec.shape, size, lo,
                             lo + sl.numel() * size)
            return
        full = torch.empty(tuple(spec.shape), dtype=sl.dtype, device=tensor.device)
        self._read_range(full.view(-1).view(torch.uint8), region_off, spec.shape, size, 0, full.numel() * size)
        dim = next(d for d in range(full.dim()) if sl.shape[d] != full.shape[d])
        start = (ptr - base) // (full.stride(dim) * size)
        tensor.copy_(full.narrow(dim, start, sl.shape[dim]), non_blocking=True)

    def _populate_shadow(self) -> None:
        """C2 only (a ``v`` run, never a timed one): the trainer's rows into the shadow, so the base reader's digests
        and C2-SLICE compare the resident weights with what the trainer ranks hold."""
        from shared_weight_store.rs_store import row_chunks

        if not getattr(self, "_shadow_registered", False):
            self._client.register(self._shadow_base, self._shadow_nbytes)
            self._shadow_registered = True
        for name in self._owner_names:
            spec = self._specs[name]
            base = self._store.region_view(0, name).data_ptr() - self._shadow_base
            for rank, off, n in row_chunks(tuple(spec.shape), spec.dtype.itemsize, len(self._ranks), 0, spec.nbytes):
                r = self._ranks[rank]
                self._client.read(r["session"], self._shadow_base + base + off, r["base"] + base + off, n)

    def _fetch_change_sets(self) -> None:
        """Each trainer rank's change set into the registered local regions: counts first, whose sum sizes idx / val.
        From there the base reader's ChangeSetReader runs unchanged (host window -> GPU by DMA)."""
        from shared_weight_store.change_set import cs_names

        remote_of = self._cs_offset_table()
        for rank, r in enumerate(self._ranks):
            cn, inm, vn = cs_names(rank)
            cv = self._cs_store.region_view(0, cn)
            self._client.read(r["session"], cv.data_ptr(), r["base"] + remote_of[cn], cv.numel() * cv.element_size())
            n = int(cv[:self._cs_reader.tiles_per_rank()[rank]].sum())
            if not n:
                continue
            for nm in (inm, vn):
                v = self._cs_store.region_view(0, nm)
                self._client.read(r["session"], v.data_ptr(), r["base"] + remote_of[nm], n * v.element_size())

    def _cs_offset_table(self) -> dict:
        """Change-set regions' offsets in a trainer rank's store: the same manifest, so the same as the shadow's."""
        if not hasattr(self, "_cs_offsets"):
            self._cs_offsets = {t.name: self._store._model.region_view(0, t.name).data_ptr() - self._shadow_base
                                for t in self._cs_specs}
        return self._cs_offsets

    def receive_weights(self, update_info, load_weights) -> None:
        self._ensure_open()
        if os.environ.get("WT_CXL_C2", "0") == "1":
            self._wait_committed(update_info.version)
            self._populate_shadow()
        super().receive_weights(update_info, load_weights)

    def _receive_sparse(self, version: int) -> None:
        from shared_weight_store.change_set import ChangeSetReader, owner_specs

        self._wait_committed(version)
        if self._cs_reader is None:
            self._cs_reader = ChangeSetReader(owner_specs(self._store.manifest), self._cs_world)
        self._fetch_change_sets()
        super()._receive_sparse(version)
