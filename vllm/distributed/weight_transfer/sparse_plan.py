# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P5 sparse placement shared by the transport engines (design p5 §1.4.3 / §3.10).

A change set carries CANONICAL element indices (the HF-shaped tensor's flat index). vLLM's
loaders are copy-based and cannot take indices, so each engine needs, per store owner, the
rule that maps canonical ``(row, col)`` onto this TP rank's live parameter: row-parallel
column slice, merged-column / fused-qkv row slice at the fused offset, padded vocab rows,
replicated identity, a MoE's fused 3D experts at this EP rank's local slots. The rule is
derived once from the live model here and applied by
``shared_weight_store.change_set.ChangeSetReader.apply``; the transport (shared store or NCCL)
only decides how the entries arrive. Anything unknown raises — never a silent mis-place.
"""

import torch


def sliceable(p) -> bool:
    """A param is TP-sliceable only if unquantized/unpacked — quantized (GPTQ/AWQ/
    Marlin/FP8) and bitsandbytes params carry a kernel-tiled/packed layout whose
    shape differs from the logical one for reasons unrelated to TP, so they must go
    through the framework loader + ``process_weights_after_loading`` (P3D)."""
    return (
        not getattr(p, "use_bitsandbytes_4bit", False)
        and getattr(p, "packed_dim", None) is None
    )


def vocab_target(name, param, modules):
    """If store owner ``name`` is a vocab-parallel weight (``embed_tokens``/
    ``lm_head`` on a ``VocabParallelEmbedding`` module), return the geometry
    ``(org_vocab_start_index, org_vocab_end_index, num_embeddings_per_partition)``;
    else ``None`` (→ unchanged full read, P3D).

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


def expert_target(name, shape, params, modules):
    """If store owner ``name`` is a fused 3D routed-expert tensor of a FusedMoE layer on this
    model (HF transformers >= 5: ``mlp.experts.gate_up_proj`` ``[E, 2I, H]`` / ``down_proj``
    ``[E, H, I]``), return ``(kind, module, param, param_name)`` with ``kind`` in {``w13``,
    ``w2``}; else ``None``. Duck-typed on the module having a ``weight_loader`` and an
    ``_expert_map`` attribute (FusedMoE); no ``isinstance`` coupling. One rule for the dense
    EP-slice read (P7) and the sparse plan below."""
    if len(shape) != 3:
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


def expert_plan(name, S, mod, p, pname):
    """The ``DstPlan`` of fused 3D expert owner ``name`` ``[E, ...]`` on ``p``, this rank's
    ``[E_local, ...]`` parameter (design p5 §1.8). A canonical row is one expert, so expert
    parallelism is a dim-0 placement: experts ``[g0, g0 + n)`` land at local slots
    ``[l0, l0 + n)``. Ownership is read off the engine's live ``_expert_map``, never recomputed
    from config (p7 §3.2). A ``DstPlan`` holds one interval, so only a placement with one run
    per rank (``linear``) is patchable, and the values must sit in the parameter in the
    canonical layout (whole experts, ``[gate; up]`` rows). Everything else raises: a change set
    has no dense fallback, and a mis-placed expert is silent."""
    from shared_weight_store.change_set import DstPlan
    from shared_weight_store.expert_plan import expert_runs

    if not sliceable(p):
        raise RuntimeError(f"P5: expert param '{pname}' is quantized/packed; not patchable")
    backend = getattr(getattr(mod, "quant_method", None), "unquantized_backend", None)
    if backend is not None and backend.name != "TRITON":
        raise RuntimeError(f"P5: '{pname}' is in the {backend.name} kernel's weight layout, not the canonical "
                           "[gate; up] one; not patchable")
    if getattr(mod, "enable_eplb", False):
        raise RuntimeError(f"P5: '{pname}' is under EPLB (the expert placement changes at runtime); not patchable")
    if tuple(p.shape[1:]) != tuple(S[1:]):
        raise RuntimeError(f"P5: '{name}' {S}: the per-expert shape of '{pname}' is {tuple(p.shape[1:])} "
                           "(TP inside the MoE layer); only whole experts are placed")
    emap = getattr(mod, "_expert_map", None)
    if emap is None:  # no expert parallelism: every expert, in order
        if p.shape[0] != S[0]:
            raise RuntimeError(f"P5: '{pname}' holds {p.shape[0]} experts, '{name}' has {S[0]}, and there is no expert map")
        return DstPlan.identity(p, S)
    emap = emap.tolist()
    if len(emap) != S[0]:
        raise RuntimeError(f"P5: the expert map of '{pname}' maps {len(emap)} experts, '{name}' has {S[0]}")
    runs = expert_runs(emap)
    if len(runs) > 1:
        raise RuntimeError(f"P5: this rank's experts of '{name}' are {len(runs)} runs (expert placement is not "
                           "linear); the sparse plan places one contiguous run")
    g0, l0, n = (runs[0].global_start, runs[0].local_start, runs[0].count) if runs else (0, 0, 0)
    if p.shape[0] != n:
        raise RuntimeError(f"P5: '{pname}' holds {p.shape[0]} experts, its expert map gives this rank {n}")
    return DstPlan.rows(p, S, g0, n, l0)


def is_tp_rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return True


def tp_rank_size() -> tuple[int, int]:
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        return 0, 1


def build_sparse_plan(model, owners, tag: str = "P5") -> list:
    """One ``DstPlan`` per store owner: where its canonical elements live on this TP rank."""
    from shared_weight_store.change_set import DstPlan
    from shared_weight_store.slice_plan import even_shard

    tp_rank, tp_size = tp_rank_size()
    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    mapping = getattr(model, "packed_modules_mapping", None) or {}
    plans, kinds = [], {"row": 0, "col": 0, "qkv": 0, "vocab": 0, "repl": 0, "expert": 0}

    def _bf16(p, name):
        if p.dtype != torch.bfloat16:
            raise RuntimeError(f"P5: '{name}' is {p.dtype}; the sparse path patches bf16 weights only")
        if not p.is_contiguous():
            raise RuntimeError(f"P5: '{name}' is not contiguous")
        return p

    for o in owners:
        name, S = o.name, tuple(int(d) for d in o.shape)
        p = params.get(name)
        if p is not None:
            _bf16(p, name)
            vg = vocab_target(name, p, modules) if len(S) == 2 else None
            if vg is not None:
                start, end, _ = vg
                plans.append(DstPlan.rows(p, S, start, end - start, 0))
                kinds["vocab"] += 1
            elif tuple(p.shape) == S:
                plans.append(DstPlan.identity(p, S))
                kinds["repl"] += 1
            elif (len(S) == 2 and p.dim() == 2 and p.shape[0] == S[0] and p.shape[1] * tp_size == S[1]
                  and getattr(p, "input_dim", None) is not None):
                start, size = even_shard(S[1], tp_rank, tp_size)
                plans.append(DstPlan.columns(p, S, start, size))
                kinds["row"] += 1
            elif p.shape[0] * tp_size == S[0] and tuple(p.shape[1:]) == tuple(S[1:]):
                start, size = even_shard(S[0], tp_rank, tp_size)  # unfused column-parallel
                plans.append(DstPlan.rows(p, S, start, size, 0))
                kinds["col"] += 1
            else:
                raise RuntimeError(f"P5: cannot place '{name}' {S} onto param {tuple(p.shape)} (tp={tp_size})")
            continue
        ep = expert_target(name, S, params, modules)
        if ep is not None:  # fused 3D routed experts of a MoE: this rank's experts, by the engine's expert map
            _, mod, ep_p, pname = ep
            plans.append(expert_plan(name, S, mod, _bf16(ep_p, pname), pname))
            kinds["expert"] += 1
            continue
        # fused constituent: q/k/v -> qkv_proj, gate/up -> gate_up_proj (weight or bias)
        placed = False
        for fused, parts in mapping.items():
            for i, c in enumerate(parts):
                needle = f".{c}."
                if needle not in name:
                    continue
                fused_name = name.replace(needle, f".{fused}.")
                fp = params.get(fused_name)
                mod = modules.get(fused_name.rsplit(".", 1)[0])
                if fp is None or mod is None:
                    continue
                _bf16(fp, fused_name)
                if not sliceable(fp):
                    raise RuntimeError(f"P5: fused '{fused_name}' is quantized/packed; not patchable")
                if hasattr(mod, "num_kv_head_replicas"):  # QKVParallelLinear (GQA replica rule)
                    hs, vhs = mod.head_size, getattr(mod, "v_head_size", mod.head_size)
                    if c.startswith("q"):
                        size, srank, doff = mod.num_heads * hs, tp_rank, 0
                    elif c.startswith("k"):
                        size, srank, doff = mod.num_kv_heads * hs, tp_rank // mod.num_kv_head_replicas, mod.num_heads * hs
                    else:
                        size, srank, doff = (mod.num_kv_heads * vhs, tp_rank // mod.num_kv_head_replicas,
                                             (mod.num_heads + mod.num_kv_heads) * hs)
                    start = srank * size
                    kinds["qkv"] += 1
                elif hasattr(mod, "output_sizes"):  # MergedColumnParallelLinear
                    tp = int(getattr(mod, "tp_size", tp_size))
                    size = int(mod.output_sizes[i]) // tp
                    start = tp_rank * size
                    doff = sum(int(s) for s in mod.output_sizes[:i]) // tp
                    kinds["col"] += 1
                else:
                    raise RuntimeError(f"P5: fused module for '{name}' has no known placement rule")
                if S[0] < start + size:
                    raise RuntimeError(f"P5: '{name}' {S}: slice [{start}, {start + size}) out of range")
                plans.append(DstPlan.rows(fp, S, start, size, doff))
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError(f"P5: store owner '{name}' has no parameter on this rank and no fused target")
    if is_tp_rank0():
        print(f"[WT-CXL] {tag} sparse plan (rank {tp_rank}/{tp_size}): {len(plans)} owners -> "
              f"{kinds}", flush=True)
    return plans


@torch.no_grad()
def plan_slice(plan, full: torch.Tensor) -> torch.Tensor:
    """This rank's slice of the canonical ``full`` tensor under ``plan`` (rows ``[r0, r1)``, cols
    ``[c0, c1)``), contiguous."""
    rows = full.numel() // plan.cols
    return (full.reshape(rows, plan.cols).narrow(0, plan.r0, plan.r1 - plan.r0)
            .narrow(1, plan.c0, plan.c1 - plan.c0).contiguous())


@torch.no_grad()
def c2_compare(plan, want_bf16: torch.Tensor) -> tuple[int, float]:
    """Tier-S oracle step: ``(mismatched elements, max |diff|)`` between the resident destination
    under ``plan`` and ``want_bf16`` (this rank's slice of the dense rendition, same shape)."""
    d = plan.dst.view(-1)
    got = d.view(d.numel() // plan.dcols, plan.dcols).narrow(0, plan.doff, plan.r1 - plan.r0).narrow(
        1, 0, plan.c1 - plan.c0).contiguous()
    if torch.equal(want_bf16.view(torch.int16), got.view(torch.int16)):
        return 0, 0.0
    n_bad = int((want_bf16.view(torch.int16) != got.view(torch.int16)).sum())
    return n_bad, float((want_bf16.float() - got.float()).abs().max())
