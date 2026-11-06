"""Public TokenGate inference API — mirrors the standard transformers flow.

The three calls match `AutoModelForCausalLM.from_pretrained` /
`tokenizer.apply_chat_template` / `model.generate`, with one extra concept
(the `{state}` placeholder marking the prunable document) and one extra
parameter (`max_kept_prefill_tokens`, the paper's K):

    from transformers import AutoTokenizer
    from tokengate import load_model, apply_chat_template

    model = load_model("liana313/tokengate-biodexr-qwen3-1.7b")
    tok = AutoTokenizer.from_pretrained(model.base_model)

    prompt = "Answer using the article below.\\n\\nArticle:\\n{state}\\n\\nAnswer:"
    ids = apply_chat_template(tok, prompt, state=article_text)
    out = model.generate([ids], tok, max_kept_prefill_tokens=1000,
                         max_new_tokens=64)
    print(out.answers[0])
"""
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    GATE_ATTN, get_full_attn_layer_indices, patch_model_with_flex_attention,
    set_attention_backends_for_layers,
)
from tokengate.e2e_gate.attn.flex_attention_utils import warmup_flex_attention
from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
from tokengate.e2e_gate.gate_module.lora import (
    collect_lora_state_dict, resolve_all_adapter_layers,
)

GATE_COMPILE = ("none", "p2", "all")

# Re-exported here so `from tokengate import apply_chat_template` works; the
# implementation (and the state sentinels) live next to the span logic.
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: F401
    STATE_END_ID,
    STATE_START_ID,
    apply_chat_template,
    split_state_span,
)

# K value meaning "keep everything" (no state token is ever dropped).
_NO_PRUNING = 10_000_000


def _resolve_gate_path(checkpoint: str) -> str:
    """A checkpoint spec → the gate .pt file.

    Accepts a direct ``*.pt`` path, a local run directory, or a Hugging Face
    Hub repo id (downloaded via ``snapshot_download``). In a directory the
    newest-stage gate wins: ``swa_gate_*`` > ``gate_final`` > ``gate_step*``;
    the adapter pairs with it by filename (``lora.find_adapter_path``).
    """
    if checkpoint.endswith(".pt"):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"gate checkpoint not found: {checkpoint}")
        return checkpoint
    ckpt_dir = checkpoint
    if not os.path.isdir(ckpt_dir):
        from huggingface_hub import snapshot_download
        print(f"[tokengate] downloading checkpoint from hf.co/{checkpoint} ...")
        ckpt_dir = snapshot_download(checkpoint)
    for pat in ("swa_gate_*.pt", "gate_final.pt", "gate_step*.pt"):
        hits = sorted(glob.glob(os.path.join(ckpt_dir, pat)))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no gate checkpoint (gate_*.pt) in {ckpt_dir}")


@dataclass
class GenerateOutput:
    """Result of ``TokenGateModel.generate`` (one entry per input sequence)."""
    answers: List[str]       # decoded, first-EOS-truncated
    n_kept: List[int]        # KV-cache entries kept per sequence
    n_state: List[int]       # state tokens offered to the gate per sequence
    kept_pos: List[List[int]]  # kept token positions (into the clean prompt ids)
    prefill_s: float         # CUDA-synced prefill wall (whole batch)
    decode_s: float          # CUDA-synced decode wall (whole batch)


class TokenGateModel:
    """A base model + trained gate + paired adapter behind one object.

    Built by ``load_model``; everything not defined here (``config``,
    ``device``, ...) delegates to the underlying transformers model.
    """

    def __init__(self, model, gate, gate_layer: int, base_model: str):
        self.model = model
        self.gate = gate
        self.gate_layer = gate_layer
        self.base_model = base_model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(
        self,
        ids: Union[List[int], List[List[int]]],
        tokenizer,
        *,
        max_new_tokens: int = 20,
        max_kept_prefill_tokens: Optional[int] = None,
    ) -> GenerateOutput:
        """Gated prefill + greedy decode (the standard ``generate`` call).

        ids : one sequence or a batch of sequences from
            ``apply_chat_template(..., state=...)`` — each must carry the
            state sentinels (ValueError otherwise).
        max_new_tokens : generation budget; the default 20 matches
            transformers' own default generation length
            (``GenerationConfig.max_length = 20``).
        max_kept_prefill_tokens : at most this many state tokens survive
            prefill into the KV cache (the paper's K). None = keep all
            (no pruning).
        """
        if ids and isinstance(ids[0], int):
            ids = [ids]
        split = [split_state_span(list(seq)) for seq in ids]
        clean_ids = [s[0] for s in split]
        starts = [s[1] for s in split]
        ends = [s[2] for s in split]
        k = (_NO_PRUNING if max_kept_prefill_tokens is None
             else max_kept_prefill_tokens)
        from tokengate.e2e_gate.decode.gate_decode import generate_batch
        res = generate_batch(
            self.model, self.gate, tokenizer, clean_ids, starts, ends,
            self.gate_layer, k, max_new_tokens)
        kept_pos = [res.kept_pos[b, :n].cpu().tolist()
                    for b, n in enumerate(res.n_kepts)]
        return GenerateOutput(
            answers=res.answers, n_kept=res.n_kepts,
            n_state=[e - s for s, e in zip(starts, ends)],
            kept_pos=kept_pos,
            prefill_s=res.prefill_s, decode_s=res.decode_s)


def load_model(
    checkpoint: str,
    base_model: Optional[str] = None,
    gate_layer: Optional[int] = None,
    gate_attn: str = "fa2",
) -> TokenGateModel:
    """Load a trained TokenGate checkpoint as ONE model object.

    checkpoint : HF Hub repo id, local run directory, or a gate ``*.pt`` path.
        The name-paired adapter file next to the gate is loaded with it.
    base_model / gate_layer : read from the checkpoint's own metadata when
        omitted (every trainer checkpoint records them); passing them
        overrides the recorded values.
    gate_attn : inference kernels — "fa2" (default; FlashAttention-2 + the
        key-bias kernel, matches the training runs) or "flex" (no custom
        kernel build needed, slower).
    """
    import torch
    gate_path = _resolve_gate_path(checkpoint)
    meta = torch.load(gate_path, map_location="cpu",
                      weights_only=False).get("metadata", {})
    if base_model is None:
        base_model = meta.get("base_model")
        if not base_model:
            raise ValueError(
                f"{gate_path} records no base_model in its metadata; pass "
                "base_model= explicitly")
    if gate_layer is None:
        gate_layer = meta.get("gate_layer")
        if gate_layer is None:
            raise ValueError(
                f"{gate_path} records no gate_layer in its metadata; pass "
                "gate_layer= explicitly")
    model, gate, _ = load_gate_model(base_model, gate_path, gate_layer,
                                     gate_attn=gate_attn)
    return TokenGateModel(model, gate, gate_layer, base_model)


def compile_layer_forward(layer, cuda_graphs: bool = False, idx=None) -> None:
    """torch.compile a decoder layer's forward under a code object of its own.

    Dynamo caches compiled graphs per code object, and every layer's graph is
    distinct (dynamo guards on the layer's slot in the KV cache), so with the
    shared Qwen3DecoderLayer.forward the cache behind one code object holds a
    graph per layer -- 28 per model, 56 with a length-dependent kernel choice,
    twice that with two models resident -- and each call walks that chain of
    guard sets until its own matches.  At 1k tokens that walk cost more than
    the layer's kernels: Phase 2 of the pruned prefill read a flat 9.5 ms for
    26 layers whatever the context.  A copy of the code object per layer gives
    each its own cache of one or two entries.
    """
    import types
    fn = type(layer).forward
    # idx names the per-layer code object; DeltaNet layers of the Qwen3.5
    # hybrid have linear_attn instead of self_attn, so never touch
    # layer.self_attn directly here
    if idx is None:
        _attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        idx = getattr(_attn, "layer_idx", "")
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_l{idx}")
    fresh = types.FunctionType(code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    fresh.__kwdefaults__ = fn.__kwdefaults__
    layer.forward = torch.compile(types.MethodType(fresh, layer), dynamic=True, fullgraph=False,
                                  mode="reduce-overhead" if cuda_graphs else None)



def _infer_lora_shape(sd, path):
    """(layer_ids, dotted_targets, rank) read off a LoRA state dict itself.

    The artifact is self-describing: keys carry the layer index and the dotted
    module path relative to the decoder layer (e.g. 'self_attn.q_proj'), and
    lora_B is stored (rank, out).  Metadata is NOT consulted — old checkpoints
    with adapter layers ≠ [lora_start_layer .. gate_layer] (e.g. the ll=all
    runs) load correctly this way.  Only alpha is non-inferable (see
    _lora_alpha_for).
    """
    import re as _re
    layer_ids = sorted({int(m.group(1)) for k in sd
                        for m in [_re.search(r"\.layers\.(\d+)\.", k)] if m})
    targets = sorted({m.group(1) for k in sd
                      for m in [_re.search(r"\.layers\.\d+\.(.+?)\.lora_[AB]$", k)] if m})
    ranks = {v.shape[0] for k, v in sd.items() if k.endswith(".lora_B")}
    if not layer_ids or not targets or len(ranks) != 1:
        raise ValueError(f"cannot infer LoRA shape from {path}: layers={layer_ids} "
                         f"targets={targets} ranks={ranks}")
    return layer_ids, targets, ranks.pop()



def _lora_alpha_for(gate_path, meta):
    """LoRA alpha for a gate checkpoint: gate meta, else the run's params.json
    (written next to the checkpoints), else fail — alpha is not stored in the
    adapter file and a silently-wrong default rescales every adapter output."""
    if meta.get("lora_alpha"):
        return float(meta["lora_alpha"])
    pj = os.path.join(os.path.dirname(os.path.abspath(gate_path)), "params.json")
    if os.path.exists(pj):
        import json as _json
        with open(pj) as f:
            la = _json.load(f).get("lora_alpha")
        if la:
            return float(la)
    raise ValueError(f"lora_alpha not in gate meta and no params.json with it next to "
                     f"{gate_path}")



def load_gate_model(base_model, gate_path, gate_layer, gate_attn="fa2",
                    merge_adapters_above_gate=False, compile_phase2="none", cuda_graphs=False,
                    *, adapter_path=None, adapter_config=None, expected_metadata=None,
                    device="cuda", warmup=True):
    """gate_attn names the inference attention kernels
    (qwen3_flex_attention.GATE_ATTN); "fa2" (FA2 flash causal + the
    FlashAttention-2 key-bias kernel) is the default and matches the final
    training runs' kernels; "flex" is training's original (slow) kernel.
    merge_adapters_above_gate folds the LoRA / delta adapters of the layers
    ABOVE gate_layer into their base weights (lora.merge_adapters_into_layers);
    the gate layer and below keep the unmerged, training-identical path so the
    selection is bit-for-bit what the checkpoint selects.
    compile_phase2="p2" torch.compiles every decoder layer above gate_layer
    (dynamic shapes; the Phase-1 layers stay eager for the same reason), with
    inductor's emulate_precision_casts so fused kernels keep eager's bf16
    roundings (without it a near-tie first token can flip)."""
    from transformers import AutoModelForCausalLM

    from tokengate.e2e_gate.gate_module.lora import find_adapter_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()

    gate, meta = load_gate(gate_path) if gate_path is not None else (None, {})
    input_all_prev_layers = False
    if gate is not None:
        full_attn = get_full_attn_layer_indices(model)
        if not 0 <= gate_layer < len(model.model.layers) or not any(i > gate_layer for i in full_attn):
            raise ValueError(f"Invalid gate layer {gate_layer}: no full attention above it")
        gate = gate.to(device=device, dtype=torch.float32).eval()
        for name, value in (expected_metadata or {}).items():
            if meta.get(name) != value:
                raise ValueError(f"Gate and adapter provenance differ: {name}")
        if expected_metadata and (gate.selection_mode != "topk" or gate.random_topk):
            raise ValueError("Evaluation requires deterministic top-K selection")
    # Adapter pairing by checkpoint name (lora.find_adapter_path): gate_step{N} ->
    # lora_step{N}, gate_final -> lora_final, swa_gate_* -> swa_lora_*.  Never a live
    # adapter for an SWA gate and never lora_final for a step gate -- the old fallbacks
    # silently scored a gate with an adapter it was not averaged / saved with.
    lora_path = adapter_path or (find_adapter_path(gate_path) if gate_path else None)
    if meta.get("adapter_type") and lora_path is None:
        raise FileNotFoundError(
            f"{gate_path} was trained with adapter_type={meta['adapter_type']!r} but no "
            f"paired adapter file exists next to it (see lora.find_adapter_path)")
    if lora_path is not None:
        from tokengate.e2e_gate.gate_module.lora import (
            apply_delta_to_layers,
            apply_lora_to_layers,
            load_lora_state_dict,
        )
        # meta["adapter_type"] disambiguates delta vs lora (absent in pre-delta
        # checkpoints → "lora").
        adapter_type = meta.get("adapter_type") or "lora"
        ls = meta.get("lora_start_layer") or 0
        layer_ids = list(range(ls, gate_layer + 1)) if gate_layer is not None else []
        if adapter_type == "delta":
            _sd = torch.load(lora_path, map_location="cpu")
            # The adapter may cover layers ABOVE gate_layer (e.g. --unfreeze-layers, which
            # applies delta to those layers). Infer ALL layers present in the checkpoint so
            # apply_delta wraps them; else load_lora_state_dict fails on unmatched tensors.
            import re as _re
            _ck = sorted({int(m.group(1)) for k in _sd for m in [_re.search(r"\.layers\.(\d+)\.", k)] if m})
            layer_ids = sorted(set(layer_ids) | set(_ck))
            apply_delta_to_layers(model, layer_ids)
            load_lora_state_dict(model, _sd)
            print(f"[gate] loaded delta adapter {lora_path} (layers={layer_ids})", flush=True)
        else:
            _sd = torch.load(lora_path, map_location="cpu")
            if adapter_config is None:
                layer_ids, lt, lr = _infer_lora_shape(_sd, lora_path)
                la = _lora_alpha_for(gate_path, meta)
            else:
                layer_ids = resolve_all_adapter_layers(model)
                lt, lr, la = (adapter_config[name] for name in ("lora_targets", "lora_rank", "lora_alpha"))
                if gate is not None:
                    for name, value in {"lora_layer_ids": layer_ids, "lora_targets": lt,
                                        "lora_rank": lr, "lora_alpha": la}.items():
                        if meta.get(name) != value:
                            raise ValueError(f"Gate and adapter provenance differ: {name}")
            apply_lora_to_layers(model, layer_ids, lr, la, lt)
            if adapter_config is not None:
                expected = collect_lora_state_dict(model)
                if set(_sd) != set(expected) or any(_sd[k].shape != v.shape for k, v in expected.items()):
                    raise ValueError("Adapter checkpoint must contain every expected tensor and shape")
            load_lora_state_dict(model, _sd)
            print(f"[gate] loaded LoRA {lora_path} (rank={lr}, alpha={la}, targets={lt}, "
                  f"layers={layer_ids[0]}..{layer_ids[-1]})", flush=True)
    else:
        print(f"[gate] WARNING: no LoRA file found next to {gate_path}", flush=True)

    model.requires_grad_(False)
    if gate is None:
        return model, None, False
    gate.requires_grad_(False)

    if merge_adapters_above_gate:
        # True / 1: layers above the gate only (the selection stays bit-exact);
        # 2: every layer (kernel parity with the baseline; changes the selection
        # at bf16 near-ties, see check_gate_equiv).
        from tokengate.e2e_gate.gate_module.lora import merge_adapters_into_layers
        _first = 0 if merge_adapters_above_gate == 2 else gate_layer + 1
        _layers = list(range(_first, len(model.model.layers)))
        _n = merge_adapters_into_layers(model, _layers)
        print(f"[gate] merged {_n} adapter modules into base weights on layers "
              f"{_layers[0]}..{_layers[-1]}"
              + ("" if _first == 0 else f"; layers 0..{gate_layer} unmerged"), flush=True)
    model = patch_model_with_flex_attention(model)
    set_attention_backends_for_layers(model, gate_attn)
    print(f"[gate] attention kernels: {gate_attn} -> causal/bias = {GATE_ATTN[gate_attn]}", flush=True)
    if compile_phase2 in ("p2", "all"):
        # "all" also compiles the Phase-1 layers 0..gate_layer (layer gate_layer
        # runs through attn_only_forward / gate_layer_mlp, which call its
        # submodules, so its layer.forward compile is inert; the submodules stay
        # eager). Whether that keeps the selection bit-identical is what
        # check_gate_equiv decides.
        import torch._inductor.config as _ic
        _ic.emulate_precision_casts = True
        # Every decoder layer compiles to its own graph (dynamo guards on the
        # layer's index into the cache), times two once a length-dependent
        # kernel choice adds a specialisation; with a second compiled model
        # resident (the TTFT bench) the default limit of 8 -- or the 64 set
        # for training -- is crossed and dynamo silently runs the rest EAGER
        # (the baseline read 27-32 ms where it measures 16).  Raise the limit
        # and make crossing it an error, never a fallback.
        import torch._dynamo.config as _dc
        _dc.recompile_limit = max(_dc.recompile_limit, 512)
        _dc.accumulated_recompile_limit = max(_dc.accumulated_recompile_limit, 4096)
        _dc.fail_on_recompile_limit_hit = True
        # cuda_graphs: the Phase-2 layers run under CUDA graphs (inductor's
        # reduce-overhead mode, one recording per distinct shape) and the
        # pruned sequence is padded to a multiple of 64 tokens so the shapes
        # are few.  Phase 2 on ~1k kept tokens was 26 x 0.33 ms of launch
        # overhead, whatever the context length.  Phase 1 stays as it is.
        from tokengate.e2e_gate.attn.qwen3_flex_attention import (
            set_static_inputs_for_layers,
        )
        from tokengate.e2e_gate.gate_module.gate_forward import set_phase2_bucket
        # cuda_graphs: 0 none; 1 inductor's reduce-overhead mode per layer above
        # the gate; 2 the whole Phase-2 loop captured once per shape
        # (phase2_graph) with the layers compiled in the default mode.
        cuda_graphs = int(cuda_graphs)
        set_phase2_bucket(64 if cuda_graphs else None, cuda_graphs=cuda_graphs)
        set_static_inputs_for_layers(model, gate_layer + 1, len(model.model.layers), bool(cuda_graphs))
        _first = 0 if compile_phase2 == "all" else gate_layer + 1
        for _i in range(_first, len(model.model.layers)):
            compile_layer_forward(model.model.layers[_i], cuda_graphs=(cuda_graphs == 1 and _i > gate_layer), idx=_i)
        # The gate layer runs as two halves (attention in Phase 1, MLP on the kept
        # tokens at Phase-2 entry) through module-level functions, not
        # layer.forward; compile those too when Phase 1 is compiled.
        from tokengate.e2e_gate.gate_module.gate_forward import (
            compile_gate_branch,
            set_split_layer_compile,
        )
        set_split_layer_compile(compile_phase2 == "all")
        # The scoring branch too, in the parity configuration: its norm,
        # activation and casts fuse (their roundings change, so not for "p2").
        if compile_phase2 == "all" and getattr(gate, "gate_branch", None) is not None:
            compile_gate_branch(gate, model, gate_layer)
        print(f"[gate] torch.compile on layers {_first}..{len(model.model.layers) - 1} "
              f"(dynamic, emulate_precision_casts"
              f"{', CUDA graphs above the gate layer, Phase 2 padded to 64' if cuda_graphs == 1 else ''}"
              f"{', Phase 2 as one CUDA graph per shape, padded to 64' if cuda_graphs == 2 else ''})"
              + ("; scoring branch compiled" if getattr(gate, "_compiled_branch_hs", None) is not None else "")
              + ("; gate layer's split halves compiled" if compile_phase2 == "all" else ""), flush=True)
    elif compile_phase2 != "none":
        raise ValueError(f"compile_phase2 {compile_phase2!r} not in {GATE_COMPILE}")
    if warmup:
        warmup_flex_attention(device)
    # load_gate() reconstructs the scoring branch itself: fp32-storage CastLinears
    # that compute in the input dtype (the bf16 backbone activations), exactly the
    # modules training used -- no rebuild from the base model, no fp32-compute drift
    # vs the training-time [eval] / [swa eval] numbers.  It fails loudly when a
    # checkpoint declares a branch it does not carry.
    _gb_mode = getattr(gate, "gate_branch_mode", None)
    if _gb_mode and gate.gate_branch is None:
        raise RuntimeError(f"gate_branch_mode={_gb_mode!r} but load_gate attached no "
                           f"scoring branch: {gate_path}")
    if _gb_mode and warmup:
        from tokengate.e2e_gate.gate_module.e2e_gate_module import freeze_cast_linears
        _nf = freeze_cast_linears(gate.gate_branch, torch.bfloat16)
        print(f"[gate] scoring branch ({_gb_mode}): "
              f"{sum(p.numel() for p in gate.gate_branch.parameters())/1e6:.2f}M params, "
              f"fp32 storage, compute in the backbone dtype ({_nf} CastLinears' bf16 copies frozen)", flush=True)
    print(f"[gate] ready (layer={gate_layer}, d_model={gate.d_model})", flush=True)
    return model, gate, input_all_prev_layers

