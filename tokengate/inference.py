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
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import load_gate_model
    model, gate, _ = load_gate_model(base_model, gate_path, gate_layer,
                                     gate_attn=gate_attn)
    return TokenGateModel(model, gate, gate_layer, base_model)
