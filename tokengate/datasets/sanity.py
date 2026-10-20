"""Two-baseline sanity check for a candidate pretraining dataset.

Works for any Task subclass + raw samples in the canonical schema.

The two baselines, both run on the target model (Qwen3-8B by default):

  (1) full-context oracle  — no compression, vanilla SDPA.
      Tests whether the base model can follow the task instruction at all.
      Threshold ("criterion 1") is intentionally weak (oracle_acc > 0) so
      it surfaces obviously broken prompts; the user is expected to also
      eyeball full_context.jsonl for degenerate generations (instruction
      not followed, repeated tokens, refusal, etc.).

  (2) init-gated at K       — FlexAttention-patched model with a zero-init
      E2EGate at gate_layer=0.  If gated_acc < oracle_acc, the gate has
      headroom to learn into; if gated_acc ~= oracle_acc, the article isn't
      being used and the dataset is a poor fit for pruning-style training.

Output:
  full_context.jsonl  — per-sample {id, gen, answer, score}
  init_gated.jsonl    — per-sample {id, gen, answer, score, n_art}
  results.json        — aggregate + pass flags for both criteria
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices, patch_model_with_flex_attention,
    set_gate_bias_for_layers,
)
from tokengate.e2e_gate.attn.flex_attention_utils import warmup_flex_attention
from tokengate.e2e_gate.gate_module.gate_forward import (
    apply_chat_template, build_gate_branch, find_article_boundaries, prefill_for_decode,
)
from tokengate.generators import _strip_thinking
from tokengate.datasets.tasks.base import Task


def _generate_with_cache(model, tokenizer, prompt_ids: List[int], device: str,
                         max_gen: int) -> List[int]:
    """Greedy generate from a vanilla (unpatched, SDPA) model with a fresh cache."""
    inner = model.model
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    L = inp.shape[1]
    with torch.no_grad():
        hs = inner.embed_tokens(inp)
        pos = torch.arange(L, device=device).unsqueeze(0)
        pe = inner.rotary_emb(hs, pos)
        cache = DynamicCache()
        cp = torch.arange(L, device=device)
        for layer in inner.layers:
            hs = layer(hs, attention_mask=None, position_ids=pos,
                       past_key_values=cache, use_cache=True,
                       cache_position=cp, position_embeddings=pe)
        out_tokens: List[int] = []
        cur_pos = L
        for _ in range(max_gen):
            hs_n = inner.norm(hs)
            tok = model.lm_head(hs_n)[:, -1].argmax(-1, keepdim=True)
            tid = tok.item()
            out_tokens.append(tid)
            if tid == tokenizer.eos_token_id:
                break
            hs = inner.embed_tokens(tok)
            pos_n = torch.tensor([[cur_pos]], device=device)
            pe = inner.rotary_emb(hs, pos_n)
            cp = torch.tensor([cur_pos], device=device)
            for layer in inner.layers:
                hs = layer(hs, attention_mask=None, position_ids=pos_n,
                           past_key_values=cache, use_cache=True,
                           cache_position=cp, position_embeddings=pe)
            cur_pos += 1
    return out_tokens


def _run_oracle(model, tokenizer, task: Task, raw_samples: List[Dict],
                device: str, max_gen: int):
    log = []
    for s in raw_samples:
        prompt = task.format_prompt(s["article"])
        prompt_ids = apply_chat_template(tokenizer, prompt)
        gen_ids = _generate_with_cache(model, tokenizer, prompt_ids, device, max_gen)
        if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
            gen_ids = gen_ids[:-1]
        gen_text = _strip_thinking(tokenizer.decode(gen_ids, skip_special_tokens=True))
        score = task.evaluate(gen_text, s["answer"], sample=s)
        log.append({"id": s.get("id"), "gen": gen_text,
                    "answer": s["answer"], "score": float(score)})
    acc = sum(r["score"] for r in log) / max(len(log), 1)
    return acc, log


def _run_init_gated(model, gate, tokenizer, task: Task, raw_samples: List[Dict],
                    gate_layer: int, K: int, device: str, max_gen: int):
    inner = model.model
    log = []
    skipped = 0
    for s in raw_samples:
        non_comp = task.get_non_compressible_context(s["article"])
        prompt_ids = apply_chat_template(tokenizer, task.format_prompt(s["article"]))
        empty_ids = apply_chat_template(tokenizer, task.format_prompt(non_comp))
        pre_len, post_len = find_article_boundaries(prompt_ids, empty_ids)
        article_start = pre_len
        article_end = len(prompt_ids) - post_len
        n_art = article_end - article_start
        if n_art <= 0:
            skipped += 1
            continue
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            first_token, cache, kept_pos, n_kept, _ = prefill_for_decode(
                model, gate, input_ids, article_start, article_end, gate_layer, K,
            )
            tokens = [first_token]
            cur_pos = (kept_pos[0, -1] + 1).to(device).view(1, 1)
            cache_next = n_kept
            # Decode runs without gate bias (kept positions are already chosen).
            set_gate_bias_for_layers(model, 0, len(inner.layers), None)
            for _ in range(max_gen - 1):
                if tokens[-1].item() == tokenizer.eos_token_id:
                    break
                cur_tok = tokens[-1]
                hs = inner.embed_tokens(cur_tok)
                pe = inner.rotary_emb(hs, cur_pos.to(hs.device))
                cp = torch.tensor([cache_next], device=hs.device)
                for layer in inner.layers:
                    hs = layer(hs, attention_mask=None,
                               position_ids=cur_pos.to(hs.device),
                               past_key_values=cache, use_cache=True,
                               cache_position=cp, position_embeddings=pe)
                hs = inner.norm(hs)
                tok = model.lm_head(hs)[:, -1].argmax(-1, keepdim=True).to(device)
                tokens.append(tok)
                cur_pos = cur_pos + 1
                cache_next += 1
        if tokens and tokens[-1].item() == tokenizer.eos_token_id:
            tokens = tokens[:-1]
        if tokens:
            gen_text = _strip_thinking(
                tokenizer.decode(torch.cat(tokens, dim=1)[0].cpu(), skip_special_tokens=True)
            )
        else:
            gen_text = ""
        score = task.evaluate(gen_text, s["answer"], sample=s)
        log.append({"id": s.get("id"), "gen": gen_text,
                    "answer": s["answer"], "score": float(score),
                    "n_art": int(n_art)})
    acc = sum(r["score"] for r in log) / max(len(log), 1)
    return acc, log, skipped


def run_sanity_check(
    *,
    model_name: str,
    task: Task,
    raw_samples: List[Dict],
    K: int = 2000,
    output_dir: Path,
    gate_layer: int = 0,
    device: str = "cuda:0",
    max_gen: int = 50,
    sanity_max_article_tokens: int = 30000,
    sanity_n: int = 20,
) -> Dict:
    """Run both baselines and write per-sample logs + aggregate results.

    The full candidate ``raw_samples`` list is first length-filtered to those
    whose article tokenizes to at most ``sanity_max_article_tokens`` tokens,
    then the first ``sanity_n`` survivors are kept. This prevents OOM on
    datasets where the leading samples are far longer than the model's
    context window (e.g. NarrativeQA, where the first story is 199K tokens
    > Qwen3-8B's 131K).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Length-filter the full pool first, then take the first sanity_n
    # survivors. We measure article-token length the cheap way (encode raw
    # article, no chat template) — the actual prompt is slightly longer but
    # well within margin at 30K.
    n_in = len(raw_samples)
    filtered: List[Dict] = []
    n_inspected = 0
    n_skipped_long = 0
    for s in raw_samples:
        n_inspected += 1
        ids = tokenizer.encode(s["article"], add_special_tokens=False)
        if len(ids) <= sanity_max_article_tokens:
            filtered.append(s)
            if len(filtered) >= sanity_n:
                break
        else:
            n_skipped_long += 1
    if n_skipped_long:
        print(
            f"[sanity] {task.name}: length-filtered first {n_inspected} of {n_in} "
            f"raw_samples → kept {len(filtered)} (dropped {n_skipped_long} > "
            f"{sanity_max_article_tokens} tokens)"
        )
    if not filtered:
        raise RuntimeError(
            f"[sanity] {task.name}: 0 samples remain after length filter "
            f"(max={sanity_max_article_tokens} tokens, scanned={n_inspected} of "
            f"{n_in}). Raise sanity_max_article_tokens."
        )
    raw_samples = filtered

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa",
    ).eval()

    print(f"[sanity] {task.name}: oracle (full context), n={len(raw_samples)}")
    oracle_acc, oracle_log = _run_oracle(
        model, tokenizer, task, raw_samples, device, max_gen,
    )
    with (output_dir / "full_context.jsonl").open("w") as f:
        for r in oracle_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sanity] {task.name}: oracle_acc = {oracle_acc:.3f}")

    # Patching is irreversible — must come after oracle.
    print(f"[sanity] {task.name}: patching with FlexAttention, init gate at K={K}")
    patch_model_with_flex_attention(model)
    n_heads = model.config.num_attention_heads
    n_layers_bias = len(
        [i for i in get_full_attn_layer_indices(model) if i > gate_layer])
    gate = E2EGate(
        d_model=model.config.hidden_size,
        n_heads=n_heads,
        n_layers_bias=n_layers_bias,
        
    ).to(device)
    gate.set_qk_norm_scales(model, gate_layer)
    build_gate_branch(model, gate, gate_layer)
    # non-trivial scores (the folded readout row is zero-init)
    torch.nn.init.normal_(gate.gate_branch.down_proj.weight, std=0.02)
    warmup_flex_attention(device=torch.device(device))

    gated_acc, gated_log, skipped = _run_init_gated(
        model, gate, tokenizer, task, raw_samples, gate_layer, K, device, max_gen,
    )
    with (output_dir / "init_gated.jsonl").open("w") as f:
        for r in gated_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results = {
        "model": model_name,
        "task": task.name,
        "n_samples": len(raw_samples),
        "n_skipped_length_filter": int(n_skipped_long),
        "sanity_max_article_tokens": int(sanity_max_article_tokens),
        "K": K,
        "gate_layer": gate_layer,
        "skipped_init_gated": skipped,
        "oracle_acc": float(oracle_acc),
        "init_gated_acc": float(gated_acc),
        "gap_oracle_minus_gated": float(oracle_acc - gated_acc),
        "criterion_1_oracle_coherent": bool(oracle_acc > 0.0),
        "criterion_2_has_headroom": bool(gated_acc < oracle_acc),
    }
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(
        f"[sanity] {task.name}: oracle={oracle_acc:.3f}  "
        f"init_gated@K={K}={gated_acc:.3f}  "
        f"gap={results['gap_oracle_minus_gated']:+.3f}  "
        f"crit1={'PASS' if results['criterion_1_oracle_coherent'] else 'FAIL'}  "
        f"crit2={'PASS' if results['criterion_2_has_headroom'] else 'FAIL'}"
    )
    return results
