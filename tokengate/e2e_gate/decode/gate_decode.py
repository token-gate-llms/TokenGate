"""Shared batched gate decode — the ONE generation-eval implementation.

Both consumers call this:
  * the trainer's in-training eval (``train_e2e_gate.evaluate``), for its
    batched chunks AND its B=1 / OOM-fallback path;
  * the benchmark harness (``benchmark_gate_e2e.gate_generate_batch`` is a
    thin compat wrapper kept for its 15+ importers);
  * the public API (``tokengate.load_model(...)`` → ``TokenGateModel.generate``).

History: the trainer's batched eval decode loop (2026-05-08) and the
benchmark's timing copy of it (2026-06-01) drifted for four months with
fixes manually cross-ported between them.  This module is the union of
every fix from both sides:
  * pad-slot dec_mask — prefill masks shorter samples' zero-padded cache
    slots via the gate bias, but the bias reset below leaves them
    attendable; catastrophic at K=full (the gatefull 0.212->0.105 collapse);
  * first-EOS truncation — batched decode appends tokens to finished rows
    until ALL rows hit EOS, so answers must be cut at the FIRST EOS (the
    old strip-trailing version destroyed token-F1, 0.391->0.244);
  * form-decode (``FormDecodeState``: decode with the same softmax form as
    prefill, default since 2026-09-13) plus a plain-decode branch for the
    legacy benchmark callers;
  * prune_below_gate=False routing (the default): layers <= gate_layer keep
    their FULL padded phase-1 KV, and get a full-width pad mask + their own
    cache_position stream at decode;
  * sharded-model device handling (masks/positions moved to hs.device each
    step rather than assuming one global CUDA device);
  * CUDA-synced prefill/decode walls (callers that don't time just ignore
    them — two syncs per call are negligible against the Python loop).

Deliberately NOT here (caller policy, do not add): batch chunking, OOM
fallback/halving, memory instrumentation, scoring.  One batch in, one
result out; ``torch.cuda.OutOfMemoryError`` propagates to the caller.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List

import torch

from tokengate.e2e_gate.gate_module.gate_forward import (
    is_linear_attention_layer,
    prefill_batch_for_decode,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    FormDecodeState,
    set_attn_scale_for_layers,
    set_gate_bias_for_layers,
    set_ssmax_ratio_for_layers,
)
from tokengate.generators import _strip_thinking


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Timer:
    """CUDA-synced wall-clock timer (context manager)."""

    def __enter__(self):
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        self.elapsed = time.perf_counter() - self._t0


@dataclass
class GateDecodeResult:
    answers: List[str]        # first-EOS-truncated, _strip_thinking applied
    n_gen: List[int]          # generated tokens per sample (pre-EOS)
    n_kepts: List[int]        # kept cache entries per sample
    kept_pos: torch.Tensor    # (B, max_n_kept) ORIGINAL positions, on device
    K_clampeds: List[int]     # effective kept-article count per sample
    prefill_s: float
    decode_s: float
    steps: int                # decode steps executed (batch-wide)
    token_ids: List[List[int]] = field(default_factory=list)  # includes first EOS


@torch.no_grad()
def generate_batch(
    model,
    gate,
    tokenizer,
    prompt_ids_list: List[List[int]],
    article_starts: List[int],
    article_ends: List[int],
    gate_layer: int,
    max_kept_prefill_tokens,
    max_new_tokens: int,
    *,
    decode_phase2_form: bool = True,
    prune_below_gate: bool = False,
    token_callback: Callable[[torch.Tensor], None] | None = None,
) -> GateDecodeResult:
    """Shared greedy generation for gated and full-context models.

    With ``gate=None``, requests use native Hugging Face generation serially;
    otherwise they use batched gate prefill and decode.

    max_kept_prefill_tokens (the K of the paper) may be a scalar or a
    per-sample list (``prefill_batch_for_decode`` accepts both); it bounds how
    many state tokens survive prefill into the KV cache.  decode_phase2_form=True (the default everywhere since
    2026-09-13) decodes with the prefill softmax form via FormDecodeState;
    False is the legacy plain decode (bias attrs cleared, pad mask only).
    prune_below_gate=False (the default) keeps the FULL phase-1 KV at
    layers <= gate_layer through decode; True slices it to the kept tokens
    (accuracy-identical per the noprunebelowL ablation).

    token_callback, when supplied, receives each batch of token IDs on device,
    starting immediately after prefill and before decode setup. Finished rows
    remain in the batch until all rows finish; token_ids in the result truncates
    each row at its first EOS, including EOS. Timing and CPU delivery are caller
    responsibilities. The callback must not modify the supplied tensor.

    No_grad is load-bearing, not hygiene: with a trainable layer-0 adapter,
    grad-mode decode retains every step's full KV-cache via grad_fn (the
    v16a step-0 OOM).
    """
    if gate is None:
        return _generate_full_context(model, tokenizer, prompt_ids_list, article_starts,
                                      article_ends, max_new_tokens, token_callback)
    inner = model.model
    device = next(model.parameters()).device
    n_layers = len(inner.layers)

    with _Timer() as t_pf:
        first_toks, cache, kept_pos_b, n_kepts, K_clampeds = prefill_batch_for_decode(
            model, gate, prompt_ids_list, article_starts, article_ends,
            gate_layer, max_kept_prefill_tokens,
            prune_below_gate=prune_below_gate,
        )
    if token_callback is not None:
        token_callback(first_toks)
    B = first_toks.shape[0]
    last_valid = torch.tensor([n - 1 for n in n_kepts], device=device)
    next_pos = (kept_pos_b[torch.arange(B), last_valid] + 1).unsqueeze(1).to(device)
    cache_next = kept_pos_b.shape[1]

    # Pad-slot mask for decode: prefill masks each shorter sample's
    # zero-padded cache slots via the gate bias, but the bias reset below
    # leaves them attendable — zero-K logits soak softmax mass.  Harmless at
    # fixed K (n_kept ~equal across the batch), catastrophic at K=full where
    # batch lengths differ by thousands (the gatefull collapse).
    # Decode-appended slots (shared tail) are valid for every sample, stay 0.
    dec_dtype = next(model.parameters()).dtype
    dec_mask = torch.zeros(B, 1, 1, cache_next + max_new_tokens,
                           dtype=dec_dtype, device=device)
    for _b, _nk in enumerate(n_kepts):
        if _nk < cache_next:
            dec_mask[_b, :, :, _nk:cache_next] = torch.finfo(dec_dtype).min

    if not prune_below_gate:
        # layers <= gate_layer kept their FULL padded phase-1 cache (width S):
        # they need their own pad mask (pads at [len_i, S); decode-appended
        # slots valid for all) and their own cache_position (append at S+step).
        S = max(len(p) for p in prompt_ids_list)
        for cached_layer in cache.layers[:gate_layer + 1]:
            keys = getattr(cached_layer, "keys", None)
            if keys is not None:
                assert keys.shape[2] == S, (
                    f"below-gate cache width {keys.shape[2]} != phase-1 padded "
                    f"len {S} — prefill/decode disagree about prune_below_gate")
        below_mask = torch.zeros(B, 1, 1, S + max_new_tokens,
                                 dtype=dec_dtype, device=device)
        for _b, _p in enumerate(prompt_ids_list):
            if len(_p) < S:
                below_mask[_b, :, :, len(_p):S] = torch.finfo(dec_dtype).min

    fds = None
    if decode_phase2_form:
        # Construct IMMEDIATELY after prefill while the bias/ssmax/attn_scale
        # attrs are live (the fold reads them); the ctor also clears them.
        fds = FormDecodeState(model, gate_layer, n_kepts, cache_next,
                              max_new_tokens, dec_mask)
    else:
        # plain decode: nothing folded — still clear the attrs so no compiled
        # code reads a per-batch-swapped attr tensor during decode.
        set_gate_bias_for_layers(model, 0, n_layers, None)
        set_ssmax_ratio_for_layers(model, gate_layer + 1, n_layers, None)
        set_attn_scale_for_layers(model, gate_layer + 1, n_layers, None)

    eos = torch.zeros(B, dtype=torch.bool, device=device)
    all_gen: List[List[torch.Tensor]] = [[first_toks[b:b + 1].to(device)]
                                         for b in range(B)]
    cur = first_toks.to(device)
    steps = 0
    with _Timer() as t_dec:
        for step_i in range(max_new_tokens - 1):
            eos |= (cur[:, 0] == tokenizer.eos_token_id)
            if eos.all():
                break
            hs = inner.embed_tokens(cur)
            hd = hs.device
            pe = inner.rotary_emb(hs, next_pos.to(hd))
            cp = torch.tensor([cache_next], device=hd)
            if fds is not None:
                fds.step(next_pos, step_i)
            # one dec_mask slice per step, not one per layer
            dm = dec_mask[:, :, :, : cache_next + 1]
            if not prune_below_gate:
                below_next = S + steps
                bm = below_mask[:, :, :, : below_next + 1]
                cp_below = torch.tensor([below_next], device=hd)
            for li in range(n_layers):
                if is_linear_attention_layer(inner.layers[li]):
                    # Every row supplies one real generated token; KV padding
                    # masks describe full-attention history, not recurrent input.
                    hs = inner.layers[li](
                        hs, attention_mask=None,
                        position_ids=next_pos.to(hd),
                        past_key_values=cache, use_cache=True,
                        position_embeddings=pe,
                    )
                    continue
                if not prune_below_gate and li <= gate_layer:
                    # full phase-1 cache at this layer: full-width pad mask +
                    # its own append slot (the form fold never covers <= L)
                    hs = inner.layers[li](
                        hs, attention_mask=bm.to(hd),
                        position_ids=next_pos.to(hd),
                        past_key_values=cache, use_cache=True,
                        cache_position=cp_below, position_embeddings=pe,
                    )
                    continue
                am = fds.mask_for(li, cache_next + 1) if fds is not None else dm
                hs = inner.layers[li](
                    hs, attention_mask=am.to(hd),
                    position_ids=next_pos.to(hd),
                    past_key_values=cache, use_cache=True,
                    cache_position=cp, position_embeddings=pe,
                )
            hs = inner.norm(hs)
            nt = model.lm_head(hs)[:, -1].argmax(-1, keepdim=True).to(device)
            if token_callback is not None:
                token_callback(nt)
            for b in range(B):
                all_gen[b].append(nt[b:b + 1])
            cur = nt
            next_pos = next_pos + 1
            cache_next += 1
            steps += 1
    if fds is not None:
        fds.finish()

    answers: List[str] = []
    n_gen: List[int] = []
    token_ids: List[List[int]] = []
    for b in range(B):
        glist = torch.cat(all_gen[b], dim=1)[0].cpu().tolist()
        # Cut at the FIRST EOS: batched decode runs until ALL rows hit EOS and
        # appends every step's token to every row, so an early-finishing row
        # looks like [..., answer, EOS, junk, junk, ...].
        if tokenizer.eos_token_id in glist:
            end = glist.index(tokenizer.eos_token_id)
            token_ids.append(glist[:end + 1])
            glist = glist[:end]
        else:
            token_ids.append(glist)
        n_gen.append(len(glist))
        answers.append(_strip_thinking(
            tokenizer.decode(glist, skip_special_tokens=True)) if glist else "")
    return GateDecodeResult(answers, n_gen, list(n_kepts), kept_pos_b,
                            list(K_clampeds), t_pf.elapsed, t_dec.elapsed, steps,
                            token_ids)


class TokenDelivery:
    """Record serial CPU tokens and their first/last delivery times.

    Hugging Face sends the input prompt to its streamer before generated tokens;
    skip that one call for SFT. TokenGate's callback sends generated tokens only.
    """

    def __init__(self, *, skip_prompt: bool = False, callback=None):
        self.callback = callback
        self.skip_prompt = skip_prompt
        self.token_ids: list[int] = []
        self.first_time: float | None = None
        self.last_time: float | None = None

    def put(self, tokens: torch.Tensor) -> None:
        if self.skip_prompt:
            self.skip_prompt = False
            return
        if tokens.numel() != 1:
            raise ValueError("Decode timing requires one token from one request")
        token = int(tokens.cpu().item())
        now = time.perf_counter()
        if self.first_time is None:
            self.first_time = now
        self.last_time = now
        self.token_ids.append(token)
        if self.callback is not None:
            self.callback(tokens.reshape(1, 1))

    def end(self) -> None:
        """Complete the Hugging Face streamer interface; no text is decoded."""

    def result(self, start: float, eos_token_id: int | None, cap: int) -> dict:
        if self.first_time is None or self.last_time is None:
            raise RuntimeError("Generation delivered no tokens")
        tokens = self.token_ids
        if not 1 <= len(tokens) <= cap or eos_token_id in tokens[:-1]:
            raise RuntimeError("Generation did not stop at the first EOS or cap")
        stopped = tokens[-1] == eos_token_id
        if not stopped and len(tokens) != cap:
            raise RuntimeError("Generation stopped before EOS or the token cap")
        steps = len(tokens) - 1
        elapsed = self.last_time - self.first_time
        return {
            "token_ids": tokens,
            "generated_tokens": len(tokens),
            "decode_tokens": steps,
            "generation_seconds": self.last_time - start,
            "decode_seconds": elapsed if steps else None,
            "decode_seconds_per_token": elapsed / steps if steps else None,
            "finish_reason": "stop" if stopped else "length",
            "stop_reason": eos_token_id if stopped else None,
        }


def _generate_full_context(model, tokenizer, prompts, starts, ends, cap, callback):
    """SFT uses Hugging Face generation, including its configured processors.

    Requests run serially; this path serves validation and batch-one timing.
    The vanilla throughput benchmark retains its explicitly batched HF path.
    """
    if callback is not None and len(prompts) != 1:
        raise ValueError("Token delivery timing requires one request")
    answers, tokens, counts = [], [], []
    prefill_seconds = decode_seconds = 0.0
    for prompt in prompts:
        delivery = TokenDelivery(skip_prompt=True, callback=callback)
        inputs = torch.tensor([prompt], device=model.device)
        _sync()
        start = time.perf_counter()
        output = model.generate(
            inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            max_new_tokens=cap, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, streamer=delivery,
        )
        ids = output[0, len(prompt):].cpu().tolist()
        if ids != delivery.token_ids:
            raise RuntimeError("CPU delivery differs from generated tokens")
        delivery.result(start, tokenizer.eos_token_id, cap)
        prefill_seconds += delivery.first_time - start
        decode_seconds += delivery.last_time - delivery.first_time
        text_ids = ids[:-1] if ids[-1] == tokenizer.eos_token_id else ids
        answers.append(_strip_thinking(tokenizer.decode(text_ids, skip_special_tokens=True)))
        tokens.append(ids)
        counts.append(len(text_ids))
    lengths = [len(prompt) for prompt in prompts]
    positions = torch.full((len(prompts), max(lengths)), -1, device=model.device, dtype=torch.long)
    for i, length in enumerate(lengths):
        positions[i, :length] = torch.arange(length, device=model.device)
    return GateDecodeResult(answers, counts, lengths, positions,
                            [end - start for start, end in zip(starts, ends)],
                            prefill_seconds, decode_seconds, max(len(t) - 1 for t in tokens), tokens)
