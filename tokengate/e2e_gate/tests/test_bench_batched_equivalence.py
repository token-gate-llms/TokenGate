"""Validate the BENCHMARK inference paths: padded batched generation must match
single-sample (B=1, unpadded) generation.

Two code paths are covered, all engines behind bench_grouped_eval:

  1. gate path    — prefill_batch_for_decode (right-padded phase-1, per-sample
                    top-K prune, zero-padded phase-2 caches) + the batched
                    greedy decode loop in gate_generate_batch.
  2. vanilla path — vanilla_generate_batch (left-padded prefill with explicit
                    position_ids) vs vanilla_generate_single.

For each path we run 4 deliberately different-length prompts once as a padded
B=4 batch and once each at B=1 (no padding), then compare:
  - kept-token positions and first decoded token (gate path),
  - the greedy token streams.

Exact bitwise equality is NOT expected: batched bf16 matmuls and FlexAttention
block boundaries differ between padded and unpadded runs (~1e-3 relative hidden
-state error — same root cause as tests/test_batched_equivalence.py), so greedy
argmax can flip at genuine near-ties and diverge from there. Thresholds below
encode "equivalent up to bf16 non-associativity"; a real padding bug (positions
shifted by pad count, pad tokens attended, cache mis-packing) produces gross
divergence — near-zero kept overlap or immediate garbage — far outside them.

Run with:
    python -m tokengate.e2e_gate.tests.test_bench_batched_equivalence
    # hybrid (gate layer must be a full-attention layer, e.g. 3):
    python -m ... --base-model Qwen/Qwen3.5-2B --gate-layer 3
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch, prefill_batch_for_decode
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices, patch_model_with_flex_attention,
)
from tokengate.e2e_gate.attn.flex_attention_utils import warmup_flex_attention
from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
    gate_generate_batch, load_vanilla_model,
    vanilla_generate_batch, vanilla_generate_single,
)
from tokengate.datasets.tasks import get_task
from tokengate.e2e_gate.tests.test_batched_equivalence import _make_batch_item

K = 50
MAX_NEW = 32
KEPT_OVERLAP_MIN = 0.95    # ≥ ceil(0.95*K) of the K kept positions must agree per sample
EXACT_STREAM_MIN = 3       # of 4 samples must decode to the IDENTICAL token stream
PREFIX_AGREE_MIN = 0.90    # mean common-prefix fraction across samples

# distinct per-position content (no repeated sentences) so gate scores have no
# near-ties for topk to flip on
def _article(n_facts, salt):
    return " ".join(
        f"Fact {i}: patient record {salt}-{i} lists measurement {(i * 7 + salt) % 97} "
        f"taken on day {(i * 3) % 28 + 1}." for i in range(n_facts)
    )


def _prefix_frac(a, b):
    if not a and not b:
        return 1.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b))


def run_test(a):
    assert torch.cuda.is_available(), "benchmark inference paths are CUDA-only"
    device = torch.device("cuda")
    model_name = a.base_model or "Qwen/Qwen3-0.6B"
    gate_layer = a.gate_layer if a.gate_layer is not None else 4
    paths = set(a.paths.split(","))
    unknown = paths - {"gate", "vanilla"}
    assert not unknown, f"unknown --paths entries: {sorted(unknown)}"
    print(f"[test] device={device}  model={model_name}  gate_layer={gate_layer}  "
          f"paths={sorted(paths)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # same tokenizer setup as bench_grouped_eval — vanilla_generate_batch REQUIRES
    # left padding (first run of this test right-padded and caught the corruption)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    task = get_task("biodex_reactions")
    # 4 samples with very different lengths → real padding in the batch
    articles = [_article(n, s) for n, s in ((40, 1), (90, 2), (150, 3), (220, 4))]
    metas = []
    for art in articles:
        item = _make_batch_item(tokenizer, task, art, "male", max_art=4000)
        metas.append({"prompt_ids": item["input_ids"][:item["ctx_len"]],
                      "article_start": item["article_start"],
                      "article_end": item["article_end"]})
    lens = [len(m["prompt_ids"]) for m in metas]
    print(f"[test] prompt lens = {lens}")
    assert len(set(lens)) == len(lens), "test premise: lengths must differ to exercise padding"

    failures = []
    tok = lambda t: tokenizer.encode(t, add_special_tokens=False)

    def _stream_check(label, ans_b, ans_s):
        exact = sum(x == y for x, y in zip(ans_b, ans_s))
        fracs = [_prefix_frac(tok(x), tok(y)) for x, y in zip(ans_b, ans_s)]
        print(f"[test] {label}: exact={exact}/4  prefix-agreement={['%.2f' % f for f in fracs]}")
        for i, (x, y) in enumerate(zip(ans_b, ans_s)):
            if x != y:
                print(f"[test]   sample {i} B=4: {x!r}\n[test]   sample {i} B=1: {y!r}")
        if exact < EXACT_STREAM_MIN:
            failures.append(f"{label}: only {exact}/4 exact streams (need {EXACT_STREAM_MIN})")
        if sum(fracs) / len(fracs) < PREFIX_AGREE_MIN:
            failures.append(f"{label}: mean prefix agreement {sum(fracs)/len(fracs):.2f} "
                            f"< {PREFIX_AGREE_MIN}")

    if "gate" in paths:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": device})
        model = patch_model_with_flex_attention(model)
        model.eval()
        warmup_flex_attention(device)

        nlb = len([i for i in get_full_attn_layer_indices(model) if i > gate_layer])
        gate = E2EGate(d_model=model.config.hidden_size,
                       n_heads=model.config.num_attention_heads,
                       n_layers_bias=nlb).to(device).to(torch.float32)
        gate.set_qk_norm_scales(model, gate_layer)
        build_gate_branch(model, gate, gate_layer)
        # Non-trivial, DETERMINISTIC scores (the folded readout row is zero-init;
        # an unseeded random readout makes near-tie top-K selections — and thus
        # the greedy decode streams — flip run to run).  std large enough to
        # separate scores beyond bf16 batched-vs-single noise.
        torch.manual_seed(20261012)
        torch.nn.init.normal_(gate.gate_branch.down_proj.weight, std=0.05)

        _pb = a.prune_below

        # ── 1a. gate path: prefill selection + first token, B=4 vs B=1 ────────
        with torch.no_grad():
            ft_b, _, kept_b, nk_b, _ = prefill_batch_for_decode(
                model, gate, [m["prompt_ids"] for m in metas],
                [m["article_start"] for m in metas], [m["article_end"] for m in metas],
                gate_layer, K, prune_below_gate=_pb)
            singles = [prefill_batch_for_decode(
                model, gate, [m["prompt_ids"]], [m["article_start"]], [m["article_end"]],
                gate_layer, K, prune_below_gate=_pb) for m in metas]
        for i, (ft_s, _, kept_s, nk_s, _) in enumerate(singles):
            assert nk_b[i] == nk_s[0], f"sample {i}: n_kept {nk_b[i]} (B=4) != {nk_s[0]} (B=1)"
            set_b = set(kept_b[i, :nk_b[i]].tolist())
            set_s = set(kept_s[0, :nk_s[0]].tolist())
            ov = len(set_b & set_s) / max(len(set_s), 1)
            ftm = ft_b[i].item() == ft_s[0].item()
            print(f"[test] sample {i}: kept overlap={ov:.3f}  first_token match={ftm}")
            if ov < KEPT_OVERLAP_MIN:
                failures.append(f"sample {i}: kept-position overlap {ov:.3f} < {KEPT_OVERLAP_MIN}")
            if not ftm:
                failures.append(f"sample {i}: first token {ft_b[i].item()} != {ft_s[0].item()}")

        # ── 1b. gate path: full greedy stream, B=4 vs B=1 ─────────────────────
        with torch.no_grad():
            ans_b, ng_b, _, _, _, _ = gate_generate_batch(
                model, gate, tokenizer, metas, gate_layer, K, MAX_NEW, False,
                prune_below_gate=_pb)
            ans_s = [gate_generate_batch(model, gate, tokenizer, [m], gate_layer, K,
                                         MAX_NEW, False, prune_below_gate=_pb)[0][0]
                     for m in metas]
        _stream_check("gate decode", ans_b, ans_s)

        if not a.prune_below:
            # ── 1c. K=full: prune vs noprune must be IDENTICAL (the kept set is
            # everything, so the two cache layouts hold the same entries at the
            # same widths — any divergence is a mask/cache_position bug) ────────
            KFULL = 10 ** 9
            with torch.no_grad():
                full_p = gate_generate_batch(model, gate, tokenizer, metas, gate_layer,
                                             KFULL, MAX_NEW, False, prune_below_gate=True)[0]
                full_np = gate_generate_batch(model, gate, tokenizer, metas, gate_layer,
                                              KFULL, MAX_NEW, False, prune_below_gate=False)[0]
            _eq = sum(x == y for x, y in zip(full_p, full_np))
            print(f"[test] gate K=full prune-vs-noprune: identical={_eq}/4")
            if _eq < len(metas):
                for i, (x, y) in enumerate(zip(full_p, full_np)):
                    if x != y:
                        print(f"[test]   sample {i} prune:   {x!r}\n[test]   sample {i} noprune: {y!r}")
                failures.append(f"gate K=full: prune vs noprune differ on {len(metas) - _eq}/4 samples")
        del model
        torch.cuda.empty_cache()

    # ── 2. vanilla path: left-padded batch vs single ──────────────────────────
    if "vanilla" in paths:
        vmodel = load_vanilla_model(model_name)
        with torch.no_grad():
            vans_b, _, _, _, _ = vanilla_generate_batch(
                vmodel, tokenizer, [m["prompt_ids"] for m in metas], MAX_NEW)
            vans_s = [vanilla_generate_single(vmodel, tokenizer, m["prompt_ids"], MAX_NEW)[0]
                      for m in metas]
        _stream_check("vanilla decode", vans_b, vans_s)
        del vmodel
        torch.cuda.empty_cache()

    print()
    if failures:
        print("[test] FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"[test] PASSED: padded batched inference ≡ single-sample inference "
          f"(within bf16 non-associativity) on benchmark paths {sorted(paths)}")


def run_real_test(a):
    """Same batched-vs-single check on REAL LongBench rows with a REAL trained
    checkpoint (gate + adapter via load_gate_model — the exact bench loader).

    Samples n_per_subset val rows from each subset, interleaves subsets so every
    batch of 4 mixes datasets (and therefore lengths), and compares the padded
    B=4 greedy streams against each sample decoded alone at K=2000.
    """
    import os, random
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
        load_val_samples, build_metas, load_gate_model)
    from tokengate.e2e_gate.bench.benchmark_sanity_check import manifest_pairs
    from tokengate.datasets.manifest import eval_only_pairs_csv

    subsets = a.subsets.split(",")
    print(f"[real] ckpt={a.gate_path}\n[real] subsets={subsets}  K={a.k}  "
          f"n/subset={a.n_per_subset}  max_new={a.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(a.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    pairs = list(manifest_pairs(a.manifest)) + [
        tuple(pr.split(":", 1)) for pr in eval_only_pairs_csv(a.manifest).split(",") if pr]
    by_tk = {tk: ds for ds, tk in pairs}
    missing = [t for t in subsets if t not in by_tk]
    assert not missing, f"subsets not in manifest {a.manifest}: {missing}"

    rng = random.Random(a.seed)
    metas_by_subset = {}
    model_name = os.path.basename(a.base_model)
    for tk in subsets:
        rows = load_val_samples(a.pretrain_data_root, by_tk[tk], tk, model_name,
                                50, a.max_article_tokens, None)
        assert rows, f"no val rows for {tk}"
        rows = rng.sample(rows, min(a.n_per_subset, len(rows)))
        ml = []
        for s_ in rows:
            task = get_task(s_["task"])
            if hasattr(task, "use_judge"):
                task.use_judge = False
            ml += build_metas(tokenizer, task, [s_])
        metas_by_subset[tk] = ml

    _lens = {tk: len(ml) for tk, ml in metas_by_subset.items()}
    assert len(set(_lens.values())) == 1, (
        f"per-subset meta counts differ (build_metas dropped rows): {_lens} — "
        "zip-interleave would silently truncate; pick different rows/seed")
    # round-robin interleave → every batch of 4 mixes subsets/lengths
    metas = [m for tup in zip(*metas_by_subset.values()) for m in tup]
    print(f"[real] {len(metas)} samples, prompt lens = "
          f"{[len(m['prompt_ids']) for m in metas]}")

    model, gate, _iapl = load_gate_model(a.base_model, a.gate_path, a.gate_layer)

    failures, exact, fracs = [], 0, []
    tok = lambda t: tokenizer.encode(t, add_special_tokens=False)
    for bi in range(0, len(metas), 4):
        batch = metas[bi:bi + 4]
        with torch.no_grad():
            ans_b, _, nk_b, _, _, _ = gate_generate_batch(
                model, gate, tokenizer, batch, a.gate_layer, a.k,
                a.max_new_tokens, False)
            singles = [gate_generate_batch(model, gate, tokenizer, [m], a.gate_layer,
                                           a.k, a.max_new_tokens, False)
                       for m in batch]
        for j, m in enumerate(batch):
            ans_s, nk_s = singles[j][0][0], singles[j][2][0]
            ds = m["sample"]["dataset"]
            if nk_b[j] != nk_s:
                failures.append(f"{ds}[{bi+j}]: n_kept {nk_b[j]} (B=4) != {nk_s} (B=1)")
            fr = _prefix_frac(tok(ans_b[j]), tok(ans_s))
            ex = ans_b[j] == ans_s
            exact += ex; fracs.append(fr)
            print(f"[real] {ds:28s} len={len(m['prompt_ids']):6d}  exact={ex}  "
                  f"prefix-agreement={fr:.2f}")
            if not ex:
                print(f"[real]   B=4: {ans_b[j][:160]!r}\n[real]   B=1: {ans_s[:160]!r}")

    n = len(fracs)
    mean_fr = sum(fracs) / n
    print(f"\n[real] exact={exact}/{n}  mean prefix-agreement={mean_fr:.3f}")
    # Real 39k-token prompts at bf16: padded vs unpadded runs legitimately differ
    # ~1e-3 in hidden states, so long greedy decodes may diverge at a genuine
    # near-tie. A padding BUG shows up as broad divergence, not an occasional flip.
    if exact < 0.6 * n:
        failures.append(f"only {exact}/{n} exact streams (need >= 60%)")
    # 0.75, not higher: long FREE-FORM generations (gov_report summaries) match for
    # ~50 tokens then flip at a genuine near-tie; short factual answers stay exact.
    if mean_fr < 0.75:
        failures.append(f"mean prefix agreement {mean_fr:.3f} < 0.75")
    if failures:
        print("[real] FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        raise SystemExit(1)
    print("[real] PASSED: batched == single-sample outputs on real LongBench rows "
          "(within bf16 non-associativity)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="run on real LongBench val rows with a real checkpoint "
                         "(default: fast synthetic test)")
    ap.add_argument("--gate-path")
    ap.add_argument("--base-model", default=None,
                    help="HF model id; default Qwen/Qwen3-0.6B (synthetic) / "
                         "Qwen/Qwen3-8B (--real). On hybrid Qwen3.5 pass a "
                         "--gate-layer that is a full-attention layer (e.g. 3)")
    ap.add_argument("--gate-layer", type=int, default=None,
                    help="gate layer; default 4 (synthetic) / 0 (--real)")
    ap.add_argument("--paths", default="gate,vanilla",
                    help="comma list of synthetic paths to check: gate,vanilla")
    ap.add_argument("--prune-below", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="run the gate path with the below-gate KV "
                         "PRUNED to the kept tokens (the pre-2026-09 behavior).  "
                         "DEFAULT OFF: keep the full below-gate KV through decode, "
                         "and additionally require K=full prune-vs-noprune streams "
                         "to be identical (they hold the same cache entries)")
    ap.add_argument("--manifest", default="iso_2wiki")
    ap.add_argument("--pretrain-data-root")
    ap.add_argument("--subsets", default="longbench_narrativeqa,longbench_hotpotqa,"
                                         "longbench_gov_report,longbench_trec")
    ap.add_argument("--n-per-subset", type=int, default=2)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-article-tokens", type=int, default=39000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.real:
        # preserve the historical --real defaults
        args.base_model = args.base_model or "Qwen/Qwen3-8B"
        args.gate_layer = args.gate_layer if args.gate_layer is not None else 0
        assert args.gate_path and args.pretrain_data_root, \
            "--real needs --gate-path and --pretrain-data-root"
        run_real_test(args)
    else:
        run_test(args)
