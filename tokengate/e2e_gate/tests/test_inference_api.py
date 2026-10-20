"""CPU tests for the public inference API (tokengate.inference).

Validates the state-sentinel contract: apply_chat_template(state=...) returns
one ids list whose sentinels recover EXACTLY the span the manual
full/empty-render diff (compute_prunable_span) produces, in chat and raw
modes; split_state_span fails loudly on ids without a valid span; and the
demo's task-derived {state} template renders token-identically to the task's
own format_prompt. Needs the Qwen3-1.7B tokenizer (HF cache), no GPU.

Run: python -m tokengate.e2e_gate.tests.test_inference_api
"""

ARTICLE = ("Within 2 days of starting oral amoxicillin the patient developed "
           "a pruritic maculopapular rash and mild facial swelling. " * 20)
PROMPT = ("Answer the question using only the biomedical article below. "
          "Answer with the adverse drug reaction terms only, separated by "
          "commas.\n\nArticle:\n{state}\n\nAnswer:")


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_sentinel_round_trip(tok):
    from tokengate import (STATE_END_ID, STATE_START_ID, apply_chat_template,
                           split_state_span)
    from tokengate.e2e_gate.gate_module.gate_forward import compute_prunable_span
    for use_chat in (True, False):
        ids = apply_chat_template(tok, PROMPT, use_chat_template=use_chat,
                                  state=ARTICLE)
        assert ids.count(STATE_START_ID) == 1 and ids.count(STATE_END_ID) == 1
        clean, s, e = split_state_span(ids)
        # reference: the manual two-render diff
        full = apply_chat_template(tok, PROMPT.replace("{state}", ARTICLE),
                                   use_chat_template=use_chat)
        empty = apply_chat_template(tok, PROMPT.replace("{state}", ""),
                                    use_chat_template=use_chat)
        ref_s, ref_e = compute_prunable_span(full, empty)
        assert clean == full, "sentinel-stripped ids != plain render"
        assert (s, e) == (ref_s, ref_e), (
            f"sentinel span {(s, e)} != diff span {(ref_s, ref_e)} "
            f"(chat={use_chat})")
        assert e - s > 100, "state span implausibly small"
    print("  [pass] sentinel span == compute_prunable_span (chat + raw)")


def test_split_fails_loudly(tok):
    from tokengate import apply_chat_template, split_state_span
    plain = apply_chat_template(tok, PROMPT.replace("{state}", ARTICLE))
    for bad in (plain,                      # no sentinels
                plain + [-1002],            # end only
                [-1001] + plain,            # start only
                [-1002] + plain + [-1001]): # reversed
        try:
            split_state_span(bad)
        except ValueError:
            continue
        raise AssertionError("split_state_span accepted ids without a valid span")
    try:
        apply_chat_template(tok, "no placeholder here", state=ARTICLE)
    except ValueError:
        pass
    else:
        raise AssertionError("apply_chat_template accepted a prompt without {state}")
    print("  [pass] missing/malformed spans and placeholders fail loudly")


def test_demo_template_matches_task(tok):
    """task.format_prompt('{state}') + state= must equal format_prompt(article)."""
    from tokengate import apply_chat_template, split_state_span
    from tokengate.datasets.tasks import get_task
    task = get_task("biodex_reactions")
    ids = apply_chat_template(tok, task.format_prompt("{state}"),
                              use_chat_template=task.use_chat_template,
                              state=ARTICLE)
    clean, s, e = split_state_span(ids)
    direct = apply_chat_template(tok, task.format_prompt(ARTICLE),
                                 use_chat_template=task.use_chat_template)
    assert clean == direct, "templated render differs from task.format_prompt"
    assert e > s
    print("  [pass] demo {state} template token-identical to task.format_prompt")


def main():
    print("[test_inference_api]")
    tok = _tok()
    test_sentinel_round_trip(tok)
    test_split_fails_loudly(tok)
    test_demo_template_matches_task(tok)
    print("[test_inference_api] ALL PASS")


if __name__ == "__main__":
    main()
