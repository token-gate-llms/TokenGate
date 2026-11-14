import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tokengate.experimental.jev import (
    _DecisionPrefillModel,
    gated_decide,
    input_pruned_decide,
    standard_decide,
    train_jev_gate_step,
)
from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.gate_module.gate_forward import (
    _apply_gate_bias, _attn_only_forward_eager, build_gate_branch, call_layer_padsafe,
)


class _Question:
    name = "refund"
    type = "noul"
    options = ["false", "true"]

    def answer(self, probabilities):
        return {"true_probability": probabilities[1]}


class _DecisionConfig:
    logits_processors = {"noul": lambda tail, scores: scores}

    def prepare_inputs(self, request, tokenizer):
        state = request["state"]
        ids = [10, 11] + ([20, 99, 21] if state else []) + [30, 31]
        return [({"input_ids": torch.tensor([ids]), "marks": torch.tensor([len(ids)])}, [_Question()])]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.proj = nn.Linear(2, 24)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(), offsets={"noul": 0}, softcap=None
        )
        self.decision_config = _DecisionConfig()

    def decide(self, input_ids, marks, decision_questions):
        assert int(marks.reshape(-1)[0]) == input_ids.shape[1]
        hit = 99 in input_ids[0].tolist()
        return [torch.tensor([0.0, 3.0] if hit else [3.0, 0.0])]


class JevBridgeTests(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.request = {"state": "needle", "questions": {"refund": {"type": "noul"}}}

    def test_baseline_and_input_pruning(self):
        full = standard_decide(self.model, None, self.request)
        short = input_pruned_decide(self.model, None, self.request, token_budget=1)
        middle = input_pruned_decide(self.model, None, self.request, token_budget=1, method="middle")
        suffix = input_pruned_decide(self.model, None, self.request, token_budget=1, method="suffix")
        random = input_pruned_decide(self.model, None, self.request, token_budget=3, method="random")
        self.assertEqual(full.state_tokens, 3)
        self.assertGreater(full.answers["refund"]["true_probability"], 0.9)
        self.assertLess(short.answers["refund"]["true_probability"], 0.1)
        self.assertGreater(middle.answers["refund"]["true_probability"], 0.9)
        self.assertLess(suffix.answers["refund"]["true_probability"], 0.1)
        self.assertEqual(random.kept_positions, list(range(7)))

    def test_gated_scores_use_decision_head_slots(self):
        raw = torch.zeros(1, 24)
        raw[0, 1] = 4
        def fake_prefill(wrapper, gate, ids, starts, ends, layer, k, **kwargs):
            self.assertEqual((starts, ends, layer, k), ([2], [5], 1, 2))
            self.assertTrue(kwargs["return_logits"])
            self.assertIsInstance(wrapper, _DecisionPrefillModel)
            return raw, None, torch.tensor([[0, 1, 3, 4, 5, 6]]), [6], [2]
        with patch("tokengate.experimental.jev.prefill_batch_for_decode", fake_prefill):
            out = gated_decide(self.model, object(), None, self.request, gate_layer=1, token_budget=2)
        self.assertGreater(out.answers["refund"]["true_probability"], 0.9)
        self.assertEqual(out.kept_state_tokens, 2)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            input_pruned_decide(self.model, None, self.request, token_budget=0)
        with self.assertRaises(ValueError):
            gated_decide(self.model, object(), None, self.request, gate_layer=1, token_budget=0)

    def test_peft_lora_is_folded_into_gate_branch(self):
        class PeftLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = nn.Linear(2, 3, bias=False)
                self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
                self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 3, bias=False)})
                self.active_adapters = ["default"]
                self.scaling = {"default": 0.5}
                self.fan_in_fan_out = False
                self.merged = False

        mlp = nn.Module()
        mlp.gate_proj = PeftLinear()
        mlp.up_proj = nn.Linear(2, 3, bias=False)
        mlp.down_proj = nn.Linear(3, 2, bias=False)
        layer = nn.Module()
        layer.mlp = mlp
        backbone = nn.Module()
        backbone.layers = nn.ModuleList([layer])
        wrapper = nn.Module()
        wrapper.model = backbone
        gate = E2EGate(d_model=2, n_heads=1, n_layers_bias=1)
        branch = build_gate_branch(wrapper, gate, 0)
        source = mlp.gate_proj
        expected = source.base_layer.weight.float() + 0.5 * (
            source.lora_B["default"].weight.float() @ source.lora_A["default"].weight.float()
        )
        self.assertTrue(torch.allclose(branch.gate_proj.weight, expected))
        self.assertEqual(branch.gate_proj.weight.dtype, torch.float32)

    def test_decision_distillation_updates_gate(self):
        class Gate(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.0))
        gate = Gate()
        optimizer = torch.optim.SGD(gate.parameters(), lr=0.1)

        def fake_forward(wrapper, actual_gate, batch, layer, **kwargs):
            self.assertTrue(kwargs["frozen_backbone"])
            self.assertEqual(batch[0]["K"], 1)
            logits = torch.cat([torch.zeros(1), actual_gate.weight.view(1), torch.zeros(22)]).view(1, 1, 24)
            return logits, torch.tensor([[True]]), torch.tensor([[True]])

        with patch("tokengate.experimental.jev.forward_with_gate_batched", fake_forward):
            loss = train_jev_gate_step(
                self.model, gate, None, self.request, optimizer,
                gate_layer=1, token_budget=1,
            )
        self.assertGreater(loss, 0)
        self.assertGreater(gate.weight.item(), 0)

    def test_new_qwen35_linear_layer_dispatch(self):
        layer = nn.Module()
        layer.block_type = "linear_attention"
        layer.input_layernorm = nn.Identity()
        layer.linear_attn = lambda hidden_states, **kwargs: hidden_states * 2
        hidden = torch.ones(1, 3, 4)
        actual = _attn_only_forward_eager(layer, hidden, None, None, None, None)
        self.assertTrue(torch.equal(actual, hidden * 3))

    def test_new_qwen35_linear_layer_padding(self):
        layer = nn.Module()
        layer.block_type = "linear_attention"
        hidden = torch.tensor([[[1.0], [2.0], [0.0]]])
        seen = []
        def capture(value):
            seen.append(value.clone())
            return value
        actual = call_layer_padsafe(layer, hidden, [1], fn=capture)
        self.assertTrue(torch.equal(seen[0], torch.tensor([[[0.0], [1.0], [2.0]]])))
        self.assertTrue(torch.equal(actual, hidden))

    def test_zero_bias_ablation_preserves_selection_bias_shape(self):
        gate = SimpleNamespace(n_layers_bias=1, zero_gate_bias=True,
                               scaled_softmax_scale=False)
        bias = torch.tensor([[1.0, -2.0]])
        with patch("tokengate.e2e_gate.gate_module.gate_forward._full_attn_layers_above", return_value=[3]), \
             patch("tokengate.e2e_gate.gate_module.gate_forward.set_ssmax_ratio_for_layers"), \
             patch("tokengate.e2e_gate.gate_module.gate_forward.set_gate_bias_for_layers") as set_bias:
            _apply_gate_bias(self.model, gate, 1, 4, bias, per_head=False)
        applied = set_bias.call_args.args[3]
        self.assertEqual(applied.shape, bias.shape)
        self.assertTrue(torch.equal(applied, torch.zeros_like(bias)))


if __name__ == "__main__":
    unittest.main()
