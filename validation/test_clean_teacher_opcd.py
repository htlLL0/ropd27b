#!/usr/bin/env python3
"""CPU-only engineering gates; randomly initialized tiny models, no model scores."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'code/src')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import b200_collection as core
TEST_ROOT = Path(os.environ.get('OPCD_TEST_TMP', str(ROOT / 'output/setup/training_tests')))
os.environ.update(core.runtime_environment(TEST_ROOT))

import torch
from r_opcd.clean_teacher_opcd import (attach_adapter, chunked_backward, clean_messages,
    response_hidden, training_example, train_step)
from r_opcd.objective import compute_full_vocab_objective
from train_clean_teacher import load_checkpoint, save_checkpoint, prepare_training

CFG = json.loads((ROOT / 'config/clean_teacher_opcd.json').read_text())


def tiny_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, layer_types=['linear_attention', 'full_attention'],
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                         'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]})
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                  out_hidden_size=32, num_position_embeddings=16, patch_size=2)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision,
                        image_token_id=61, video_token_id=62, vision_start_token_id=63)
    cfg._attn_implementation = 'sdpa'
    return Qwen3_5ForConditionalGeneration(cfg)


class CleanTeacherTests(unittest.TestCase):
    def test_clean_view_ignores_gold_attack_and_labels(self):
        row = dict(user_query='task', clean_context='original input', contaminated_context='ATTACK',
                   sanitized_context='NOT THE CLEAN INPUT', target_answer='GOLD', malicious_span='ATTACK',
                   attack_success_label=True, risk_weight=0.2)
        expected = [{'role': 'user', 'content': 'task'}, {'role': 'input', 'content': 'original input'}]
        self.assertEqual(clean_messages(row), expected)
        for key in list(row):
            if key not in ('user_query', 'clean_context'):
                changed = copy.deepcopy(row)
                changed[key] = 'MUTATED'
                self.assertEqual(clean_messages(changed), expected)

    def test_empty_clean_context_is_valid(self):
        self.assertEqual(clean_messages({'user_query': 'task', 'clean_context': ''})[1]['content'], '')

    def test_all_response_tokens_and_no_truncation(self):
        item = {'index': 0, 'source': {'id': 'x'}, 'prompt': {
            'prompt_token_ids': [1, 3, 4], 'prompt_attention_mask': [1, 1, 1]}}
        clean = {'prompt_token_ids': [1], 'prompt_attention_mask': [1]}
        response = [7, 8, 2]
        out = training_example(item, clean, response, max_sequence_tokens=6, vocab_size=16)
        self.assertEqual(out['response_ids'], response)
        self.assertEqual(out['student_prompt_ids'], out['teacher_minus_prompt_ids'])
        with self.assertRaises(ValueError):
            training_example(item, clean, response, max_sequence_tokens=5, vocab_size=16)
        with self.assertRaises(ValueError):
            training_example(item, clean, [], max_sequence_tokens=6, vocab_size=16)

    def test_chunk_loss_and_gradient_match_full_vocab_reference(self):
        torch.manual_seed(23)
        for top_k in (None, 5):
            s = torch.randn(1, 7, 6, requires_grad=True)
            m, p = torch.randn_like(s), torch.randn_like(s)
            head = torch.nn.Linear(6, 17, bias=False).requires_grad_(False)
            objective = {**CFG['objective'], 'teacher_plus_top_k': top_k}
            expected = compute_full_vocab_objective(head(s), head(m), head(p),
                attack_gate=torch.ones(1), reliability_weight=torch.ones(1), **objective)
            expected.loss.backward()
            grad, loss = s.grad.clone(), expected.loss.item()
            s2 = s.detach().clone().requires_grad_(True)
            actual = chunked_backward(s2, m, p, head, objective, chunk_size=3)
            self.assertAlmostEqual(actual['loss'], loss, places=6)
            torch.testing.assert_close(s2.grad, grad, atol=1e-6, rtol=1e-5)

    def test_no_disagreement_yields_zero_not_sft(self):
        s = torch.randn(1, 4, 6, requires_grad=True)
        m = torch.randn_like(s)
        head = torch.nn.Linear(6, 17).requires_grad_(False)
        metrics = chunked_backward(s, m, m, head, CFG['objective'], 3)
        self.assertEqual(metrics['selected_tokens'], 0)
        self.assertEqual(metrics['loss'], 0)
        self.assertEqual(s.grad.abs().sum().item(), 0)

    def test_simulated_generation_rejected_before_training(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            generation = Path(temporary) / 'generation'
            output = Path(temporary) / 'train'
            core.save(generation / 'completion.json', {'status': 'complete', 'samples': 9600, 'server_released': True})
            core.save(generation / 'hardware.json', {'simulation_only': True, 'bf16_probe': 'NOT_RUN_SIMULATION'})
            with self.assertRaisesRegex(RuntimeError, 'real B200 generation gate'):
                prepare_training(generation, output, CFG)

    def test_bfloat16_hybrid_model_update(self):
        torch.manual_seed(24)
        model = attach_adapter(tiny_model().to(torch.bfloat16), CFG)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        example = dict(student_prompt_ids=[1, 5, 6, 7, 9, 10], teacher_minus_prompt_ids=[1, 5, 6, 7, 9, 10],
                       teacher_plus_prompt_ids=[1, 5, 6], response_ids=[11, 12, 2])
        metrics = train_step(model, optimizer, example, CFG)
        self.assertGreater(metrics['grad_norm'], 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics['loss'])))
        self.assertEqual(model.get_base_model().lm_head.weight.dtype, torch.bfloat16)

    def test_actual_hybrid_model_update_alignment_frozen_teacher_and_resume(self):
        torch.manual_seed(24)
        base = tiny_model()
        cfg = copy.deepcopy(CFG)
        cfg['runtime']['response_chunk_size'] = 2
        model = attach_adapter(base, cfg)
        attacked, clean, response = [1, 5, 6, 7, 9, 10], [1, 5, 6], [11, 12, 2]
        example = dict(student_prompt_ids=attacked, teacher_minus_prompt_ids=attacked,
                       teacher_plus_prompt_ids=clean, response_ids=response)
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            teacher_before = response_hidden(model, clean, response).clone()
            ids = torch.tensor([attacked + response])
            full = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            direct = base.lm_head(response_hidden(model, attacked, response))
            torch.testing.assert_close(direct, full[:, len(attacked)-1:-1], atol=1e-6, rtol=1e-5)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        metrics = train_step(model, optimizer, example, cfg)
        self.assertGreater(metrics['grad_norm'], 0)
        for n, p in model.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p, frozen[n], atol=0, rtol=0)
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            torch.testing.assert_close(response_hidden(model, clean, response), teacher_before, atol=0, rtol=0)
        parent = TEST_ROOT
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            output = Path(temporary)
            binding = {'config': cfg, 'fixture': 'random_tiny_model_cpu_only'}
            checkpoint = save_checkpoint(model, optimizer, output, 1, binding, metrics)
            train_step(model, optimizer, example, cfg)
            uninterrupted = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            step, _ = load_checkpoint(model, optimizer, output, binding)
            self.assertEqual(step, 1)
            train_step(model, optimizer, example, cfg)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p, uninterrupted[n], atol=0, rtol=0)
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'corrupted')
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, optimizer, output, binding)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
