#!/usr/bin/env python3
"""CPU AgentDojo/protocol/transport checks; all simulated responses are fixtures."""
import copy
import argparse
import contextlib
import io
import http.server
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'validation')]
import b200_collection as core
TEST_ROOT = ROOT/'output/setup/agentdojo_tests'
os.environ.update(core.runtime_environment(TEST_ROOT))
import agentdojo_runtime as dojo
from test_txt_transfer import TransferTests


class AgentDojoTests(unittest.TestCase):
    def test_vllm_server_import_as_numeric_docker_user(self):
        import getpass
        from vllm.entrypoints.openai import api_server
        self.assertTrue(getpass.getuser())
        self.assertTrue(callable(api_server.run_server))

    def test_complete_frozen_protocol(self):
        report = dojo.check_protocol()
        self.assertEqual(report['cases_per_model'], 1081)
        self.assertEqual(report['counts'], {'benign':97,'injection_utility':35,'attacked':949})

    def test_real_native_ground_truth_scorers_four_suites(self):
        from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
        for name, suite in dojo.suites().items():
            task = next(iter(suite.user_tasks.values()))
            utility, _ = suite.run_task_with_pipeline(GroundTruthPipeline(task), task, None, {})
            self.assertTrue(utility, name)
            injection = next(iter(suite.injection_tasks.values()))
            # Executing the injection goal tests security=True == attacker succeeded.
            attacker_pipeline = GroundTruthPipeline(injection)
            _, security = suite.run_task_with_pipeline(attacker_pipeline, task, injection, {})
            self.assertTrue(security, name)

    def test_error_fallback_is_preserved_but_not_confirmed_asr(self):
        valid = {'status':'valid','utility':True,'security':False}
        failed = {'status':'infrastructure_error','utility':False,'security':True}
        result = dojo.metrics([valid,failed], 2, attacked=True)
        self.assertIsNone(result['asr'])
        self.assertEqual(result['official_raw_asr'], .5)
        self.assertEqual(result['asr_on_valid_cases'], 0)
        self.assertEqual(result['infrastructure_errors'], 1)

    def test_failure_snapshot_contains_logs_and_preserves_status(self):
        from evaluate_agentdojo import export_run
        import txt_transfer
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)/'run'; root.mkdir()
            core.save(root/'summary.json', {'status':'partial_or_errors','simulation_only':True})
            core.save(root/'failure.json', {'type':'SIMULATED_TEST_ERROR','scientific_result':None})
            (root/'supervisor.log').write_text('模拟过程日志😀\n'*20000)
            (root/'adapter_snapshot').mkdir()
            (root/'adapter_snapshot/adapter_model.safetensors').write_bytes(b'fixture weight, do not transport')
            report = export_run(root)
            dest = Path(temporary)/'restored'
            txt_transfer.verify(report['directory'],restored=dest)
            self.assertEqual(core.read(dest/'summary.json')['status'],'partial_or_errors')
            self.assertEqual((dest/'supervisor.log').read_bytes(),(root/'supervisor.log').read_bytes())
            self.assertTrue((dest/'failure.json').is_file())
            self.assertFalse((dest/'adapter_snapshot').exists())
            self.assertTrue(all(p.stat().st_size <= 90000 for p in Path(report['directory']).iterdir()))

    def test_runner_preflight_failure_automatically_exports(self):
        import shutil
        from unittest.mock import patch
        import evaluate_agentdojo
        import txt_transfer
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            project = Path(temporary)
            (project/'config').mkdir(); (project/'inputs').mkdir()
            shutil.copyfile(ROOT/'config/agentdojo.json',project/'config/agentdojo.json')
            shutil.copyfile(ROOT/'inputs/agentdojo_cases.jsonl',project/'inputs/agentdojo_cases.jsonl')
            (project/'bundle_manifest.json').write_text('{"simulation_only":true}')
            out = project/'output/eval_failure'; out.mkdir(parents=True)
            args = argparse.Namespace(run_id='eval_failure',training_run_id='clean_opcd',models=['base'],gpu='0',resume=False)
            with patch.object(core,'ROOT',project), patch.object(core,'select_gpu',side_effect=RuntimeError('SIMULATED_ENV_BLOCK')):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = evaluate_agentdojo.run(args)
            self.assertEqual(code,1)
            self.assertEqual(core.read(out/'progress.json')['stage'],'failed')
            self.assertEqual(core.read(out/'summary.json')['status'],'partial_or_errors')
            latest = core.read(out/'transfer_latest.json')
            self.assertEqual(txt_transfer.verify(out/latest['directory'])['status'],'pass')
            self.assertTrue(core.read(out/'owned_processes_released.json')['server_released'])

    def test_local_http_native_trace_and_model_pairing(self):
        # A loopback-only synthetic model exercises the REAL AgentDojo evaluator.
        captured = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append(request)
                body = json.dumps({'id':'fixture','object':'chat.completion','created':1,
                    'model':'opcd-agentdojo-target','simulation_only':True,
                    'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'<think>Fixture only.</think> I cannot complete this task.'}}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
                self.send_response(200); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        server = http.server.ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            case = core.rows(ROOT/'inputs/agentdojo_cases.jsonl')[0]
            with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
                for label in ('base','trained'):
                    out = Path(tmp)/label; out.mkdir()
                    result = dojo.run_case(case,out,f'http://127.0.0.1:{server.server_port}/v1','opcd-agentdojo-target')
                    self.assertEqual(result['status'],'valid',result['error'])
                    self.assertIs(type(result['utility']),bool)
                    self.assertTrue(list((out/'traces').rglob('*.json')))
                    self.assertTrue(list((out/'requests').rglob('raw_response.json')))
                    with self.assertRaisesRegex(RuntimeError, 'Simulated responses'):
                        dojo.checked_result(out/'result.json',case)
                    self.assertEqual(dojo.checked_result(out/'result.json',case,allow_simulation=True)['utility'],result['utility'])
                    wrong_score = copy.deepcopy(result); wrong_score['utility'] = not result['utility']
                    core.save(out/'result.json',wrong_score)
                    with self.assertRaisesRegex(RuntimeError, 'native AgentDojo trace'):
                        dojo.checked_result(out/'result.json',case,allow_simulation=True)
                    core.save(out/'result.json',result)
                self.assertEqual(captured[0],captured[1])
                self.assertEqual(captured[0]['temperature'],0.0)
                self.assertNotIn('max_tokens',captured[0])
                self.assertTrue(captured[0]['chat_template_kwargs']['enable_thinking'])
                request_path = next((Path(tmp)/'base/requests').rglob('request.json'))
                request_path.write_text('{}')
                with self.assertRaises(RuntimeError): dojo.checked_result(Path(tmp)/'base/result.json',case,allow_simulation=True)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_adapter_merge_matches_unmerged_tiny_model(self):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        from test_clean_teacher_opcd import tiny_model, CFG
        from r_opcd.clean_teacher_opcd import attach_adapter, train_step
        torch.manual_seed(25)
        base = tiny_model()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
            root = Path(tmp)
            base.save_pretrained(root/'base')
            model = attach_adapter(base, copy.deepcopy(CFG))
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
            example = dict(student_prompt_ids=[1,5,6,9],teacher_minus_prompt_ids=[1,5,6,9],
                           teacher_plus_prompt_ids=[1,5],response_ids=[11,12,2])
            train_step(model,optimizer,example,CFG)
            model.eval(); model.save_pretrained(root/'adapter',save_embedding_layers=False)
            ids = torch.tensor([[1,5,6,9,11]])
            with torch.no_grad(): expected = model(input_ids=ids).logits
            fresh = Qwen3_5ForConditionalGeneration.from_pretrained(root/'base')
            loaded = PeftModel.from_pretrained(fresh,root/'adapter')
            merged = loaded.merge_and_unload(safe_merge=True).eval()
            with torch.no_grad(): actual = merged(input_ids=ids).logits
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
