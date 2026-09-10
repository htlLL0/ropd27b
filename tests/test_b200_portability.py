"""CPU tests for hardware refusal, portable commands and exact-token recovery."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b200_collection', ROOT / 'scripts/b200_collection.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class B200PortabilityTests(unittest.TestCase):
    def gpu(self):
        return {'uuid': 'GPU-test', 'name': 'NVIDIA B200', 'memory_total_mib': 180000,
                'memory_used_mib': 0, 'driver_version': '580.1'}

    def test_reject_a100_before_any_cuda_allocation(self):
        row = self.gpu()
        row['name'] = 'NVIDIA A100-PCIE-40GB'
        with self.assertRaisesRegex(RuntimeError, 'B200 required'):
            runner.validate_gpu_row(row)

    def test_reject_busy_gpu(self):
        row = self.gpu()
        row['memory_used_mib'] = 20000
        with self.assertRaisesRegex(RuntimeError, 'already in use'):
            runner.validate_gpu_row(row)

    def test_reject_mig_capacity(self):
        row = self.gpu()
        row['memory_total_mib'] = 23000
        with self.assertRaisesRegex(RuntimeError, 'full B200'):
            runner.validate_gpu_row(row)

    def test_reject_multiple_gpu_selector_without_running_command(self):
        with patch.object(runner, 'command') as cmd:
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                runner.select_gpu('0,1')
            cmd.assert_not_called()

    def test_run_id_cannot_escape_output_root(self):
        for value in ('../outside', '/tmp/out', 'two words', 'a/b'):
            with self.assertRaises(RuntimeError):
                runner.output_for(value)

    def test_all_runs_are_below_output(self):
        self.assertEqual(runner.output_for('main'), ROOT / 'output/main')
        self.assertEqual(runner.output_for('run2'), ROOT / 'output/run2')

    def test_runtime_caches_and_temporary_files_stay_with_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'output/main'
            env = runner.runtime_environment(output)
            for name, value in env.items():
                self.assertTrue(Path(value).is_relative_to(output), name)
                self.assertTrue(Path(value).is_dir(), name)

    def test_stdout_stderr_and_uncaught_errors_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'output/main'
            code = (
                'import importlib.util, os\n'
                f's=importlib.util.spec_from_file_location("runner", {str(ROOT / "scripts/b200_collection.py")!r})\n'
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n'
                f'm.redirect_run_logs(m.Path({str(output)!r}))\n'
                'print("python stdout marker", flush=True)\n'
                'os.write(1,b"native stdout marker\\n")\n'
                'os.write(2,b"native stderr marker\\n")\n'
                'raise RuntimeError("synthetic logging check")\n'
            )
            result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, '')
            self.assertEqual(result.stderr, '')
            text = (output / 'supervisor.log').read_text()
            for marker in ['python stdout marker', 'native stdout marker', 'native stderr marker',
                           'Traceback', 'synthetic logging check']:
                self.assertIn(marker, text)

    def test_docker_writes_only_to_output_mount(self):
        spec = importlib.util.spec_from_file_location('docker_launcher_test', ROOT / 'scripts/docker_launcher.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'b200_collection': runner}):
            spec.loader.exec_module(module)
        with patch.dict(os.environ, {'OPCD_DOCKER_BUNDLE': '/host path/bundle'}):
            self.assertEqual(module.mount_args(), ['--mount',
                'type=bind,src=/host path/bundle,dst=/bundle,readonly', '--mount',
                'type=bind,src=/host path/bundle/output,dst=/bundle/output'])

    def test_server_only_changes_topology_and_location(self):
        cmd = runner.server_command(12345)
        pairs = {'--tensor-parallel-size': '1', '--dtype': 'bfloat16', '--max-num-seqs': '16',
                 '--max-model-len': '32768', '--max-num-batched-tokens': '4096',
                 '--generation-config': 'vllm', '--host': '127.0.0.1', '--port': '12345'}
        for flag, value in pairs.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--enforce-eager', cmd)
        for text in ('--quantization', '--speculative-config', '--reasoning-parser'):
            self.assertNotIn(text, cmd)
        cfg = runner.config()
        with patch.object(runner, 'ROOT', Path('/some other machine/portable bundle')), patch.object(runner, 'config', return_value=cfg):
            cmd = runner.server_command(12345)
            self.assertEqual(cmd[cmd.index('--model') + 1], '/some other machine/portable bundle/model')

    def test_config_only_changes_locations_metadata_and_gpu_topology(self):
        old = runner.read(ROOT / 'provenance/source_config.json')
        new = runner.config()
        changed = {k for k in new if old.get(k) != new[k]}
        self.assertEqual(changed, {'schema', 'created_at_utc', 'model_path', 'data_path', 'model_source_manifest',
                                  'tensor_parallel_size', 'concurrency', 'physical_gpus', 'gpu_uuids'})
        self.assertEqual(new['concurrency'], old['concurrency'] * 2)

    def test_exact_tokens_and_resume_reject_tampering(self):
        from transformers import AutoTokenizer
        from r_opcd.secopd_aligned_collection import make_record
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
        with (ROOT / 'inputs/prepared.jsonl').open() as stream:
            item = json.loads(next(stream))
        cfg = runner.config()
        # Synthetic parser fixture only; never counted as a generated model response.
        tokens = tokenizer.encode('A brief thought.</think>\nThe answer.', add_special_tokens=False) + [cfg['eos_token_ids'][0]]
        raw = {'model': cfg['served_model_name'], 'choices': [{'token_ids': tokens,
               'prompt_token_ids': item['request']['prompt'], 'finish_reason': 'stop'}],
               'usage': {'prompt_tokens': len(item['request']['prompt']), 'completion_tokens': len(tokens)}}
        record = make_record(item, raw, cfg, tokenizer, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            case = out / 'cases/00000'
            runner.save(case / 'request.json', {'messages': item['messages'], 'body': item['request']})
            runner.save(case / 'raw_response.json', raw)
            runner.save(case / 'trajectory.json', record)
            self.assertEqual(set(runner.completed_records(out, [item], cfg, tokenizer)), {0})
            wrong = deepcopy(raw)
            wrong['choices'][0]['prompt_token_ids'] = [1]
            runner.save(case / 'raw_response.json', wrong)
            with self.assertRaisesRegex(ValueError, 'different prompt'):
                runner.completed_records(out, [item], cfg, tokenizer)
            runner.save(case / 'raw_response.json', raw)
            record['model_revision'] = 'wrong-checkpoint'
            runner.save(case / 'trajectory.json', record)
            with self.assertRaisesRegex(RuntimeError, 'model mismatch'):
                runner.completed_records(out, [item], cfg, tokenizer)

    def test_resume_does_not_silently_retry_incomplete_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'cases/00000').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'No silent re-generation'):
                runner.completed_records(out, [{}], runner.config(), None)

    def test_process_identity_has_stable_start_fingerprint(self):
        first = runner.process_identity(os.getpid())
        self.assertEqual(first, runner.process_identity(os.getpid()))
        self.assertTrue(first['start_ticks'].isdigit())
        self.assertEqual(first['pid'], os.getpid())


if __name__ == '__main__':
    unittest.main(verbosity=2)
