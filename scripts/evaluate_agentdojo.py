#!/usr/bin/env python3
"""Single-B200 base/trained AgentDojo evaluation with automatic TXT snapshots."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import b200_collection as core
import agentdojo_runtime as dojo
import txt_transfer


def environment():
    value = core.check_environment()
    for name, expected in {'agentdojo': '0.1.30', 'peft': '0.20.0', 'openai': '2.44.0', 'pydantic': '2.13.4',
                           'google-cloud-bigtable': '2.31.0', 'protobuf': '5.29.6'}.items():
        value[name] = importlib.metadata.version(name)
        core.require(value[name] == expected, f'Wrong {name} version; rebuild Dockerfile.agentdojo')
    import agentdojo
    core.require(Path(agentdojo.__file__).resolve().is_relative_to(core.ROOT / 'third_party/agentdojo'),
                 'Must load the pinned bundled AgentDojo source')
    return value


def checkpoint_snapshot(training, output):
    """Pin the completed final adapter, independent of future training-directory changes."""
    completion = core.read(training / 'completion.json')
    latest = core.read(training / 'latest_checkpoint.json')
    core.require(completion['status'] == 'complete' and completion['steps'] == 9600
                 and completion['checkpoint'] == latest, 'A completed final training checkpoint is required')
    path = (training / latest['path']).resolve()
    core.require(path.parent == (training / 'checkpoints').resolve(), 'Invalid checkpoint path')
    core.require(core.sha(path / 'commit.json') == latest['commit_sha256'], 'Checkpoint commit hash mismatch')
    commit, state = core.read(path / 'commit.json'), core.read(path / 'state.json')
    core.require(state['step'] == commit['step'] == 9600 and
                 state['binding']['config']['model_revision'] == core.config()['model_revision'], 'Checkpoint/base mismatch')
    for name, sha in commit['files_sha256'].items():
        core.require(Path(name).name == name and core.sha(path / name) == sha, 'Checkpoint payload changed')
    folder = output / 'adapter_snapshot'
    folder.mkdir(exist_ok=True)
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        target = folder / name
        if target.exists():
            core.require(core.sha(target) == commit['files_sha256'][name], 'Existing snapshot differs')
        else:
            shutil.copyfile(path / name, target)
    identity = {'training_run': training.name, 'checkpoint_path': latest['path'], 'step': state['step'],
        'commit_sha256': latest['commit_sha256'], 'adapter_files_sha256': {
            name: core.sha(folder / name) for name in ('adapter_model.safetensors', 'adapter_config.json')},
        'model_revision': core.config()['model_revision'], 'checkpoint_selection': 'completed final; no test selection'}
    core.save(output / 'checkpoint_identity.json', identity)
    return identity


def merge_model(output):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    target = output / 'merged_model'
    core.require(not target.exists(), 'Merged model already exists')
    pending = output / 'tmp/merged_model_pending'
    core.require(not pending.exists(), 'Incomplete merge exists in tmp/merged_model_pending; preserve and inspect it before retrying')
    size = sum(v['bytes'] for k, v in core.read(core.ROOT / 'bundle_manifest.json')['files'].items() if k.startswith('model/') and k.endswith('.safetensors'))
    core.require(shutil.disk_usage(output).free > size + 10*1024**3, 'Need space for a separate ~56 GB merged model plus 10 GiB reserve')
    identity = core.read(output / 'checkpoint_identity.json')
    for name, sha in identity['adapter_files_sha256'].items():
        core.require(core.sha(output / 'adapter_snapshot' / name) == sha, 'Adapter snapshot changed')
    base = Qwen3_5ForConditionalGeneration.from_pretrained(core.ROOT / 'model', local_files_only=True,
                dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, output / 'adapter_snapshot', is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(pending, safe_serialization=True, max_shard_size='4GB')
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': core.sha(p)} for p in sorted(pending.iterdir()) if p.is_file()}
    pending.rename(target)
    core.save(output / 'merge_receipt.json', {'checkpoint': identity, 'files': files,
              'original_base_unchanged': True, 'tokenizer': 'original bundled model tokenizer', 'at_utc': core.now()})


def verify_merge(output, identity):
    receipt = core.read(output / 'merge_receipt.json')
    core.require(receipt['checkpoint'] == identity, 'Merged checkpoint identity differs')
    for name, entry in receipt['files'].items():
        core.require(Path(name).name == name, 'Invalid merged model file')
        path = output / 'merged_model' / name
        core.require(path.stat().st_size == entry['bytes'] and core.sha(path) == entry['sha256'], 'Merged model file changed')


def server_command(model_path, port):
    cfg = dojo.config()
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model_path),
        '--tokenizer', str(core.ROOT / 'model'), '--served-model-name', 'opcd-agentdojo-target',
        '--host', '127.0.0.1', '--port', str(port), '--tensor-parallel-size', '1', '--dtype', 'bfloat16',
        '--max-model-len', str(cfg['max_model_len']), '--gpu-memory-utilization', '0.90', '--max-num-seqs', '1',
        '--max-num-batched-tokens', '4096', '--language-model-only', '--enforce-eager',
        '--disable-custom-all-reduce', '--generation-config', 'vllm', '--seed', str(cfg['seed'])]


def export_run(output):
    """Called after owned children stop and logs are flushed; exports failures too."""
    logs = output / 'launcher_logs'
    logs.mkdir(exist_ok=True)
    for path in (core.ROOT / 'output/logs').glob('agentdojo_*.log'):
        shutil.copyfile(path, logs / path.name)
    sys.stdout.flush()
    sys.stderr.flush()
    destination = output / 'transfers' / f'snapshot_{time.time_ns()}'
    receipt = txt_transfer.pack(output, destination)
    core.save(output / 'transfer_latest.json', {'directory': str(destination.relative_to(output)),
              'parts': receipt['parts'], 'files': receipt['files'], 'stream_sha256': receipt['stream_sha256'],
              'max_txt_bytes': receipt['max_file_bytes'], 'at_utc': core.now()})
    return receipt


def run(args):
    output = core.output_for(args.run_id)
    lock = (output / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = dojo.config()
    core.require(not (output / 'completion.json').exists(), 'Evaluation already finished; use a new run ID')
    saved = output / 'run_config.json'
    run_cfg = {'evaluation': cfg, 'models': args.models, 'training_run_id': args.training_run_id,
               'bundle_manifest_sha256': core.sha(core.ROOT / 'bundle_manifest.json')}
    if saved.exists():
        core.require(args.resume and core.read(saved) == run_cfg, 'Resume requires the same config/models/inputs/code')
    else:
        core.require(not args.resume, 'No previous evaluation')
        core.save(saved, run_cfg)
    stopping, server, child = False, None, None
    results = []
    cases = core.rows(core.ROOT / 'inputs/agentdojo_cases.jsonl')
    start, baseline = time.monotonic(), 0
    current_model, current_case = None, None
    stage = 'preflight'

    def stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def progress():
        elapsed, count = time.monotonic()-start, len(results)-baseline
        core.save(output / 'progress.json', {'stage': stage, 'model': current_model, 'completed': len(results),
            'active_case': current_case,
            'total': len(cases)*len(args.models), 'errors': sum(r['status'] != 'valid' for r in results),
            'elapsed_seconds_this_session': elapsed,
            'estimated_remaining_seconds': elapsed/count*(len(cases)*len(args.models)-len(results)) if count >= 10 else None,
            'updated_at_utc': core.now()})

    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(15):
            progress()

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    exit_code = 0
    try:
        progress()
        gpu = core.select_gpu(args.gpu)
        core.save(output / 'asset_gate.json', core.verify_assets())
        core.save(output / 'environment.json', environment())
        core.save(output / 'protocol.json', dojo.check_protocol())
        core.save(output / 'hardware.json', core.probe_gpu(gpu))
        identity = checkpoint_snapshot(core.output_for(args.training_run_id), output) if 'trained' in args.models else None
        binding = {'config': run_cfg, 'base_model': core.config()['model_name'], 'base_revision': core.config()['model_revision'],
                   'checkpoint': identity, 'cases_sha256': core.sha(core.ROOT / 'inputs/agentdojo_cases.jsonl')}
        if (output / 'run_binding.json').exists():
            core.require(core.read(output / 'run_binding.json') == binding, 'Evaluation binding changed')
        else:
            core.save(output / 'run_binding.json', binding)
        for model in args.models:
            for case in cases:
                committed = sorted((output / model / 'cases' / case['case_id']).glob('attempt_*/result.json'))
                if committed:
                    saved_result = dojo.checked_result(committed[-1], case, binding=binding, model=model)
                    if saved_result['status'] == 'valid':
                        results.append(saved_result)
        baseline, start = len(results), time.monotonic()
        for model in args.models:
            current_model = model
            pending = []
            for case in cases:
                committed = sorted((output/model/'cases'/case['case_id']).glob('attempt_*/result.json'))
                if not committed or dojo.checked_result(committed[-1], case, binding=binding, model=model)['status'] != 'valid':
                    pending.append(case)
            if not pending or stopping:
                continue
            if model == 'trained':
                if not (output / 'merge_receipt.json').exists():
                    stage = 'merging_adapter'
                    progress()
                    command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), 'merge', '--run-id', args.run_id]
                    core.save(output / 'merge_command.json', command)
                    with (output / 'merge.log').open('a') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        while child.poll() is None:
                            progress()
                            time.sleep(2)
                    core.require(child.returncode == 0, 'Adapter merge failed; inspect merge.log')
                    child = None
                verify_merge(output, identity)
                if stopping:
                    break
            stage = 'model_loading'
            progress()
            folder = output / model
            folder.mkdir(exist_ok=True)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            command = server_command(core.ROOT/'model' if model == 'base' else output/'merged_model', port)
            core.save(folder / 'server_command.json', command)
            with (folder / 'server.log').open('a') as log:
                server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+1800
            endpoint = f'http://127.0.0.1:{port}/v1'
            while True:
                core.require(server.poll() is None and time.monotonic() < deadline, 'vLLM server failed or timed out')
                if stopping:
                    break
                try:
                    with urllib.request.urlopen(endpoint+'/models', timeout=5) as reply:
                        models = json.load(reply)
                    core.require(any(m['id'] == 'opcd-agentdojo-target' for m in models['data']), 'Wrong model endpoint')
                    break
                except (OSError, TimeoutError):
                    time.sleep(2)
            stage = 'evaluating'
            for case in pending:
                if stopping:
                    break
                current_case = case['case_id']
                progress()
                case_dir = folder / 'cases' / case['case_id']
                attempt = case_dir / f'attempt_{time.time_ns()}'
                attempt.mkdir(parents=True)
                core.save(attempt / 'context.json', {'case': case, 'model': model, 'binding': binding,
                                                    'at_utc': core.now(), 'evaluation_is_real': True})
                result = dojo.run_case(case, attempt, endpoint, 'opcd-agentdojo-target')
                results.append(result)
                progress()
                print(json.dumps({'model': model, 'case': case['case_id'], 'status': result['status'],
                                  'utility': result['utility'], 'security': result['security']}), flush=True)
                if result['status'] != 'valid':
                    # Stop on infrastructure failure; retain the original upstream outcome.
                    raise RuntimeError('AgentDojo infrastructure failure; see case requests and native trace')
                current_case = None
            core.shutdown_child(server)
            core.save(folder / 'server_release.json', {'returncode': server.returncode, 'at_utc': core.now()})
            server = None
        stage = 'paused' if stopping else 'evaluation_complete'
    except BaseException as exc:
        exit_code, stage = 1, 'failed'
        traceback.print_exc()
        core.save(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'at_utc': core.now()})
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join()
        core.shutdown_child(server)
        core.shutdown_child(child)
        core.save(output / 'owned_processes_released.json', {'server_released': True, 'merge_process_released': True, 'at_utc': core.now()})
        try:
            summary = dojo.summarize(output, args.models, cases)
        except Exception as exc:
            exit_code, stage = 1, 'failed'
            summary = {'status': 'integrity_error', 'models': {}, 'error': str(exc)}
            traceback.print_exc()
        core.save(output / 'summary.json', summary)
        overview = ['AgentDojo native metrics; security=True means attack success.', f"Status: {summary['status']}"]
        for model, values in summary['models'].items():
            overview.append(model + ': ' + json.dumps(values['groups'], ensure_ascii=False))
        (output / 'RESULTS.txt').write_text('\n'.join(overview)+'\n')
        if stage == 'evaluation_complete':
            if summary['status'] == 'complete':
                core.save(output / 'completion.json', {'status': 'complete', 'models': args.models,
                    'cases_per_model': len(cases), 'server_released': True, 'at_utc': core.now()})
            else:
                exit_code, stage = 1, 'failed'
        progress()
        try:
            export_run(output)
        except Exception as exc:
            exit_code, stage = 1, 'failed'
            core.save(output / 'transfer_failure.json', {'type': type(exc).__name__, 'message': str(exc), 'at_utc': core.now()})
            traceback.print_exc()
            progress()
    return exit_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'preflight', 'run', 'merge', 'export'])
    p.add_argument('--gpu', default='0')
    p.add_argument('--run-id', default='agentdojo')
    p.add_argument('--training-run-id', default='clean_opcd')
    p.add_argument('--models', nargs='+', choices=['base', 'trained'], default=['base', 'trained'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    core.require(len(args.models) == len(set(args.models)), 'Duplicate models')
    output = core.output_for(args.run_id)
    core.require(args.run_id not in ('main', args.training_run_id), 'Use a separate evaluation directory')
    if args.mode == 'export':
        core.require(output.is_dir(), 'Evaluation output does not exist')
        print(json.dumps(export_run(output)))
        return
    os.environ.update(core.runtime_environment(output if args.mode in ('run', 'merge') else core.ROOT/'output/setup/agentdojo_tests'))
    if args.mode == 'check':
        core.verify_assets(quick=True)
        print(json.dumps({'environment': environment(), 'protocol': dojo.check_protocol()}))
    elif args.mode == 'preflight':
        gpu = core.select_gpu(args.gpu)
        print(json.dumps({'environment': environment(), 'hardware': core.probe_gpu(gpu)}))
    elif args.mode == 'merge':
        merge_model(output)
    else:
        core.redirect_run_logs(output)
        raise SystemExit(run(args))


if __name__ == '__main__':
    main()
