#!/usr/bin/env python3
"""Portable single-B200 collection using the unchanged frozen sampling/parser code."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/tools')]
STOP = threading.Event()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def config():
    return read(ROOT / 'config/generation.json')


def output_for(run_id):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id), 'Invalid run ID')
    return ROOT / 'output' / run_id


def redirect_run_logs(output):
    """Persist Python, native-library stdout/stderr and uncaught tracebacks."""
    output.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    with (output / 'supervisor.log').open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


def runtime_environment(output):
    """Keep framework caches and temporary runtime files with this run."""
    directories = {
        'HF_HOME': output / 'cache/huggingface',
        'XDG_CACHE_HOME': output / 'cache/xdg',
        'VLLM_CACHE_ROOT': output / 'cache/vllm',
        'TORCHINDUCTOR_CACHE_DIR': output / 'cache/torchinductor',
        'TRITON_CACHE_DIR': output / 'cache/triton',
        'CUDA_CACHE_PATH': output / 'cache/cuda',
        'TMPDIR': output / 'tmp',
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return {key: str(path) for key, path in directories.items()}


def command(args, timeout=60):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()


def verify_assets(quick=False):
    manifest = read(ROOT / 'bundle_manifest.json')
    weights_skipped = 0
    for name, entry in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid manifest path')
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), 'Missing file or symlink: ' + name)
        require(path.stat().st_size == entry['bytes'], 'File size mismatch: ' + name)
        if quick and name.endswith('.safetensors'):
            weights_skipped += 1
        else:
            require(sha(path) == entry['sha256'], 'SHA256 mismatch: ' + name)
    receipt = {'status': 'quick_size_only_for_weights' if quick else 'pass',
               'files': len(manifest['files']), 'weight_hashes_skipped': weights_skipped,
               'bundle_manifest_sha256': sha(ROOT / 'bundle_manifest.json'), 'at_utc': now(),
               'B200_execution_verified': False}
    print(json.dumps(receipt), flush=True)
    return receipt


def check_environment():
    import torch
    expected = {'vllm': '0.24.0', 'transformers': '5.12.1', 'torch': '2.11.0+cu130'}
    actual = {name: importlib.metadata.version(name) for name in expected}
    require(actual == expected, f'Environment differs from pinned image: {actual}; expected {expected}')
    require(torch.version.cuda == '13.0', 'Use the pinned CUDA 13.0 environment')
    arch = torch._C._cuda_getArchFlags()
    require('sm_100' in arch.split(), 'PyTorch wheel lacks B200 sm_100 support')
    return {**actual, 'cuda': torch.version.cuda, 'arch_flags': arch, 'at_utc': now()}


def validate_gpu_row(row):
    require('B200' in row['name'], 'B200 required; selected GPU is ' + row['name'])
    require(row['memory_total_mib'] >= 170000, 'A full B200 GPU is required (not a small MIG slice)')
    require(row['memory_used_mib'] <= 256, 'Selected B200 is already in use; no existing process will be stopped')
    return row


def select_gpu(selector):
    require(re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', selector), 'Select exactly one GPU index or UUID')
    values = command(['nvidia-smi', '--id=' + selector,
                      '--query-gpu=uuid,name,memory.total,memory.used,driver_version',
                      '--format=csv,noheader,nounits']).splitlines()
    require(len(values) == 1, 'Select exactly one B200')
    parts = [v.strip() for v in values[0].split(',')]
    require(len(parts) == 5, 'Unexpected nvidia-smi output')
    row = validate_gpu_row({'uuid': parts[0], 'name': parts[1], 'memory_total_mib': int(parts[2]),
                            'memory_used_mib': int(parts[3]), 'driver_version': parts[4]})
    # Configure visibility before torch initializes CUDA, including for native execution.
    os.environ['CUDA_VISIBLE_DEVICES'] = row['uuid']
    return row


def probe_gpu(row):
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'CUDA must expose exactly one working GPU')
    require('B200' in torch.cuda.get_device_name(0), 'CUDA selected the wrong physical GPU')
    require(torch.cuda.get_device_capability(0) == (10, 0), 'Expected B200 compute capability 10.0')
    require(torch.cuda.get_device_properties(0).total_memory >= 170000 * 1024 * 1024, 'CUDA exposes a partial GPU')
    require(torch.cuda.is_bf16_supported(), 'Native BF16 unavailable')
    a = torch.ones((128, 128), dtype=torch.bfloat16, device='cuda')
    b = a @ a
    torch.cuda.synchronize()
    require(bool(torch.all(b == 128).item()), 'B200 BF16 matrix multiplication probe failed')
    del a, b
    torch.cuda.empty_cache()
    return {**row, 'visible_device_count': 1, 'bf16_probe': 'pass', 'compute_capability': [10, 0], 'at_utc': now()}


def check_inputs():
    from transformers import AutoTokenizer
    from r_opcd.frontier_collection import canonical_sha256
    from r_opcd.secopd_aligned_collection import make_request, student_messages, teacher_messages, tokenize_messages
    cfg = config()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    items = rows(ROOT / 'inputs/prepared.jsonl')
    views = rows(ROOT / 'inputs/teacher_views.jsonl')
    source = rows(ROOT / 'source_data/attack_candidates.jsonl')
    require(len(items) == len(views) == len(source) == 9600, 'Expected exactly 9600 aligned inputs')
    require(sha(ROOT / 'source_data/attack_candidates.jsonl') == cfg['source_sha256'], 'Source file changed')
    require(len({x['source']['id'] for x in items}) == 9600, 'Duplicate source IDs')
    counts = Counter()
    for index, (item, view, row) in enumerate(zip(items, views, source)):
        require(item['index'] == index and item['source'] == row, 'Source/order mismatch')
        require(item['messages'] == student_messages(row), 'user/input roles or content changed')
        require(item['prompt'] == tokenize_messages(tokenizer, item['messages']), 'Native Student prompt mismatch')
        require(view['source_id'] == row['id'] and view['parent_source_id'] == row['parent_source_id']
                and view['student_equals_tminus'] is True, 'T+ source binding mismatch')
        require(view['messages'] == teacher_messages(row), 'T+ messages changed')
        require(view['teacher_plus'] == tokenize_messages(tokenizer, view['messages'], span=row['malicious_span']),
                'T+ exact-q1 prompt/mask mismatch')
        require(item['request'] == make_request(item, cfg), 'Sampling request changed')
        require(len(item['request']['prompt']) + cfg['max_new_tokens'] <= cfg['max_model_len'], 'Prompt truncation forbidden')
        counts[row['secopd_attack_plan_cell']] += 1
        if (index + 1) % 1000 == 0:
            print(f'Native prompt gate {index + 1}/9600', flush=True)
    selected = read(ROOT / 'inputs/smoke_selection.json')
    require(len(selected) == len(set(selected)) == 12 and all(0 <= i < 9600 for i in selected), 'Invalid canary IDs')
    require(Counter(items[i]['source']['secopd_attack_plan_cell'] for i in selected) ==
            {'straightforward_prepend': 4, 'straightforward_append': 4, 'completion': 4}, 'Canary coverage changed')
    require(counts == {'straightforward_prepend': 4350, 'straightforward_append': 4290, 'completion': 960}, 'Attack mixture changed')
    return tokenizer, items, {'status': 'pass', 'samples': 9600, 'counts': dict(counts),
                              'same_sampling_requests': True, 'Tplus_forward_verified': False, 'at_utc': now()}


def server_command(port):
    cfg = config()
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(ROOT / 'model'),
            '--served-model-name', cfg['served_model_name'], '--host', '127.0.0.1', '--port', str(port),
            '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-model-len', '32768',
            '--gpu-memory-utilization', '0.90', '--max-num-seqs', '16', '--max-num-batched-tokens', '4096',
            '--language-model-only', '--enforce-eager', '--disable-custom-all-reduce',
            '--generation-config', 'vllm', '--seed', str(cfg['seed'])]


def process_identity(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19], 'cmdline': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}


def shutdown_child(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


def completed_records(output, items, cfg, tokenizer):
    from r_opcd.frontier_collection import canonical_sha256
    from collect_secopd_qwen36_9600 import parse_tokens
    completed = {}
    for case in sorted((output / 'cases').glob('*')):
        require(case.is_dir() and case.name.isdecimal(), 'Unexpected case directory')
        index = int(case.name)
        require(0 <= index < len(items), 'Unknown case index')
        require((case / 'trajectory.json').exists() and not (case / 'error.json').exists(),
                f'Incomplete/error case {index}; inspect it before resuming. No silent re-generation.')
        item, record, raw = items[index], read(case / 'trajectory.json'), read(case / 'raw_response.json')
        require(read(case / 'request.json') == {'messages': item['messages'], 'body': item['request']}, 'Saved request mismatch')
        tokens, reason = parse_tokens(raw, item['request']['prompt'], cfg)
        require(raw['model'] == cfg['served_model_name'] and record['model_revision'] == cfg['model_revision'], 'Saved model mismatch')
        require(record['generation_config_sha256'] == canonical_sha256(cfg), 'Saved config mismatch')
        require(record['source_id'] == item['source']['id'] and record['source_row_sha256'] == canonical_sha256(item['source']), 'Saved source mismatch')
        require(record['response_token_ids'] == tokens and record['stop_reason'] == reason and
                record['response_token_ids_sha256'] == canonical_sha256(tokens), 'Saved token/stop mismatch')
        require(record['training_ready'] is False, 'Unexpected training-ready flag')
        completed[index] = record
    return completed


def run_collection(args):
    output, cfg = output_for(args.run_id), config()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (output / 'completion.json').exists(), 'This run is already complete')
    if (output / 'config.json').exists():
        require(args.resume and read(output / 'config.json') == cfg, 'Existing run: use --resume with unchanged config')
    else:
        require(not args.resume, 'No previous run to resume')
        require(not (output / 'cases').exists(), 'Unidentified existing case files')
        save(output / 'config.json', cfg)
    STOP.clear()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    save(output / 'supervisor.json', {**process_identity(os.getpid()), 'started_at_utc': now(),
         'run_id': args.run_id, 'docker_container': os.environ.get('OPCD_CONTAINER_NAME'),
         'stop_signal': 'SIGUSR1: stop submitting and drain in-flight requests', 'training_started': False})
    server = None
    completed, in_flight = {}, set()
    begin, collect_start, baseline_count, baseline_tokens = time.monotonic(), None, 0, 0
    stage = 'preflight'
    errors = []

    def progress(current):
        elapsed = time.monotonic() - begin
        tokens = sum(r['generated_tokens_including_terminal'] for r in completed.values())
        duration = time.monotonic() - collect_start if collect_start is not None else 0
        finished = len(completed) - baseline_count
        eta = duration / finished * (9600 - len(completed)) if finished >= 100 else None
        save(output / 'progress.json', {'stage': current, 'completed': len(completed), 'total': 9600,
             'in_flight': len(in_flight), 'elapsed_seconds': elapsed, 'collection_seconds_this_session': duration,
             'generated_tokens': tokens, 'tokens_per_second_this_session': (tokens - baseline_tokens) / duration if duration else None,
             'estimated_remaining_seconds': eta, 'counts': dict(Counter(r['stop_reason'] for r in completed.values())),
             'missing_think_end': sum(r['thinking_end_count'] == 0 for r in completed.values()),
             'empty_final': sum(not r['final_answer'] for r in completed.values()),
             'errors': errors, 'supervisor_pid': os.getpid(), 'training_started': False, 'updated_at_utc': now()})

    try:
        progress(stage)
        # Refuse wrong/busy hardware before allocating any CUDA memory.
        gpu = select_gpu(args.gpu)
        save(output / 'asset_gate.json', verify_assets())
        save(output / 'environment.json', check_environment())
        from r_opcd.secopd_aligned_collection import make_record
        tokenizer, items, input_gate = check_inputs()
        save(output / 'prompt_gate.json', input_gate)
        save(output / 'hardware.json', probe_gpu(gpu))
        completed = completed_records(output, items, cfg, tokenizer)
        baseline_count = len(completed)
        baseline_tokens = sum(r['generated_tokens_including_terminal'] for r in completed.values())
        require(not STOP.is_set(), 'Pause requested before generation')
        # Private localhost port; never connect to an unrelated existing server.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
            port = probe.getsockname()[1]
        cmd = server_command(port)
        server_log = (output / 'server.log').open('a')
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
               'DO_NOT_TRACK': '1', 'VLLM_ENABLE_CUDA_COMPATIBILITY': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        server = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=server_log, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=True)
        save(output / 'server.json', {**process_identity(server.pid), 'command': cmd, 'gpu': gpu, 'port': port, 'at_utc': now()})
        stage, loading = 'model_loading', time.monotonic()
        while True:
            require(not STOP.is_set(), 'Pause requested during loading')
            require(server.poll() is None, 'Model server exited; inspect server.log')
            require(time.monotonic() - loading < 1800, 'Model health timeout')
            progress(stage)
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as reply:
                    if reply.status == 200:
                        break
            except (OSError, TimeoutError):
                STOP.wait(5)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10) as reply:
            models = json.load(reply)
        require(any(x['id'] == cfg['served_model_name'] for x in models['data']), 'Wrong served model identity')
        save(output / 'server_models.json', models)
        (output / 'cases').mkdir(exist_ok=True)
        collect_start = time.monotonic()

        def one(item, phase):
            index = item['index']
            case = output / 'cases' / f'{index:05d}'
            case.mkdir(exist_ok=False)
            save(case / 'execution.json', {'index': index, 'phase': phase, 'started_at_utc': now(),
                 'gpu_uuid': gpu['uuid'], 'tensor_parallel_size': 1, 'source_lane_before_migration': item['lane']})
            save(case / 'request.json', {'messages': item['messages'], 'body': item['request']})
            start = time.monotonic()
            try:
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',
                    data=json.dumps(item['request']).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=cfg['request_timeout_seconds']) as reply:
                    raw = json.load(reply)
                save(case / 'raw_response.json', raw)
                record = make_record(item, raw, cfg, tokenizer, time.monotonic() - start)
                record['execution_provenance'] = {'hardware': 'single_B200', 'gpu_uuid': gpu['uuid'],
                    'tensor_parallel_size': 1, 'max_concurrency': 16, 'phase': phase,
                    'source_lane_before_migration': item['lane']}
                save(case / 'trajectory.json', record)
                return index, record
            except BaseException as exc:
                STOP.set()
                save(case / 'error.json', {'type': type(exc).__name__, 'message': str(exc), 'at_utc': now(),
                                         'semantic_label': None, 'automatic_retry': False})
                raise

        selected = set(read(ROOT / 'inputs/smoke_selection.json'))
        for phase in ('smoke', 'bulk'):
            stage = phase
            if phase == 'bulk':
                require(read(output / 'smoke_gate.json')['status'] == 'pass', 'Canary gate missing')
            pending_items = iter(item for item in items if item['index'] not in completed
                                 and ((item['index'] in selected) == (phase == 'smoke')))
            with (output / 'completed.jsonl').open('a') as journal, ThreadPoolExecutor(max_workers=16) as pool:
                futures = {}

                def fill():
                    while len(futures) < 16 and not STOP.is_set():
                        item = next(pending_items, None)
                        if item is None:
                            break
                        in_flight.add(item['index'])
                        futures[pool.submit(one, item, phase)] = item['index']

                fill()
                while futures:
                    done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = futures.pop(future)
                        in_flight.remove(index)
                        try:
                            _, record = future.result()
                            completed[index] = record
                            journal.write(json.dumps({'index': index, 'source_id': record['source_id'], 'at_utc': now()}) + '\n')
                            journal.flush()
                            os.fsync(journal.fileno())
                        except BaseException as exc:
                            errors.append({'index': index, 'type': type(exc).__name__, 'message': str(exc)})
                            STOP.set()
                    if time.monotonic() - begin > cfg['run_timeout_seconds']:
                        errors.append({'type': 'RunTimeout', 'message': '48-hour limit reached'})
                        STOP.set()
                    if server.poll() is not None:
                        errors.append({'type': 'ServerExited', 'message': 'Inspect server.log'})
                        STOP.set()
                    fill()
                    progress('draining' if STOP.is_set() else phase)
            require(not errors, 'Generation error; preserved all completed responses and typed errors')
            if STOP.is_set():
                break
            if phase == 'smoke':
                require(selected.issubset(completed), 'Incomplete canary coverage')
                require(all(completed[i]['stop_reason'] == 'eos_token' and completed[i]['thinking_end_count'] == 1
                            and completed[i]['final_answer'] for i in selected), 'Canary failed: cap/thinking boundary/empty final')
                save(output / 'smoke_gate.json', {'status': 'pass', 'count': 12, 'indices': sorted(selected),
                     'counts_toward_9600': True, 'not_A_U_evidence': True, 'at_utc': now()})
        if STOP.is_set():
            save(output / 'pause.json', {'status': 'user_paused', 'completed': len(completed), 'in_flight': 0,
                 'automatic_resume': False, 'at_utc': now()})
            stage = 'paused'
        else:
            checked = completed_records(output, items, cfg, tokenizer)
            require(set(checked) == set(range(9600)), 'Full coverage audit failed')
            verify_assets(quick=True)
            destination = output / 'generations.jsonl'
            tmp = destination.with_suffix('.jsonl.tmp')
            with tmp.open('w') as stream:
                for index in range(9600):
                    stream.write(json.dumps(checked[index], ensure_ascii=False, separators=(',', ':')) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(destination)
            save(output / 'generation_summary.json', {'status': 'complete', 'samples': 9600,
                 'generations_sha256': sha(destination), 'training_ready': False, 'at_utc': now(),
                 'counts': dict(Counter(r['stop_reason'] for r in checked.values())),
                 'pending': ['new_A_U_labels', 'cap_disposition', 'Tplus_model_forward_gate', 'reliability_calibration', 'training_integration']})
            stage = 'generation_complete'
    except BaseException as exc:
        stage = 'failed'
        save(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'at_utc': now(),
                                      'automatic_retry': False, 'training_started': False})
        raise
    finally:
        shutdown_child(server)
        if server is not None:
            save(output / 'server_release.json', {'pid': server.pid, 'returncode': server.poll(), 'at_utc': now()})
        progress(stage)
        if stage == 'generation_complete':
            save(output / 'completion.json', {'status': 'complete', 'samples': 9600, 'server_released': True,
                                             'training_started': False, 'at_utc': now()})


def show_status(output):
    path = output / 'progress.json'
    if not path.exists():
        print('Not started: ' + str(output))
        return
    p = read(path)
    stamp = datetime.fromisoformat(p['updated_at_utc']).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai'))
    print(f"Beijing {stamp:%m-%d %H:%M:%S} | {p['stage']}")
    print(f"Completed {p['completed']}/{p['total']} | in flight {p['in_flight']} | errors {len(p['errors'])}")
    rate, eta = p.get('tokens_per_second_this_session'), p.get('estimated_remaining_seconds')
    print(f"Output tokens/s: {rate:.1f}" if rate else 'Output tokens/s: pending')
    print(f"ETA: {eta / 3600:.2f} hours (observed throughput estimate)" if eta else 'ETA: pending 100 completed samples')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['verify', 'check', 'preflight', 'run', 'start', 'status', 'pause'])
    p.add_argument('--gpu', default=os.environ.get('OPCD_GPU', '0'))
    p.add_argument('--run-id', default='main')
    p.add_argument('--port', type=int, default=0, help='0 selects an unused localhost port')
    p.add_argument('--resume', action='store_true', help='Explicitly resume a clean paused run; never retry incomplete/error cases')
    p.add_argument('--quick', action='store_true', help='verify only: skip weight content hashes, still check their sizes')
    args = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    output = output_for(args.run_id)
    if args.mode == 'verify':
        verify_assets(args.quick)
    elif args.mode == 'check':
        verify_assets(args.quick)
        env = check_environment()
        _, _, gate = check_inputs()
        print(json.dumps({'environment': env, 'input_gate': gate, 'B200_execution_verified': False}))
    elif args.mode == 'preflight':
        gpu = select_gpu(args.gpu)
        print(json.dumps({'environment': check_environment(), 'hardware': probe_gpu(gpu)}))
    elif args.mode == 'run':
        redirect_run_logs(output)
        os.environ.update(runtime_environment(output))
        run_collection(args)
    elif args.mode == 'status':
        show_status(output)
    elif args.mode == 'pause':
        receipt = read(output / 'supervisor.json')
        require(not receipt.get('docker_container'), 'For Docker use ./docker_b200.sh pause')
        actual = process_identity(receipt['pid'])
        require(actual is not None and actual['start_ticks'] == receipt['start_ticks'] and
                str(Path(__file__).resolve()) in actual['cmdline'], 'Supervisor identity changed or exited')
        os.kill(receipt['pid'], signal.SIGUSR1)
        print('Pause requested; in-flight requests will drain. Inspect status before moving files.')
    else:
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
        else:
            require((output / 'config.json').exists(), 'No previous run to resume')
        launch_lock = (output / 'launch.lock').open('a')
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read(output / 'supervisor.json') if (output / 'supervisor.json').exists() else None
        if old:
            actual = process_identity(old['pid'])
            require(actual is None or actual['start_ticks'] != old['start_ticks'], 'Supervisor is still running')
        cmd = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), 'run', '--gpu', args.gpu,
               '--run-id', args.run_id, '--port', str(args.port)] + (['--resume'] if args.resume else [])
        with (output / 'supervisor.log').open('a') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        save(output / 'launch.json', {'pid': child.pid, 'command': cmd, 'at_utc': now()})
        print(json.dumps({'status': 'launched_preflight_pending', 'pid': child.pid, 'output': str(output)}))


if __name__ == '__main__':
    main()
