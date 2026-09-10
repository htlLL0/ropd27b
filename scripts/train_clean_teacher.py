#!/usr/bin/env python3
"""Train the opt-in clean-T+ all-OPCD variant from audited initial rollouts."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import time

import b200_collection as core

CONFIG_PATH = core.ROOT / 'config/clean_teacher_opcd.json'


def training_config():
    cfg = core.read(CONFIG_PATH)
    core.require(cfg['attack_gate'] == cfg['reliability_weight'] == 1 and
                 cfg['clean_sft_weight'] == 0 and not cfg['semantic_label_filter'], 'Unlabeled objective contract changed')
    core.require(cfg['epochs'] == 1 and cfg['samples'] == 9600, 'Expected one complete 9600-row cycle')
    generation = core.config()
    core.require(all(cfg[k] == generation[k] for k in ('model_name', 'model_revision')), 'Training and rollout bases must match')
    runtime = cfg['runtime']
    core.require(runtime['dtype'] == 'bfloat16' and runtime['attention_implementation'] == 'sdpa'
                 and runtime['micro_batch_size'] == runtime['gradient_accumulation_steps'] == 1,
                 'This trainer implements BF16/SDPA and one trajectory per update')
    core.require(runtime['save_every'] >= 1 and runtime['keep_last_checkpoints'] >= 2
                 and runtime['response_chunk_size'] >= 1, 'Invalid checkpoint/chunk configuration')
    return cfg


def check_environment():
    result = core.check_environment()
    for package, expected in training_config()['versions'].items():
        result[package] = importlib.metadata.version(package)
        core.require(result[package] == expected, f'{package} must be {expected}; build Dockerfile.training')
    return result


def check_clean_inputs():
    from r_opcd.clean_teacher_opcd import clean_prompt
    from r_opcd.secopd_aligned_collection import student_messages, tokenize_messages, make_request
    from transformers import AutoTokenizer
    cfg = core.config()
    tokenizer = AutoTokenizer.from_pretrained(core.ROOT / 'model', local_files_only=True)
    items = core.rows(core.ROOT / 'inputs/prepared.jsonl')
    source = core.rows(core.ROOT / 'source_data/attack_candidates.jsonl')
    core.require(len(items) == len(source) == 9600, 'Expected all 9600 source rows')
    core.require(core.sha(core.ROOT / 'source_data/attack_candidates.jsonl') == cfg['source_sha256'], 'Source changed')
    core.require(len({r['id'] for r in source}) == 9600, 'Duplicate sources')
    prompts, lengths = [], []
    for index, (item, row) in enumerate(zip(items, source)):
        core.require(item['index'] == index and item['source'] == row, 'Source/order mismatch')
        core.require(item['messages'] == student_messages(row), 'Original attacked input changed')
        core.require(item['prompt'] == tokenize_messages(tokenizer, item['messages']), 'Attacked prompt token mismatch')
        core.require(item['request'] == make_request(item, cfg), 'Frozen sampling request changed')
        plus = clean_prompt(row, tokenizer)
        for prompt in (item['prompt'], plus):
            core.require(prompt['prompt_tokens'] + cfg['max_new_tokens'] <= cfg['max_model_len'], 'Truncation forbidden')
        prompts.append(plus)
        lengths.append(plus['prompt_tokens'])
        if (index + 1) % 1000 == 0:
            print(f'Clean/attacked native prompt gate {index + 1}/9600', flush=True)
    gate = {'status': 'pass', 'samples': 9600, 'max_clean_prompt_tokens': max(lengths),
            'student_equals_tminus': True, 'teacher_plus_fields': ['user_query', 'clean_context'],
            'label_filter': False, 'clean_sft': False, 'B200_forward_verified': False, 'at_utc': core.now()}
    return tokenizer, items, prompts, gate


def prepare_training(generation, output, cfg):
    """Re-audit every raw response and its export; never accept synthetic fixtures."""
    from r_opcd.clean_teacher_opcd import training_example
    from r_opcd.secopd_aligned_collection import make_record
    core.require(generation != output, 'Use a separate training run ID')
    receipt = core.read(generation / 'completion.json')
    core.require(receipt['status'] == 'complete' and receipt['samples'] == cfg['samples'] and
                 receipt['server_released'] is True, 'Generation must be fully complete and GPU released')
    hardware = core.read(generation / 'hardware.json')
    core.require(hardware.get('bf16_probe') == 'pass' and hardware.get('compute_capability') == [10, 0]
                 and not hardware.get('simulation_only'), 'A real B200 generation gate is required')
    gcfg = core.config()
    core.require(core.read(generation / 'config.json') == gcfg, 'Generation config changed')
    summary = core.read(generation / 'generation_summary.json')
    core.require(summary['generations_sha256'] == core.sha(generation / 'generations.jsonl'), 'Export hash mismatch')
    tokenizer, items, plus_prompts, gate = check_clean_inputs()
    case_names = {p.name for p in (generation / 'cases').iterdir()}
    core.require(case_names == {f'{i:05d}' for i in range(cfg['samples'])}, 'Missing/extra cases')
    data = output / 'training_samples.jsonl'
    tmp = data.with_suffix('.jsonl.tmp')
    counts, offsets = Counter(), []
    with (generation / 'generations.jsonl').open() as exported, tmp.open('wb') as dst:
        for item, plus in zip(items, plus_prompts):
            case = generation / 'cases' / f"{item['index']:05d}"
            core.require(not (case / 'error.json').exists(), 'Case error cannot be silently ignored')
            record, raw = core.read(case / 'trajectory.json'), core.read(case / 'raw_response.json')
            core.require(core.read(case / 'request.json') == {'messages': item['messages'], 'body': item['request']},
                         'Saved generation request mismatch')
            rebuilt = make_record(item, raw, gcfg, tokenizer, record['generation_elapsed_seconds'])
            core.require(all(record.get(k) == v for k, v in rebuilt.items()), 'Raw response/trajectory mismatch')
            core.require(record == json.loads(next(exported)), 'Export/case mismatch')
            execution = record.get('execution_provenance', {})
            core.require(execution.get('hardware') == 'single_B200' and
                         re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', execution.get('gpu_uuid', '')),
                         'Missing real B200 generation provenance')
            core.require(not any(record.get(k) or raw.get(k) for k in ('simulation_only', 'synthetic', 'not_for_training_or_evaluation')),
                         'Synthetic validation outputs cannot train')
            example = training_example(item, plus, record['response_token_ids'],
                max_sequence_tokens=cfg['runtime']['max_sequence_tokens'], vocab_size=len(tokenizer))
            example.update(stop_reason=record['stop_reason'], response_token_ids_sha256=record['response_token_ids_sha256'])
            offsets.append(dst.tell())
            dst.write((json.dumps(example, separators=(',', ':')) + '\n').encode())
            counts[record['stop_reason']] += 1
        core.require(not exported.read().strip(), 'Extra export records')
        dst.flush()
        os.fsync(dst.fileno())
    digest = core.sha(tmp)
    if data.exists():
        core.require(core.sha(data) == digest, 'Existing prepared training data changed')
        tmp.unlink()
    else:
        tmp.replace(data)
    binding = {'config': cfg, 'bundle_manifest_sha256': core.sha(core.ROOT / 'bundle_manifest.json'),
               'generation_run_id': generation.name, 'generation_sha256': summary['generations_sha256'],
               'training_samples_sha256': digest, 'samples': len(offsets)}
    core.save(output / 'input_gate.json', {**gate, 'raw_cases_reaudited': len(offsets), 'counts': dict(counts),
         'training_ready_for_this_variant_only': True, 'binding': binding})
    return data, offsets, binding


def save_checkpoint(model, optimizer, output, step, binding, metrics):
    import torch
    directory = output / 'checkpoints'
    directory.mkdir(exist_ok=True)
    name = f'step_{step:06d}_{time.time_ns()}'
    pending, final = directory / (name + '.partial'), directory / name
    pending.mkdir()
    model.save_pretrained(pending, safe_serialization=True, save_embedding_layers=False)
    card = pending / 'README.md'
    if card.exists():
        card.rename(pending / 'ADAPTER_CARD.md')
    torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'python_rng': random.getstate()}, pending / 'optimizer_rng.pt')
    core.save(pending / 'state.json', {'step': step, 'binding': binding, 'metrics': metrics, 'at_utc': core.now()})
    files = {p.name: core.sha(p) for p in pending.iterdir() if p.is_file()}
    core.save(pending / 'commit.json', {'step': step, 'files_sha256': files})
    # Flush payloads before publishing the pointer. Incomplete writes stay .partial.
    for path in pending.iterdir():
        with path.open('rb') as handle:
            os.fsync(handle.fileno())
    pending.rename(final)
    core.save(output / 'latest_checkpoint.json', {'path': str(final.relative_to(output)), 'step': step,
              'commit_sha256': core.sha(final / 'commit.json')})
    committed = sorted((p for p in directory.glob('step_*') if p.is_dir() and not p.name.endswith('.partial')),
                       key=lambda p: p.name)
    keep = binding['config']['runtime']['keep_last_checkpoints']
    for old in committed[:-keep]:
        shutil.rmtree(old)
    return final


def load_checkpoint(model, optimizer, output, binding):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    latest = core.read(output / 'latest_checkpoint.json')
    path = (output / latest['path']).resolve()
    core.require(path.parent == (output / 'checkpoints').resolve(), 'Invalid checkpoint path')
    core.require(core.sha(path / 'commit.json') == latest['commit_sha256'], 'Checkpoint commit hash mismatch')
    commit = core.read(path / 'commit.json')
    for name, digest in commit['files_sha256'].items():
        core.require(Path(name).name == name and core.sha(path / name) == digest, 'Checkpoint payload hash mismatch')
    state = core.read(path / 'state.json')
    core.require(state['binding'] == binding and state['step'] == latest['step'] == commit['step'], 'Resume provenance mismatch')
    weights = load_file(str(path / 'adapter_model.safetensors'))
    result = set_peft_model_state_dict(model, weights)
    core.require(not result.unexpected_keys and not any('lora_' in n for n in result.missing_keys), 'Adapter keys mismatch')
    # Only a locally created, hash-verified checkpoint is accepted here.
    saved = torch.load(path / 'optimizer_rng.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng']:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    return state['step'], state['metrics']


def run_training(args):
    output = core.output_for(args.run_id)
    generation = core.output_for(args.generation_run_id)
    cfg = training_config()
    lock = (output / 'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    core.require(not (output / 'completion.json').exists(), 'Training already complete')
    if (output / 'config.json').exists():
        core.require(args.resume and core.read(output / 'config.json') == cfg, 'Existing training: use --resume with same config')
    else:
        core.require(not args.resume, 'No training run to resume')
        core.save(output / 'config.json', cfg)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    core.save(output / 'supervisor.json', {**core.process_identity(os.getpid()), 'at_utc': core.now()})
    step, metrics = 0, {}
    start = time.monotonic()
    baseline = 0

    def progress(stage):
        elapsed, done = time.monotonic() - start, step - baseline
        core.save(output / 'progress.json', {'stage': stage, 'step': step, 'total_steps': cfg['samples'],
             'metrics': metrics, 'elapsed_seconds_this_session': elapsed,
             'estimated_remaining_seconds': elapsed / done * (cfg['samples'] - step) if done >= 10 else None,
             'updated_at_utc': core.now(), 'generation_run_id': args.generation_run_id})

    try:
        progress('preflight')
        gpu = core.select_gpu(args.gpu)
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        from r_opcd.clean_teacher_opcd import attach_adapter, train_step
        core.save(output / 'asset_gate.json', core.verify_assets())
        core.save(output / 'environment.json', check_environment())
        data, offsets, binding = prepare_training(generation, output, cfg)
        if (output / 'binding.json').exists():
            core.require(core.read(output / 'binding.json') == binding, 'Run binding changed')
        else:
            core.save(output / 'binding.json', binding)
        if stopping:
            progress('paused_before_model_load')
            return
        core.save(output / 'hardware.json', core.probe_gpu(gpu))
        torch.manual_seed(cfg['seed'])
        random.seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        progress('model_loading')
        base = Qwen3_5ForConditionalGeneration.from_pretrained(core.ROOT / 'model', local_files_only=True,
                 dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
        model = attach_adapter(base, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        core.save(output / 'adapter_layout.json', {'trainable_parameters': sum(p.numel() for p in params),
            'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad],
            'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'shared_teacher_base': True})
        optimizer = torch.optim.AdamW(params, lr=cfg['optimizer']['learning_rate'], weight_decay=cfg['optimizer']['weight_decay'])
        if args.resume and (output / 'latest_checkpoint.json').exists():
            step, metrics = load_checkpoint(model, optimizer, output, binding)
            core.save(output / f'resume.{time.time_ns()}.json', {
                'resumed_from_step': step, 'at_utc': core.now(),
                'note': 'Append-only metrics can include updates after this checkpoint; these are recomputed.'})
        elif args.resume:
            core.require(not (output / 'metrics.jsonl').exists(), 'Updates exist but no committed checkpoint; inspect failure')
        order = list(range(len(offsets)))
        random.Random(cfg['seed']).shuffle(order)
        core.require(0 <= step <= len(order), 'Invalid resume cursor')
        baseline, start = step, time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        progress('training')
        with data.open('rb') as stream, (output / 'metrics.jsonl').open('a') as journal:
            while step < len(order) and not stopping:
                index = order[step]
                stream.seek(offsets[index])
                example = json.loads(stream.readline())
                tick = time.monotonic()
                metrics = train_step(model, optimizer, example, cfg)
                torch.cuda.synchronize()
                metrics.update(index=index, source_id=example['source_id'], step_seconds=time.monotonic() - tick,
                               peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                               peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
                # First update is a real 27B forward/backward gate, counted in the epoch.
                if step == 0:
                    core.require(metrics['grad_norm'] > 0 and metrics['selected_tokens'] > 0, 'First update has no corrective gradient')
                step += 1
                journal.write(json.dumps({'step': step, 'at_utc': core.now(), **metrics}) + '\n')
                journal.flush()
                if step == 1 or step % cfg['runtime']['save_every'] == 0 or stopping or step == len(order):
                    save_checkpoint(model, optimizer, output, step, binding, metrics)
                if step == 1:
                    core.save(output / 'first_update_gate.json', {'status': 'pass', 'actual_model': cfg['model_name'],
                        'actual_gpu': gpu, 'metrics': metrics, 'scientific_evaluation': False, 'at_utc': core.now()})
                progress('training')
                print(json.dumps({'step': step, 'total': len(order), **metrics}), flush=True)
        if stopping:
            # Even if the signal arrived between steps, the exact cursor is committed.
            save_checkpoint(model, optimizer, output, step, binding, metrics)
            progress('paused')
        else:
            progress('training_complete')
            core.save(output / 'completion.json', {'status': 'complete', 'steps': step, 'samples': len(order),
                 'all_samples_opcd': True, 'semantic_labels_used': False, 'clean_sft_used': False,
                 'checkpoint': core.read(output / 'latest_checkpoint.json'), 'at_utc': core.now(),
                 'trajectory_mode': cfg['trajectory_mode'], 'scientific_evaluation': False})
    except BaseException as exc:
        core.save(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'step': step, 'resume_from_last_committed_checkpoint_only': True, 'at_utc': core.now()})
        progress('failed')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'preflight', 'run'])
    p.add_argument('--gpu', default='0')
    p.add_argument('--run-id', default='clean_opcd')
    p.add_argument('--generation-run-id', default='main')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = core.output_for(args.run_id)
    if args.mode == 'check':
        core.verify_assets(quick=True)
        env = check_environment()
        _, _, _, gate = check_clean_inputs()
        print(json.dumps({'environment': env, 'input_gate': gate}))
    elif args.mode == 'preflight':
        gpu = core.select_gpu(args.gpu)
        print(json.dumps({'environment': check_environment(), 'hardware': core.probe_gpu(gpu)}))
    else:
        core.redirect_run_logs(output)
        os.environ.update(core.runtime_environment(output))
        run_training(args)


if __name__ == '__main__':
    main()
