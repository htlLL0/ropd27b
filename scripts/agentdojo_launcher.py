#!/usr/bin/env python3
"""Portable Docker controls for paired AgentDojo evaluation."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import b200_collection as core
from docker_launcher import base_args


def image_tag():
    raw = (core.ROOT/'Dockerfile.agentdojo').read_bytes()+(core.ROOT/'requirements-agentdojo.txt').read_bytes()
    return 'opcd-agentdojo:'+hashlib.sha256(raw).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['build', 'check', 'preflight', 'start', 'status', 'pause', 'logs', 'export'])
    p.add_argument('--gpu', default=os.environ.get('OPCD_GPU', '0'))
    p.add_argument('--run-id', default='agentdojo')
    p.add_argument('--training-run-id', default='clean_opcd')
    p.add_argument('--models', nargs='+', choices=['base', 'trained'], default=['base', 'trained'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = core.output_for(args.run_id)
    core.require(args.run_id not in ('main', args.training_run_id), 'Use a separate evaluation run ID')
    image = image_tag()
    if args.mode == 'build':
        context = core.ROOT/'output/setup/agentdojo_build'
        context.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(core.ROOT/'Dockerfile.agentdojo', context/'Dockerfile')
        shutil.copyfile(core.ROOT/'requirements-agentdojo.txt', context/'requirements-agentdojo.txt')
        subprocess.run(['docker', 'build', '-t', image, str(context)], check=True)
        return
    if args.mode == 'status':
        if (output/'progress.json').exists():
            value = core.read(output/'progress.json')
            stamp = datetime.fromisoformat(value['updated_at_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
            print(f"Beijing {stamp:%m-%d %H:%M:%S} | {value['stage']} | model={value['model']} | {value['completed']}/{value['total']} | errors={value['errors']}")
            eta = value.get('estimated_remaining_seconds')
            print(f'ETA {eta/3600:.2f} hours' if eta else 'ETA pending 10 completed cases this session')
        else:
            print('Not started: '+str(output))
        if (output/'docker.json').exists():
            receipt = core.read(output/'docker.json')
            print(core.command(['docker', 'inspect', receipt['id'], '--format', '{{json .State}}']))
        if (output/'transfer_latest.json').exists():
            print('TXT transfer: '+str(output/core.read(output/'transfer_latest.json')['directory']))
        return
    if args.mode == 'logs':
        subprocess.run(['tail', '-n', '100', '-F', str(output/'supervisor.log')], check=True)
        return
    if args.mode in ('pause', 'export'):
        receipt = core.read(output/'docker.json')
        actual = json.loads(core.command(['docker', 'inspect', receipt['id']]))[0]
        core.require(actual['Id'] == receipt['id'] and actual['Config']['Labels'].get('research.scope') == receipt['scope'],
                     'Container identity/scope mismatch')
        if args.mode == 'pause':
            core.require(actual['State']['Running'], 'Container already stopped')
            subprocess.run(['docker', 'kill', '--signal=SIGUSR1', receipt['id']], check=True)
            print('Pause requested: complete the current task, stop owned processes, then export TXT files.')
            return
        core.require(not actual['State']['Running'], 'Pause evaluation and wait for container exit before export')
        lock = (output/'evaluation.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from evaluate_agentdojo import export_run
        print(json.dumps(export_run(output), ensure_ascii=False))
        return
    core.command(['docker', 'image', 'inspect', image])
    (core.ROOT/'output').mkdir(exist_ok=True)
    if args.mode == 'check':
        for script, extra in [('scripts/evaluate_agentdojo.py', ['check']), ('validation/test_agentdojo_delivery.py', [])]:
            subprocess.run(['docker', 'run', '--rm', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
                            *base_args(), image, '-u', '-B', '/bundle/'+script, *extra], check=True)
        return
    gpu = core.select_gpu(args.gpu)
    gpu_args = ['--runtime', 'nvidia', '--gpus', 'device='+gpu['uuid']]
    if args.mode == 'preflight':
        subprocess.run(['docker', 'run', '--rm', *gpu_args, *base_args(), image, '-u', '-B',
            '/bundle/scripts/evaluate_agentdojo.py', 'preflight', '--gpu', gpu['uuid']], check=True)
        return
    if 'trained' in args.models:
        completion = core.read(core.output_for(args.training_run_id)/'completion.json')
        core.require(completion['status'] == 'complete' and completion['steps'] == 9600,
                     'Finish training first, or use --models base for the original-model-only run')
    if args.resume:
        core.require((output/'run_config.json').exists(), 'No evaluation to resume')
        previous = core.read(output/'docker.json')
        actual = json.loads(core.command(['docker', 'inspect', previous['id']]))[0]
        core.require(not actual['State']['Running'], 'Previous evaluator is still running')
        core.save(output/f'docker.previous.{time.time_ns()}.json', previous)
    else:
        output.mkdir(parents=True, exist_ok=False)
    scope = 'opcd-agentdojo-'+core.sha(core.ROOT/'bundle_manifest.json')[:12]+'-'+args.run_id
    name = scope+'-'+str(time.time_ns())[-10:]
    command = ['docker', 'run', '-d', '--init', '--name', name, '--restart', 'no', '--label', 'research.scope='+scope,
        *gpu_args, *base_args(), image, '-u', '-B', '/bundle/scripts/evaluate_agentdojo.py', 'run',
        '--gpu', gpu['uuid'], '--run-id', args.run_id, '--training-run-id', args.training_run_id, '--models', *args.models]
    if args.resume:
        command.append('--resume')
    core.save(output/'docker.command.json', command)
    cid = core.command(command, timeout=180)
    actual = json.loads(core.command(['docker', 'inspect', cid]))[0]
    core.require(actual['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == [gpu['uuid']], 'Docker GPU assignment mismatch')
    core.save(output/'docker.json', {'id': cid, 'scope': scope, 'image': image, 'image_id': actual['Image'],
                                  'gpu': gpu, 'models': args.models, 'at_utc': core.now()})
    print(json.dumps({'status': 'launched_preflight_pending', 'models': args.models, 'output': str(output)}))


if __name__ == '__main__':
    main()
