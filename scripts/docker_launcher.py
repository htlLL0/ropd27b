#!/usr/bin/env python3
"""Host-side Docker controls; no model/Python packages required on the host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import b200_collection as core


def mount_args(writable_output=True):
    # Override only when the Docker daemon sees a different host-side mount path.
    source = os.environ.get('OPCD_DOCKER_BUNDLE', str(core.ROOT))
    core.require(',' not in source and '\n' not in source, 'Docker bind path cannot contain commas/newlines')
    args = ['--mount', f'type=bind,src={source},dst=/bundle,readonly']
    if writable_output:
        args += ['--mount', f'type=bind,src={source}/output,dst=/bundle/output']
    return args


def base_args(writable_output=True):
    return ['--network', 'none', '--ipc', 'host', '--security-opt', 'label=disable',
            '--user', f'{os.getuid()}:{os.getgid()}', '--entrypoint', 'python3',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
            '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
            '-e', 'HF_HOME=/tmp/opcd-hf', '-e', 'XDG_CACHE_HOME=/tmp/opcd-cache',
            '-e', 'VLLM_CACHE_ROOT=/tmp/opcd-vllm', '-e', 'OMP_NUM_THREADS=1',
            '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'VLLM_NO_USAGE_STATS=1',
            '-e', 'DO_NOT_TRACK=1', '-e', 'VLLM_ENABLE_CUDA_COMPATIBILITY=1', *mount_args(writable_output)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['pull', 'check', 'preflight', 'start', 'status', 'pause', 'logs'])
    p.add_argument('--gpu', default=os.environ.get('OPCD_GPU', '0'))
    p.add_argument('--run-id', default='main')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--quick', action='store_true', help='check only: skip full weight SHA256, still validate inputs')
    args = p.parse_args()
    output = core.output_for(args.run_id)
    image = core.config()['image']
    if args.mode == 'pull':
        subprocess.run(['docker', 'pull', image], check=True)
        return
    if args.mode == 'status':
        core.show_status(output)
        if (output / 'docker.json').exists():
            receipt = core.read(output / 'docker.json')
            print(core.command(['docker', 'inspect', receipt['id'], '--format', '{{json .State}}']))
        return
    if args.mode == 'logs':
        log = output / 'supervisor.log'
        core.require(log.exists(), 'Supervisor log has not been created yet; inspect output/logs/docker_start.log')
        subprocess.run(['tail', '-n', '100', '-F', str(log)], check=True)
        return
    if args.mode == 'pause':
        receipt = core.read(output / 'docker.json')
        actual = json.loads(core.command(['docker', 'inspect', receipt['id']]))[0]
        core.require(actual['Id'] == receipt['id'] and actual['Config']['Labels'].get('research.scope') == receipt['scope'],
                     'Container identity/scope mismatch')
        core.require(actual['State']['Running'], 'Container is already stopped')
        subprocess.run(['docker', 'kill', '--signal=SIGUSR1', receipt['id']], check=True)
        print('Pause requested. Requests drain before GPU release; use status to confirm.')
        return
    core.command(['docker', 'image', 'inspect', image])  # No implicit network pull.
    (core.ROOT / 'output').mkdir(exist_ok=True)
    if args.mode == 'check':
        subprocess.run(['docker', 'run', '--rm', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
            *base_args(False), image, '-u', '-B', '/bundle/scripts/b200_collection.py', 'check']
            + (['--quick'] if args.quick else []), check=True)
        return
    gpu = core.select_gpu(args.gpu)
    gpu_args = ['--runtime', 'nvidia', '--gpus', 'device=' + gpu['uuid']]
    if args.mode == 'preflight':
        subprocess.run(['docker', 'run', '--rm', *gpu_args, *base_args(False), image,
                        '-u', '-B', '/bundle/scripts/b200_collection.py', 'preflight', '--gpu', gpu['uuid']], check=True)
        return
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        core.require((output / 'config.json').exists(), 'No previous run to resume')
        if (output / 'docker.json').exists():
            previous = core.read(output / 'docker.json')
            actual = json.loads(core.command(['docker', 'inspect', previous['id']]))[0]
            core.require(not actual['State']['Running'], 'Previous collection is still running')
            core.save(output / f'docker.previous.{time.time_ns()}.json', previous)
    scope = 'opcd-q38-b200-' + core.sha(core.ROOT / 'bundle_manifest.json')[:12] + '-' + args.run_id
    name = scope + '-' + str(time.time_ns())[-10:]
    cmd = ['docker', 'run', '-d', '--init', '--name', name, '--restart', 'no', '--label', 'research.scope=' + scope,
           *gpu_args, *base_args(), '-e', 'OPCD_CONTAINER_NAME=' + name, image, '-u', '-B',
           '/bundle/scripts/b200_collection.py', 'run', '--gpu', gpu['uuid'], '--run-id', args.run_id]
    if args.resume:
        cmd.append('--resume')
    core.save(output / 'docker.command.json', cmd)
    cid = core.command(cmd, timeout=180)
    ident = json.loads(core.command(['docker', 'inspect', cid]))[0]
    actual_devices = ident['HostConfig']['DeviceRequests'][0]['DeviceIDs']
    core.require(actual_devices == [gpu['uuid']], 'Docker GPU assignment mismatch')
    receipt = {'id': cid, 'name': name, 'scope': scope, 'gpu': gpu, 'image': image,
               'image_id': ident['Image'], 'created_at_utc': core.now(), 'training_started': False}
    core.save(output / 'docker.json', receipt)
    print(json.dumps({'status': 'launched_preflight_pending', **receipt, 'output': str(output)}))


if __name__ == '__main__':
    main()
