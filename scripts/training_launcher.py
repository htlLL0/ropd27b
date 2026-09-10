#!/usr/bin/env python3
"""Host Docker controls for the added clean-teacher training route."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import time
from zoneinfo import ZoneInfo

import b200_collection as core
from docker_launcher import base_args


def image_tag():
    return 'opcd-clean-training:' + core.sha(core.ROOT / 'Dockerfile.training')[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['build', 'check', 'preflight', 'start', 'status', 'pause', 'logs'])
    p.add_argument('--gpu', default=os.environ.get('OPCD_GPU', '0'))
    p.add_argument('--run-id', default='clean_opcd')
    p.add_argument('--generation-run-id', default='main')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output, generation = core.output_for(args.run_id), core.output_for(args.generation_run_id)
    core.require(output != generation, 'Training and generation need different run IDs')
    if args.mode == 'build':
        # Dockerfile via stdin: never send the 56 GB model as build context.
        with (core.ROOT / 'Dockerfile.training').open('rb') as stream:
            subprocess.run(['docker', 'build', '-t', image_tag(), '-'], stdin=stream, check=True)
        return
    if args.mode == 'status':
        if (output / 'progress.json').exists():
            value = core.read(output / 'progress.json')
            stamp = datetime.fromisoformat(value['updated_at_utc']).astimezone(ZoneInfo('Asia/Shanghai'))
            print(f"Beijing {stamp:%m-%d %H:%M:%S} | {value['stage']} | {value['step']}/{value['total_steps']}")
            print(json.dumps(value.get('metrics', {}), ensure_ascii=False))
            eta = value.get('estimated_remaining_seconds')
            print(f'ETA: {eta / 3600:.2f} hours (observed estimate)' if eta else 'ETA: pending 10 updates this session')
        else:
            print('Training not started: ' + str(output))
        if (output / 'docker.json').exists():
            receipt = core.read(output / 'docker.json')
            print(core.command(['docker', 'inspect', receipt['id'], '--format', '{{json .State}}']))
        return
    if args.mode == 'logs':
        subprocess.run(['tail', '-n', '100', '-F', str(output / 'supervisor.log')], check=True)
        return
    if args.mode == 'pause':
        receipt = core.read(output / 'docker.json')
        actual = json.loads(core.command(['docker', 'inspect', receipt['id']]))[0]
        core.require(actual['Id'] == receipt['id'] and actual['Config']['Labels'].get('research.scope') == receipt['scope'],
                     'Container identity mismatch')
        core.require(actual['State']['Running'], 'Training container already stopped')
        subprocess.run(['docker', 'kill', '--signal=SIGUSR1', receipt['id']], check=True)
        print('Pause requested: finish the current update, save adapter/optimizer/cursor, then exit.')
        return
    image = image_tag()
    core.command(['docker', 'image', 'inspect', image])
    (core.ROOT / 'output').mkdir(exist_ok=True)
    if args.mode == 'check':
        subprocess.run(['docker', 'run', '--rm', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
            *base_args(False), image, '-u', '-B', '/bundle/scripts/train_clean_teacher.py', 'check'], check=True)
        # Real tiny hybrid Qwen CPU forward/backward and numerical regression gates.
        subprocess.run(['docker', 'run', '--rm', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
            *base_args(), '-e', 'OPCD_TEST_TMP=/bundle/output/setup/training_tests', image,
            '-u', '-B', '/bundle/validation/test_clean_teacher_opcd.py'], check=True)
        return
    gpu = core.select_gpu(args.gpu)
    gpu_args = ['--runtime', 'nvidia', '--gpus', 'device=' + gpu['uuid']]
    if args.mode == 'preflight':
        subprocess.run(['docker', 'run', '--rm', *gpu_args, *base_args(False), image, '-u', '-B',
            '/bundle/scripts/train_clean_teacher.py', 'preflight', '--gpu', gpu['uuid']], check=True)
        return
    core.require((generation / 'completion.json').exists(), 'Finish generation first: ./docker_b200.sh status')
    finished = core.read(generation / 'completion.json')
    core.require(finished['status'] == 'complete' and finished['samples'] == 9600 and finished['server_released'],
                 'Generation is not complete or server has not released GPU')
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        core.require((output / 'config.json').exists(), 'No previous training run')
        core.require(not (output / 'completion.json').exists(), 'Training already complete')
        previous = core.read(output / 'docker.json')
        actual = json.loads(core.command(['docker', 'inspect', previous['id']]))[0]
        core.require(not actual['State']['Running'], 'Previous training container is still running')
        core.save(output / f'docker.previous.{time.time_ns()}.json', previous)
    scope = 'opcd-clean-train-' + core.sha(core.ROOT / 'bundle_manifest.json')[:12] + '-' + args.run_id
    name = scope + '-' + str(time.time_ns())[-10:]
    command = ['docker', 'run', '-d', '--init', '--name', name, '--restart', 'no', '--label', 'research.scope=' + scope,
        *gpu_args, *base_args(), image, '-u', '-B', '/bundle/scripts/train_clean_teacher.py', 'run',
        '--gpu', gpu['uuid'], '--run-id', args.run_id, '--generation-run-id', args.generation_run_id]
    if args.resume:
        command.append('--resume')
    core.save(output / 'docker.command.json', command)
    cid = core.command(command, timeout=180)
    actual = json.loads(core.command(['docker', 'inspect', cid]))[0]
    core.require(actual['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == [gpu['uuid']], 'GPU assignment mismatch')
    receipt = {'id': cid, 'scope': scope, 'gpu': gpu, 'image': image, 'image_id': actual['Image'],
               'generation_run_id': args.generation_run_id, 'at_utc': core.now()}
    core.save(output / 'docker.json', receipt)
    print(json.dumps({'status': 'launched_training_preflight_pending', 'output': str(output), **receipt}))


if __name__ == '__main__':
    main()
