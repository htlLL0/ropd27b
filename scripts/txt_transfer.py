#!/usr/bin/env python3
"""Lossless readable TXT transport. Every file, including headers, <=90,000 bytes."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

FORMAT = 'opcd-txt-transfer-v1'
MAGIC = b'OPCD-READABLE-FILES-V1\n'
CONTROL = '000000_CONTROL.txt'
LIMIT = 90000
EXCLUDED = {'transfers', 'cache', 'tmp', 'merged_model', 'adapter_snapshot', '__pycache__'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def line(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


def safe_relative(value):
    p = PurePosixPath(value)
    require(value and not p.is_absolute() and '..' not in p.parts and '\\' not in value
            and str(p) == value and value != '.', 'Unsafe or noncanonical archived path')
    return p


def selected_files(root):
    files = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part in EXCLUDED for part in rel.parts) or rel.name == 'transfer_latest.json':
            continue
        if path.is_symlink():
            raise ValueError('Symlink cannot be exported: ' + str(rel))
        if path.is_file():
            require(path.suffix not in ('.safetensors', '.pt', '.bin'), 'Model/binary weights are not result logs: ' + str(rel))
            files.append(path)
    priority = {'RESULTS.txt': 0, 'summary.json': 1, 'progress.json': 2, 'protocol.json': 3, 'run_binding.json': 4}
    return sorted(files, key=lambda p: (priority.get(str(p.relative_to(root)), 5), str(p)))


def pack(root, destination, max_bytes=LIMIT):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(2048 <= max_bytes <= LIMIT, 'Maximum TXT size must be 2048..90000 bytes')
    require(root.is_dir() and not destination.exists(), 'Source missing or destination already exists')
    files = selected_files(root)
    require(files, 'No result/log files to export')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.transfer-', dir=destination.parent) as tmp:
        temp = Path(tmp)
        stream_path = temp / 'stream'
        with stream_path.open('wb') as stream:
            stream.write(MAGIC)
            for path in files:
                stat = path.stat()
                raw = path.read_bytes()
                after = path.stat()
                require((stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and len(raw) == stat.st_size,
                        'A source log changed during export; stop its writer before export')
                try:
                    raw.decode('utf-8')
                    payload, encoding = raw, 'utf-8'
                except UnicodeDecodeError:
                    payload, encoding = base64.b64encode(raw), 'base64'
                stream.write(line({'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': digest(raw),
                                   'encoding': encoding, 'stored_bytes': len(payload)}))
                stream.write(payload)
                stream.write(b'\n')
            stream.write(line({'end': True, 'files': len(files)}))
        # Record boundaries without making a second on-disk payload copy.
        pieces = []
        with stream_path.open('rb') as stream:
            while True:
                offset = stream.tell()
                block = stream.read(max_bytes - 1024)
                if not block:
                    break
                try:
                    block.decode('utf-8')
                except UnicodeDecodeError as exc:
                    require(len(block) - exc.start <= 3, 'Internal stream is not UTF-8')
                    stream.seek(-(len(block) - exc.start), os.SEEK_CUR)
                    block = block[:exc.start]
                pieces.append((offset, len(block)))
        checksum = file_hash(stream_path)
        identity = checksum[:24]
        ready = temp / 'ready'
        ready.mkdir()
        previous = '0' * 64
        with stream_path.open('rb') as stream:
            for index, (offset, size) in enumerate(pieces, 1):
                stream.seek(offset)
                payload = stream.read(size)
                current = digest(payload)
                header = {'format': FORMAT, 'transfer_id': identity, 'part': index, 'parts': len(pieces),
                          'payload_bytes': len(payload), 'payload_sha256': current, 'previous_sha256': previous}
                body = line(header) + payload
                require(len(body) <= max_bytes, 'TXT size exceeds limit')
                (ready / f'part-{index:06d}-of-{len(pieces):06d}.txt').write_bytes(body)
                previous = current
        control = {'format': FORMAT, 'transfer_id': identity, 'parts': len(pieces), 'files': len(files),
                   'max_file_bytes': max_bytes, 'stream_bytes': stream_path.stat().st_size,
                   'stream_sha256': checksum, 'last_payload_sha256': previous,
                   'algorithm': 'SHA256 per part + ordered hash chain + complete stream + per original file',
                   'excluded_directories': sorted(EXCLUDED), 'text_encoding': 'UTF-8 without BOM'}
        require(len(line(control)) <= max_bytes, 'Control file too large')
        (ready / CONTROL).write_bytes(line(control))
        # Verify parts and original-file checksums without another full disk copy.
        with open(os.devnull, 'wb') as sink:
            concatenate(ready, sink)
        with stream_path.open('rb') as stream:
            parse_stream(stream, control)
        ready.rename(destination)
    return {**control, 'directory': str(destination)}


def concatenate(source, stream):
    source = Path(source)
    control_path = source / CONTROL
    require(control_path.is_file() and not control_path.is_symlink(), 'Missing control file')
    control_bytes = control_path.read_bytes()
    require(len(control_bytes) <= LIMIT, 'Control TXT exceeds 90,000 bytes')
    control = json.loads(control_bytes)
    require(control['format'] == FORMAT and 2048 <= control['max_file_bytes'] <= LIMIT, 'Wrong transfer format/limit')
    count = control['parts']
    require(type(count) is int and count >= 1, 'Invalid part count')
    actual = {p.name for p in source.iterdir()}
    require(len(actual) == count+1, f'Missing/duplicate/unexpected files: expected {count+1}, found {len(actual)}')
    names = {CONTROL} | {f'part-{i:06d}-of-{count:06d}.txt' for i in range(1, count+1)}
    require(actual == names, f'Missing/duplicate/unexpected files: missing={sorted(names-actual)[:10]}, extra={sorted(actual-names)[:10]}')
    previous, total, h = '0' * 64, 0, hashlib.sha256()
    for index in range(1, count+1):
        path = source / f'part-{index:06d}-of-{count:06d}.txt'
        require(path.is_file() and not path.is_symlink(), 'Invalid part file')
        body = path.read_bytes()
        require(len(body) <= control['max_file_bytes'], 'TXT exceeds size limit')
        header, payload = body.split(b'\n', 1)
        metadata = json.loads(header)
        payload.decode('utf-8')
        require(metadata['format'] == FORMAT and metadata['transfer_id'] == control['transfer_id']
                and metadata['part'] == index and metadata['parts'] == count, 'Part identity/index mismatch')
        require(metadata['payload_bytes'] == len(payload) and metadata['payload_sha256'] == digest(payload),
                'Damaged/truncated part payload')
        require(metadata['previous_sha256'] == previous, 'Broken part hash chain')
        previous = metadata['payload_sha256']
        total += len(payload)
        h.update(payload)
        stream.write(payload)
    require(total == control['stream_bytes'] and h.hexdigest() == control['stream_sha256']
            and previous == control['last_payload_sha256'] and h.hexdigest()[:24] == control['transfer_id'],
            'Complete stream integrity mismatch')
    return control


def parse_stream(stream, control, restore=None):
    require(stream.readline() == MAGIC, 'Bad joined stream header')
    seen = set()
    while True:
        header = stream.readline(65536)
        require(header.endswith(b'\n'), 'Missing/truncated file metadata')
        meta = json.loads(header)
        if meta.get('end') is True:
            require(meta['files'] == len(seen) == control['files'] and not stream.read(1), 'Wrong file count/trailing bytes')
            return
        relative = safe_relative(meta['path'])
        require(str(relative) not in seen, 'Duplicate archived file')
        seen.add(str(relative))
        size = meta['stored_bytes']
        require(type(size) is int and 0 <= size <= control['stream_bytes'], 'Invalid file size')
        data = stream.read(size)
        require(len(data) == size and stream.read(1) == b'\n', 'Truncated archived file')
        require(meta['encoding'] in ('utf-8', 'base64'), 'Unsupported encoding')
        if meta['encoding'] == 'utf-8':
            data.decode('utf-8')
        else:
            data = base64.b64decode(data, validate=True)
        require(len(data) == meta['bytes'] and digest(data) == meta['sha256'], 'Original file hash/length mismatch')
        if restore:
            path = restore / str(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def verify(source, *, joined=None, restored=None):
    source = Path(source).resolve()
    destination = Path(joined or restored).resolve() if joined or restored else None
    require(not (joined and restored), 'Select join or restore')
    if destination:
        require(not destination.exists() and source not in destination.parents, 'Destination exists or is inside transfer folder')
        destination.parent.mkdir(parents=True, exist_ok=True)
    # Temporary files are not placed in the input transfer directory.
    parent = destination.parent if destination else source.parent
    with tempfile.TemporaryDirectory(prefix='.verify-', dir=parent) as tmp:
        stream_path = Path(tmp) / 'joined.txt'
        with stream_path.open('wb') as stream:
            control = concatenate(source, stream)
        restore = Path(tmp) / 'restored' if restored else None
        if restore:
            restore.mkdir()
        with stream_path.open('rb') as stream:
            parse_stream(stream, control, restore)
        if joined:
            stream_path.rename(destination)
        elif restored:
            restore.rename(destination)
    return {'status': 'pass', **control}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['pack', 'verify', 'join', 'restore'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path)
    args = p.parse_args()
    if args.mode != 'verify':
        p.error('--destination is required') if args.destination is None else None
    if args.mode == 'pack':
        result = pack(args.source, args.destination)
    else:
        result = verify(args.source, joined=args.destination if args.mode == 'join' else None,
                        restored=args.destination if args.mode == 'restore' else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
