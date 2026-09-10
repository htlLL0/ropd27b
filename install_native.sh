#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$bundle_dir/output/setup"
exec > >(tee -a "$bundle_dir/output/setup/install_native.log") 2>&1
export PIP_CACHE_DIR="$bundle_dir/output/setup/pip-cache"
python3 -m venv "$bundle_dir/.venv"
"$bundle_dir/.venv/bin/python" -m pip install --upgrade pip
"$bundle_dir/.venv/bin/python" -m pip install -r "$bundle_dir/requirements-b200.txt"
"$bundle_dir/.venv/bin/python" -m pip check
printf '%s\n' 'Installed. Next: ./run_b200.sh preflight --gpu 0'
