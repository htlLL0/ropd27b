#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${OPCD_PYTHON:-}" ]]; then
  bundle_python="$OPCD_PYTHON"
elif [[ -x "$bundle_dir/.venv/bin/python" ]]; then
  bundle_python="$bundle_dir/.venv/bin/python"
else
  bundle_python=python3
fi
if [[ $# -eq 0 ]]; then set -- start; fi
case "$1" in
  status)
    exec "$bundle_python" -B "$bundle_dir/scripts/b200_collection.py" "$@"
    ;;
  verify|check|preflight|run|start|pause)
    mkdir -p "$bundle_dir/output/logs"
    "$bundle_python" -u -B "$bundle_dir/scripts/b200_collection.py" "$@" 2>&1 |
      tee -a "$bundle_dir/output/logs/native_$1.log"
    ;;
  *)
    exec "$bundle_python" -B "$bundle_dir/scripts/b200_collection.py" "$@"
    ;;
esac
