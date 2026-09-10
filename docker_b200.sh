#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then set -- start; fi
case "$1" in
  status|logs)
    exec python3 -B "$bundle_dir/scripts/docker_launcher.py" "$@"
    ;;
  pull|check|preflight|start|pause)
    mkdir -p "$bundle_dir/output/logs"
    python3 -u -B "$bundle_dir/scripts/docker_launcher.py" "$@" 2>&1 |
      tee -a "$bundle_dir/output/logs/docker_$1.log"
    ;;
  *)
    exec python3 -B "$bundle_dir/scripts/docker_launcher.py" "$@"
    ;;
esac
