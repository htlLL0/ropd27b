#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then set -- --help; fi
case "$1" in
  status|logs|--help|-h)
    exec python3 -B "$bundle_dir/scripts/agentdojo_launcher.py" "$@" ;;
  *)
    mkdir -p "$bundle_dir/output/logs"
    python3 -u -B "$bundle_dir/scripts/agentdojo_launcher.py" "$@" 2>&1 |
      tee -a "$bundle_dir/output/logs/agentdojo_$1.log" ;;
esac
