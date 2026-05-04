#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/venv/bin/python"
if [[ -x "${PY}" ]]; then
  exec "${PY}" "${ROOT}/video_streamer.py" "$@"
fi
echo "No venv at ${ROOT}/venv. Create it and install deps, then re-run:" >&2
echo "  python3 -m venv ${ROOT}/venv" >&2
echo "  ${ROOT}/venv/bin/python -m pip install -r ${ROOT}/requirements.txt" >&2
echo "  ${ROOT}/run_video_streamer.sh" >&2
exit 1
