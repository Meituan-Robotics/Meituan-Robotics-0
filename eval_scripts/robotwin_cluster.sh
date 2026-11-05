#!/usr/bin/env bash
# Forward evaluation options to the portable RoboTwin launcher.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOTWIN_SERVER_TIMEOUT="${ROBOTWIN_SERVER_TIMEOUT:-1200}"
export ROBOTWIN_JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-2}"
exec bash "${REPO_ROOT}/examples/Robotwin/eval_files/start_eval.sh" "$@"
