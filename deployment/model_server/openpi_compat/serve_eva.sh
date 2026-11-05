#!/usr/bin/env bash
# Usage: CHECKPOINT_PATH=/path/to/run/checkpoints/model.pt bash serve_eva.sh
# Activate your MR0 environment first, or set MR0_PYTHON explicitly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MR0_REPO="${MR0_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd -P)}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the policy checkpoint}"
export PYTHONPATH="$MR0_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}"
args=(--ckpt_path "$CHECKPOINT_PATH" --port "${PORT:-6000}")
if [[ -n "${UNNORM_KEY:-}" ]]; then args+=(--unnorm_key "$UNNORM_KEY"); fi
if [[ "${USE_BF16:-false}" == "true" ]]; then args+=(--use_bf16); fi
cd "$MR0_REPO"
exec "${MR0_PYTHON:-python}" -m deployment.model_server.openpi_compat.server "${args[@]}" "$@"
