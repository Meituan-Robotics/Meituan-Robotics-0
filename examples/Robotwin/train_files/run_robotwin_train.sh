#!/bin/bash
# Example: train MR0 on RoboTwin.
#
# Modify the paths at the top and pick a config, then run:
#   bash examples/Robotwin/train_files/run_robotwin_train.sh
#
# For multi-node, set NNODES / NPROC_PER_NODE / NODE_RANK / MASTER_ADDR / MASTER_PORT
# environment variables — see train_scripts/launch_training.sh.
# Use the same RUN_ID on every node.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ─── Choose training config (Stage 1 fast, Stage 2 mmdit) ───
CONFIG=${CONFIG:-examples/Robotwin/train_files/mr0_fast_robotwin_qwen35_bs8_32gpu.yaml}

# ─── Override paths / run id / batch size on the command line if needed ───
BASE_VLM=${BASE_VLM:-/PATH/TO/Qwen3.5-4B-Action}
DATA_ROOT=${DATA_ROOT:-/PATH/TO/RoboTwin}
RUN_ROOT=${RUN_ROOT:-/PATH/TO/checkpoints}
RUN_ID=${RUN_ID:-mr0_fast_robotwin}

bash train_scripts/launch_training.sh \
  --config_yaml "$CONFIG" \
  --framework.qwenvl.base_vlm "$BASE_VLM" \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --run_root_dir "$RUN_ROOT" \
  --run_id "$RUN_ID" \
  "$@"
