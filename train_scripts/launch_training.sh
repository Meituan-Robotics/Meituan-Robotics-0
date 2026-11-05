#!/bin/bash
# Generic multi-node torchrun launcher for MR0 training.
#
# Example: 4 nodes x 8 GPUs (run on every node with NODE_RANK=0, 1, 2, or 3).
# Set MASTER_ADDR to node 0's reachable IP; share the same port and run ID.
#   NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#   bash train_scripts/launch_training.sh \
#     --config_yaml examples/Robotwin/train_files/mr0_fast_robotwin_qwen35_bs8_32gpu.yaml \
#     --run_id mr0_fast_robotwin
#
# All arguments are forwarded to the training entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
ulimit -n 80000

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l | tr -d ' ')}

ENTRY=${ENTRY:-mr0/training/train_mr0.py}

echo "==============================================="
echo "  MR0 training launcher"
echo "  entry:            $ENTRY"
echo "  nnodes:           $NNODES"
echo "  node_rank:        $NODE_RANK"
echo "  nproc_per_node:   $NPROC_PER_NODE"
echo "  master:           ${MASTER_ADDR}:${MASTER_PORT}"
echo "  args:             $@"
echo "==============================================="

torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ENTRY" \
  "$@"
