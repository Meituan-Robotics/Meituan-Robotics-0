# Getting Started with MR0

[Back to the project overview](README.md)

[Installation](#installation) · [Model weights](#model-weights) · [Simulation evaluation](#simulation-evaluation) · [Training](#training) · [Serving](#serving--real-robot) · [Repository layout](#repository-layout)

## Installation

Run these commands from the repository root. Use Python 3.11 and install FFmpeg for video decoding.

```bash
conda create -n mr0 python=3.11 -y
conda activate mr0
conda install -c conda-forge 'ffmpeg<8' -y

# Install the matching CUDA build of the pinned PyTorch / TorchVision pair.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

The commands install the `mr0` package in editable mode and pin Transformers 5.7.0 for Qwen3.5. TorchCodec 0.3.0 is paired with PyTorch 2.7.1; see the [TorchCodec compatibility table](https://github.com/meta-pytorch/torchcodec#installing-torchcodec). If you choose a different PyTorch build, update TorchVision and TorchCodec together. Flash Attention is optional; the VLM wrapper falls back to PyTorch SDPA when it is unavailable.

## Model weights

Download the pretrained model or a benchmark-specific model from the [model table](README.md#models). A complete Qwen3.5 VLA checkpoint already contains the Qwen backbone and action head. Keep its matching configuration, dataset statistics, and Qwen tokenizer/processor metadata alongside it; a separate Qwen weight file is unnecessary.

For a downloaded MR0 or RoboTwin release, load the complete checkpoint with:

```python
import os
from pathlib import Path
from mr0.model.framework.base_framework import baseframework

root = Path("/PATH/TO/DOWNLOADED_MODEL").resolve()
os.environ["MR0_RELEASE_DIR"] = str(root)
model = baseframework.from_pretrained(root / "checkpoints/model.pt").eval().to("cuda")
```

The loader reads Qwen weights directly from the complete checkpoint and checks the full model with strict state-dict loading. `base_vlm/Qwen3.5-4B-Action/` supplies configuration and tokenizer/processor files only. For RoboDojo and RoboCasa365, follow the benchmark-specific setup below.

## Simulation evaluation

<a id="robotwin-evaluation"></a>

### RoboTwin

Install the RoboTwin simulator and its environment separately, and set `ROBOTWIN_PATH` to its checkout. Download [Meituan-Robotics-0-RoboTwin](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboTwin), keeping its checkpoint, configuration, and dataset statistics together. The launcher supports both clean and randomized evaluation:

```bash
export ROBOTWIN_PATH=/PATH/TO/RoboTwin
export ROBOTWIN_MR0_ENV=mr0

bash examples/Robotwin/eval_files/start_eval.sh \
  --mode demo_randomized \
  --name mr0_robotwin \
  --ckpt /PATH/TO/Meituan-Robotics-0-RoboTwin/checkpoints/model.pt \
  all
```

Use `--mode demo_clean` for the clean setting. The `eval_files/` directory contains the policy server and simulator interface. See [RoboTwin setup](examples/Robotwin/README.md) for environment instructions. The selected native policy must use its matching image preprocessing; the bundled 224 × 224 training example is not a substitute.

### RoboDojo

Refer to the [XPolicyLab MR0 policy](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Meituan_Robotics_0) for RoboDojo environment setup and evaluation. The [Meituan-Robotics-0-RoboDojo model card](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboDojo) documents the matching policy adapter, complete checkpoint, and required CD-LAM auxiliary checkpoint.

### RoboCasa365

Refer to [robocasa365-eval](https://github.com/linglingxiansen/robocasa365-eval) for environment setup and evaluation commands. Use [Meituan-Robotics-0-RoboCasa365](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboCasa365) and point the evaluator at `checkpoints/model.pt` in the downloaded release, keeping its configuration, dataset statistics, and tokenizer/processor metadata together.

## Training

Choose a RoboTwin training configuration:

| Recipe | Configuration |
| :--- | :--- |
| MMDiT downstream training from the pretrained MR0 model | [mr0_mmdit_robotwin_qwen35_bs8_32gpu.yaml](examples/Robotwin/train_files/mr0_mmdit_robotwin_qwen35_bs8_32gpu.yaml) |
| FAST action-token training with Qwen3.5 | [mr0_fast_robotwin_qwen35_bs8_32gpu.yaml](examples/Robotwin/train_files/mr0_fast_robotwin_qwen35_bs8_32gpu.yaml) |

Both examples use **4 nodes × 8 GPUs = 32 GPUs**, with a per-GPU batch size of 8 (global batch size 256). The MMDiT example uses 224 × 224 images; use the released RoboTwin model's matching configuration for evaluation.

### Configure the run

Set `framework.qwenvl.base_vlm`, `datasets.vla_data.data_root_dir`, and `run_root_dir` in your chosen configuration. Use the same configuration and accessible data/model paths on all four nodes.

For MMDiT downstream training, download [Meituan-Robotics-0](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0), set both `framework.qwenvl.checkpoint_path` and `trainer.pretrained_checkpoint` to its `checkpoints/model.pt`, and point `framework.qwenvl.base_vlm` at its `base_vlm/Qwen3.5-4B-Action/` directory. Training from a standalone pretrained Qwen model uses `base_vlm` with Qwen weights and leaves `checkpoint_path` unset.

```bash
${EDITOR:-vi} examples/Robotwin/train_files/mr0_mmdit_robotwin_qwen35_bs8_32gpu.yaml
```

<a id="multi-node-training"></a>

### Launch on multiple nodes

Run the following command **on each of the four nodes**. Set `NODE_RANK` to **0, 1, 2, or 3** for the corresponding node. Use the same `MASTER_ADDR`, `MASTER_PORT`, and `run_id` everywhere; `MASTER_ADDR` must be the reachable address of node 0.

```bash
conda activate mr0

export NNODES=4
export NPROC_PER_NODE=8
export NODE_RANK=0                 # Change to 1, 2, or 3 on the other nodes
export MASTER_ADDR=10.0.0.1        # Replace with node 0's reachable IP
export MASTER_PORT=29500

bash train_scripts/launch_training.sh \
  --config_yaml examples/Robotwin/train_files/mr0_mmdit_robotwin_qwen35_bs8_32gpu.yaml \
  --run_id mr0_mmdit_robotwin
```

### Overrides and VLM co-training

Append configuration overrides to the launch command, such as `--datasets.vla_data.per_device_batch_size 4` or `--trainer.max_train_steps 100000`. Use the same overrides on every node.

For VLM co-training (adds a VQA auxiliary loss), set `ENTRY=mr0/training/train_mr0_cotrain.py` when invoking the training launcher and register your VQA datasets in `mr0/dataloader/qwenvl_llavajson/qwen_data_config.py`.

## Serving / Real Robot

An OpenPI-compatible policy server lives under `deployment/model_server/openpi_compat/`. It exposes an inference endpoint over the OpenPI websocket protocol so that any OpenPI-compatible controller can drive the model.

Real-robot post-training scaffolding is under `examples/AgileX_Fold/`. The `data_registry/` and `gen_block_modality_shadow.py` there show how to plug a new real-robot dataset into the loader; adapt to your own robot's data schema and add the corresponding mixture to `mr0/dataloader/gr00t_lerobot/registry.py`.

## Repository Layout

```
mr0/
  model/
    framework/VLM4A/       QwenFast, QwenMMDiT          (VLA frameworks; auto-registered)
    modules/
      vlm/                 Qwen3.5 VL backbone
      action_model/        FAST + MMDiT action heads
  dataloader/              LeRobot dataset loader, dataset registry, VLM datasets
  training/
    train_mr0.py         VLA-only training entry point
    train_mr0_cotrain.py VLA + VLM co-training (multimodal auxiliary loss)
    trainer_utils/           Trainer helpers, checkpoint load audit, distributed utilities

examples/
  Robotwin/                Reference RoboTwin training + evaluation
    train_files/           Stage 1 (fast) and Stage 2 (mmdit) yaml configs
    eval_files/            Policy server, model interface, RoboTwin deploy specs
  AgileX_Fold/             Real-robot post-training scaffolding

deployment/model_server/openpi_compat/    OpenPI-compatible websocket policy server
tests/                                    Unit tests (dataloader, checkpoint audit, MMDiT, ...)
tools/                                    Dataset statistics, HF LeRobot download, AV1→H.264 conversion
train_scripts/launch_training.sh          Generic torchrun launcher
eval_scripts/robotwin_cluster*.sh         RoboTwin evaluation drivers
```


## Local validation

```bash
pip install -e '.[dev]'
python -m pytest tests -q
ruff check --select E9,F821,F822,F823 mr0 deployment examples tools tests
python -m build --wheel
```

The tests use synthetic inputs and small modules; model weights and the RoboTwin simulator are required separately for full evaluation. The Python package is `mr0`; use `MR0_PYTHON` and `ROBOTWIN_MR0_ENV` for evaluation scripts. Existing state-dict checkpoint keys and framework names (`QwenFast`, `QwenMMDiT`) are unchanged by the package rename.

See the [release validation record](docs/release_validation.md) for the tested environment and coverage.
