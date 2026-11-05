# MR0 on RoboTwin

Training configurations and evaluation adapters for MR0. See the [Getting Started guide](../../GETTING_STARTED.md) for model setup and multi-node training.

## 📦 1. Environment Setup

Please first follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to create the base `robotwin` environment.

Then prepare the two runtime environments once:

1. Install the MR0 dependencies in the `mr0` environment.

```bash
conda activate mr0
pip install -r requirements.txt
pip install -e .
```

2. Install the RoboTwin eval-side dependencies in the `robotwin` environment.

```bash
conda activate robotwin
pip install -r examples/Robotwin/eval_files/requirements.txt
```

3. Point the launcher to your local RoboTwin checkout.

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
```

4. Because RoboTwin is a third-party repository, patch your own local RoboTwin checkout so `script/eval_policy.py` accepts `--policy_ckpt_path`.

Apply the following change in your own RoboTwin repo:

```diff
diff --git a/script/eval_policy.py b/script/eval_policy.py
index eded198..9fb36e3 100644
--- a/script/eval_policy.py
+++ b/script/eval_policy.py
@@ -69,6 +69,7 @@ def main(usr_args):
     # checkpoint_num = usr_args['checkpoint_num']
     policy_name = usr_args["policy_name"]
     instruction_type = usr_args["instruction_type"]
+    policy_ckpt_path = usr_args["policy_ckpt_path"]
     save_dir = None
     video_save_dir = None
     video_size = None
@@ -81,6 +82,7 @@ def main(usr_args):
     args['task_name'] = task_name
     args["task_config"] = task_config
     args["ckpt_setting"] = ckpt_setting
+    args["policy_ckpt_path"] = policy_ckpt_path

     embodiment_type = args.get("embodiment")
     embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
@@ -327,11 +329,13 @@ def eval_policy(task_name,
 def parse_args_and_config():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, required=True)
     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
     args = parser.parse_args()

     with open(args.config, "r", encoding="utf-8") as f:
         config = yaml.safe_load(f)
+    config["policy_ckpt_path"] = args.policy_ckpt_path

     # Parse overrides
     def parse_override_pairs(pairs):
```

This patch is intentionally documented here rather than vendored into `mr0`, because RoboTwin is maintained in a separate repository. The MR0 launcher passes `--policy_ckpt_path` at runtime; without this patch, RoboTwin cannot forward the checkpoint path into `model2robotwin_interface.py`.

Optional:

- If you need the scripts to run the bootstrap `pip install` steps for you, set `export ROBOTWIN_AUTO_INSTALL_DEPS=1`.
- If your conda env names are different, set `ROBOTWIN_MR0_ENV` and `ROBOTWIN_ENV`.

## 🚀 2. Evaluation Workflow

### Recommended: `start_eval.sh` (unified entrypoint)

`start_eval.sh` is the main launcher. It starts the policy server, waits for readiness, runs the RoboTwin eval, streams per-episode success rates to the terminal, and cleans up all processes on exit (including Ctrl+C).

```
bash start_eval.sh -m <mode> -n <policy_name> -c <ckpt_path> [options] <tasks...>
```

#### Required flags

| Flag | Description |
|------|-------------|
| `-m`, `--mode` | Eval mode: `demo_clean` or `demo_randomized` |
| `-n`, `--name` | Policy name (used for log directory naming, forwarded to RoboTwin as `ckpt_setting`) |
| `-c`, `--ckpt` | Path to the MR0 checkpoint file |

#### Tasks (positional arguments)

All remaining arguments after flags are treated as tasks. You can specify:

- One or more task names: `adjust_bottle open_laptop lift_pot`
- The keyword `all` to evaluate all 50 RoboTwin 2.0 tasks
- A task-list file (one task per line): `task_list.txt`

#### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `-s`, `--seed` | `0` | Eval seed (also via `ROBOTWIN_SEED`) |
| `-j`, `--jobs-per-gpu` | `1` | Concurrent jobs per visible GPU (also via `ROBOTWIN_JOBS_PER_GPU`) |
| `-p`, `--base-port` | `5694` | First port to allocate (also via `ROBOTWIN_BASE_PORT`) |
| `--server-timeout` | `600` | Seconds to wait for the policy server to start (also via `ROBOTWIN_SERVER_TIMEOUT`) |
| `--install-deps` | off | Run pip install bootstrap steps once (also via `ROBOTWIN_AUTO_INSTALL_DEPS=1`) |
| `-h`, `--help` | | Show help message |

Flags take priority over environment variables when both are set.

#### Examples

Single task, clean mode:

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n test1 \
    -c /path/to/checkpoint.pt \
    adjust_bottle
```

Multiple tasks:

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_randomized -n my_run \
    -c /path/to/checkpoint.pt \
    adjust_bottle open_laptop lift_pot place_shoe
```

All 50 tasks with custom seed and 2 jobs per GPU:

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n full_eval -s 42 -j 2 \
    -c /path/to/checkpoint.pt \
    all
```

Tasks from a file:

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n my_run \
    -c /path/to/checkpoint.pt \
    task_list.txt
```

### Multi-GPU scheduling

The launcher auto-detects visible GPUs (via `CUDA_VISIBLE_DEVICES` or `nvidia-smi`) and runs one policy-server + eval pair per GPU by default. Ports are allocated automatically starting from `--base-port`.

8-GPU example:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_randomized -n full_eval \
    -c /path/to/checkpoint.pt \
    all
```

This schedules all 50 tasks across 8 GPUs, running up to 8 tasks in parallel. When a task finishes on a GPU, the next pending task is dispatched to that slot.

### Runtime output

During evaluation, per-episode success rates are streamed to stdout in real time:

```
[RESULT] adjust_bottle: Success rate: 1/1 => 100.0%, current seed: 100001
[RESULT] adjust_bottle: Success rate: 2/2 => 100.0%, current seed: 100002
[RESULT] adjust_bottle: Success rate: 3/3 => 100.0%, current seed: 100005
```

Full eval output (including per-step logs) is always saved to log files.

### Process cleanup

Pressing Ctrl+C (or sending SIGINT/SIGTERM) triggers a recursive cleanup that kills the entire process tree — including all policy servers and RoboTwin eval subprocesses. A SIGTERM is sent first, followed by SIGKILL after 2 seconds for any remaining processes.

### Logs

Logs are written under the checkpoint directory by default:

```
<ckpt_dir>/robotwin_eval_logs/<name>_<mode>_<ckpt_stem>_<timestamp>/
    <task>_<mode>_slot<N>_gpu<G>_port<P>_server.log
    <task>_<mode>_slot<N>_gpu<G>_port<P>_eval.log
```

Override the log root with `ROBOTWIN_LOG_ROOT`.

### Environment variables

These environment variables are read when the corresponding flag is not set:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOTWIN_PATH` | — | Path to the local RoboTwin repository (required) |
| `ROBOTWIN_MR0_ENV` | `mr0` | Conda env name for the policy server (used to auto-detect Python) |
| `ROBOTWIN_ENV` | `robotwin` | Conda env name for RoboTwin eval (used to auto-detect Python) |
| `MR0_PYTHON` | auto | Explicit path to the mr0 Python binary (skips conda env lookup) |
| `ROBOTWIN_PYTHON` | auto | Explicit path to the robotwin Python binary (skips conda env lookup) |
| `ROBOTWIN_SEED` | `0` | Eval seed (overridden by `-s`) |
| `ROBOTWIN_JOBS_PER_GPU` | `1` | Concurrent jobs per GPU (overridden by `-j`) |
| `ROBOTWIN_BASE_PORT` | `5694` | First port to allocate (overridden by `-p`) |
| `ROBOTWIN_SERVER_TIMEOUT` | `600` | Server startup timeout in seconds (overridden by `--server-timeout`) |
| `ROBOTWIN_AUTO_INSTALL_DEPS` | `0` | Set to `1` to bootstrap pip deps (overridden by `--install-deps`) |
| `ROBOTWIN_LOG_ROOT` | auto | Override the log output directory |

The launcher does **not** use `conda activate`. Instead, it locates the Python binary directly from the conda env directory. It searches `CONDA_EXE`, `CONDA_PREFIX`, `~/miniconda3/envs/`, `~/anaconda3/envs/`, etc. If auto-detection fails, set `MR0_PYTHON` and `ROBOTWIN_PYTHON` explicitly.

### `deploy_policy.yml` configuration

`examples/Robotwin/eval_files/deploy_policy.yml` is treated as a template. The following fields are read from it at runtime:

| Field | Description |
|-------|-------------|
| `normalization_mode` | Normalization mode: `min_max` or `q99` |
| `unnorm_key` | Unnormalization key for the embodiment |
| `action_mode` | Action mode (e.g. `abs`) |

`host` and `port` are overridden at runtime by the launcher. If your checkpoint was trained with percentile normalization, set `normalization_mode: "q99"`.

### Low-level manual mode

If you prefer to manage the policy server and eval processes yourself:

1. Start the policy server (in the `mr0` conda env):

```bash
bash examples/Robotwin/eval_files/run_policy_server.sh /path/to/checkpoint.pt [gpu_id] [port]
```

2. Run evaluation (in the `robotwin` conda env):

```bash
conda activate robotwin
cd examples/Robotwin/eval_files
bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <ckpt_path> [port] [host]
```

Example:

```bash
bash eval.sh adjust_bottle demo_clean my_eval 0 0 /path/to/checkpoint.pt 5694
```

### RoboTwin 2.0 task list

All tasks in RoboTwin 2.0 include:

```txt
adjust_bottle
beat_block_hammer
blocks_ranking_rgb
blocks_ranking_size
click_alarmclock
click_bell
dump_bin_bigbin
grab_roller
handover_block
handover_mic
hanging_mug
lift_pot
move_can_pot
move_pillbottle_pad
move_playingcard_away
move_stapler_pad
open_laptop
open_microwave
pick_diverse_bottles
pick_dual_bottles
place_a2b_left
place_a2b_right
place_bread_basket
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_container_plate
place_dual_shoes
place_empty_cup
place_fan
place_mouse_pad
place_object_basket
place_object_scale
place_object_stand
place_phone_stand
place_shoe
press_stapler
put_bottles_dustbin
put_object_cabinet
rotate_qrcode
scan_object
shake_bottle_horizontally
shake_bottle
stack_blocks_three
stack_blocks_two
stack_bowls_three
stack_bowls_two
stamp_seal
turn_switch
```

All modes include `demo_clean` and `demo_randomized`.
