"""Run launcher control flow with fake executables; never starts training or a robot."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fake_python(tmp_path):
    executable = tmp_path / "fake-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'path': os.environ.get('PYTHONPATH', '')}))\n"
    )
    executable.chmod(0o755)
    return executable


def run(script, args, tmp_path, env):
    return subprocess.run(
        ["bash", str(ROOT / script), *args], cwd=tmp_path,
        env={**os.environ, **env}, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("rank", range(4))
def test_training_launcher_uses_renamed_entry_and_distributed_ranks(tmp_path, rank):
    executable = fake_python(tmp_path)
    (tmp_path / "torchrun").symlink_to(executable)
    capture = tmp_path / "capture.json"
    result = run("train_scripts/launch_training.sh", ["--config_yaml", "my config.yaml"], tmp_path, {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "CAPTURE": str(capture),
        "NNODES": "4", "NPROC_PER_NODE": "8", "NODE_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500", "ENTRY": "mr0/training/train_mr0.py",
    })
    assert result.returncode == 0, result.stderr
    data = json.loads(capture.read_text())
    assert "--nnodes=4" in data["args"]
    assert f"--node_rank={rank}" in data["args"]
    assert "--nproc_per_node=8" in data["args"]
    assert data["args"][-3:] == ["mr0/training/train_mr0.py", "--config_yaml", "my config.yaml"]
    assert str(ROOT) in data["path"].split(os.pathsep)


def test_eval_launcher_forwards_paths_with_spaces(tmp_path):
    robotwin = tmp_path / "RoboTwin checkout"
    (robotwin / "script").mkdir(parents=True)
    (robotwin / "script/eval_policy.py").write_text("# policy_ckpt_path\n")
    capture = tmp_path / "capture.json"
    result = run("examples/Robotwin/eval_files/eval.sh", [
        "adjust_bottle", "demo_clean", "mr0", "42", "0", "/tmp/model weights.pt", "5694",
    ], tmp_path, {"ROBOTWIN_PATH": str(robotwin), "ROBOTWIN_PYTHON": str(fake_python(tmp_path)), "CAPTURE": str(capture)})
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())["args"]
    assert args[args.index("--policy_ckpt_path") + 1] == "/tmp/model weights.pt"
    assert args[args.index("--task_name") + 1] == "adjust_bottle"
    assert not Path(args[args.index("--config") + 1]).exists()  # temporary policy YAML cleaned up


def test_openpi_launcher_forwards_checkpoint_and_extra_options(tmp_path):
    capture = tmp_path / "capture.json"
    result = run("deployment/model_server/openpi_compat/serve_eva.sh", ["--normalization_mode", "min_max"], tmp_path, {
        "CHECKPOINT_PATH": "/tmp/model weights.pt", "MR0_PYTHON": str(fake_python(tmp_path)),
        "CAPTURE": str(capture), "USE_BF16": "true",
    })
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())["args"]
    assert args[:2] == ["-m", "deployment.model_server.openpi_compat.server"]
    assert args[args.index("--ckpt_path") + 1] == "/tmp/model weights.pt"
    assert "--use_bf16" in args
    assert args[-2:] == ["--normalization_mode", "min_max"]


@pytest.mark.parametrize("suffix", ["", "_320x240", "_640x480"])
def test_cluster_wrapper_help_is_portable(tmp_path, suffix):
    result = run(f"eval_scripts/robotwin_cluster{suffix}.sh", ["--help"], tmp_path, {})
    assert result.returncode == 0, result.stderr
    assert "MR0_PYTHON" in result.stderr
