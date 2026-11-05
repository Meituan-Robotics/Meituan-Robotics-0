"""Exercise renamed imports and framework/data discovery without loading weights."""
import importlib
from pathlib import Path

from omegaconf import OmegaConf

from mr0.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from mr0.model.framework.base_framework import _auto_import_framework_modules
from mr0.model.tools import FRAMEWORK_REGISTRY


def test_frameworks_register_under_mr0():
    _auto_import_framework_modules()
    for framework in ("QwenFast", "QwenMMDiT"):
        model_class = FRAMEWORK_REGISTRY[framework]
        assert model_class.__module__.startswith("mr0.")


def test_training_and_server_entrypoints_import():
    for name in (
        "mr0.training.train_mr0", "mr0.training.train_mr0_cotrain",
        "deployment.model_server.server_policy", "deployment.model_server.openpi_compat.server",
    ):
        assert callable(importlib.import_module(name).main)


def test_example_dataset_registries_resolve():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "examples/Robotwin/train_files").glob("mr0_*.yaml"):
        cfg = OmegaConf.load(path)
        mixture = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix]
        assert mixture
        for entry in mixture:
            assert entry[2] in ROBOT_TYPE_CONFIG_MAP


def test_qwen35_backbone_module_imports():
    module = importlib.import_module("mr0.model.modules.vlm.QWen3_5")
    assert module.Qwen3_5ForConditionalGeneration.__name__ == "Qwen3_5ForConditionalGeneration"
