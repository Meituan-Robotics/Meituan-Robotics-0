import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from mr0.training.trainer_utils.trainer_tools import TrainerUtils


class TinyVLA(nn.Module):
    def __init__(self, action_width=3):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(2, 2)
        self.action_model = nn.Linear(2, action_width)


class CheckpointLoadAuditTest(unittest.TestCase):
    def test_full_load_reports_complete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, target = TinyVLA(), TinyVLA()
            checkpoint_path = Path(tmp_dir) / "model.pt"
            audit_path = Path(tmp_dir) / "audit.json"
            torch.save(source.state_dict(), checkpoint_path)
            TrainerUtils.load_pretrained_backbones(target, checkpoint_path, audit_path=audit_path)
            report = json.loads(audit_path.read_text())
            self.assertEqual(report["missing_keys"], [])
            self.assertEqual(report["unexpected_keys"], [])
            self.assertEqual(report["shape_mismatches"], [])
            self.assertEqual(report["coverage"]["qwen_vl_interface"]["coverage"], 1.0)
            self.assertEqual(report["coverage"]["action_model"]["coverage"], 1.0)

    def test_full_load_fails_before_loading_on_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, target = TinyVLA(4), TinyVLA(3)
            original_target = {key: value.clone() for key, value in target.state_dict().items()}
            checkpoint_path = Path(tmp_dir) / "model.pt"
            audit_path = Path(tmp_dir) / "audit.json"
            torch.save(source.state_dict(), checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "full checkpoint audit failed"):
                TrainerUtils.load_pretrained_backbones(target, checkpoint_path, audit_path=audit_path)
            report = json.loads(audit_path.read_text())
            self.assertEqual(
                {item["key"] for item in report["shape_mismatches"]},
                {"action_model.bias", "action_model.weight"},
            )
            for key, value in original_target.items():
                torch.testing.assert_close(target.state_dict()[key], value)

    def test_partial_load_is_strict_and_leaves_action_model_initialized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, target = TinyVLA(), TinyVLA()
            original_action = {key: value.clone() for key, value in target.action_model.state_dict().items()}
            checkpoint_path = Path(tmp_dir) / "model.pt"
            audit_path = Path(tmp_dir) / "audit.json"
            torch.save(source.state_dict(), checkpoint_path)
            TrainerUtils.load_pretrained_backbones(
                target,
                checkpoint_path,
                reload_modules="qwen_vl_interface",
                audit_path=audit_path,
            )
            self.assertEqual(json.loads(audit_path.read_text())["reload_modules"], ["qwen_vl_interface"])
            for key, value in original_action.items():
                torch.testing.assert_close(target.action_model.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
