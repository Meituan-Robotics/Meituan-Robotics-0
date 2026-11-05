"""AgileX fold-cloth (eef.6d) data config.

state / action are both 20-dim eef-6d vectors:
  Left arm:  xyz(3) + 6D_rot(6) + gripper(1) = 10
  Right arm: xyz(3) + 6D_rot(6) + gripper(1) = 10
"""
import os

from mr0.dataloader.gr00t_lerobot.datasets import ModalityConfig
from mr0.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from mr0.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from mr0.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from mr0.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoRandomRotation,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class AgileXFoldEef6dDataConfig:
    video_keys = [
        "video.primary_image",
        "video.left_wrist",
        "video.right_wrist",
    ]
    state_keys = [
        "state.left_xyz",
        "state.left_rot6d",
        "state.left_gripper",
        "state.right_xyz",
        "state.right_rot6d",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_xyz",
        "action.left_rot6d",
        "action.left_gripper",
        "action.right_xyz",
        "action.right_rot6d",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(
        self, binary_gripper: bool = True, image_augmentation: bool = False, normalization_mode: str = "min_max"
    ):
        # normalization_mode controls the non-gripper dims: "min_max" (default, legacy) or
        # "q99" (eva-openpi quantile norm: q01/q99 -> [-1,1]). Gripper dims stay binary
        # when binary_gripper=True, else fall back to normalization_mode as well.
        # MR0_NORMALIZATION_MODE env var (recap chain + eva_adapter.py serving) takes
        # precedence: when set, it forces that mode on ALL dims including grippers, matching
        # the uniform normalization the recap chain trains and serves with.
        env_mode = os.environ.get("MR0_NORMALIZATION_MODE", None)
        if env_mode is not None:
            if env_mode not in {"min_max", "q99"}:
                raise ValueError(f"MR0_NORMALIZATION_MODE must be one of [min_max, q99], got {env_mode!r}")
            normalization_mode = env_mode
            binary_gripper = False
        gripper_norm = "binary" if binary_gripper else normalization_mode
        state_norm = {k: (gripper_norm if "gripper" in k else normalization_mode) for k in self.state_keys}
        action_norm = {k: (gripper_norm if "gripper" in k else normalization_mode) for k in self.action_keys}

        # openpi-style image augmentation (train-only; reproduces eva-openpi model.py:173-192).
        # Random ops return None in eval mode (VideoTransform.get_transform), so inference is
        # unaffected. Off by default; driven by the yaml flag datasets.vla_data.image_augmentation.
        video_augs = []
        if image_augmentation:
            # Primary (non-wrist) cam gets the geometric augs; wrist cams do not — matching openpi.
            primary_keys = [k for k in self.video_keys if "wrist" not in k]
            # Native cam resolution is 480x640 (H,W) for all 3 views (dataset info.json cam shape
            # [3, 480, 640]). Crop shrinks the primary view, so we Resize it back to native before
            # the shared ColorJitter, which concatenates all views and requires a uniform H,W.
            video_augs = [
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=primary_keys, scale=0.95),
                VideoResize(apply_to=primary_keys, height=480, width=640, interpolation="linear"),
                VideoRandomRotation(apply_to=primary_keys, degrees=(-5, 5)),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.0,
                ),
                VideoToNumpy(apply_to=self.video_keys),
            ]

        return ComposedModalityTransform(
            transforms=[
                *video_augs,
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_norm,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_norm,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "agile_x_eef_6d": AgileXFoldEef6dDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "agile_x_eef_6d": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "agilex_fold_v1": [
        ("AgileXPP_Fold_Cloth_Norm_Dual_Pos", 1.0, "agile_x_eef_6d"),
    ],
    "agilex_fold_v2": [
        ("AgileXPP_Fold_Cloth_Norm_Dual_Pos_V2", 1.0, "agile_x_eef_6d"),
    ],
    "agilex_fold_cmagic": [
        ("CMagic_Fold_Cloth_Norm_Dual_Base", 1.0, "agile_x_eef_6d"),
    ],
    "agilex_fold_all": [
        ("AgileXPP_Fold_Cloth_Norm_Dual_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_Fold_Cloth_Norm_Dual_Pos_V2", 1.0, "agile_x_eef_6d"),
        ("CMagic_Fold_Cloth_Norm_Dual_Base", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_Fold_Cloth_Norm_Dual_Pos_V3", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_Fold_Cloth_Hil_Dual_Pos", 3.0, "agile_x_eef_6d"),
    ],
    "block_new": [
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V2", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V3", 1.0, "agile_x_eef_6d"),
    ],
    # block_260707_mask_still_with_HIL: 7 sub-datasets (2 HIL entities + 5 Norm symlinks),
    # each parquet carries a `collector` column (1=moving, 0=static) for static-frame masking
    # (enable collector_full_chunk_only in the data cfg). Equal-weight mix, matching the
    # eva-openpi reference (agile_x_block_eef6d_h50_masked_still_hil) parent-dir co-training.
    "block_260707_hil": [
        ("AgileXPP_PnP_Block_HiL_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_HiL_Dual_Ori_Pos_V2", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V2", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V3", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V4", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V5", 1.0, "agile_x_eef_6d"),
    ],
    # block_260715_340Nsft: 4 pure-Norm SFT sub-datasets (no HIL), 340 episodes total
    # (100 + 50 + 90 + 100). Same eef.6d layout + `collector` static-frame column as
    # block_260707_hil, so collector_full_chunk_only masking still applies. Equal-weight mix.
    "block_260715_340Nsft": [
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_PosV0", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V3", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_v2", 1.0, "agile_x_eef_6d"),
    ],
    # RECAP target: all 11 children of mixed/mr0_cable_v1.
    # Each subset preserves its own episode-local indices and image-feature parquet schema.
    "mr0_cable_v1_recap": [
        ("hil70", 1.0, "agile_x_eef_6d"),
        ("hil80", 1.0, "agile_x_eef_6d"),
        ("hil100", 1.0, "agile_x_eef_6d"),
        ("il0615", 1.0, "agile_x_eef_6d"),
        ("il0616", 1.0, "agile_x_eef_6d"),
        ("il061718", 1.0, "agile_x_eef_6d"),
        ("il0622", 1.0, "agile_x_eef_6d"),
        ("il0626", 1.0, "agile_x_eef_6d"),
        ("il0707", 1.0, "agile_x_eef_6d"),
        ("teleop_lancable", 1.0, "agile_x_eef_6d"),
        ("teleop_norm", 1.0, "agile_x_eef_6d"),
    ],
    # RECAP target: all 13 children of mixed/mr0_cable_v2 (mr0_cable_v1 的 11 个 + V11/V12 新增 2 个).
    "mr0_cable_v2_recap": [
        ("hil70", 1.0, "agile_x_eef_6d"),
        ("hil80", 1.0, "agile_x_eef_6d"),
        ("hil100", 1.0, "agile_x_eef_6d"),
        ("il0615", 1.0, "agile_x_eef_6d"),
        ("il0616", 1.0, "agile_x_eef_6d"),
        ("il061718", 1.0, "agile_x_eef_6d"),
        ("il0622", 1.0, "agile_x_eef_6d"),
        ("il0626", 1.0, "agile_x_eef_6d"),
        ("il0707", 1.0, "agile_x_eef_6d"),
        ("teleop_lancable", 1.0, "agile_x_eef_6d"),
        ("teleop_norm", 1.0, "agile_x_eef_6d"),
        ("mr0_150k_0722", 1.0, "agile_x_eef_6d"),
        ("mr0_150k_0723", 1.0, "agile_x_eef_6d"),
    ],
    # RECAP target: all 14 children of mixed/mr0_cable_v3 (v2 + recap_0724).
    "mr0_cable_v3_recap": [
        ("hil70", 1.0, "agile_x_eef_6d"),
        ("hil80", 1.0, "agile_x_eef_6d"),
        ("hil100", 1.0, "agile_x_eef_6d"),
        ("il0615", 1.0, "agile_x_eef_6d"),
        ("il0616", 1.0, "agile_x_eef_6d"),
        ("il061718", 1.0, "agile_x_eef_6d"),
        ("il0622", 1.0, "agile_x_eef_6d"),
        ("il0626", 1.0, "agile_x_eef_6d"),
        ("il0707", 1.0, "agile_x_eef_6d"),
        ("teleop_lancable", 1.0, "agile_x_eef_6d"),
        ("teleop_norm", 1.0, "agile_x_eef_6d"),
        ("mr0_150k_0722", 1.0, "agile_x_eef_6d"),
        ("mr0_150k_0723", 1.0, "agile_x_eef_6d"),
        ("recap_0724", 1.0, "agile_x_eef_6d"),
    ],
    # block_260706_masked_still: 5 pure-Norm SFT subsets (no HIL), real dirs (not symlinks),
    # each carrying a `collector` static-frame column + 20-D eef.6d layout, so
    # collector_full_chunk_only masking still applies. Equal-weight mix.
    "block_260706_500Osft": [
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V2", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V3", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V4", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V5", 1.0, "agile_x_eef_6d"),
    ],
    # block_260716_500Nsft: 6 pure-Norm SFT subsets (no HIL) = block_260715_340Nsft's 4
    # New-Norm subsets + two extra real dirs (V4, V5). Same 20-D eef.6d layout + `collector`
    # static-frame column, so collector_full_chunk_only masking still applies. Equal-weight mix.
    "block_260716_500Nsft": [
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_PosV0", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V3", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V4", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V5", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_v2", 1.0, "agile_x_eef_6d"),
    ],
    # block_260723_500Nsft_50HIL_MR0: block_260716_500Nsft's 6 pure-Norm SFT subsets +
    # one MR0 HIL real dir (AgileXPP_PnP_Block_MR0HiL_New_Dual_Ori_Pos_V1, 58 episodes).
    # Same 20-D eef.6d layout + `collector` static-frame column, so collector_full_chunk_only
    # masking still applies. Equal-weight mix.
    "block_260723_500Nsft_50HIL_MR0": [
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_PosV0", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V3", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V4", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V5", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_v2", 1.0, "agile_x_eef_6d"),
        ("AgileXPP_PnP_Block_MR0HiL_New_Dual_Ori_Pos_V1", 1.0, "agile_x_eef_6d"),
    ],
}
