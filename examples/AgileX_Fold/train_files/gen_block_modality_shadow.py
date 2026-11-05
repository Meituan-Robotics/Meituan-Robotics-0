#!/usr/bin/env python3
"""Generate modality.json shadow files for block_260707_mask_still_with_HIL.

The target dataset (7 sub-datasets: 2 HIL entities + 5 Norm symlinks) has no
meta/modality.json, and its symlinked Norm subsets point at a shared source we
must not pollute. So we write the modality metadata into a *shadow* directory
that mirrors the dataset layout; the dataloader resolves it via
``modality_root_dir`` (see LeRobotSingleDataset._get_shadow_meta_path):

    shadow file = <modality_root_dir> / <subset> / meta / modality.json

where <subset> = dataset_path.relative_to(data_root_dir). So pass
``--shadow_root`` equal to the yaml's ``modality_root_dir`` (already
repo-specific); this script writes <shadow_root>/<subset>/meta/modality.json.

All 7 subsets share the same 20-D eef.6d layout (state/action both map
observation.state.eef.6d / action.eef.6d), so one template is reused.

Idempotent: existing files are left untouched unless --force is given.
"""
import argparse
import json
from pathlib import Path

SUBSETS = [
    "AgileXPP_PnP_Block_HiL_Dual_Ori_Pos",
    "AgileXPP_PnP_Block_HiL_Dual_Ori_Pos_V2",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V2",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V3",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V4",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V5",
]

# block_260715_340Nsft: 4 pure-Norm SFT subsets (same 20-D eef.6d layout).
SUBSETS_260715_340NSFT = [
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_PosV0",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V3",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_v2",
]

# block_260706_masked_still: 5 pure-Norm SFT subsets (same 20-D eef.6d layout).
SUBSETS_260706_500OSFT = [
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V2",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V3",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V4",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_V5",
]

# block_260716_500Nsft: 6 pure-Norm SFT subsets (same 20-D eef.6d layout) =
# block_260715_340Nsft's 4 New-Norm subsets + two extra real dirs (V4, V5).
SUBSETS_260716_500NSFT = [
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_PosV0",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V3",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V4",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V5",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_v2",
]

# block_260723_500Nsft_50HIL_MR0: block_260716_500Nsft's 6 pure-Norm SFT subsets +
# one MR0 HIL real dir (AgileXPP_PnP_Block_MR0HiL_New_Dual_Ori_Pos_V1). Same 20-D eef.6d
# layout + `collector` static-frame column, so collector_full_chunk_only masking applies.
SUBSETS_260723_500NSFT_50HIL_MR0 = [
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_PosV0",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V3",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V4",
    "AgileXPP_PnP_Block_New_Norm_Dual_Ori_Pos_V5",
    "AgileXPP_PnP_Block_Norm_Dual_Ori_Pos_v2",
    "AgileXPP_PnP_Block_MR0HiL_New_Dual_Ori_Pos_V1",
]


def _state_action_block(original_key: str) -> dict:
    """20-D eef.6d layout: left(xyz3+rot6+grip1) + right(xyz3+rot6+grip1)."""
    return {
        "left_xyz": {"start": 0, "end": 3, "absolute": True, "dtype": "float32",
                     "original_key": original_key, "rotation_type": "rotation_6d"},
        "left_rot6d": {"start": 3, "end": 9, "absolute": True, "dtype": "float32",
                       "original_key": original_key, "rotation_type": "rotation_6d"},
        "left_gripper": {"start": 9, "end": 10, "absolute": True, "dtype": "float32",
                         "original_key": original_key},
        "right_xyz": {"start": 10, "end": 13, "absolute": True, "dtype": "float32",
                      "original_key": original_key},
        "right_rot6d": {"start": 13, "end": 19, "absolute": True, "dtype": "float32",
                        "original_key": original_key, "rotation_type": "rotation_6d"},
        "right_gripper": {"start": 19, "end": 20, "absolute": True, "dtype": "float32",
                          "original_key": original_key},
    }


MODALITY_TEMPLATE = {
    "state": _state_action_block("observation.state.eef.6d"),
    "action": _state_action_block("action.eef.6d"),
    "video": {
        "primary_image": {"original_key": "observation.images.cam_high"},
        "left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shadow_root", required=True,
        help="= the yaml's modality_root_dir. Files go to <shadow_root>/<subset>/meta/modality.json.",
    )
    parser.add_argument(
        "--subset_group",
        default="block_260707_hil",
        choices=[
            "block_260707_hil",
            "block_260715_340Nsft",
            "block_260706_500Osft",
            "block_260716_500Nsft",
            "block_260723_500Nsft_50HIL_MR0",
        ],
        help="named fallback subset list; ignored when --subsets is supplied.",
    )
    parser.add_argument("--subsets", nargs="+", help="explicit current-training subset names, e.g. hil70 hil100 il0615.")
    parser.add_argument("--force", action="store_true", help="overwrite existing modality.json")
    args = parser.parse_args()

    if args.subsets:
        subsets = args.subsets
    elif args.subset_group == "block_260707_hil":
        subsets = SUBSETS
    elif args.subset_group == "block_260715_340Nsft":
        subsets = SUBSETS_260715_340NSFT
    elif args.subset_group == "block_260716_500Nsft":
        subsets = SUBSETS_260716_500NSFT
    elif args.subset_group == "block_260723_500Nsft_50HIL_MR0":
        subsets = SUBSETS_260723_500NSFT_50HIL_MR0
    else:
        subsets = SUBSETS_260706_500OSFT

    written, skipped = 0, 0
    for subset in subsets:
        out = Path(args.shadow_root) / subset / "meta" / "modality.json"
        if out.exists() and not args.force:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(MODALITY_TEMPLATE, f, indent=2)
        written += 1
        print(f"  wrote {out}")

    print(f"modality.json shadow: {written} written, {skipped} already present "
          f"(root={args.shadow_root})")


if __name__ == "__main__":
    main()
