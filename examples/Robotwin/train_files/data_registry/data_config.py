"""RobotWin benchmark — data config, embodiment tags, and mixtures."""

from pathlib import Path

from mr0.dataloader.gr00t_lerobot.datasets import ModalityConfig
from mr0.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from mr0.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from mr0.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin, action_indices=16)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 (action_indices=50)
# ---------------------------------------------------------------------------
class AgilexData50Config(AgilexDataConfig):
    action_indices = list(range(50))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — ARX X5
# ---------------------------------------------------------------------------
class ArxX5DataConfig:
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex N (action_indices=N), reuses Data50 transform (binary_threshold=0.49).
# Used for action-chunk ablations: dataloader returns exactly N future action steps,
# so model's actions[:, -(N): ] aligns with actions[:, :N] = the immediately-next N steps.
# ---------------------------------------------------------------------------
class AgilexData8Config(AgilexData50Config):
    action_indices = list(range(8))


class AgilexData16Config(AgilexData50Config):
    action_indices = list(range(16))


class AgilexData32Config(AgilexData50Config):
    action_indices = list(range(32))


class AgilexData100Config(AgilexData50Config):
    action_indices = list(range(100))


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
    "robotwin8": AgilexData8Config(),
    "robotwin16": AgilexData16Config(),
    "robotwin32": AgilexData32Config(),
    "robotwin50": AgilexData50Config(),
    "robotwin100": AgilexData100Config(),
    "arx_x5": ArxX5DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Uses NEW_EMBODIMENT fallback (not in base embodiment_tags.py)
}

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "robotwin_all": [
        ("Clean/adjust_bottle", 1.0, "robotwin"), ("Randomized/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"), ("Randomized/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"), ("Randomized/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"), ("Randomized/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"), ("Randomized/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"), ("Randomized/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"), ("Randomized/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"), ("Randomized/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"), ("Randomized/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"), ("Randomized/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"), ("Randomized/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"), ("Randomized/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"), ("Randomized/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"), ("Randomized/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"), ("Randomized/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"), ("Randomized/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"), ("Randomized/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"), ("Randomized/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"), ("Randomized/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"), ("Randomized/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"), ("Randomized/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"), ("Randomized/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"), ("Randomized/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"), ("Randomized/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"), ("Randomized/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"), ("Randomized/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"), ("Randomized/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"), ("Randomized/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"), ("Randomized/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"), ("Randomized/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"), ("Randomized/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"), ("Randomized/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"), ("Randomized/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"), ("Randomized/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"), ("Randomized/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"), ("Randomized/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"), ("Randomized/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"), ("Randomized/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"), ("Randomized/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"), ("Randomized/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"), ("Randomized/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"), ("Randomized/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"), ("Randomized/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all_50": [
        ("Clean/adjust_bottle", 1.0, "robotwin50"), ("Randomized/adjust_bottle", 1.0, "robotwin50"),
        ("Clean/beat_block_hammer", 1.0, "robotwin50"), ("Randomized/beat_block_hammer", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin50"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"), ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"), ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/click_bell", 1.0, "robotwin50"), ("Randomized/click_bell", 1.0, "robotwin50"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin50"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Clean/grab_roller", 1.0, "robotwin50"), ("Randomized/grab_roller", 1.0, "robotwin50"),
        ("Clean/handover_block", 1.0, "robotwin50"), ("Randomized/handover_block", 1.0, "robotwin50"),
        ("Clean/handover_mic", 1.0, "robotwin50"), ("Randomized/handover_mic", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"), ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Clean/lift_pot", 1.0, "robotwin50"), ("Randomized/lift_pot", 1.0, "robotwin50"),
        ("Clean/move_can_pot", 1.0, "robotwin50"), ("Randomized/move_can_pot", 1.0, "robotwin50"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin50"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Clean/move_playingcard_away", 1.0, "robotwin50"), ("Randomized/move_playingcard_away", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"), ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/open_laptop", 1.0, "robotwin50"), ("Randomized/open_laptop", 1.0, "robotwin50"),
        ("Clean/open_microwave", 1.0, "robotwin50"), ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin50"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin50"), ("Randomized/pick_dual_bottles", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"), ("Randomized/place_a2b_left", 1.0, "robotwin50"),
        ("Clean/place_a2b_right", 1.0, "robotwin50"), ("Randomized/place_a2b_right", 1.0, "robotwin50"),
        ("Clean/place_bread_basket", 1.0, "robotwin50"), ("Randomized/place_bread_basket", 1.0, "robotwin50"),
        ("Clean/place_bread_skillet", 1.0, "robotwin50"), ("Randomized/place_bread_skillet", 1.0, "robotwin50"),
        ("Clean/place_burger_fries", 1.0, "robotwin50"), ("Randomized/place_burger_fries", 1.0, "robotwin50"),
        ("Clean/place_can_basket", 1.0, "robotwin50"), ("Randomized/place_can_basket", 1.0, "robotwin50"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin50"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Clean/place_container_plate", 1.0, "robotwin50"), ("Randomized/place_container_plate", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"), ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_empty_cup", 1.0, "robotwin50"), ("Randomized/place_empty_cup", 1.0, "robotwin50"),
        ("Clean/place_fan", 1.0, "robotwin50"), ("Randomized/place_fan", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"), ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/place_object_basket", 1.0, "robotwin50"), ("Randomized/place_object_basket", 1.0, "robotwin50"),
        ("Clean/place_object_scale", 1.0, "robotwin50"), ("Randomized/place_object_scale", 1.0, "robotwin50"),
        ("Clean/place_object_stand", 1.0, "robotwin50"), ("Randomized/place_object_stand", 1.0, "robotwin50"),
        ("Clean/place_phone_stand", 1.0, "robotwin50"), ("Randomized/place_phone_stand", 1.0, "robotwin50"),
        ("Clean/place_shoe", 1.0, "robotwin50"), ("Randomized/place_shoe", 1.0, "robotwin50"),
        ("Clean/press_stapler", 1.0, "robotwin50"), ("Randomized/press_stapler", 1.0, "robotwin50"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin50"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"), ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/rotate_qrcode", 1.0, "robotwin50"), ("Randomized/rotate_qrcode", 1.0, "robotwin50"),
        ("Clean/scan_object", 1.0, "robotwin50"), ("Randomized/scan_object", 1.0, "robotwin50"),
        ("Clean/shake_bottle", 1.0, "robotwin50"), ("Randomized/shake_bottle", 1.0, "robotwin50"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin50"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"), ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/stack_blocks_two", 1.0, "robotwin50"), ("Randomized/stack_blocks_two", 1.0, "robotwin50"),
        ("Clean/stack_bowls_three", 1.0, "robotwin50"), ("Randomized/stack_bowls_three", 1.0, "robotwin50"),
        ("Clean/stack_bowls_two", 1.0, "robotwin50"), ("Randomized/stack_bowls_two", 1.0, "robotwin50"),
        ("Clean/stamp_seal", 1.0, "robotwin50"), ("Randomized/stamp_seal", 1.0, "robotwin50"),
        ("Clean/turn_switch", 1.0, "robotwin50"), ("Randomized/turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_clean_50": [
        ("adjust_bottle", 1.0, "robotwin50"),
        ("beat_block_hammer", 1.0, "robotwin50"),
        ("blocks_ranking_rgb", 1.0, "robotwin50"),
        ("blocks_ranking_size", 1.0, "robotwin50"),
        ("click_alarmclock", 1.0, "robotwin50"),
        ("click_bell", 1.0, "robotwin50"),
        ("dump_bin_bigbin", 1.0, "robotwin50"),
        ("grab_roller", 1.0, "robotwin50"),
        ("handover_block", 1.0, "robotwin50"),
        ("handover_mic", 1.0, "robotwin50"),
        ("hanging_mug", 1.0, "robotwin50"),
        ("lift_pot", 1.0, "robotwin50"),
        ("move_can_pot", 1.0, "robotwin50"),
        ("move_pillbottle_pad", 1.0, "robotwin50"),
        ("move_playingcard_away", 1.0, "robotwin50"),
        ("move_stapler_pad", 1.0, "robotwin50"),
        ("open_laptop", 1.0, "robotwin50"),
        ("open_microwave", 1.0, "robotwin50"),
        ("pick_diverse_bottles", 1.0, "robotwin50"),
        ("pick_dual_bottles", 1.0, "robotwin50"),
        ("place_a2b_left", 1.0, "robotwin50"),
        ("place_a2b_right", 1.0, "robotwin50"),
        ("place_bread_basket", 1.0, "robotwin50"),
        ("place_bread_skillet", 1.0, "robotwin50"),
        ("place_burger_fries", 1.0, "robotwin50"),
        ("place_can_basket", 1.0, "robotwin50"),
        ("place_cans_plasticbox", 1.0, "robotwin50"),
        ("place_container_plate", 1.0, "robotwin50"),
        ("place_dual_shoes", 1.0, "robotwin50"),
        ("place_empty_cup", 1.0, "robotwin50"),
        ("place_fan", 1.0, "robotwin50"),
        ("place_mouse_pad", 1.0, "robotwin50"),
        ("place_object_basket", 1.0, "robotwin50"),
        ("place_object_scale", 1.0, "robotwin50"),
        ("place_object_stand", 1.0, "robotwin50"),
        ("place_phone_stand", 1.0, "robotwin50"),
        ("place_shoe", 1.0, "robotwin50"),
        ("press_stapler", 1.0, "robotwin50"),
        ("put_bottles_dustbin", 1.0, "robotwin50"),
        ("put_object_cabinet", 1.0, "robotwin50"),
        ("rotate_qrcode", 1.0, "robotwin50"),
        ("scan_object", 1.0, "robotwin50"),
        ("shake_bottle", 1.0, "robotwin50"),
        ("shake_bottle_horizontally", 1.0, "robotwin50"),
        ("stack_blocks_three", 1.0, "robotwin50"),
        ("stack_blocks_two", 1.0, "robotwin50"),
        ("stack_bowls_three", 1.0, "robotwin50"),
        ("stack_bowls_two", 1.0, "robotwin50"),
        ("stamp_seal", 1.0, "robotwin50"),
        ("turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin_hard10_50": [
        ("Clean/open_microwave", 1.0, "robotwin50"), ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"), ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"), ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"), ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"), ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"), ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"), ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"), ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"), ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"), ("Randomized/place_a2b_left", 1.0, "robotwin50"),
    ],
    "robotwin_task1": [("adjust_bottle", 1.0, "robotwin")],
    "robotwin_task2": [("place_a2b_left", 1.0, "robotwin"), ("place_a2b_right", 1.0, "robotwin")],
    "arx_x5": [("arx_x5", 1.0, "arx_x5")],
}

# Auto-derive action-chunk variants of robotwin_all_50 by remapping robot_type.
# Used by chunk{8,16,32} ablations so the dataloader returns N action steps and
# model's [:, -N:] slice aligns with the next N steps (not steps t+42..t+49).
for _N in (8, 16, 32, 100):
    DATASET_NAMED_MIXTURES[f"robotwin_all_{_N}"] = [
        (path, w, f"robotwin{_N}") for (path, w, _rt) in DATASET_NAMED_MIXTURES["robotwin_all_50"]
    ]


# Real-ratio version of robotwin_all_50: Clean and Randomized weighted to match
# their actual trajectory counts (50 vs 500 per task → 1:10). Default mixture
# was 1:1 (over-sampling Clean ~10x). Combined with balance_dataset_weights=False
# (default), this samples Clean:Randomized at 1:10 within each task.
DATASET_NAMED_MIXTURES["robotwin_all_50_realratio"] = [
    (path, 10.0 if path.startswith("Randomized/") else 1.0, rt)
    for (path, _w, rt) in DATASET_NAMED_MIXTURES["robotwin_all_50"]
]

# ---------------------------------------------------------------------------
# RoboDojo LeRobot v2.1. Set datasets.vla_data.data_root_dir to the parent directory.
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES["robodojo_35"] = [
    ("RoboDojo_lerobot_v21_video", 1.0, "robotwin50"),
]
DATASET_NAMED_MIXTURES["robodojo_35_chunk100"] = [
    ("RoboDojo_lerobot_v21_video", 1.0, "robotwin100"),
]
