"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

import json
import os
import re
from typing import Tuple

import numpy as np
import torch
from accelerate.logging import get_logger

logger = get_logger(__name__)


class TrackerMixin:
    """Mixin providing multi-backend metric logging (wandb, tensorboard)."""

    def _init_trackers(self, group="train"):
        self._trackers_active = {"wandb": False, "tensorboard": False}
        trackers = getattr(self.config, "trackers", ["jsonl"])
        if isinstance(trackers, str):
            trackers = [trackers]

        if not (hasattr(self, "accelerator") and self.accelerator.is_main_process):
            return

        if "wandb" in trackers:
            import wandb

            os.makedirs(os.path.join(self.config.output_dir, "wandb"), exist_ok=True)
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=group,
            )
            self._trackers_active["wandb"] = True

        if "tensorboard" in trackers:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(self.config.output_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_dir)
            self._trackers_active["tensorboard"] = True

    def _log_to_trackers(self, metrics, step):
        if not (hasattr(self, "accelerator") and self.accelerator.is_main_process):
            return

        if self._trackers_active.get("wandb"):
            import wandb

            wandb.log(metrics, step=step)

        if self._trackers_active.get("tensorboard"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, step)
            self._tb_writer.flush()

    def _finish_trackers(self):
        if not (hasattr(self, "accelerator") and self.accelerator.is_main_process):
            return

        if self._trackers_active.get("wandb"):
            import wandb

            wandb.finish()

        if self._trackers_active.get("tensorboard"):
            self._tb_writer.close()


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    # Normalize: bool ``true`` from YAML freezes the VLM backbone;
    # bool ``false`` means no freeze; list/tuple values are joined into a comma-separated string.
    if isinstance(freeze_modules, bool):
        freeze_modules = "qwen_vl_interface" if freeze_modules else ""
    if isinstance(freeze_modules, (list, tuple)):
        freeze_modules = ",".join(str(m) for m in freeze_modules)
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters
            params = [p for p in module.parameters() if id(p) not in frozen_params]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            logger.warning(f"⚠️ module path `{module_name}` not found in vla, its params will use base_lr")

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from PIL import Image
from torchvision.ops import box_iou


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


class TrainerUtils(TrackerMixin):
    @staticmethod
    def _raw_lr_scheduler(lr_scheduler):
        return getattr(lr_scheduler, "scheduler", lr_scheduler)

    def _configure_lr_scheduler_step(self):
        """Keep scheduler steps aligned with this repo's global optimizer step count."""
        if hasattr(self.lr_scheduler, "split_batches"):
            self.lr_scheduler.split_batches = True

    def _ensure_scheduler_state_for_resume(self, ckpt_dir):
        """Create or repair scheduler.bin before accelerator.load_state()."""
        sched_file = os.path.join(ckpt_dir, "scheduler.bin")
        expected_step = int(self.completed_steps)
        repair_reason = None

        if os.path.exists(sched_file):
            try:
                state = torch.load(sched_file, map_location="cpu")
                last_epoch = state.get("last_epoch")
                if last_epoch == expected_step:
                    return
                repair_reason = f"scheduler.bin last_epoch={last_epoch}, expected={expected_step}"
            except Exception as exc:
                repair_reason = f"could not read scheduler.bin ({exc})"
        else:
            repair_reason = "scheduler.bin missing"

        if self.accelerator.is_main_process:
            logger.warning(f"{repair_reason} in {ckpt_dir}; synthesizing scheduler state")
            raw_sched = self._raw_lr_scheduler(self.lr_scheduler)
            for _ in range(expected_step):
                raw_sched.step()
            torch.save(raw_sched.state_dict(), sched_file)

        self.accelerator.wait_for_everyone()

    def _sync_optimizer_lr_to_scheduler(self):
        raw_sched = self._raw_lr_scheduler(self.lr_scheduler)
        optimizer = getattr(raw_sched, "optimizer", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups is None or not hasattr(raw_sched, "get_last_lr"):
            return

        last_lrs = raw_sched.get_last_lr()
        if len(param_groups) != len(last_lrs):
            logger.warning(
                f"Cannot sync optimizer lr: {len(param_groups)} param groups vs {len(last_lrs)} scheduler lrs"
            )
            return

        for group, lr in zip(param_groups, last_lrs):
            group["lr"] = lr

    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#" * 30)
        print(freeze_modules)
        # Normalize freeze_modules: accept str, list, or bool.
        # bool True freezes the VLM backbone; bool False means no freeze.
        if isinstance(freeze_modules, bool):
            freeze_modules = "qwen_vl_interface" if freeze_modules else ""
        if isinstance(freeze_modules, (list, tuple)):
            freeze_modules = ",".join(str(m) for m in freeze_modules)
        if freeze_modules and isinstance(freeze_modules, str):
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if dist.get_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(
        model,
        checkpoint_path=None,
        reload_modules=None,
        audit_path=None,
        allowed_shape_mismatch_keys=None,
    ):
        """Load pretrained weights and emit an auditable initialization report.

        Partial initialization (``reload_modules``) is strict for every named
        module. Full-model initialization is strict by default: missing,
        unexpected, and shape-mismatched keys fail before any weights are
        loaded. Shape adaptation is only allowed for explicitly listed keys.
        """
        if not checkpoint_path:
            return model

        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}") from e

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"❌ checkpoint must contain a state dict, got {type(checkpoint).__name__}")

        model_state = model.state_dict()
        allowed_mismatches = set(allowed_shape_mismatch_keys or [])

        # ── Resolve tied embeddings ──────────────────────────────────────────
        # safetensors deduplicates tied weights (e.g. lm_head.weight shares data
        # with embed_tokens.weight).  When loading from a safetensors checkpoint,
        # lm_head.weight may be absent even though embed_tokens.weight is present.
        # We detect this and backfill the missing key from its tied partner.
        tied_embedding_pairs = [
            ("qwen_vl_interface.model.lm_head.weight", "qwen_vl_interface.model.model.language_model.embed_tokens.weight"),
        ]
        tied_recovered = []
        for missing_key, partner_key in tied_embedding_pairs:
            if missing_key not in checkpoint and partner_key in checkpoint and missing_key in model_state:
                if checkpoint[partner_key].shape == model_state[missing_key].shape:
                    checkpoint[missing_key] = checkpoint[partner_key].clone()
                    tied_recovered.append(missing_key)
                    if is_main_process:
                        print(f"🔗 tied embedding recovered: {missing_key} <- {partner_key}")

        report = {
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "mode": "partial" if reload_modules else "full",
            "reload_modules": [],
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
            "allowed_shape_mismatch_keys": sorted(allowed_mismatches),
            "tied_embedding_recovered": tied_recovered,
            "coverage": {},
        }

        if reload_modules:
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            if not module_paths:
                raise ValueError("reload_modules was provided but contains no module paths")
            for path in module_paths:
                module = model
                try:
                    for module_name in path.split("."):
                        module = getattr(module, module_name)
                except AttributeError as e:
                    raise RuntimeError(f"❌ cannot find module path: {path}") from e

                prefix = path + "."
                sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                if not sub_state_dict:
                    raise RuntimeError(f"❌ parameters not found in checkpoint for module '{path}'")
                try:
                    module.load_state_dict(sub_state_dict, strict=True)
                except RuntimeError as e:
                    raise RuntimeError(f"❌ strict loading failed for module '{path}': {e}") from e

                report["reload_modules"].append(path)
                report["coverage"][path] = {
                    "loaded_keys": len(sub_state_dict),
                    "model_keys": len(module.state_dict()),
                    "coverage": 1.0,
                }
                if is_main_process:
                    print(f"✅ strictly loaded module '{path}' ({len(sub_state_dict)} keys)")
            if is_main_process:
                print("ℹ️ modules not listed in reload_modules retain their initialization")
        else:
            checkpoint_keys = set(checkpoint)
            model_keys = set(model_state)
            missing_keys = sorted(model_keys - checkpoint_keys)
            unexpected_keys = sorted(checkpoint_keys - model_keys)
            shape_mismatches = [
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint[key].shape),
                    "model_shape": list(model_state[key].shape),
                    "allowed": key in allowed_mismatches,
                }
                for key in sorted(model_keys & checkpoint_keys)
                if checkpoint[key].shape != model_state[key].shape
            ]
            report["missing_keys"] = missing_keys
            report["unexpected_keys"] = unexpected_keys
            report["shape_mismatches"] = shape_mismatches

            # Match the reference full-load behavior: allowlisted shape mismatches are
            # omitted from the checkpoint and retain the model constructor's random init.
            report["shape_adaptations"] = [
                {
                    "key": item["key"],
                    "method": "keep_random_init",
                    "checkpoint_shape": item["checkpoint_shape"],
                    "model_shape": item["model_shape"],
                }
                for item in shape_mismatches
                if item["allowed"]
            ]

            for prefix in ("qwen_vl_interface", "action_model"):
                prefix_with_dot = prefix + "."
                group_keys = {key for key in model_keys if key.startswith(prefix_with_dot)}
                loaded_keys = {
                    key
                    for key in group_keys & checkpoint_keys
                    if checkpoint[key].shape == model_state[key].shape
                }
                report["coverage"][prefix] = {
                    "loaded_keys": len(loaded_keys),
                    "model_keys": len(group_keys),
                    "coverage": len(loaded_keys) / len(group_keys) if group_keys else 1.0,
                }

            disallowed_mismatches = [item for item in shape_mismatches if not item["allowed"]]
            unapproved_missing = [key for key in missing_keys if key not in allowed_mismatches]
            unknown_allowlist = sorted(allowed_mismatches - {item["key"] for item in shape_mismatches})
            if unexpected_keys or unapproved_missing or disallowed_mismatches or unknown_allowlist:
                report["unknown_allowed_shape_mismatch_keys"] = unknown_allowlist
                model._checkpoint_load_audit = report
                TrainerUtils._write_checkpoint_load_audit(report, audit_path, is_main_process)
                raise RuntimeError(
                    "❌ full checkpoint audit failed: "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
                    f"mismatch={len(shape_mismatches)}, unknown_allowlist={len(unknown_allowlist)}"
                )

            loadable_checkpoint = {
                key: value for key, value in checkpoint.items() if key not in allowed_mismatches
            }
            incompatible = model.load_state_dict(loadable_checkpoint, strict=False)
            residual_missing = sorted(key for key in incompatible.missing_keys if key not in allowed_mismatches)
            if residual_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "❌ full checkpoint load produced unaudited incompatibilities: "
                    f"missing={residual_missing}, unexpected={sorted(incompatible.unexpected_keys)}"
                )
            if is_main_process:
                print(
                    "✅ audited full-model load: "
                    f"VLM coverage={report['coverage']['qwen_vl_interface']['coverage']:.2%}, "
                    f"action coverage={report['coverage']['action_model']['coverage']:.2%}"
                )

        model._checkpoint_load_audit = report
        TrainerUtils._write_checkpoint_load_audit(report, audit_path, is_main_process)
        return model

    @staticmethod
    def _write_checkpoint_load_audit(report, audit_path, is_main_process):
        if not audit_path or not is_main_process:
            return
        audit_path = os.fspath(audit_path)
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"📋 checkpoint load audit saved to {audit_path}")

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number.

        Supports both full-state directories (steps_N/) created by
        accelerator.save_state() and legacy model-only files
        (steps_N_pytorch_model.pt / steps_N_model.safetensors).
        Prefers directory-based checkpoints when both exist for the same step.
        """
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        candidates = {}

        for entry in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, entry)
            m_dir = re.match(r"steps_(\d+)$", entry)
            if m_dir and os.path.isdir(full_path):
                step = int(m_dir.group(1))
                candidates[step] = full_path
                continue
            m_file = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", entry)
            if m_file and os.path.isfile(full_path):
                step = int(m_file.group(1))
                if step not in candidates:
                    candidates[step] = full_path

        if not candidates:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        latest_step = max(candidates)
        latest_path = candidates[latest_step]
        self.accelerator.print(f"Latest checkpoint found: {latest_path} (step {latest_step})")
        return latest_path, latest_step


import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")
