from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


class _FakeTensor:
    def __init__(self, value, dtype=None, device=None):
        self.array = np.asarray(value, dtype=dtype)
        self.device = device
        self.dtype = self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def size(self, dim=None):
        return self.array.shape if dim is None else self.array.shape[dim]

    def view(self, *shape):
        return _FakeTensor(self.array.reshape(shape))

    def reshape(self, *shape):
        return _FakeTensor(self.array.reshape(shape))

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self.array, axis=dim))

    def expand(self, *shape):
        target = []
        for i, dim in enumerate(shape):
            target.append(self.array.shape[i] if dim == -1 else dim)
        return _FakeTensor(np.broadcast_to(self.array, tuple(target)))

    def to(self, device):
        self.device = device
        return self

    def long(self):
        return _FakeTensor(self.array.astype(np.int64), device=self.device)

    def cumsum(self, dim=-1):
        return _FakeTensor(np.cumsum(self.array, axis=dim), device=self.device)

    def masked_fill_(self, mask, value):
        self.array[np.asarray(mask)] = value
        return self

    def max(self, axis=None, keepdim=False):
        return _FakeTensor(np.max(self.array, axis=axis, keepdims=keepdim), device=self.device)

    def squeeze(self, axis=None):
        return _FakeTensor(np.squeeze(self.array, axis=axis), device=self.device)

    def flatten(self):
        return _FakeTensor(self.array.flatten(), device=self.device)

    def tolist(self):
        return self.array.tolist()

    def item(self):
        return self.array.item()

    def copy(self):
        return _FakeTensor(self.array.copy(), device=self.device)

    def __array__(self, dtype=None):
        return self.array.astype(dtype) if dtype is not None else self.array

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        for item in self.array:
            yield _FakeTensor(item, device=self.device)

    def __getitem__(self, item):
        return _FakeTensor(self.array[item], device=self.device)

    def __setitem__(self, item, value):
        self.array[item] = np.asarray(value)

    def __eq__(self, other):
        return _FakeTensor(self.array == np.asarray(other), device=self.device)

    def __add__(self, other):
        return _FakeTensor(self.array + np.asarray(other), device=self.device)

    def __radd__(self, other):
        return _FakeTensor(np.asarray(other) + self.array, device=self.device)

    def __sub__(self, other):
        return _FakeTensor(self.array - np.asarray(other), device=self.device)

    def __rsub__(self, other):
        return _FakeTensor(np.asarray(other) - self.array, device=self.device)

    def __mul__(self, other):
        return _FakeTensor(self.array * np.asarray(other), device=self.device)


def _install_torch_stub() -> None:
    try:
        import torch  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _FakeTensor
    torch_mod.LongTensor = _FakeTensor
    torch_mod.long = np.int64
    torch_mod.int32 = np.int32
    torch_mod.float32 = np.float32

    def tensor(value, dtype=None, device=None):
        return _FakeTensor(value, dtype=dtype, device=device)

    def arange(*args, dtype=None, device=None):
        return _FakeTensor(np.arange(*args, dtype=dtype or np.int64), device=device)

    torch_mod.tensor = tensor
    torch_mod.arange = arange
    torch_mod.zeros = lambda shape, dtype=None, device=None: _FakeTensor(
        np.zeros(shape, dtype=dtype or np.float32), device=device
    )
    torch_mod.ones = lambda *shape, dtype=None, device=None: _FakeTensor(
        np.ones(shape, dtype=dtype or np.float32), device=device
    )
    torch_mod.ones_like = lambda value: _FakeTensor(np.ones_like(np.asarray(value)))
    torch_mod.stack = lambda values, dim=0: _FakeTensor(np.stack([np.asarray(v) for v in values], axis=dim))
    torch_mod.cat = lambda values, dim=0: _FakeTensor(np.concatenate([np.asarray(v) for v in values], axis=dim))
    torch_mod.cumsum = lambda value, dim=0, dtype=None: _FakeTensor(
        np.cumsum(np.asarray(value), axis=dim, dtype=dtype)
    )
    torch_mod.argwhere = lambda value: _FakeTensor(np.argwhere(np.asarray(value)))

    nn_mod = types.ModuleType("torch.nn")
    functional_mod = types.ModuleType("torch.nn.functional")
    functional_mod.pad = lambda value, pad, mode="constant", value_constant=0: value
    rnn_mod = types.ModuleType("torch.nn.utils.rnn")
    rnn_mod.pad_sequence = lambda values, batch_first=False, padding_value=0: _FakeTensor(
        np.stack([np.asarray(v) for v in values], axis=0 if batch_first else 1)
    )
    utils_nn_mod = types.ModuleType("torch.nn.utils")
    utils_nn_mod.rnn = rnn_mod
    nn_mod.functional = functional_mod
    nn_mod.utils = utils_nn_mod
    torch_mod.nn = nn_mod

    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.is_initialized = lambda: False
    dist_mod.get_rank = lambda: 0
    dist_mod.get_world_size = lambda: 1
    dist_mod.barrier = lambda: None
    torch_mod.distributed = dist_mod

    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    data_mod.DataLoader = object
    data_mod.get_worker_info = lambda: None
    utils_mod = types.ModuleType("torch.utils")
    utils_mod.data = data_mod
    torch_mod.utils = utils_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod
    sys.modules["torch.nn.utils"] = utils_nn_mod
    sys.modules["torch.nn.utils.rnn"] = rnn_mod
    sys.modules["torch.distributed"] = dist_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod


def _install_optional_dependency_stubs() -> None:
    if "numpydantic" not in sys.modules:
        try:
            import numpydantic  # noqa: F401
        except ModuleNotFoundError:
            numpydantic_mod = types.ModuleType("numpydantic")
            numpydantic_mod.NDArray = Any
            sys.modules["numpydantic"] = numpydantic_mod

    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError:
            transformers_mod = types.ModuleType("transformers")
            transformers_mod.PreTrainedTokenizer = object
            transformers_mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
            transformers_mod.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
            sys.modules["transformers"] = transformers_mod

    if "decord" not in sys.modules:
        try:
            import decord  # noqa: F401
        except ModuleNotFoundError:
            decord_mod = types.ModuleType("decord")
            decord_mod.VideoReader = object
            sys.modules["decord"] = decord_mod

    if "omegaconf" not in sys.modules:
        try:
            import omegaconf  # noqa: F401
        except ModuleNotFoundError:
            omegaconf_mod = types.ModuleType("omegaconf")
            omegaconf_mod.OmegaConf = types.SimpleNamespace(to_container=lambda cfg, resolve=True: cfg, load=lambda p: None)
            sys.modules["omegaconf"] = omegaconf_mod


def _install_package_stubs() -> None:
    package_paths = {
        "mr0": REPO_ROOT / "mr0",
        "mr0.dataloader": REPO_ROOT / "mr0" / "dataloader",
        "mr0.dataloader.gr00t_lerobot": REPO_ROOT / "mr0" / "dataloader" / "gr00t_lerobot",
        "mr0.dataloader.qwenvl_llavajson": REPO_ROOT / "mr0" / "dataloader" / "qwenvl_llavajson",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    transform_mod = types.ModuleType("mr0.dataloader.gr00t_lerobot.transform")

    class ComposedModalityTransform:
        def __init__(self, transforms=None):
            self.transforms = transforms or []
            self.metadata = None

        def __call__(self, data):
            return data

        def set_metadata(self, metadata):
            self.metadata = metadata

    transform_mod.ComposedModalityTransform = ComposedModalityTransform
    sys.modules["mr0.dataloader.gr00t_lerobot.transform"] = transform_mod

    video_mod = types.ModuleType("mr0.dataloader.gr00t_lerobot.video")
    video_mod.get_all_frames = lambda *args, **kwargs: None
    video_mod.get_frames_by_timestamps = lambda *args, **kwargs: None
    sys.modules["mr0.dataloader.gr00t_lerobot.video"] = video_mod


def _clear_mr0_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "mr0" or module_name.startswith("mr0."):
            sys.modules.pop(module_name, None)


def _load_dataloader_modules():
    try:
        gr00t = importlib.import_module("mr0.dataloader.gr00t_lerobot.datasets")
        vlm = importlib.import_module("mr0.dataloader.vlm_datasets")
        return gr00t, vlm
    except ModuleNotFoundError:
        # Lightweight CI images may not include torch/decord/video deps. Fall back
        # to stubs only for those dependency gaps; real envs exercise real imports.
        _clear_mr0_modules()

    _install_torch_stub()
    _install_optional_dependency_stubs()
    _install_package_stubs()
    gr00t = importlib.import_module("mr0.dataloader.gr00t_lerobot.datasets")
    vlm = importlib.import_module("mr0.dataloader.vlm_datasets")
    return gr00t, vlm


GD, VLM = _load_dataloader_modules()


class _TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"
    model_max_length = 1024
    max_len_single_sentence = 1024
    chat_template = None

    def __init__(self, max_len=1024):
        self.model_max_length = max_len
        self.max_len_single_sentence = max_len

    def apply_chat_template(self, messages):
        text = "\n".join(f"{msg['role']}:{msg['content']}" for msg in messages)
        return [1, 2, 3] + [(ord(ch) % 97) + 4 for ch in text]


class _DummyDataset:
    def __init__(self, tag, metadata=None, num_steps=0):
        self.tag = tag
        self.metadata = metadata
        self.dataset_name = f"dummy-{tag}"
        self.all_steps = [(0, i) for i in range(num_steps)]
        self.received_metadata = None

    def __len__(self):
        # Mirrors LeRobotSingleDataset.__len__ for the non-lazy sampling path
        # (self._lazy_step_sampling is False), which returns len(self.all_steps).
        return len(self.all_steps)

    def set_transforms_metadata(self, metadata):
        self.received_metadata = metadata


def _make_modality_metadata():
    return GD.LeRobotModalityMetadata.model_validate(
        {
            "state": {"x": {"start": 0, "end": 1, "original_key": "observation.state"}},
            "action": {"x": {"start": 0, "end": 1, "original_key": "action"}},
            "video": {"cam": {"original_key": "observation.images.cam"}},
            "annotation": {"task": {"original_key": "tasks"}},
        }
    )


def _stat(mean):
    value = np.asarray([mean], dtype=np.float32)
    return {
        "min": value - 1.0,
        "max": value + 1.0,
        "mean": value,
        "std": np.asarray([0.0], dtype=np.float32),
        "q01": value - 0.5,
        "q99": value + 0.5,
    }


def _make_dataset_metadata(tag, state_mean, action_mean):
    return GD.DatasetMetadata.model_validate(
        {
            "statistics": {
                "state": {"state.x": _stat(state_mean)},
                "action": {"action.x": _stat(action_mean)},
            },
            "modalities": {
                "video": {"video.cam": {"resolution": (2, 2), "channels": 3, "fps": 30.0}},
                "state": {
                    "state.x": {
                        "absolute": True,
                        "rotation_type": None,
                        "shape": (1,),
                        "continuous": True,
                    }
                },
                "action": {
                    "action.x": {
                        "absolute": True,
                        "rotation_type": None,
                        "shape": (1,),
                        "continuous": True,
                    }
                },
            },
            "embodiment_tag": tag,
        }
    )


def _make_sequential_mixture(num_steps, mode="train"):
    mix = object.__new__(GD.LeRobotMixtureDataset)
    mix.datasets = [_DummyDataset(GD.EmbodimentTag.FRANKA.value, num_steps=num_steps)]
    mix._dataset_sampling_weights = np.asarray([1.0], dtype=np.float64)
    mix._sequential_step_sampling = True
    mix._step_order = [np.arange(num_steps)]
    mix._step_pos = [0]
    mix._step_cycle = [0]
    mix._worker_shard_ready = False
    mix.mode = mode
    mix.epoch = 0
    mix.seed = 123
    return mix


class _WorkerContext:
    def __init__(self, worker_id, num_workers, rank=0, world_size=1):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size

    def __enter__(self):
        import torch.utils.data as torch_data

        self.torch_data = torch_data
        self.old_get_worker_info = getattr(torch_data, "get_worker_info", None)
        self.old_rank = os.environ.get("RANK")
        self.old_world_size = os.environ.get("WORLD_SIZE")
        self.old_dist_is_initialized = GD.dist.is_initialized
        torch_data.get_worker_info = lambda: types.SimpleNamespace(id=self.worker_id, num_workers=self.num_workers)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        GD.dist.is_initialized = lambda: False

    def __exit__(self, exc_type, exc, tb):
        if self.old_get_worker_info is None:
            delattr(self.torch_data, "get_worker_info")
        else:
            self.torch_data.get_worker_info = self.old_get_worker_info
        if self.old_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = self.old_rank
        if self.old_world_size is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = self.old_world_size
        GD.dist.is_initialized = self.old_dist_is_initialized


class VlmDataloaderCorrectnessTest(unittest.TestCase):
    def test_validate_visual_inputs_rejects_silent_media_tag_mismatches(self):
        dataset = object.__new__(VLM.LazySupervisedDataset)

        image_files, video_files = dataset._validate_visual_inputs(
            {
                "images": ["a.jpg", "b.jpg"],
                "conversations": [{"from": "human", "value": "<image> first <image> second"}],
            }
        )
        self.assertEqual(image_files, ["a.jpg", "b.jpg"])
        self.assertEqual(video_files, [])

        with self.assertRaisesRegex(ValueError, "Image count/tag mismatch"):
            dataset._validate_visual_inputs(
                {"images": ["a.jpg"], "conversations": [{"from": "human", "value": "no visual tag"}]}
            )
        with self.assertRaisesRegex(ValueError, "Video count/tag mismatch"):
            dataset._validate_visual_inputs(
                {"videos": ["a.mp4", "b.mp4"], "conversations": [{"from": "human", "value": "<video>"}]}
            )
        with self.assertRaisesRegex(ValueError, "both images and videos"):
            dataset._validate_visual_inputs(
                {
                    "images": ["a.jpg"],
                    "videos": ["a.mp4"],
                    "conversations": [{"from": "human", "value": "<image>"}],
                }
            )

    def test_preprocess_qwen_visual_requires_one_grid_per_visual_tag(self):
        sources = [
            [
                {"from": "human", "value": "look <image> then <image>"},
                {"from": "gpt", "value": "done"},
            ]
        ]

        with self.assertRaisesRegex(ValueError, "more <image> tags"):
            VLM.preprocess_qwen_2_visual(sources, _TinyTokenizer(), grid_thw=[1], visual_type="image")

        with self.assertRaisesRegex(ValueError, "Found 1 <image> tags but processed 2"):
            VLM.preprocess_qwen_2_visual(
                [[{"from": "human", "value": "look <image>"}, {"from": "gpt", "value": "done"}]],
                _TinyTokenizer(),
                grid_thw=[1, 1],
                visual_type="image",
            )

        output = VLM.preprocess_qwen_2_visual(
            [[{"from": "human", "value": "look <image>"}, {"from": "gpt", "value": "done"}]],
            _TinyTokenizer(),
            grid_thw=[2],
            visual_type="image",
        )
        self.assertEqual(output["input_ids"].shape[0], 1)
        self.assertEqual(output["labels"].shape, output["input_ids"].shape)

    def test_get_item_drops_overlong_samples_instead_of_truncating(self):
        dataset = object.__new__(VLM.LazySupervisedDataset)
        samples = [
            {
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "world"},
                ]
            }
        ]
        # __init__ is bypassed, so mirror the attributes _get_raw_sample/__len__
        # rely on: list_data_dict aliases _eager_samples, there is no jsonl
        # source, and _total_length is eager + jsonl counts.
        dataset._eager_samples = samples
        dataset.list_data_dict = samples
        dataset._jsonl_total_length = 0
        dataset._total_length = len(samples)
        dataset.tokenizer = _TinyTokenizer(max_len=4)

        original_preprocess = VLM.preprocess_qwen_2_visual

        def fake_preprocess(*args, **kwargs):
            return {
                "input_ids": VLM.torch.tensor([[1, 2, 3, 4, 5]], dtype=VLM.torch.long),
                "labels": VLM.torch.tensor([[1, 2, 3, 4, 5]], dtype=VLM.torch.long),
            }

        try:
            VLM.preprocess_qwen_2_visual = fake_preprocess
            with self.assertRaisesRegex(ValueError, "exceeds model limit"):
                dataset._get_item(0)
        finally:
            VLM.preprocess_qwen_2_visual = original_preprocess


class LeRobotDataloaderCorrectnessTest(unittest.TestCase):
    def test_stats_cache_config_changes_with_schema_indices_and_parquet_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            parquet_a = data_dir / "episode_000000.parquet"
            parquet_b = data_dir / "episode_000001.parquet"
            parquet_a.write_bytes(b"first")
            parquet_b.write_bytes(b"second")
            meta = _make_modality_metadata()

            base = GD._build_stats_cache_config(
                action_mode="abs",
                lerobot_version="v2.1",
                action_keys_full=["action.x"],
                state_keys_full=["state.x"],
                action_indices=[0],
                state_indices=[0],
                action_mode_apply_keys=["action.x"],
                action_mode_state_map={"action.x": "state.x"},
                parquet_paths=[parquet_a, parquet_b],
                dataset_path=root,
                lerobot_modality_meta=meta,
            )
            changed_indices = GD._build_stats_cache_config(
                action_mode="abs",
                lerobot_version="v2.1",
                action_keys_full=["action.x"],
                state_keys_full=["state.x"],
                action_indices=None,
                state_indices=[0],
                action_mode_apply_keys=["action.x"],
                action_mode_state_map={"action.x": "state.x"},
                parquet_paths=[parquet_a, parquet_b],
                dataset_path=root,
                lerobot_modality_meta=meta,
            )
            parquet_a.write_bytes(b"first-plus-more-bytes")
            changed_file = GD._build_stats_cache_config(
                action_mode="abs",
                lerobot_version="v2.1",
                action_keys_full=["action.x"],
                state_keys_full=["state.x"],
                action_indices=[0],
                state_indices=[0],
                action_mode_apply_keys=["action.x"],
                action_mode_state_map={"action.x": "state.x"},
                parquet_paths=[parquet_a, parquet_b],
                dataset_path=root,
                lerobot_modality_meta=meta,
            )

        self.assertEqual(base["parquet_files"][0]["path"], "data/episode_000000.parquet")
        self.assertIn("modality_metadata", base)
        self.assertNotEqual(base, changed_indices)
        self.assertNotEqual(base["parquet_files"], changed_file["parquet_files"])

    def test_delta_and_rel_action_modes_fail_fast_without_matching_state(self):
        dataset = object.__new__(GD.LeRobotSingleDataset)
        dataset._action_mode = "delta"
        dataset._action_mode_apply_keys = None
        dataset._action_mode_state_map = {}
        dataset._modality_keys = {"action": ["action.x"], "state": []}

        with self.assertRaisesRegex(ValueError, "Add action_mode_state_map"):
            dataset._apply_action_mode({"action.x": np.zeros((2, 1), dtype=np.float32)})

        dataset._modality_keys = {"action": ["action.x"], "state": ["state.x"]}
        data = {
            "action.x": np.asarray([[2.0, 3.0], [5.0, 7.0], [9.0, 8.0]], dtype=np.float32),
            "state.x": np.asarray([[1.0, 2.0]], dtype=np.float32),
        }
        out = dataset._apply_action_mode(data.copy())
        np.testing.assert_allclose(
            out["action.x"],
            np.asarray([[1.0, 1.0], [3.0, 4.0], [4.0, 1.0]], dtype=np.float32),
        )

        dataset._action_mode = "rel"
        out = dataset._apply_action_mode(data.copy())
        np.testing.assert_allclose(
            out["action.x"],
            np.asarray([[1.0, 1.0], [4.0, 5.0], [8.0, 6.0]], dtype=np.float32),
        )

    def test_pack_sample_accepts_concat_transform_aggregated_keys(self):
        dataset = object.__new__(GD.LeRobotSingleDataset)
        dataset.data_cfg = {"image_size": None, "include_state": True}
        dataset._modality_keys = {
            "video": ["video.cam0", "video.cam1"],
            "action": ["action.x"],
            "state": ["state.x"],
            "language": ["annotation.task"],
        }
        video = np.arange(1 * 2 * 3 * 4 * 3, dtype=np.uint8).reshape(1, 2, 3, 4, 3)
        data = {
            "video": video,
            "action": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "state": np.asarray([[5.0, 6.0]], dtype=np.float32),
            "annotation.task": ["pick object"],
        }

        sample = dataset._pack_sample(data)

        self.assertEqual(sample["lang"], "pick object")
        self.assertEqual(len(sample["image"]), 2)
        self.assertEqual(sample["image"][0].shape, (3, 4, 3))
        self.assertEqual(sample["action"].dtype, np.float16)
        self.assertEqual(sample["state"].dtype, np.float16)
        np.testing.assert_allclose(sample["action"], data["action"].astype(np.float16))
        np.testing.assert_allclose(sample["state"], data["state"].astype(np.float16))

    def test_metadata_merge_uses_weights_from_each_embodiment_tag_only(self):
        franka = GD.EmbodimentTag.FRANKA.value
        gr1 = GD.EmbodimentTag.GR1.value
        d0 = _DummyDataset(franka, _make_dataset_metadata(franka, state_mean=0.0, action_mean=0.0))
        d1 = _DummyDataset(franka, _make_dataset_metadata(franka, state_mean=10.0, action_mean=10.0))
        d2 = _DummyDataset(gr1, _make_dataset_metadata(gr1, state_mean=100.0, action_mean=100.0))

        mix = object.__new__(GD.LeRobotMixtureDataset)
        mix.datasets = [d0, d1, d2]
        mix._dataset_sampling_weights = np.asarray([0.25, 0.75, 0.99], dtype=np.float64)

        mix.update_metadata({"percentile_mixing_method": "weighted_average"})

        franka_mean = np.asarray(mix.merged_metadata[franka].statistics.action["action.x"].mean)
        gr1_mean = np.asarray(mix.merged_metadata[gr1].statistics.action["action.x"].mean)
        np.testing.assert_allclose(franka_mean, np.asarray([7.5], dtype=np.float32))
        np.testing.assert_allclose(gr1_mean, np.asarray([100.0], dtype=np.float32))
        self.assertIs(d0.received_metadata, mix.merged_metadata[franka])
        self.assertIs(d1.received_metadata, mix.merged_metadata[franka])
        self.assertIs(d2.received_metadata, mix.merged_metadata[gr1])

    def test_sequential_worker_shards_partition_global_order_across_ranks(self):
        original = np.arange(10)
        shards = []
        for rank in range(2):
            for worker_id in range(3):
                mix = _make_sequential_mixture(num_steps=10, mode="eval")
                with _WorkerContext(worker_id=worker_id, num_workers=3, rank=rank, world_size=2):
                    mix._init_worker_shard()
                global_worker_id = rank * 3 + worker_id
                np.testing.assert_array_equal(mix._step_order[0], original[global_worker_id::6])
                shards.extend(mix._step_order[0].tolist())

        self.assertEqual(sorted(shards), list(range(10)))
        self.assertEqual(len(shards), len(set(shards)))

    def test_sequential_reshuffle_cycle_partitions_even_when_shard_lengths_differ(self):
        mixes = []
        for worker_id in range(4):
            mix = _make_sequential_mixture(num_steps=10, mode="train")
            with _WorkerContext(worker_id=worker_id, num_workers=4):
                mix._init_worker_shard()
            mixes.append(mix)

        cycle0 = []
        for mix in mixes:
            while mix._step_pos[0] < len(mix._step_order[0]):
                _, _, base_index = mix.sample_step(index=0)
                cycle0.append(base_index)
        self.assertEqual(sorted(cycle0), list(range(10)))
        self.assertEqual(len(cycle0), len(set(cycle0)))

        cycle1 = []
        for mix in mixes:
            _, _, base_index = mix.sample_step(index=0)
            cycle1.append(base_index)
            while mix._step_pos[0] < len(mix._step_order[0]):
                _, _, base_index = mix.sample_step(index=0)
                cycle1.append(base_index)

        self.assertEqual(sorted(cycle1), list(range(10)))
        self.assertEqual(len(cycle1), len(set(cycle1)))
        self.assertTrue(all(mix._step_cycle[0] == 1 for mix in mixes))

    def test_sequential_sharding_rejects_empty_worker_shards(self):
        mix = _make_sequential_mixture(num_steps=5, mode="eval")
        with _WorkerContext(worker_id=0, num_workers=6):
            with self.assertRaisesRegex(ValueError, "at least one step per global worker"):
                mix._init_worker_shard()


if __name__ == "__main__":
    unittest.main()
