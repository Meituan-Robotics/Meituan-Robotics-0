"""Load an actual small Qwen backbone with only metadata beside the VLA checkpoint."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from mr0.model.modules.vlm import QWen3_5


@pytest.fixture
def backbone(tmp_path, monkeypatch):
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "layer_types": ["full_attention"], "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "out_hidden_size": 32, "num_position_embeddings": 16, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
        tie_word_embeddings=True,
    )
    model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    directory = tmp_path / "Qwen3.5-tiny"
    model.save_pretrained(directory)
    tokenizer = SimpleNamespace(unk_token_id=None, convert_tokens_to_ids=lambda token: None)
    monkeypatch.setattr(
        QWen3_5.AutoProcessor, "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=tokenizer),
    )
    cfg = OmegaConf.create({"framework": {"qwenvl": {"base_vlm": str(directory)}}, "datasets": {}})
    return model, directory, cfg


@pytest.mark.parametrize("serialization", ["pt", "legacy_pt", "safetensors"])
def test_complete_checkpoint_needs_no_backbone_weight_file(backbone, tmp_path, serialization):
    source, directory, cfg = backbone
    torch.manual_seed(123)
    baseline = QWen3_5._QWen3_5_VL_Interface(cfg).model
    baseline_rng = torch.get_rng_state().clone()
    state = {"qwen_vl_interface.model." + key: value.clone() for key, value in source.state_dict().items()}
    state["action_model.example_weight"] = torch.ones(2, 2)
    checkpoint = tmp_path / ("model.safetensors" if serialization == "safetensors" else "model.pt")
    if serialization == "safetensors":
        from safetensors.torch import save_file

        save_file(state, checkpoint)
    else:
        torch.save(state, checkpoint, _use_new_zipfile_serialization=serialization != "legacy_pt")
    (directory / "model.safetensors").unlink()
    cfg.framework.qwenvl.checkpoint_path = str(checkpoint)
    torch.manual_seed(123)
    loaded = QWen3_5._QWen3_5_VL_Interface(cfg).model
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == baseline.state_dict()[name].dtype
        torch.testing.assert_close(tensor, source.state_dict()[name], rtol=0, atol=0)
    baseline_buffers = dict(baseline.named_buffers())
    for name, tensor in loaded.named_buffers():
        torch.testing.assert_close(tensor, baseline_buffers[name], rtol=0, atol=0)


def test_checkpoint_without_backbone_is_rejected(backbone, tmp_path):
    _, _, cfg = backbone
    checkpoint = tmp_path / "head_only.pt"
    torch.save({"action_model.example_weight": torch.ones(2)}, checkpoint)
    cfg.framework.qwenvl.checkpoint_path = str(checkpoint)
    with pytest.raises(ValueError, match="no Qwen backbone weights"):
        QWen3_5._QWen3_5_VL_Interface(cfg)
