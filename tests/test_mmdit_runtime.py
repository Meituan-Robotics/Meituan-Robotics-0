"""Small real-tensor training/inference and checkpoint round-trip checks."""
import io

import pytest
import torch
from omegaconf import OmegaConf

from mr0.model.modules.action_model.MMDiT_ActionHeader import MMDiTConfig, MMDiTFlowmatchingActionHead


@pytest.mark.parametrize("prediction_type", ["sample", "velocity"])
@pytest.mark.parametrize("state_placement", ["condition", "noise"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mmdit_backward_and_checkpoint_roundtrip(monkeypatch, prediction_type, state_placement, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setitem(MMDiTConfig, "test_tiny", {"hidden_dim": 32, "num_heads": 4, "num_blocks": 2})
    cfg = OmegaConf.create({"framework": {"action_model": {
        "action_model_type": "test_tiny", "action_dim": 4, "state_dim": 4,
        "action_horizon": 3, "num_inference_timesteps": 2, "vl_feature_dim": 8,
        "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
        "num_timestep_buckets": 1000, "prediction_type": prediction_type,
        "state_placement": state_placement,
    }}})
    head = MMDiTFlowmatchingActionHead(cfg).to(device)
    vl_embs = torch.randn(2, 5, 8, device=device)
    state = torch.randn(2, 1, 4, device=device)
    actions = torch.randn(2, 3, 4, device=device)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool, device=device)
    loss = head(vl_embs, actions, state=state, action_mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    torch.optim.AdamW(head.parameters(), lr=1e-4).step()
    checkpoint = io.BytesIO()
    torch.save(head.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = MMDiTFlowmatchingActionHead(cfg).to(device)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    head.eval()
    restored.eval()
    noise = torch.randn_like(actions)
    expected = head.predict_action(vl_embs, state=state, noise=noise)
    actual = restored.predict_action(vl_embs, state=state, noise=noise)
    assert actual.shape == (2, 3, 4) and torch.isfinite(actual).all()
    torch.testing.assert_close(expected, actual)
