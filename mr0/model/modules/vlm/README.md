# Qwen3.5 VLM wrapper

`QWen3_5.py` provides the Qwen3.5 backbone used by MR0. The wrapper loads a model and processor, prepares multi-view image and instruction inputs, and exposes forward and generation methods to the FAST and MMDiT frameworks.

```python
from mr0.model.modules.vlm import get_vlm_model

vlm = get_vlm_model(config)
```

Set `config.framework.qwenvl.base_vlm` to a Qwen3.5 model directory or model ID. FAST training requires the 2,048 robot action tokens; see [token preparation](tools/add_qwen_special_tokens/README.md).
