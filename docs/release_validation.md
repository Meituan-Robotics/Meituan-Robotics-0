# MR0 package migration and release validation

## Complete-checkpoint loading — November 5, 2026

Qwen3.5 can now initialize directly from the backbone tensors inside a complete VLA checkpoint. Its separate metadata directory needs no Qwen weight file. The final full-model load remains strict.

- Eight targeted regression and package tests passed, including a real small Qwen model loaded from PyTorch, legacy PyTorch, and Safetensors checkpoints with the standalone backbone weight file removed. The tests check exact weights, dtypes, tied embeddings, nonpersistent buffers, RNG state, and rejection of a head-only checkpoint.
- The selected full-size MR0 and RoboTwin checkpoints each loaded all 929 state-dict entries successfully with metadata-only backbone directories.
- For both models, seeded synthetic-input GPU inference produced exactly the same actions as the previous loader. CPU/CUDA RNG states, nonpersistent buffers, and dtypes also matched. Output shapes were `(1, 16, 20)` and `(1, 100, 14)` respectively.
- Undefined-name and syntax checks passed for the changed Python files.

These checks used the dependency environment listed below and an H100 GPU. They do not remeasure simulator benchmark scores. RoboDojo uses its separate policy adapter and CD-LAM dependency; its model repository supplies the matching loading instructions.

## Package migration — November 4, 2026

Validated on November 4, 2026, after renaming the Python package from `starVLA` to `mr0`.

## Changes covered

- Renamed package imports, dynamic framework discovery, training entrypoints, configuration paths, environment variables, and installation metadata.
- Included dataset registries, DeepSpeed configurations, and FAST tokenizer data in the wheel. Runtime dependencies are declared by the Python package.
- Corrected the RoboTwin evaluation shell syntax, removed references to an unavailable cluster setup script, and made the OpenPI launcher configurable.
- Restored the local adaptive action ensemble helper so the evaluation option no longer silently disables itself.
- Fixed policy-server envelope routing and invalid-message handling. Updated the debug client to the supported request API and preserved proxy environment variables when connecting directly.
- Corrected the Qwen3.5 dependency stack, removed duplicate/conflicting and unused legacy dependencies, and updated installation and serving documentation.
- Retained upstream copyright notices, LICENSE, CITATION.cff, and project acknowledgements.

## Verification

| Check | Result |
| :--- | :--- |
| Complete test suite | **96 passed**, no skipped tests; two dependency deprecation warnings |
| MMDiT runtime | CPU and CUDA forward/backward, optimizer step, strict state-dict round trip, deterministic inference with fixed noise |
| Framework and data discovery | QwenFast, QwenMMDiT, Qwen3.5 backbone import, and example dataset registries resolve under `mr0` |
| Launchers | Four node ranks, paths containing spaces, evaluation argument forwarding, OpenPI options, and cluster-wrapper help |
| WebSocket transport | Actual local client/server exchange, NumPy serialization, metadata handshake, envelope and flat requests, malformed requests |
| CLI entrypoints | Separate-process `--help` checks passed for both trainers, both policy servers, and the debug client |
| Packaging | Editable installation and wheel build; wheel imported from outside the source checkout; packaged Python files match source |
| Static checks | Python parsing, Shell syntax, undefined names, repository imports, and local documentation links |
| Release scan | No internal mount paths or common private-key/token patterns found in released source; 120 historical text blobs also scanned |

The final dependency validation used Python 3.11.5, PyTorch 2.7.1 + CUDA 12.6, TorchVision 0.22.1, TorchCodec 0.3.0, Transformers 5.7.0, Accelerate 1.5.2, NumPy 1.26.4, Pydantic 2.10.6, and PyArrow 14.0.1. The validation venv reused the installed CUDA/PyTorch runtime and installed all declared requirements in its own environment; it was not a clean operating-system container build. DeepSpeed was installed with `DS_BUILD_OPS=0` for these checks.

MMDiT model implementation bytes are unchanged by the package move. Framework identifiers (`QwenFast`, `QwenMMDiT`) and state-dict keys remain unchanged. Use `MR0_PYTHON`, `ROBOTWIN_MR0_ENV`, and `MR0_NORMALIZATION_MODE` in new launch configurations.

## Reproduce

After following [installation](../GETTING_STARTED.md#installation):

```bash
pip install -e '.[dev]'
python -m pytest tests -q
make check
python -m build --wheel
```

Tests use small models, synthetic observations, and temporary checkpoints. This validation does not replace full pretrained-model inference, a multi-node training run, RoboTwin/RoboDojo benchmark evaluation, or real-robot testing. Common-pattern credential scanning is not a comprehensive security audit.
