# Selected model releases

The following three checkpoint selections are fixed for the initial MR0 model release. Public model repository IDs and download URLs are pending. The names below are display names, not existing Hugging Face repository IDs.

## Stage1-FAST

- Framework: `QwenFast`, Qwen3.5-4B backbone with FAST action tokens.
- Source run: `vla_dualarm5synbuy_224_qwen35vl4b_fast_cotrain_lishuovqa0630_bs8_256gpu`.
- Selected file: `steps_250000_pytorch_model.pt` (10,368,683,293 bytes; approximately 9.66 GiB).
- Training: VLA/VQA co-training, 256 GPUs; selected at 250k steps.
- Role: pretrained representation used by the historical RoboTwin downstream line.

## Stage2-MMDiT

- Framework: `QwenMMDiT`, MMDiT-Psi0 / JiT continuous-action head.
- Source run: `vla_dualarm5synbuy_224_qwen35vl4b_mmdit_psi0_cond_jit_fast256synbuy430kinit_cotrain_lishuovqa0630_langmix_bs8_128gpu`.
- Selected file: `steps_250000_pytorch_model.pt` (11,365,578,710 bytes; approximately 10.59 GiB).
- Initialization: **430k** checkpoint of the Stage 1 FAST run above, confirmed by the saved training configuration and the checkpoint-loading log. The 250k in this release name is the Stage 2 training step.
- Language sampling: current instruction 50%, task description 50%.
- Action/state dimensions: 20; action horizon: 16. This is a pretraining checkpoint, not the RoboTwin 14-dimensional, horizon-100 endpoint.

## MR0-RoboTwin

- Source run: `robotwin_all50_native_qwen35vl4b_mmdit_psi0_cond_jit_dualarm5synbuyfast250kinit_chunk100_hardw_avgA92_bs8_32gpu`.
- Selected file: **`steps_70000_pytorch_model.pt`**.
- Model: Qwen3.5-4B + MMDiT-Psi0 / JiT; 14-dimensional actions/state, action horizon 100.
- Lineage: Stage 1 FAST 250k → RoboTwin 224-pixel downstream training → average of downstream 130k/140k/150k (`avgA`) → hard-task weighted native-resolution fine-tuning → **70k single checkpoint**.
- Image preprocessing: preserve aspect ratio through native-resolution smart resize (`max_pixels=100352`, `min_pixels=784`; 480 × 640 inputs become 256 × 352).
- Completed evaluation: 50 tasks × 100 episodes per task, unseen instructions, separately for `demo_clean` and `demo_randomized`.
- Recomputed mean success rates: **93.22% clean**, **93.88% randomized**.

The 224-pixel `steps_avgTop4_70_90_100_110_pytorch_model.pt` (93.10% clean / 93.60% randomized) is not the selected RoboTwin release checkpoint.

## Packaging

Keep each model in a separate directory: Stage 1 and Stage 2 have the same checkpoint basename. Each complete VLA checkpoint includes the Qwen backbone and action head. Include the matching configuration, dataset statistics, and tokenizer/processor metadata alongside it; do not package a second copy of the Qwen weights. `baseframework.from_pretrained` initializes Qwen from the complete checkpoint before strictly loading the full VLA. Use the native checkpoint's own preprocessing configuration; the repository's generic 224-pixel training example is not its release configuration.

No weight upload has been performed by this README update. Download links remain placeholders.

## Benchmark provenance

- Selected RoboTwin endpoint: saved experiment configuration, historical experiment/evaluation registry, and all 50 per-task `_result.txt` files per mode, checked on 2026-11-04.
- Project overview and reported benchmark snapshot: website export revision 2026-11-03, `source/data/project-results.json`, `UPDATE-20261103.md`, and `main-overview-20261103.svg`.
- Reported RoboTwin result: 93.6% (Table 4). Do not replace this value in the imported paper figure with the separately verified native-policy result.
- Reported RoboDojo result: 14.63% average success rate (Table 5), the equal-weight average of two complete evaluations of one checkpoint. This evidence does not establish that the evaluated RoboDojo checkpoint is one of the three selected releases.
