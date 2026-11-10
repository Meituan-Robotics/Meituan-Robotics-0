<h1 align="center">Meituan-Robotics-0</h1>
<p align="center"><strong>A Vision-Language-Action Foundation Model for Desktop Manipulation</strong></p>
<p align="center">Meituan Robotics</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.08577" title="Read the technical report on arXiv"><img src="https://img.shields.io/static/v1?label=Paper&amp;message=Technical_Report&amp;color=red&amp;logo=arxiv" alt="Paper: Technical Report"></a>
  <a href="https://huggingface.co/MeituanRobotics" title="Model weights and datasets on Hugging Face"><img src="https://img.shields.io/static/v1?label=Weights%20%26%20Data&amp;message=HuggingFace&amp;color=yellow&amp;logo=huggingface" alt="Weights &amp; Data: HuggingFace"></a>
  <a href="https://meituan-robotics.github.io/Meituan-Robotics-0/" title="Meituan-Robotics-0 project page"><img src="https://img.shields.io/static/v1?label=Project&amp;message=Page&amp;color=brightgreen&amp;logo=googlechrome" alt="Project Page"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#models">Models</a> ·
  <a href="#simulation-benchmarks">Simulation</a> ·
  <a href="GETTING_STARTED.md">Documentation</a> ·
  <a href="#quick-start">Quick start</a>
</p>

<a id="real-world-manipulation"></a>

## 🤖 Real-world manipulation

| Letter block placement | Garment folding | Network cable insertion |
| :---: | :---: | :---: |
| ![MR0 placing letter blocks](assets/mr0/letter-block-placement.gif) | ![MR0 folding a garment](assets/mr0/garment-folding.gif) | ![MR0 inserting a network cable](assets/mr0/network-cable-insertion.gif) |

*Demonstrations shown at 6×, 6×, and 3× speed, respectively.*

<a id="overview"></a>

## 🌟 Overview

![MR0 overview: multimodal and robot data, staged policy training, embodied agents, and manipulation evaluation](assets/mr0/overview.png)

| Learn representations | Transfer to control | Adapt to a task suite |
| :--- | :--- | :--- |
| **Stage 1 · FAST** | **Stage 2 · MMDiT** | **Downstream policies** |
| Discrete action-token prediction with vision-language co-training. | Continuous flow-matching actions, initialized from a pretrained FAST representation. | Adapt pretrained VLA models to downstream manipulation tasks. |

- 🌐 **Large-scale multimodal pretraining.** Combine **147M vision-language samples** with **15,060 hours of robot trajectories** spanning **12 embodiments**, bringing visual, language, and action supervision into a shared training pipeline.
- ⚡ **FAST action representation learning.** Stage1-FAST learns discrete action-token prediction alongside vision-language co-training, connecting multimodal representations to robot behavior.
- 🧠 **MMDiT continuous control.** Stage2-MMDiT transfers the pretrained FAST representation into a continuous flow-matching action model, bridging action-token learning and continuous robot control.
- 🎯 **Downstream task adaptation.** Adapt pretrained VLA models to downstream manipulation tasks, with real-world demonstrations covering letter block placement, garment folding, and network cable insertion.
- 📊 **Simulation evaluation and deployment.** Evaluate on **RoboTwin**, **RoboDojo**, and **RoboCasa365**, with MR0-RoboTwin reaching **93.22%** clean and **93.88%** randomized success rates. The repository includes training configurations, a RoboTwin evaluation interface, and an OpenPI-compatible policy server.

<a id="models"></a>

## 📦 Models

| Model / Data | Download |
| :--- | :--- |
| **Meituan-Robotics-0** | [Hugging Face](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0) |
| **Meituan-Robotics-0-RoboTwin** | [Hugging Face](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboTwin) |
| **Meituan-Robotics-0-RoboDojo** | [Hugging Face](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboDojo) |
| **Meituan-Robotics-0-RoboCasa365** | [Hugging Face](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboCasa365) |
| **RoboDesk (Pretraining Data)** | [Hugging Face](https://huggingface.co/datasets/MeituanRobotics/RoboDesk) |

<a id="simulation-benchmarks"></a>

## 🎯 Simulation benchmarks

### RoboTwin

**MR0-RoboTwin** is evaluated on **50 tasks × 100 episodes per task**, with unseen instructions:

| Model | `demo_clean` SR | `demo_randomized` SR |
| :--- | ---: | ---: |
| **MR0-RoboTwin** | **93.22%** | **93.88%** |

### RoboDojo

| Reported model | Average success rate | Protocol |
| :--- | ---: | :--- |
| **MR0** | **14.63%** | Equal-weight average of two complete evaluations of the same checkpoint. |

### RoboCasa365

| Model | Success rate (SR) |
| :--- | ---: |
| **Meituan-Robotics-0-RoboCasa365** | **55.0%** |

---

<a id="quick-start"></a>

## 🚀 Quick Start

See the [Getting Started guide](GETTING_STARTED.md) for setup and runnable commands.

<a id="citation-and-acknowledgements"></a>

## 📜 Citation and acknowledgements

Technical report: [*Meituan-Robotics-0: A Vision-Language-Action Foundation Model for Desktop Manipulation*](https://arxiv.org/abs/2611.08577) (Meituan Robotics Team, 2026), arXiv:2611.08577.

MR0 builds on the StarVLA training and deployment stack. For the underlying software, see [CITATION.cff](CITATION.cff). We thank the [StarVLA](https://github.com/starVLA/starVLA), [Qwen](https://github.com/QwenLM), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), and [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) communities.

<a id="license"></a>

## ⚖️ License

See [LICENSE](LICENSE) for the code license and the linked Hugging Face model and dataset cards for their respective licenses and release details.
