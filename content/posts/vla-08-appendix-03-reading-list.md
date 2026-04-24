---
title: "推荐阅读列表"
date: 2026-04-20T17:05:25.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 推荐阅读列表

## 模块一：前置基础

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| Learning Transferable Visual Models From Natural Language Supervision (**CLIP**) | Radford et al. | 2021 | ICML | 对比学习连接视觉和语言 |
| Sigmoid Loss for Language Image Pre-Training (**SigLIP**) | Zhai et al. | 2023 | ICCV | Sigmoid 替代 Softmax，更高效的对比学习 |
| DINOv2: Learning Robust Visual Features without Supervision | Oquab et al. | 2023 | TMLR | 自监督视觉编码器，空间细节强 |
| Visual Instruction Tuning (**LLaVA**) | Liu et al. | 2023 | NeurIPS | VLM 视觉指令微调范式 |
| PaLI-X: On Scaling up a Multilingual Vision and Language Model | Chen et al. | 2023 | arXiv | 55B 多模态大模型，RT-2 的基座 |
| A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (**DAgger**) | Ross et al. | 2011 | AISTATS | 迭代数据聚合缓解分布偏移 |
| Implicit Behavioral Cloning (**IBC**) | Florence et al. | 2022 | CoRL | 能量模型处理多模态动作分布 |

## 模块二/三：架构与核心模型

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| **Do As I Can, Not As I Say (SayCan)** | Ahn et al. | 2022 | RSS | LLM 规划 + Affordance 可行性评估 |
| Code as Policies: Language Model Programs for Embodied Control (**CaP**) | Liang et al. | 2022 | ICRA 2023 | LLM 直接生成机器人控制代码 |
| Inner Monologue: Embodied Reasoning through Planning with LMs | Huang et al. | 2022 | CoRL | LLM 闭环推理指导机器人 |
| **RT-1: Robotics Transformer** | Brohan et al. | 2022 | RSS 2023 | 35M Transformer + 130K 真机数据，TokenLearner |
| **RT-2: Vision-Language-Action Models** | Brohan et al. | 2023 | CoRL | **VLA 范式开创**，涌现能力验证 |
| RT-2-X: Large-Scale Cross-Embodiment Robotic Learning | O'Brien et al. | 2024 | ICRA | 跨具身体 VLA + Open X-Embodiment |
| **Octo: An Open-Source Generalist Robot Policy** | Team et al. | 2024 | RSS | 开源跨机器人基础模型，Readout Token |
| **OpenVLA: An Open-Source Vision-Language-Action Model** | Kim et al. | 2024 | CoRL | 开源 7B VLA（SigLIP+DINOv2+Llama2） |
| **π₀: A Vision-Language-Action Flow Model for General Robot Control** | Black et al. | 2024 | arXiv | Flow Matching 动作头 + 灵巧操作突破 |
| π₀-FAST: Efficient Action Tokenization for VLAs | Pertsch et al. | 2025 | arXiv | 加速 π₀ 的动作 Token 化 |
| GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation | Wu et al. | 2024 | ICLR | MAE 视频预训练→动作微调 |
| GR-2: Generative Video-Language-Action Model for Robot Manipulation | Cheang et al. | 2024 | arXiv | 3B 参数扩展，潜空间预测 |
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (**ALOHA**) | Zhao et al. | 2023 | RSS | 低成本双臂遥操作 + ACT 策略 |
| ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation | Aldaco et al. | 2024 | arXiv | ALOHA 改进版 |
| **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** | Chi et al. | 2023 | RSS 2024 | 扩散模型生成机器人动作序列 |
| Helix: A Foundation Vision-Language-Action Model for Generalist Humanoid Control | Figure AI | 2025 | 公司发布 | 双系统人形 VLA，BMW 工厂部署 |
| **GR00T N1: An Open Foundation Model for Generalist Humanoid Robots** | NVIDIA | 2025 | GTC | 双系统 + Isaac Sim 仿真生态协同 |
| Gemini Robotics: Bringing AI into the Physical World | Google DeepMind | 2025 | 公司发布 | 通用大模型直接扩展到机器人 |
| QUAR-VLA: Vision-Language-Action Model for Quadruped Robots | MiLAB | 2024 | arXiv | VLA 扩展到四足运动 |
| RoboFlamingo: An Open-source Framework for Robot Learning with Flamingo | Li et al. | 2024 | arXiv | Perceiver+Cross-Attention 视觉历史注入 |
| LLARVA: Vision-Action Instruction Tuning with Visual Affordances | Niu et al. | 2024 | arXiv | 2D 视觉轨迹作为中间表示 |
| 3D-VLA: 3D Vision-Language-Action Generative World Model | Zhen et al. | 2024 | arXiv | 点云输入 + 3D 世界模型 |

## 模块四：训练技术

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| **Open X-Embodiment: Robotic Learning Datasets and RT-X Models** | Collaboration | 2024 | ICRA | 21 机构跨机器人数据集（~100 万轨迹） |
| DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | Khazatsky et al. | 2024 | RSS | 分布式真机采集 |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | ICLR 2022 | 高效微调方法 |
| QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers et al. | 2023 | NeurIPS | 4-bit 量化 + LoRA |
| Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics | Zhao et al. | 2020 | — | Sim-to-Real 综述 |
| LeRobot: Open-Source Robotics Framework for AI | HuggingFace | 2024 | 开源 | 机器人学习训练/部署框架 |
| robomimic: A Framework for Robot Learning from Demonstration | Mandlekar et al. | 2021 | CoRL | 标准化 BC/IL 评测框架 |

## 模块五：机器人应用

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| Humanoid Locomotion as Next Token Prediction | Radosavovic et al. | 2024 | arXiv | 人形运动的自回归预测 |
| Voyager: An Open-Ended Embodied Agent with LLMs | Wang et al. | 2023 | arXiv | Minecraft 自驱动探索 Agent（技能库积累） |
| DEPS: Describe, Explain, Plan and Select | Wang et al. | 2023 | arXiv | 解决 LLM 规划粒度问题 |
| RoboAgent: Generalization and Efficiency in Robot Manipulation | Bharadhwaj et al. | 2023 | arXiv | 少样本 + 语义增强的泛化 |
| NaVILA: Unified VLA for Mobile Manipulation | Cheng et al. | 2024 | arXiv | 导航+操作统一 VLA |
| Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Hardware | Fu et al. | 2024 | arXiv | 移动底盘 + ALOHA 双臂 + Co-training |

## 模块六：自动驾驶

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| DriveVLM: The Convergence of AD and Large VLMs | Tian et al. | 2024 | arXiv | CoT 三阶段驾驶推理（InternVL-26B） |
| LMDrive: Closed-Loop End-to-End Driving with LLMs | Shao et al. | 2024 | CVPR | 语言条件闭环驾驶（CARLA DS=76.1） |
| DriveGPT4: Interpretable E2E Autonomous Driving via LLM | Xu et al. | 2024 | arXiv | 视频输入 + 可解释决策生成 |
| GPT-Driver: Learning to Drive with GPT | Mao et al. | 2023 | arXiv | LLM 直接输出轨迹坐标（纯文本输入） |
| Dolphins: Multimodal Language Model for Driving | Ma et al. | 2024 | arXiv | 多粒度驾驶 VLM（CIDEr=95.7） |
| DriveLM: Driving with Graph Visual QA | Sima et al. | 2024 | ECCV | QA 图结构驾驶推理数据集（443K 对） |
| EMMA: End-to-End Multimodal Model for Autonomous Driving | Hwang et al. | 2024 | arXiv | Gemini-based 端到端驾驶（Waymo Open） |
| UniAD: Planning-oriented Autonomous Driving | Hu et al. | 2023 | CVPR | 统一端到端自动驾驶基线 |
| VAD: Vectorized Scene Representation for Efficient AD | Jiang et al. | 2023 | ICCV | 向量化场景表示端到端驾驶 |

## 模块七：大模型融合

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| PaLM-E: An Embodied Multimodal Language Model | Driess et al. | 2023 | ICML | 多模态大模型做具身推理 |
| UniSim: Learning Interactive Real-World Simulators | Yang et al. | 2024 | ICLR | 通用交互式世界模拟器（统一动作接口） |
| Genie: Generative Interactive Environments | Bruce et al. | 2024 | ICML | 从视频学习可控世界模型（11B, 200K hr） |
| Dreamer-v3: Mastering Diverse Domains through World Models | Hafner et al. | 2023 | — | 潜空间世界模型（RSSM） |
| Scaling Laws for Neural Language Models | Kaplan et al. | 2020 | arXiv | LLM Scaling Laws 奠基论文 |

## 综合参考

### 优质博客与教程
- **LearnOpenCV**: "Vision-Language-Action Models & LeRobot Policy" — VLA 入门指南（含 ALOHA/π₀/Helix/GR00T/Gemini Robotics）
- **HuggingFace LeRobot 文档** — 开源机器人学习框架实践（含 ACT/Diffusion Policy 训练）
- **Google DeepMind Blog**: RT-1/RT-2/ALOHA/Gemini Robotics 系列技术博客
- **Physical Intelligence Blog**: π₀/π₀-FAST 技术博客
- **Figure AI Blog**: Helix 技术博客 + BMW 部署案例
- **NVIDIA GTC 2025**: GR00T N1 + Isaac Sim 演示

### 综述论文
- "A Survey on Vision-Language-Action Models for Embodied AI" (2024)
- "Foundation Models in Robotics: Applications, Challenges, and the Future" (2024)
- "Robot Learning in the Era of Foundation Models: A Survey" (2024)
- "A Survey on Diffusion Models for Robotics" (2024)
- "Large Language Models for Autonomous Driving: A Survey" (2024)

### 经典教材/教程
- **CS 236: Deep Generative Models** (Stanford) — Diffusion/Flow Matching 理论基础
- **CS 224R: Deep Reinforcement Learning** (Stanford) — RL 基础 + 机器人应用
- **robomimic Tutorials** — BC/BC-RNN/Diffusion Policy 实践教程
- **LeRobot Getting Started** — 从零搭建机器人 VLA 训练流水线
