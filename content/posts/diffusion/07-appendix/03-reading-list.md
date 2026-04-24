---
title: "推荐阅读列表"
date: 2026-04-20T17:40:09.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 推荐阅读列表

## 模块一：基础理论

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| Deep Unsupervised Learning using Nonequilibrium Thermodynamics | Sohl-Dickstein et al. | 2015 | ICML | 扩散概率模型的奠基论文 |
| **Denoising Diffusion Probabilistic Models (DDPM)** | Ho et al. | 2020 | NeurIPS | 简化训练目标 $\|\epsilon-\epsilon_\theta\|^2$，扩散模型走向实用 |
| Improved Denoising Diffusion Probabilistic Models | Nichol & Dhariwal | 2021 | ICML | 余弦调度、可学习方差、Improved DDPM |
| Generative Modeling by Estimating Gradients of the Data Distribution (NCSN) | Song & Ermon | 2019 | NeurIPS | 多尺度得分匹配 + 退火朗之万采样 |
| **Score-Based Generative Modeling through SDEs** | Song et al. | 2021 | ICLR (Oral) | SDE 统一 DDPM 和 Score-based，里程碑论文 |
| Variational Diffusion Models | Kingma et al. | 2021 | NeurIPS | 连续时间 VLB 优化、可学习噪声调度 |
| Elucidating the Design Space of Diffusion-Based Generative Models | Karras et al. | 2022 | NeurIPS | 系统消融分析 + Karras 噪声调度 |
| Common Diffusion Noise Schedules and Sample Steps are Flawed | Lin et al. | 2023 | WACV | Zero Terminal SNR + 偏移噪声 |
| Efficient Training of Diffusion Models via Min-SNR Weighting | Hang et al. | 2023 | ICCV | Min-SNR 加权 $w(t)=\min(\text{SNR},\gamma)$ |

## 模块二：模型架构

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| **Denoising Diffusion Implicit Models (DDIM)** | Song et al. | 2020 | ICLR 2021 | 确定性采样、步数加速、DDIM Inversion |
| **High-Resolution Image Synthesis with Latent Diffusion Models (LDM)** | Rombach et al. | 2022 | CVPR | 潜空间扩散 + Cross-Attention 条件注入 → Stable Diffusion |
| **Consistency Models** | Song et al. | 2023 | ICML | ODE 轨迹自一致性，单步生成 |
| Improved Techniques for Training Consistency Models (iCT) | Song & Dhariwal | 2024 | ICML | 连续时间 + Pseudo-Huber + lognormal 时间 |
| **Flow Matching for Generative Modeling** | Lipman et al. | 2022 | ICLR 2023 | 条件流匹配框架 |
| Flow Straight and Fast (Rectified Flow) | Liu et al. | 2022 | ICLR 2023 | Reflow 拉直路径 |
| Stochastic Interpolants | Albergo & Vanden-Eijnden | 2022 | TMLR | Flow Matching 的统一理论框架 |
| **Scalable Diffusion Models with Transformers (DiT)** | Peebles & Xie | 2023 | ICCV | Transformer 替代 U-Net + Scaling Laws (FID 2.27) |
| Scaling Rectified Flow Transformers (SD3) | Esser et al. | 2024 | arXiv | MM-DiT + 三编码器 + Rectified Flow |

## 模块三：条件生成

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| **Diffusion Models Beat GANs on Image Synthesis (ADM)** | Dhariwal & Nichol | 2021 | NeurIPS | Classifier Guidance，首次超越 GAN (FID 4.59) |
| **Classifier-Free Diffusion Guidance** | Ho & Salimans | 2022 | NeurIPS Workshop | CFG，去掉分类器，成为所有文生图标配 |
| CFG++: Manifold-constrained Classifier Free Guidance | Chung et al. | 2024 | arXiv | 在 $x_0$ 空间引导，减少过度饱和 |
| **Adding Conditional Control to T2I Diffusion Models (ControlNet)** | Zhang et al. | 2023 | ICCV | 零卷积 + 编码器副本，空间结构控制 |
| **IP-Adapter: Text Compatible Image Prompt Adapter** | Ye et al. | 2023 | arXiv | 解耦交叉注意力，图像条件注入 (~22M 参数) |
| T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability | Mou et al. | 2023 | AAAI | 轻量结构控制 (~77M 参数) |
| GLIGEN: Open-Set Grounded Text-to-Image Generation | Li et al. | 2023 | CVPR | Gated Self-Attention 布局控制 |
| An Image is Worth One Word: Textual Inversion | Gal et al. | 2022 | ICLR 2023 | 学习新词嵌入 (~1K 参数) |
| **DreamBooth: Fine Tuning Text-to-Image Diffusion Models** | Ruiz et al. | 2023 | CVPR | 先验保持损失 + 3-5 张图个性化 |
| **LoRA: Low-Rank Adaptation of Large Language Models** | Hu et al. | 2021 | ICLR 2022 | 低秩适配 $W'=W+BA$，后广泛用于 SD |

## 模块四：加速与部署

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| **DPM-Solver: A Fast ODE Solver for Diffusion** | Lu et al. | 2022 | NeurIPS | 半线性 ODE 指数积分器，10-20 步高质量 |
| DPM-Solver++: Fast Solver for Guided Sampling | Lu et al. | 2022 | arXiv | 数据预测参数化，CFG 友好 |
| UniPC: A Unified Predictor-Corrector Framework | Zhao et al. | 2023 | NeurIPS | 统一多步+预测校正，无额外 NFE |
| Progressive Distillation for Fast Sampling | Salimans & Ho | 2022 | ICLR | 迭代减半步数（1024→4 步） |
| **Latent Consistency Models (LCM)** | Luo et al. | 2023 | arXiv | 潜空间一致性蒸馏 + 增广 PF-ODE |
| LCM-LoRA: Universal Stable-Diffusion Distillation | Luo et al. | 2023 | arXiv | LoRA 形式的一致性蒸馏 (~67M 参数) |
| Adversarial Diffusion Distillation (ADD / SDXL-Turbo) | Sauer et al. | 2023 | arXiv | 蒸馏+对抗训练，单步生成 |
| Distribution Matching Distillation (DMD2) | Yin et al. | 2024 | CVPR | 分布匹配 + 对抗蒸馏 |
| Q-Diffusion: Quantizing Diffusion Models | Li et al. | 2023 | ICCV | 分步校准的扩散 PTQ |
| DeepCache: Accelerating Diffusion via Deep Feature Caching | Ma et al. | 2024 | CVPR | 相邻步特征缓存 ~2× 加速 |
| **FlashAttention: Fast and Memory-Efficient Exact Attention** | Dao et al. | 2022 | NeurIPS | IO 感知分块注意力，$O(N^2)→O(N)$ 内存 |
| FlashAttention-2: Faster Attention with Better Parallelism | Dao | 2023 | arXiv | 优化并行性和工作分区 |
| Token Merging for Stable Diffusion (ToMe) | Bolya & Hoffman | 2023 | arXiv | 合并相似 Token ~1.5× 加速 |

## 模块五：大模型应用

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| Photorealistic Text-to-Image Diffusion Models (**Imagen**) | Saharia et al. | 2022 | NeurIPS | T5-XXL 编码器 + 级联超分 |
| Hierarchical Text-Conditional Image Generation (**DALL-E 2**) | Ramesh et al. | 2022 | arXiv | CLIP 先验 + 扩散解码器 |
| **DALL-E 3** | Betker et al. | 2023 | OpenAI | Prompt 重写 + Caption 改进 |
| **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis** | Podell et al. | 2023 | arXiv | 双编码器 + 2.6B U-Net + Refiner |
| **FLUX** | Black Forest Labs | 2024 | 开源 | 12B DiT + Flow Matching |
| **Video generation models as world simulators (Sora)** | OpenAI | 2024 | 技术报告 | Spacetime DiT，可变分辨率/时长 |
| Stable Video Diffusion | Blattmann et al. | 2023 | arXiv | 图生视频基础模型 |
| CogVideoX | Hong et al. | 2024 | arXiv | 3D VAE + Expert Transformer |
| SDEdit: Guided Image Synthesis and Editing with SDEs | Meng et al. | 2021 | ICLR 2022 | 加噪再去噪编辑 |
| **Prompt-to-Prompt Image Editing with Cross Attention Control** | Hertz et al. | 2022 | ICLR 2023 | 操控注意力 map 编辑（词替换/添加/权重） |
| **InstructPix2Pix** | Brooks et al. | 2023 | CVPR | GPT-3 生成指令对 + 端到端指令编辑 |
| Null-text Inversion | Mokady et al. | 2023 | CVPR | 优化 null embedding 精确反演 |
| Image Super-Resolution via Iterative Refinement (**SR3**) | Saharia et al. | 2022 | TPAMI | Diffusion 超分，人类混淆率 50.2% |
| StableSR: Exploiting Diffusion Prior for Real-World Image SR | Wang et al. | 2024 | IJCV | 潜空间超分 + 盲退化处理 |
| **DreamFusion: Text-to-3D using 2D Diffusion** | Poole et al. | 2022 | ICLR 2023 | SDS 梯度优化 NeRF |
| **ProlificDreamer: High-Fidelity Text-to-3D (VSD)** | Wang et al. | 2023 | NeurIPS | 变分得分蒸馏，LoRA 基线替代随机噪声 |
| Zero-1-to-3: Zero-shot One Image to 3D Object | Liu et al. | 2023 | ICCV | 视角条件微调 SD |
| MVDream: Multi-view Diffusion for 3D Generation | Shi et al. | 2023 | arXiv | 4 视角联合生成 + 3D 自注意力 |
| DreamGaussian: Generative 3D Gaussian Splatting | Tang et al. | 2023 | ICLR 2024 | 3DGS + SDS，分钟级 3D 生成 |
| **Transfusion: Predict the Next Token and Diffuse Images** | Meta | 2024 | arXiv | 自回归文本 + 扩散图像统一训练 |
| Chameleon: Mixed-Modal Early-Fusion Foundation Models | Meta | 2024 | arXiv | 离散 Token 统一文本+图像 (34B) |
| Show-o: One Single Transformer to Unify Multimodal Understanding and Generation | Xie et al. | 2024 | arXiv | 自回归+离散扩散混合 |

## 模块六：自动驾驶

| 论文 | 作者 | 年份 | 会议 | 核心贡献 |
|------|------|------|------|---------|
| **Planning with Diffusion for Flexible Behavior Synthesis (Diffuser)** | Janner et al. | 2022 | ICML | 首个 Diffusion 规划器（轨迹去噪+奖励引导） |
| **Is Conditional Generative Modeling All You Need for Decision Making? (Decision Diffuser)** | Ajay et al. | 2023 | ICLR | CFG 替代 Classifier Guidance 做规划 |
| **MID: Motion Indeterminacy Diffusion** | Gu et al. | 2022 | CVPR | 首个 Diffusion 轨迹预测 |
| CTG: Controllable Traffic Generation with Diffusion | Zhong et al. | 2023 | CoRL | 联合多智能体 + 约束引导 |
| Diffusion-ES: Gradient-free Planning with Diffusion | Zhong et al. | 2024 | arXiv | Diffusion + 进化策略混合 |
| SafeDiffuser: Safe Planning with Diffusion Probabilistic Models | Xiao et al. | 2024 | arXiv | 安全约束 Diffusion 规划 |
| BEVGen: Street-View Image Generation from BEV Layout | Swerdlow et al. | 2024 | arXiv | BEV 空间 Diffusion 场景布局 |
| **MagicDrive: Street View Generation with Diverse 3D Geometry Control** | Gao et al. | 2023 | arXiv | 3D Bbox + HD Map 多视角街景生成 |
| DriveDreamer: Towards Real-world-driven World Models | Wang et al. | 2023 | arXiv | 两阶段驾驶场景生成 |
| DriveDreamer-2: LLM-Enhanced World Models | Wu et al. | 2024 | arXiv | LLM 结构化场景描述 + 视频 Diffusion |
| **GAIA-1: A Generative World Model for Autonomous Driving** | Hu et al. | 2023 | arXiv | 9B 参数驾驶世界模型（视频+动作+文本条件） |
| GenAD: Generative End-to-End Autonomous Driving | Yang et al. | 2024 | arXiv | 潜空间世界模型 + 规划统一 |
| UniAD: Planning-oriented Autonomous Driving | Hu et al. | 2023 | CVPR (Best Paper) | 端到端自动驾驶基线 |

## 综合参考

### 优质博客与教程
- **Lilian Weng**: "What are Diffusion Models?" — 最经典的 Diffusion 入门博客
- **Yang Song**: "Generative Modeling by Estimating Gradients" — Score-based 视角的权威解读
- **Hugging Face Diffusion Course** — 实践导向的扩散模型教程
- **The Annotated Diffusion Model** — 代码级别的 DDPM 逐行解读
- **Stability AI Blog** — SD 1.x → SDXL → SD3 系列技术博客
- **OpenAI Sora Technical Report** — 视频生成里程碑

### 综述论文
- "Diffusion Models: A Comprehensive Survey of Methods and Applications" (Yang et al., 2023)
- "Understanding Diffusion Models: A Unified Perspective" (Luo, 2022)
- "A Survey on Diffusion Models for Time Series and Spatio-Temporal Data" (2024)
- "Diffusion Models for Autonomous Driving: A Survey" (2024)

### 经典教材/教程
- **CS 236: Deep Generative Models** (Stanford) — Diffusion/Flow/VAE/GAN 理论基础
- **diffusers 文档** (HuggingFace) — Stable Diffusion 实践标准库
- **Score Matching Tutorial** (Yang Song) — 得分匹配理论深度解读
