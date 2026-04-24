---
title: "Diffusion 扩散模型系列"
date: 2026-04-24T00:00:00.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Diffusion 扩散模型系列

> 扩散模型（Diffusion Model）是当前最主流的生成式模型架构之一。本文档系统整理了扩散模型的理论与实践知识，涵盖基础原理、模型演进、条件生成、加速部署、大模型时代应用以及自动驾驶场景下的生成研究。

---

## 📚 系列目录

### 01 基础原理（Fundamentals）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [扩散模型核心直觉](./diffusion-01-fundamentals-01-core-intuition) | 直觉理解：前向/反向过程 |
| 02 | [前向过程](./diffusion-01-fundamentals-02-forward-process) | 逐步加噪的数学描述 |
| 03 | [反向过程](./diffusion-01-fundamentals-03-reverse-process) | 神经网络学习去噪 |
| 04 | [采样算法](./diffusion-01-fundamentals-04-sampling-algorithms) | DDPM、DDIM、概率流 ODE |
| 05 | [噪声调度](./diffusion-01-fundamentals-05-noise-schedule) | β 调度与扩散效率 |
| 06 | [分数匹配](./diffusion-01-fundamentals-06-score-matching) | Score Matching 视角 |
| 07 | [SDE 与 ODE](./diffusion-01-fundamentals-07-sde-ode) | 随机微分方程统一框架 |

### 02 Models Zoo

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [DDPM](./diffusion-02-models-zoo-01-ddpm) | 原版扩散概率模型 |
| 02 | [DDIM](./diffusion-02-models-zoo-02-ddim) | 加速采样的里程碑 |
| 03 | [Score-Based](./diffusion-02-models-zoo-03-score-based) | 分数匹配生成模型 |
| 04 | [Latent Diffusion](./diffusion-02-models-zoo-04-latent-diffusion) | Stable Diffusion 核心技术 |
| 05 | [Consistency Models](./diffusion-02-models-zoo-05-consistency-models) | 一致性模型：蒸馏加速 |
| 06 | [Flow Matching](./diffusion-02-models-zoo-06-flow-matching) | 连续归一化流的简洁框架 |
| 07 | [Evolution Map](./diffusion-02-models-zoo-07-evolution-map) | 模型演进全景图 |

### 03 条件生成（Conditional Generation）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [条件生成概述](./diffusion-03-conditional-generation-01-overview) | 分类引导与概念 |
| 02 | [Classifier Guidance](./diffusion-03-conditional-generation-02-classifier-guidance) | 梯度引导生成 |
| 03 | [CFG](./diffusion-03-conditional-generation-03-cfg) | 无分类器引导 |
| 04 | [ControlNet](./diffusion-03-conditional-generation-04-controlnet) | 可控生成的利器 |
| 05 | [IP-Adapter](./diffusion-03-conditional-generation-05-ip-adapter) | 图像提示适配器 |
| 06 | [Adapters](./diffusion-03-conditional-generation-06-adapters) | 轻量级适配方法汇总 |
| 07 | [对比总结](./diffusion-03-conditional-generation-07-comparison) | 主流方法横向对比 |

### 04 加速与部署（Acceleration & Deployment）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [加速部署概述](./diffusion-04-acceleration-deployment-01-overview) | 推理效率优化全景 |
| 02 | [ODE 求解器](./diffusion-04-acceleration-deployment-02-ode-solvers) | 数值求解器选型 |
| 03 | [蒸馏技术](./diffusion-04-acceleration-deployment-03-distillation) | 知识蒸馏加速扩散 |
| 04 | [量化](./diffusion-04-acceleration-deployment-04-quantization) | INT8/INT4 量化方案 |
| 05 | [推理优化](./diffusion-04-acceleration-deployment-05-inference-optimization) | CUDA、FlashAttention 等 |

### 05 大模型时代（LLM Era）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [文生图](./diffusion-05-llm-era-01-text-to-image) | 文本到图像生成 |
| 02 | [DiT](./diffusion-05-llm-era-02-dit) | Diffusion Transformer |
| 03 | [文生视频](./diffusion-05-llm-era-03-text-to-video) | 视频生成模型 |
| 04 | [图像编辑](./diffusion-05-llm-era-04-image-editing) | 基于扩散的编辑 |
| 05 | [超分辨率](./diffusion-05-llm-era-05-super-resolution) | 扩散 + 超分 |
| 06 | [3D 生成](./diffusion-05-llm-era-06-3d-generation) | 扩散模型 + 3D |
| 07 | [多模态](./diffusion-05-llm-era-07-multimodal) | 多模态扩散模型 |

### 06 自动驾驶（Autonomous Driving）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [场景生成概述](./diffusion-06-autonomous-driving-01-overview) | 扩散 + 自动驾驶 |
| 02 | [场景生成](./diffusion-06-autonomous-driving-02-scene-generation) | 驾驶场景合成 |
| 03 | [轨迹预测](./diffusion-06-autonomous-driving-03-trajectory-prediction) | 轨迹扩散模型 |
| 04 | [运动规划](./diffusion-06-autonomous-driving-04-motion-planning) | 扩散 + 运动规划 |
| 05 | [数据增强](./diffusion-06-autonomous-driving-05-data-augmentation) | 扩散数据增强 |
| 06 | [世界模型](./diffusion-06-autonomous-driving-06-world-model) | 扩散世界模型 |

### 07 附录（Appendix）

| 序号 | 文章 | 说明 |
|------|------|------|
| — | [索引总览](./diffusion-07-appendix-01-index) | 系列内容索引 |
| — | [术语表](./diffusion-07-appendix-02-glossary) | 核心术语解释 |
| — | [阅读清单](./diffusion-07-appendix-03-reading-list) | 必读论文列表 |
