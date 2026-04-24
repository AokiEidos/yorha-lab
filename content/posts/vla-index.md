---
title: "VLA 从入门到精通系列"
date: 2026-04-24T00:00:00.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# VLA 从入门到精通系列

> VLA（Vision-Language-Action Models）是将视觉感知、语言理解与动作执行统一在端到端神经网络中的模型，是当前机器人与自动驾驶领域最活跃的研究方向。本系列面向有深度学习基础的读者，系统梳理 VLA 的理论、架构、训练与应用。

---

## 📚 系列目录

### 01 基础（Foundations）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [VLM 视觉-语言模型回顾](./vla-01-foundations-01-vlm-review) | CLIP、ALIGN 与视觉-语言对齐 |
| 02 | [具身感知](./vla-01-foundations-02-embodied-perception) | 视觉与物理世界的连接 |
| 03 | [观测与动作空间](./vla-01-foundations-03-observation-action-space) | 机器人输入输出定义 |
| 04 | [模仿学习](./vla-01-foundations-04-imitation-learning) | 行为克隆与 DAGGER |
| 05 | [强化学习入门](./vla-01-foundations-05-reinforcement-learning) | Policy Gradient、PPO |

### 02 架构（Architecture）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [VLM 到 VLA](./vla-02-architecture-01-vlm-to-vla) | 视觉-语言模型扩展为 VLA |
| 02 | [动作表示](./vla-02-architecture-02-action-representation) | 连续动作、离散动作、扩散 |
| 03 | [动作头](./vla-02-architecture-03-action-head) | VLA 输出头设计 |
| 04 | [多模态融合](./vla-02-architecture-04-multimodal-fusion) | 视觉-语言-动作融合策略 |
| 05 | [架构对比](./vla-02-architecture-05-architecture-comparison) | RT-2/π₀/Octo 等架构分析 |

### 03 Models Zoo

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [SayCan](./vla-03-models-zoo-01-saycan) | 语言+视觉做机器人操纵 |
| 02 | [RT-1](./vla-03-models-zoo-02-rt1) | 首个真实机器人 VLA |
| 03 | [RT-2](./vla-03-models-zoo-03-rt2) | 视觉-语言-动作 co-tuning |
| 04 | [Octo](./vla-03-models-zoo-04-octo) | 开源开源 VLA 基座 |
| 05 | [OpenVLA](./vla-03-models-zoo-05-openvla) | 7B 参数开源 VLA |
| 06 | [π₀](./vla-03-models-zoo-06-pi-zero) | 扩散动作的 VLA |
| 07 | [GR 系列](./vla-03-models-zoo-07-gr-series) | GR-1/GR-2/GR-3 |
| 08 | [ALOHA & ACT](./vla-03-models-zoo-08-aloha-act) | 灵巧手 + 行为策略 |
| 09 | [Helix](./vla-03-models-zoo-09-helix) | 具身 AI 的多任务 VLA |
| 10 | [GROOT N1](./vla-03-models-zoo-10-groot-n1) | Figure 的端到端 VLA |
| 11 | [Gemini Robotics](./vla-03-models-zoo-11-gemini-robotics) | Gemini 机器人大模型 |
| 12 | [其他模型](./vla-03-models-zoo-12-other-models) | 国内外 VLA 工作汇总 |
| 13 | [演进地图](./vla-03-models-zoo-13-evolution-map) | VLA 发展脉络全景 |

### 04 训练（Training）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [数据集](./vla-04-training-01-datasets) | 机器人数据与开放数据集 |
| 02 | [数据采集](./vla-04-training-02-data-collection) | 远程操纵、演示、仿真 |
| 03 | [预训练与微调](./vla-04-training-03-pretrain-finetune) | 训练策略与课程学习 |
| 04 | [跨具身](./vla-04-training-04-cross-embodiment) | 跨机器人形态泛化 |
| 05 | [Sim-to-Real](./vla-04-training-05-sim-to-real) | 仿真到真实迁移 |
| 06 | [高效微调](./vla-04-training-06-efficient-finetuning) | LoRA、Adapter 等 |
| 07 | [工具链](./vla-04-training-07-toolchains) | 训练框架与工具链 |

### 05 机器人应用（Robotics）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [桌面操作](./vla-05-robotics-01-tabletop) | 桌面级物体操纵 |
| 02 | [灵巧手](./vla-05-robotics-02-dexterous) | 精细操作与手部控制 |
| 03 | [移动机器人](./vla-05-robotics-03-mobile) | 移动操作与导航 |
| 04 | [人形机器人](./vla-05-robotics-04-humanoid) | 人形 + VLA |
| 05 | [长时序任务](./vla-05-robotics-05-long-horizon) | 多步规划与任务执行 |

### 06 自动驾驶（Autonomous Driving）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [VLA 驾驶范式](./vla-06-autonomous-driving-01-vla-driving-paradigm) | VLA 在自动驾驶中的应用 |
| 02 | [DriveVLM](./vla-06-autonomous-driving-02-drivevlm) | 视觉-语言模型驾驶 |
| 03 | [其他驾驶 VLA](./vla-06-autonomous-driving-03-other-driving-vla) | EMMA、LAV 等 |
| 04 | [场景理解](./vla-06-autonomous-driving-04-scene-understanding) | VLA 场景感知能力 |
| 05 | [局限性](./vla-06-autonomous-driving-05-limitations) | 安全、泛化、效率挑战 |

### 07 基础模型（Foundation Models）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [LLM 推理能力](./vla-07-foundation-models-01-llm-reasoning) | LLM 作为机器人“大脑” |
| 02 | [VLM 主干](./vla-07-foundation-models-02-vlm-backbone) | VLA 视觉-语言主干 |
| 03 | [扩散动作](./vla-07-foundation-models-03-diffusion-action) | 扩散策略 + VLA |
| 04 | [具身 Agent](./vla-07-foundation-models-04-embodied-agent) | 视觉-语言-动作 Agent |
| 05 | [世界模型](./vla-07-foundation-models-05-world-model) | VLA + 世界模型 |
| 06 | [未来展望](./vla-07-foundation-models-06-future) | 发展方向与挑战 |

### 08 附录（Appendix）

| 序号 | 文章 | 说明 |
|------|------|------|
| — | [索引总览](./vla-08-appendix-01-index) | 系列内容索引 |
| — | [术语表](./vla-08-appendix-02-glossary) | 核心术语解释 |
| — | [阅读清单](./vla-08-appendix-03-reading-list) | 必读论文列表 |
