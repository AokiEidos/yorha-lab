---
title: "VLA 从入门到精通系列"
hidden: true
---

# VLA 从入门到精通系列

> VLA（Vision-Language-Action Models）是将视觉感知、语言理解与动作执行统一在端到端神经网络中的模型，是当前机器人与自动驾驶领域最活跃的研究方向。本系列面向有深度学习基础的读者，系统梳理 VLA 的理论、架构、训练与应用。

---

## 系列目录


### 基础（Foundations）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [VLM Review](vla/01-foundations/01-vlm-review) | CLIP/SigLIP/DINOv2、VLM 三组件（视觉编码器+投影层+LLM） |
| 02 | [Embodied Perception](vla/01-foundations/02-embodied-perception) | RGB 相机、深度传感、点云处理、多模态融合架构 |
| 03 | [Observation & Action Space](vla/01-foundations/03-observation-action-space) | Franka 7 关节规格、笛卡尔 vs 关节空间、256-bin 量化、Action Chunking |
| 04 | [Imitation Learning](vla/01-foundations/04-imitation-learning) | BC 损失推导、DAgger 伪代码、IBC 能量模型 |
| 05 | [Reinforcement Learning](vla/01-foundations/05-reinforcement-learning) | Bellman 方程、REINFORCE/PPO、Reward Hacking |


### 架构（Architecture）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [From VLM to VLA](vla/02-architecture/01-vlm-to-vla) | 5 层知识迁移、RT-2 涌现验证、VLM→VLA 代码示例 |
| 02 | [Action Representation](vla/02-architecture/02-action-representation) | 离散 Token vs 连续回归 vs Diffusion/FM 三路线 |
| 03 | [Action Head Design](vla/02-architecture/03-action-head) | 四种解码头 PyTorch 伪代码、Action Chunking 时序集成 |
| 04 | [Multimodal Fusion](vla/02-architecture/04-multimodal-fusion) | 早期 vs 晚期融合、FiLM/Cross-Attention/Perceiver Resampler |
| 05 | [Architecture Comparison](vla/02-architecture/05-architecture-comparison) | 10 模型全维度对比、4 种设计模式、决策树选型指南 |


### Models Zoo

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [SayCan](vla/03-models-zoo/01-saycan) | Affordance 评分、551 技能库 |
| 02 | [RT-1](vla/03-models-zoo/02-rt1) | EfficientNet→FiLM→TokenLearner→Transformer 架构 |
| 03 | [RT-2](vla/03-models-zoo/03-rt2) | PaLI-X/PaLM-E co-tuning、动作 Token 化、涌现能力 |
| 04 | [Octo](vla/03-models-zoo/04-octo) | Octo-S/B 架构、Readout Token、跨具身体微调 |
| 05 | [OpenVLA](vla/03-models-zoo/05-openvla) | Prismatic 双编码器、LoRA 微调、HuggingFace 推理 |
| 06 | [π₀](vla/03-models-zoo/06-pi-zero) | Flow Matching 动作头、~500M Action Expert、两阶段训练 |
| 07 | [GR Series](vla/03-models-zoo/07-gr-series) | MAE 视频预训练、联合损失公式、GR-2 3B 参数 |
| 08 | [ALOHA & ACT](vla/03-models-zoo/08-aloha-act) | CVAE 架构、时间集成、ALOHA 2 硬件改进 |
| 09 | [Helix](vla/03-models-zoo/09-helix) | System 1/2 双系统、GoalVector 通信协议、三级安全机制 |
| 10 | [GR00T N1](vla/03-models-zoo/10-groot-n1) | Diffusion Transformer、Isaac Sim 数据生成、NVIDIA Jetson Thor |
| 11 | [Gemini Robotics](vla/03-models-zoo/11-gemini-robotics) | Gemini 2.0 机器人控制、ER+NT 双系统 |
| 12 | [Other Models](vla/03-models-zoo/12-other-models) | LAPA/TidyBot/RoboCat/Mobile ALOHA 等 |
| 13 | [Evolution Map](vla/03-models-zoo/13-evolution-map) | VLA 模型演化全景图与时间线 |


### 训练（Training）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Datasets](vla/04-training/01-datasets) | Open X-Embodiment（55 机器人 100 万条） |
| 02 | [Data Collection](vla/04-training/02-data-collection) | 遥操作（ALOHA/UMI）、mocap 动捕、真机 vs 仿真 |
| 03 | [Pretrain & Finetune](vla/04-training/03-pretrain-finetune) | VLM 预训练+机器人微调两阶段、Co-finetuning |
| 04 | [Cross-Embodiment](vla/04-training/04-cross-embodiment) | 具身体差异分析、对抗域随机化、跨具身体 Adapter |
| 05 | [Sim-to-Real](vla/04-training/05-sim-to-real) | 域随机化、Isaac Gym PPO、sim-to-real 转移率 |
| 06 | [Efficient Finetuning](vla/04-training/06-efficient-finetuning) | LoRA/QLORA、权重合并、RT-2/Octo/OpenVLA |
| 07 | [Open Source Toolchains](vla/04-training/07-toolchains) | LeRobot/ALOHA/ACT/Octo 开源工具链 |


### 机器人（Robotics）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Tabletop Manipulation](vla/05-robotics/01-tabletop) | 抓取/放置/插入、6-DoF vs 7-DoF 机械臂 |
| 02 | [Dexterous Manipulation](vla/05-robotics/02-dexterous) | 灵巧手硬件（Allegro/Leap Hand）、多指抓取规划 |
| 03 | [Mobile Manipulation](vla/05-robotics/03-mobile) | SayCan/UMI-Desktop、导航+操作联合规划 |
| 04 | [Humanoid Control](vla/05-robotics/04-humanoid) | 双足行走+上肢协调、GR-2 人形全身控制 |
| 05 | [Long-Horizon Planning](vla/05-robotics/05-long-horizon) | 任务分解（LLM Planner）、Hierarchical RL |


### 自动驾驶（Autonomous Driving）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [VLA Driving Paradigm](vla/06-autonomous-driving/01-vla-driving-paradigm) | 端到端 vs 模块化栈、闭环 vs 开环评估 |
| 02 | [DriveVLM](vla/06-autonomous-driving/02-drivevlm) | CoT 驾驶推理、双系统（VLM+规则） |
| 03 | [Other Driving VLAs](vla/06-autonomous-driving/03-other-driving-vla) | LMDrive/GPT-Driver/ThinkDrive |
| 04 | [Scene Understanding](vla/06-autonomous-driving/04-scene-understanding) | 占用栅格、BEVFormer/MapTR |
| 05 | [Limitations & Outlook](vla/06-autonomous-driving/05-limitations) | 开环局限、RSS 安全框架、反事实测试 |


### Foundation Models

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [LLM Reasoning for Robotics](vla/07-foundation-models/01-llm-reasoning) | CoT/ToT/GoT、LLM 作为任务规划器 |
| 02 | [VLM Backbones](vla/07-foundation-models/02-vlm-backbone) | SigLIP/DINOv2/CLIP、分辨率/视野 trade-off |
| 03 | [Diffusion for Action](vla/07-foundation-models/03-diffusion-action) | Diffusion Policy、π₀/Octo 的 Diffusion 动作头 |
| 04 | [Embodied Agent](vla/07-foundation-models/04-embodied-agent) | Agent 框架（ReAct/NSat）、VLA 作为执行器 |
| 05 | [World Models](vla/07-foundation-models/05-world-model) | GAIA-1/DriveDreamer-2、Uncertainty 量化 |
| 06 | [VLA Future Outlook](vla/07-foundation-models/06-future) | Scaling Laws、触觉/力控多模态拓展 |


### 附录（Appendix）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Table of Contents](vla/08-appendix/01-index) | 全系列索引、阅读路径推荐 |
| 02 | [Glossary](vla/08-appendix/02-glossary) | 核心术语中英对照 |
| 03 | [Reading List](vla/08-appendix/03-reading-list) | 推荐论文与阅读顺序 |
