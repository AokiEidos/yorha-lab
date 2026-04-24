---
title: "扩散模型系列"
hidden: true
---

# 扩散模型系列

> 扩散模型（Diffusion Models）通过逐步加噪和去噪学习数据分布，是当前生成式 AI 的主流范式。本系列从直觉到数学、从基础理论到高级应用，系统梳理扩散模型的原理与实践。

---

## 系列目录


### 基础（Fundamentals）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Core Intuition](diffusion/01-fundamentals/01-core-intuition) | 墨水扩散类比、加噪→去噪直觉、GAN/VAE/Flow 对比 |
| 02 | [Forward Process](diffusion/01-fundamentals/02-forward-process) | 马尔可夫链、逐步加噪公式、封闭解 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ |
| 03 | [Reverse Process & Training](diffusion/01-fundamentals/03-reverse-process) | 贝叶斯后验推导、三种参数化（$\epsilon/x_0/v$-预测）对比 |
| 04 | [Sampling Algorithms](diffusion/01-fundamentals/04-sampling-algorithms) | DDPM 采样、祖先采样 vs 确定性采样、$\sigma_t$ 选择 |
| 05 | [Noise Schedule](diffusion/01-fundamentals/05-noise-schedule) | 线性 vs 余弦调度、SNR 统一视角、Min-SNR 加权 |
| 06 | [Score Matching](diffusion/01-fundamentals/06-score-matching) | 得分函数几何直觉、DSM 推导、与 DDPM 等价性 |
| 07 | [SDE & ODE Unified View](diffusion/01-fundamentals/07-sde-ode) | VP-SDE/VE-SDE/sub-VP、概率流 ODE、Euler-Maruyama |


### Models Zoo

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [DDPM](diffusion/02-models-zoo/01-ddpm) | U-Net 架构（ResBlock+Attention+时间步嵌入） |
| 02 | [DDIM](diffusion/02-models-zoo/02-ddim) | 非马尔可夫前向过程、$\eta$ 参数控制随机性、DDIM Inversion |
| 03 | [Score-Based Models](diffusion/02-models-zoo/03-score-based) | NCSN 多尺度设计、退火朗之万、Song et al. SDE 统一 |
| 04 | [Latent Diffusion](diffusion/02-models-zoo/04-latent-diffusion) | 像素→潜空间压缩比、VAE+条件 U-Net+文本编码器 |
| 05 | [Consistency Models](diffusion/02-models-zoo/05-consistency-models) | 自一致性约束公式、CD 和 CT 训练伪代码、LCM |
| 06 | [Flow Matching](diffusion/02-models-zoo/06-flow-matching) | CNF 框架、CFM 训练目标、Rectified Flow + Reflow |
| 07 | [Evolution Map](diffusion/02-models-zoo/07-evolution-map) | DDPM→DDIM→Score SDE→LDM→SD→FM 技术脉络 |


### 条件生成（Conditional Generation）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Conditional Generation Overview](diffusion/03-conditional-generation/01-overview) | 贝叶斯分解→两大范式、3 大类 12 种方法 |
| 02 | [Classifier Guidance](diffusion/03-conditional-generation/02-classifier-guidance) | 贝叶斯→得分函数→噪声预测、ADM 架构（554M） |
| 03 | [Classifier-Free Guidance](diffusion/03-conditional-generation/03-cfg) | 隐式分类器等价性、Guidance Scale 效果表、CFG++ |
| 04 | [ControlNet](diffusion/03-conditional-generation/04-controlnet) | 零卷积梯度流、各层通道维度 320/640/1280 |
| 05 | [IP-Adapter](diffusion/03-conditional-generation/05-ip-adapter) | 解耦交叉注意力、Scale $\lambda$ 调优、FaceID |
| 06 | [Adapters & Fine-tuning](diffusion/03-conditional-generation/06-adapters) | LoRA 数学（$W'=W+BA$）、DreamBooth 先验保持 |
| 07 | [Method Comparison](diffusion/03-conditional-generation/07-comparison) | 12 种方法 7 维对比表、选型决策树 |


### 加速与部署（Acceleration & Deployment）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Acceleration & Deployment Overview](diffusion/04-acceleration-deployment/01-overview) | 蒸馏/量化/求解器/推理优化四大方向 |
| 02 | [ODE Solvers](diffusion/04-acceleration-deployment/02-ode-solvers) | DPM-Solver/UniPC/Heun、多步 vs 单步 trade-off |
| 03 | [Distillation](diffusion/04-acceleration-deployment/03-distillation) | 渐进蒸馏、LCM（Latent Consistency Model） |
| 04 | [Quantization](diffusion/04-acceleration-deployment/04-quantization) | INT8/INT4/FP8、GPTQ vs AWQ |
| 05 | [Inference Optimization](diffusion/04-acceleration-deployment/05-inference-optimization) | TensorRT/FlashAttention/DeepCache |


### LLM 时代（LLM Era）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Text-to-Image](diffusion/05-llm-era/01-text-to-image) | DALL-E/SD/Imagen 演化、CLIP vs LLM（T5/LLaMA） |
| 02 | [DiT](diffusion/05-llm-era/02-dit) | Diffusion Transformer（MM-DiT）、adaLN、Scaling Laws |
| 03 | [Text-to-Video](diffusion/05-llm-era/03-text-to-video) | 时空注意力机制、Video DiT vs GAN/VAE |
| 04 | [Image Editing](diffusion/05-llm-era/04-image-editing) | SDEdit/Inpainting、Prompt-to-Prompt |
| 05 | [Super Resolution](diffusion/05-llm-era/05-super-resolution) | ESRGAN/Real-ESRGAN/SDSR、4× 超分 pipeline |
| 06 | [3D Generation](diffusion/05-llm-era/06-3d-generation) | NeRF/Gaussian Splatting/Point-E |
| 07 | [Multimodal LLMs](diffusion/05-llm-era/07-multimodal) | Flamingo/LLaVA/GPT-4V |


### 自动驾驶（Autonomous Driving）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [AD Overview](diffusion/06-autonomous-driving/01-overview) | 三层架构（感知/规划/控制）、Diffusion 在 AD 中的应用 |
| 02 | [Scene Generation](diffusion/06-autonomous-driving/02-scene-generation) | BEVGen/MagicDrive/DriveDreamer、3D bbox/HD Map |
| 03 | [Trajectory Prediction](diffusion/06-autonomous-driving/03-trajectory-prediction) | 多模态轨迹预测、MID/CTG/Diffusion-ES |
| 04 | [Motion Planning](diffusion/06-autonomous-driving/04-motion-planning) | Diffuser/Decision Diffuser、Classifier Guidance |
| 05 | [Data Augmentation](diffusion/06-autonomous-driving/05-data-augmentation) | LiDAR 点云增强、Sim-to-Real 定量分析 |
| 06 | [World Models](diffusion/06-autonomous-driving/06-world-model) | GAIA-1（9B 三路编码 Video Diffusion） |


### 附录（Appendix）

| 序号 | 文章 | 说明 |
|------|------|------|
| 01 | [Table of Contents](diffusion/07-appendix/01-index) | 全系列索引、阅读路径推荐 |
| 02 | [Glossary](diffusion/07-appendix/02-glossary) | 核心术语中英对照 |
| 03 | [Reading List](diffusion/07-appendix/03-reading-list) | 推荐论文与阅读顺序 |
