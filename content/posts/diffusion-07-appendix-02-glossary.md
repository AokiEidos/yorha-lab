---
title: "术语表 (Glossary)"
date: 2026-04-20T17:38:38.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 术语表 (Glossary)

> 按首字母排序，包含全系列出现的专业术语。格式：**术语（English）** — 解释

---

## A
- **adaGN（Adaptive Group Normalization）** — ADM 中用时间步和类别信息调制 GroupNorm 的 scale/shift 参数
- **adaLN-Zero（Adaptive Layer Norm - Zero）** — DiT 中的条件注入机制，输出 6 个调制参数（γ₁,β₁,α₁,γ₂,β₂,α₂），gate α 初始化为零确保训练初期恒等映射
- **ADD（Adversarial Diffusion Distillation，对抗扩散蒸馏）** — 结合蒸馏损失+DINOv2 特征判别器的加速方法，SDXL-Turbo 实现单步生成
- **ADM（Ablated Diffusion Model）** — Dhariwal & Nichol 2021 提出的 U-Net 架构（~554M 参数），首次让扩散模型超越 GAN

## B
- **BEV（Bird's Eye View，鸟瞰图）** — 从上方俯视的 2D 表示，自动驾驶中的核心空间表示
- **布朗运动 / 维纳过程（Wiener Process）** — 连续时间白噪声的数学模型，SDE 的随机驱动项 $dw$

## C
- **CFG（Classifier-Free Guidance，无分类器引导）** — 训练时以 10-20% 概率丢弃条件，推理时 $\hat\epsilon = \epsilon(\varnothing) + w[\epsilon(c) - \epsilon(\varnothing)]$，当前所有文生图模型标配
- **CFG++** — Chung et al. 2024，在 $x_0$ 预测空间做引导（替代 $\epsilon$ 空间），减少高 $w$ 时的过度饱和
- **CNF（Continuous Normalizing Flow，连续标准化流）** — 用 ODE $dx/dt = v_\theta(x,t)$ 定义的连续可逆变换
- **条件流匹配（Conditional Flow Matching, CFM）** — 用线性插值 $x_t=(1-t)x_0+t\epsilon$ 定义条件路径，训练匹配速度 $v=\epsilon-x_0$
- **ControlNet** — Zhang et al. 2023，复制 U-Net 编码器作为控制分支，通过零卷积连接，~361M 参数，支持边缘/深度/姿态等条件
- **ControlNet-XS** — ControlNet 的轻量变体，减少 85% 参数
- **Cross-Attention（交叉注意力）** — Q 来自图像特征，K/V 来自条件（文本/图像），LDM 中文本注入的核心机制
- **CUDA Graph** — NVIDIA GPU 优化技术，消除 kernel launch 开销，适合固定计算图的推理加速
- **一致性模型（Consistency Models）** — Song et al. 2023，利用 ODE 轨迹自一致性 $f(x_t,t)=f(x_{t'},t')$ 实现 1-2 步生成

## D
- **DDIM（Denoising Diffusion Implicit Models）** — 非马尔可夫采样，$\eta=0$ 确定性、$\eta=1$ 等价 DDPM，可子序列加速
- **DDIM Inversion（DDIM 反演）** — 利用 DDIM 确定性可逆性将真实图像映射回噪声空间，图像编辑的基础
- **DDPM（Denoising Diffusion Probabilistic Models）** — Ho et al. 2020，简化训练目标 $\|\epsilon-\epsilon_\theta\|^2$，扩散模型实用化里程碑
- **DDRM / DPS / DDNM** — 无训练后验采样方法，利用预训练扩散模型+已知退化算子做图像修复
- **DeepCache** — Ma et al. 2024，利用相邻步深层特征相似度（余弦>0.99），每 N 步完整计算，中间步复用缓存，~2× 加速
- **得分函数（Score Function）** — $\nabla_x \log p(x)$，指向数据密度增大的方向
- **得分匹配（Score Matching）** — 训练网络估计得分函数，DSM 等价于预测噪声
- **DiT（Diffusion Transformer）** — Peebles & Xie 2023，用 Transformer 替代 U-Net，遵循 Scaling Laws（33M→675M，FID 68.4→2.27）
- **DMD2（Distribution Matching Distillation）** — 分布匹配蒸馏，比 ADD 更稳定的对抗蒸馏方案
- **DPM-Solver** — Lu et al. 2022，利用扩散 ODE 的半线性结构设计的指数积分器，DPM-Solver-2/3 为多步法
- **DPM-Solver++** — 数据预测（$x_0$-参数化）版本，CFG 下更稳定
- **DreamBooth** — Ruiz et al. 2023，3-5 张图微调学习特定概念，先验保持损失防遗忘
- **DreamFusion** — Poole et al. 2022，首个 SDS 方法，用 2D 扩散梯度优化 NeRF 3D 表示

## E
- **ELBO / VLB** — 数据对数似然的变分下界，扩散模型的理论训练目标
- **EMA（Exponential Moving Average）** — 参数指数移动平均，$\theta_{ema} \leftarrow 0.9999\theta_{ema}+0.0001\theta$，提升采样质量
- **指数积分器（Exponential Integrator）** — DPM-Solver 的核心，精确处理半线性 ODE 的线性部分

## F
- **FID（Fréchet Inception Distance）** — 在 Inception 特征空间比较生成/真实分布的距离，越低越好
- **FiLM（Feature-wise Linear Modulation）** — 用条件特征的 $\gamma/\beta$ 调制目标特征，RT-1 和 Octo 使用
- **Flash Attention** — Dao et al. 2022，分块计算注意力避免 $N^2$ 矩阵写入 HBM，内存 $O(N^2)→O(N)$，SD 中 2-4× 加速
- **Flow Matching（流匹配）** — 学习速度场 $v_\theta$ 连接噪声和数据，训练 $\|v_\theta-(ε-x_0)\|^2$，比 Diffusion 更简洁
- **前向过程（Forward Process）** — $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$
- **FVD（Fréchet Video Distance）** — 视频生成质量指标，在 I3D 特征空间比较

## G
- **GAIA-1** — Wayve 2023，9B 参数驾驶世界模型（视频+动作+文本条件）
- **GAN（Generative Adversarial Network）** — 生成器+判别器对抗训练的生成模型
- **高斯噪声（Gaussian Noise）** — $\epsilon \sim \mathcal{N}(0,I)$
- **Guidance Scale（引导强度）** — CFG 中的 $w$，SD 常用 7-8.5，过大导致过度饱和

## H
- **Heun 方法** — 二阶 ODE 求解器（预测-校正法），每步 2 次 NFE，精度 $O(\Delta t^2)$

## I
- **InstructPix2Pix** — Brooks et al. 2023，指令式图像编辑（GPT-3 生成指令对 + P2P 生成图像对训练）
- **IP-Adapter** — Ye et al. 2023，解耦交叉注意力注入图像条件，~22M 参数
- **IS（Inception Score）** — 生成质量+多样性指标，越高越好

## J
- **Joint Attention（联合注意力）** — MM-DiT 中图像和文本 Token 各自 QKV 投影后 concat 做联合注意力

## K
- **Karras 噪声调度** — Karras et al. 2022，$\sigma_i = (\sigma_{max}^{1/\rho} + i/(N-1)(\sigma_{min}^{1/\rho}-\sigma_{max}^{1/\rho}))^\rho$，$\rho=7$
- **KL 散度（KL Divergence）** — $D_{KL}(p\|q) = \mathbb{E}_p[\log(p/q)]$

## L
- **LAION-5B** — 50 亿图文对的开源数据集，SD 系列的训练数据基础
- **朗之万动力学（Langevin Dynamics）** — $x_{k+1} = x_k + \delta/2 \nabla_x \log p(x_k) + \sqrt{\delta}z_k$
- **Latent Diffusion Model (LDM)** — Rombach et al. 2022，VAE 压缩→潜空间扩散→VAE 解码，~48× 计算节省
- **LCM（Latent Consistency Model）** — Luo et al. 2023，在潜空间做一致性蒸馏，增广 PF-ODE 集成 CFG
- **LCM-LoRA** — LCM 的 LoRA 版本（~67M 参数），可与社区风格 LoRA 叠加
- **长尾场景（Long-tail Scenarios）** — 罕见但安全关键的场景（行人闯入<0.5%、暴雨<1%）
- **LoRA（Low-Rank Adaptation）** — $W'=W+BA$，$r \ll d$，SD 中 rank=8 约 ~4M 参数 (<0.5%)
- **LPIPS（Learned Perceptual Image Patch Similarity）** — 感知相似度指标，越低越好

## M
- **马尔可夫链（Markov Chain）** — $q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1})$
- **Min-SNR 加权** — $w(t) = \min(\text{SNR}(t), \gamma)$，避免高 SNR 时间步主导训练
- **MM-DiT（Multimodal Diffusion Transformer）** — SD3 架构，图像+文本 Token 各自独立 QKV 后联合注意力
- **模式崩塌（Mode Collapse）** — GAN 中生成器只学会少数模式的退化现象
- **MSE（Mean Squared Error）** — 扩散模型的核心训练损失 $\|\epsilon-\epsilon_\theta\|^2$

## N
- **NCSN（Noise Conditional Score Network）** — Song & Ermon 2019，多噪声水平得分网络
- **NF4（NormalFloat4）** — QLoRA 提出的 4-bit 格式，量化级别按正态分布间隔
- **NFE（Network Function Evaluation）** — 一次网络前向传播，衡量采样计算量的标准单位
- **噪声调度（Noise Schedule）** — 定义 $\beta_t$ 随时间步的变化策略（线性/余弦/Karras）
- **Null-text Inversion** — Mokady et al. 2023，通过优化无条件嵌入 $\varnothing_t$ 消除 DDIM Inversion 误差

## O
- **ODE（Ordinary Differential Equation）** — $dx/dt = f(x,t)$，确定性微分方程
- **偏移噪声（Offset Noise）** — Lin et al. 2023，额外添加低频噪声确保模型能生成纯色图像
- **最优传输（Optimal Transport, OT）** — 以最小代价映射分布，Flow Matching 的直线路径近似 OT 路径

## P
- **配分函数（Partition Function）** — $Z = \int p_{unnorm}(x)dx$，得分函数绕过了计算 Z 的需要
- **Prompt-to-Prompt (P2P)** — Hertz et al. 2022，操控 Cross-Attention map 实现精细图像编辑
- **PTQ（Post-Training Quantization）** — 训练后量化，需要校准数据确定量化参数
- **Pseudo-Huber 损失** — iCT 使用，$\rho(x) = \sqrt{x^2+c^2}-c$，比 MSE 对离群值更鲁棒
- **概率流 ODE（Probability Flow ODE）** — $dx/dt = f(x,t) - g^2(t)/2 \cdot \nabla_x \log p_t(x)$，与 SDE 同边际分布的确定性 ODE

## Q
- **Q-Diffusion** — Li et al. 2023，分步校准的扩散模型 PTQ 方法
- **QAT（Quantization-Aware Training）** — 训练中用 STE（直通估计器）模拟量化误差

## R
- **Range Image（距离图）** — LiDAR 3D 点云的 2D 球面投影表示，用于 2D Diffusion 生成点云
- **Rectified Flow（校正流）** — 通过 Reflow 迭代拉直路径，SD3 使用
- **Reflow** — 用当前模型生成 $(x_0,x_1)$ 配对→重新训练→路径更直，2-3 次迭代后接近 1 步生成
- **反向过程（Reverse Process）** — $p_\theta(x_{t-1}|x_t)$，从噪声逐步去噪
- **重参数化技巧（Reparameterization Trick）** — $x = \mu + \sigma\epsilon$，$\epsilon \sim \mathcal{N}(0,I)$

## S
- **SDE（Stochastic Differential Equation）** — $dx = f(x,t)dt + g(t)dw$，含随机项
- **SDEdit** — Meng et al. 2021，加噪到 $t_0$ 再用新条件去噪的编辑方法
- **SDS（Score Distillation Sampling）** — $\nabla_\theta L = E[w(t)(\epsilon_\phi-\epsilon)\partial x/\partial\theta]$，用 2D 扩散梯度优化 3D
- **半线性 ODE（Semi-linear ODE）** — $dx/d\lambda = A(\lambda)x + B(\lambda)\epsilon_\theta$，DPM-Solver 利用此结构
- **SigLIP** — Google 改进 CLIP，Sigmoid 损失替代 Softmax，不依赖大 batch
- **SNR（Signal-to-Noise Ratio）** — $\bar\alpha_t/(1-\bar\alpha_t)$，统一描述噪声水平
- **SR3** — Saharia et al. 2022，通道拼接条件的 Diffusion 超分辨率，人类混淆率 50.2%
- **世界模型（World Model）** — 预测 $\hat{s}_{t+1}=f_\theta(s_t,a_t)$ 的环境动态模型

## T
- **T5-XXL** — Google 的 4.7B 纯语言编码器，4096 维/256 token，Imagen/SD3/FLUX 使用
- **Textual Inversion** — Gal et al. 2022，冻结模型，学习新词嵌入向量 $v_*$（~1K 参数）
- **时间闪烁（Temporal Flickering）** — 视频帧间不一致导致的画面闪烁
- **时空注意力（Spatial-Temporal Attention）** — 空间帧内 + 时间帧间分离注意力，视频 Diffusion 核心
- **ToMe（Token Merging）** — Bolya 2023，合并相似 Token 缩短序列，SD 中 30-50% 合并 ~1.5× 加速
- **TokenLearner** — RT-1 中将 81 个空间 Token 压缩为 8 个的可学习模块

## U
- **UniPC（Unified Predictor-Corrector）** — Zhao et al. 2023，统一多步法+预测-校正法框架，无需额外 NFE

## V
- **VAE（Variational Autoencoder）** — 编码器-解码器生成模型，LDM 中用于像素↔潜空间转换
- **VP-SDE（Variance Preserving SDE）** — DDPM 对应的连续 SDE：$dx = -\beta(t)/2 \cdot x \, dt + \sqrt{\beta(t)} \, dw$
- **VE-SDE（Variance Exploding SDE）** — NCSN 对应的连续 SDE：$dx = \sqrt{d[\sigma^2(t)]/dt} \, dw$
- **VSD（Variational Score Distillation）** — ProlificDreamer，用同步训练 LoRA 替代随机噪声作为基线，3D 质量大幅提升
- **$v$-预测（Velocity Prediction）** — $v_t = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_0$，SD 2.x 使用

## X
- **xFormers** — Meta 高效 Transformer 库，memory-efficient attention

## Z
- **Zero Terminal SNR** — Lin et al. 2023，确保 $\bar\alpha_T=0$（SNR=0），避免训练/推理分布不匹配
- **零卷积（Zero Convolution）** — ControlNet 中权重/偏置初始化为零的 1×1 卷积，渐进式引入控制信号
- **零初始化膨胀（Zero-Init Inflation）** — 视频 Diffusion 中将新插入的时间注意力层初始化为零，保护预训练图像能力
