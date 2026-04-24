---
title: "蒸馏方法"
date: 2026-04-20T17:23:25.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 蒸馏方法

> ⚙️ 进阶 | 前置知识：[高阶 ODE 求解器](./02-ode-solvers.md)，[Consistency Models](../02-models-zoo/05-consistency-models.md)

## 为什么需要蒸馏 🔰

即使最先进的 ODE 求解器（DPM-Solver++、UniPC）在 10 步以下时质量也会明显下降。根本原因在于：ODE 求解器是**通用数值方法**，不了解特定模型的去噪轨迹结构。

**蒸馏（Distillation）** 采用根本不同的策略：训练一个**学生模型（Student Model）**，直接学习在更少步数中复现**教师模型（Teacher Model）**（多步采样）的输出。蒸馏后的模型可以在 1-4 步内生成高质量图像。

### 蒸馏方法的分类

```
              扩散模型蒸馏方法
                   │
       ┌───────────┼───────────┐
       │           │           │
   轨迹蒸馏      一致性蒸馏    对抗蒸馏
   (Trajectory)  (Consistency) (Adversarial)
       │           │           │
  Progressive    LCM         ADD/SDXL-Turbo
  Distillation   LCM-LoRA    DMD2
  Guided Distill  CTM        LADD
```

- **轨迹蒸馏**：学生学习复现教师多步采样的中间轨迹或最终结果
- **一致性蒸馏**：学生学习将 ODE 轨迹上任意点映射到同一终点
- **对抗蒸馏**：引入判别器，用对抗训练保证生成质量

## 渐进式蒸馏（Progressive Distillation） ⚙️

Salimans & Ho (2022) 提出的经典方法，核心思想是**迭代减半步数**。

### 算法流程

```
第 0 轮: 教师 = 预训练模型 (1024 步)
         训练学生：1 步 ≈ 教师 2 步
         → 学生成为 512 步模型

第 1 轮: 教师 = 上一轮学生 (512 步)
         训练新学生：1 步 ≈ 教师 2 步
         → 学生成为 256 步模型

         ...（重复）

第 N 轮: 教师 = 上一轮学生 (8 步)
         训练新学生：1 步 ≈ 教师 2 步
         → 学生成为 4 步模型
```

### 单轮蒸馏的数学目标

给定教师模型 $\epsilon_T$ 和学生模型 $\epsilon_S$（初始化为教师权重），在时间区间 $[t, t']$ 上：

教师执行 2 步：

$$x_{\text{mid}} = \text{DDIM-step}(\epsilon_T, x_t, t, t_{\text{mid}})$$
$$x_{\text{target}} = \text{DDIM-step}(\epsilon_T, x_{\text{mid}}, t_{\text{mid}}, t')$$

学生执行 1 步并匹配教师的 2 步结果：

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[\| \epsilon_S(x_t, t) - \hat{\epsilon}_{\text{target}} \|_2^2 \right]$$

其中 $\hat{\epsilon}_{\text{target}}$ 是从教师的 $(x_t, x_{\text{target}})$ 对反推出的等效噪声预测：

$$\hat{\epsilon}_{\text{target}} = \frac{x_t - \alpha_{t'} x_{\text{target}}}{\sigma_{t'}}$$

### 核心 PyTorch 伪代码

```python
def progressive_distillation_step(student, teacher, x_0, t, t_mid, t_end, noise_schedule):
    """单步渐进式蒸馏训练（学生1步 ≈ 教师2步）"""
    eps = torch.randn_like(x_0)
    x_t = noise_schedule.q_sample(x_0, t, eps)
    
    # 教师两步（无梯度）
    with torch.no_grad():
        eps_1 = teacher(x_t, t)
        x_mid = noise_schedule.ddim_step(x_t, eps_1, t, t_mid)
        eps_2 = teacher(x_mid, t_mid)
        x_target = noise_schedule.ddim_step(x_mid, eps_2, t_mid, t_end)
    
    # 反推目标噪声，学生一步匹配
    eps_target = (x_t - noise_schedule.alpha(t_end) * x_target) / noise_schedule.sigma(t_end)
    eps_student = student(x_t, t)
    return F.mse_loss(eps_student, eps_target)
```

### 渐进式蒸馏的训练开销

| 轮次 | 教师步数 → 学生步数 | 训练迭代 | 训练耗时 (8×A100) |
|------|---------------------|---------|-------------------|
| 1 | 1024 → 512 | 50K | ~4h |
| 2 | 512 → 256 | 50K | ~4h |
| ... | ... | ... | ... |
| 8 | 8 → 4 | 50K | ~4h |
| 9 | 4 → 2 | 100K | ~8h |
| 10 | 2 → 1 | 200K | ~16h |
| **总计** | 1024 → 1 | ~650K | **~52h** |

后面的轮次因为质量损失更敏感，通常需要更多迭代。

## 引导蒸馏（Guided Distillation） ⚙️

### CFG 的推理开销问题

在标准推理中，**CFG（Classifier-Free Guidance）** 每步需要两次网络评估：

$$\tilde{\epsilon}(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot [\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)]$$

其中 $w$ 是引导强度（通常 $w = 7.5$）。这意味着 NFE 直接翻倍。

### 引导蒸馏的数学目标

**引导蒸馏（Guided Distillation）** 的目标是将 CFG 的效果"烘焙（Bake）"到模型中：

$$\mathcal{L}_{\text{guided}} = \mathbb{E}_{x_0, \epsilon, t, c} \left[\| \epsilon_S(x_t, t, c) - \tilde{\epsilon}_T(x_t, t, c; w) \|_2^2 \right]$$

教师输出 = CFG 组合后的预测（2 次 NFE），学生输出 = 单次前向传播直接输出等效结果（1 次 NFE）。

效果：推理时不再需要两次前向传播，**速度直接翻倍**（叠加在步数减少之上）。

### 联合蒸馏

实际中常将步数蒸馏和引导蒸馏**联合进行**：

$$\mathcal{L}_{\text{joint}} = \mathbb{E}\left[\| \epsilon_S^{\text{1-step}}(x_t, t, c) - \text{CFG}(\epsilon_T^{\text{2-step}}, x_t, t, c; w) \|_2^2 \right]$$

学生的 1 步（无 CFG） ≈ 教师的 2 步（有 CFG），实现 4 倍 NFE 削减。

## LCM：潜空间一致性模型 🔬

### 核心思想

**LCM（Latent Consistency Model，潜空间一致性模型）**（Luo et al., 2023）将 **Consistency Model** 的框架迁移到潜空间扩散模型（如 Stable Diffusion），实现 1-4 步高质量生成。

### 与一般 Consistency Distillation 的关键区别

| 对比维度 | Consistency Distillation | LCM |
|---------|------------------------|-----|
| 工作空间 | 像素空间 | 潜空间（VAE 编码后） |
| 基础模型 | 从头训练或从 DDPM 蒸馏 | 从预训练 SD 蒸馏 |
| CFG 处理 | 不涉及 | **增广 PF-ODE（Augmented PF-ODE）** |
| 参数化 | $f_\theta(x_t, t) \to x_0$ | 同，但在 latent 空间 |
| 训练数据 | 需要原始数据 | 可用合成数据 |

### 增广概率流 ODE（Augmented PF-ODE）

LCM 的核心技术创新是将 CFG 集成到 ODE 定义中，定义增广概率流：

$$\frac{dz_t}{dt} = f(t)z_t + \frac{g^2(t)}{2\sigma_t}\left[(1+w)\epsilon_\theta(z_t, t, c) - w\epsilon_\theta(z_t, t, \varnothing)\right]$$

这样 CFG 不再是采样时的后处理，而是 ODE 本身的一部分。一致性蒸馏直接在这个增广 ODE 上进行。

### LCM 训练目标

$$\mathcal{L}_{\text{LCM}} = \mathbb{E}_{z_0, c, t, t'}\left[d(f_\theta(z_t, t, c), f_{\theta^-}(\hat{z}_{t'}, t', c))\right]$$

其中：
- $f_\theta$：学生一致性函数（将 $z_t$ 映射到估计的 $z_0$）
- $f_{\theta^-}$：EMA 教师（指数移动平均）
- $\hat{z}_{t'}$：使用 ODE solver（DPM-Solver）从 $z_t$ 求解一步到 $t'$
- $d(\cdot, \cdot)$：距离度量（Huber loss 或 LPIPS）

### LCM 训练伪代码

```python
def lcm_training_step(student, ema_student, teacher_solver, z_0, c, t, noise_schedule, cfg=7.5):
    """LCM 单步训练：一致性映射 + 增广 PF-ODE"""
    t_prev = torch.clamp(t - 20, min=0)  # 跳步参数 k=20
    eps = torch.randn_like(z_0)
    z_t = noise_schedule.q_sample(z_0, t, eps)
    
    # 教师：用 DPM-Solver 做一步增广 ODE 求解（含 CFG）
    with torch.no_grad():
        eps_cfg = teacher_solver.model(z_t, t, None) + cfg * (
            teacher_solver.model(z_t, t, c) - teacher_solver.model(z_t, t, None))
        z_t_prev = teacher_solver.step(z_t, eps_cfg, t, t_prev)
        z_0_pred_ema = ema_student(z_t_prev, t_prev, c)  # EMA 一致性映射
    
    z_0_pred_student = student(z_t, t, c)  # 学生一致性映射
    return F.huber_loss(z_0_pred_student, z_0_pred_ema, delta=1.0)
```

### LCM-LoRA：轻量级一致性适配

**LCM-LoRA** 的核心洞察：一致性蒸馏只需修改模型的少量参数。用 **LoRA（Low-Rank Adaptation）** 只训练约 67M 参数（vs SD 1.5 全部 860M）：

- 训练成本从 8×A100 数天降低到单卡 A100 数小时
- LoRA 权重可以与社区已有的风格 LoRA **叠加使用**
- 切换时只需加载/卸载 LoRA 权重

```python
# LCM-LoRA 使用示例
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 加载风格 LoRA
pipe.load_lora_weights("style-lora-anime", adapter_name="style")
# 叠加 LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
pipe.set_adapters(["style", "lcm"], adapter_weights=[1.0, 1.0])

# 使用 LCM 调度器，4 步生成
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
image = pipe("anime girl", num_inference_steps=4, guidance_scale=1.0).images[0]
```

> 注意：使用 LCM-LoRA 时 `guidance_scale=1.0`（CFG 已烘焙）。

## SDXL-Turbo / ADD：对抗蒸馏 ⚙️

### ADD 架构

**ADD（Adversarial Diffusion Distillation，对抗扩散蒸馏）**（Sauer et al., 2023）结合蒸馏损失和对抗损失实现单步生成。

### 训练目标

ADD 的总损失包含两部分：

$$\mathcal{L}_{\text{ADD}} = \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{distill}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{adv}}$$

**蒸馏损失**（Score Distillation）：

$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{x_0, \epsilon, t, c}\left[\| \epsilon_S(x_t, t, c) - \text{sg}[\epsilon_T^{\text{DPM-multi}}(x_t, t, c)] \|_2^2 \right]$$

其中教师使用 DPM-Solver 多步采样的结果作为目标。

**对抗损失**（Adversarial Loss）：

$$\mathcal{L}_{\text{adv}} = \mathbb{E}_{x_0}\left[-\log D_\phi(\hat{x}_0)\right] + \mathbb{E}_{x_{\text{real}}}\left[-\log(1 - D_\phi(x_{\text{real}}))\right]$$

判别器 $D_\phi$ 使用预训练的视觉特征（如 DINOv2 特征），不仅判断全局真伪，还在多个尺度上做判别。

### ADD 的关键设计

| 设计选择 | 说明 | 原因 |
|---------|------|------|
| 预训练特征判别器 | 用 DINOv2/CLIP 特征而非原始像素 | 避免 GAN 的模式坍塌 |
| 多尺度判别 | 在特征图的多个层级判别 | 捕获局部和全局质量 |
| 渐进训练 | 先蒸馏为主，逐渐增加对抗权重 | 稳定训练 |
| 噪声注入 | 学生输入带少量噪声 | 防止过拟合单一噪声模式 |

### SDXL-Turbo 效果

| 配置 | 步数 | FID (COCO 30K) | CLIP Score | 延迟 (A100) |
|------|------|---------------|------------|------------|
| SDXL 50步 + CFG | 50 | 23.5 | 0.310 | ~6.5s |
| SDXL-Turbo 4步 | 4 | 24.8 | 0.305 | ~0.52s |
| SDXL-Turbo 1步 | 1 | 28.1 | 0.295 | ~0.13s |

单步质量已经接近多步采样，但细节纹理和一致性仍有差距。

## DMD2：分布匹配蒸馏 🔬

### 核心思想

**DMD2（Distribution Matching Distillation 2）**（Yin et al., 2024）从分布匹配的视角进行蒸馏，不是匹配单个样本的轨迹，而是匹配整体分布。

### 训练目标

DMD2 使用**回归损失 + 分布匹配损失**：

$$\mathcal{L}_{\text{DMD2}} = \mathcal{L}_{\text{regression}} + \lambda \cdot \mathcal{L}_{\text{distribution}}$$

**分布匹配损失**基于 score matching：

$$\mathcal{L}_{\text{distribution}} = \mathbb{E}_{t, \hat{x}_0}\left[\| \epsilon_{\text{fake}}(\hat{x}_0^{(t)}, t) - \epsilon_{\text{real}}(x_0^{(t)}, t) \|_2^2 \right]$$

其中用一个额外的"fake score"网络来估计学生生成分布的 score，并将其对齐到真实分布的 score（由预训练教师提供）。

### DMD2 vs ADD

| 维度 | ADD | DMD2 |
|------|-----|------|
| 对抗信号来源 | 独立判别器 | Score 函数差异 |
| 训练稳定性 | 中（需要GAN技巧） | 高（纯回归） |
| 额外网络 | 判别器 | Fake score 网络 |
| 单步质量 (COCO FID) | 28.1 | 24.2 |
| 训练成本 | 高（8×A100, 数天） | 中高（8×A100, 1-2天） |

## 综合对比 🔬

### 方法对比表

| 方法 | 步数 | FID (COCO) | 需要教师 | 训练成本 | LoRA 兼容 | 无需 CFG |
|------|------|-----------|---------|---------|----------|---------|
| Progressive Distillation | 4-8 | 25.5 | 是 | 高 (8×A100, 52h) | 需重蒸馏 | 可选 |
| Guided Distillation | 4-8 | 24.8 | 是 | 高 | 需重蒸馏 | 是 |
| LCM | 2-4 | 25.2 | 是(SD) | 中 (1×A100, 32h) | 需重蒸馏 | 是 |
| LCM-LoRA | 2-4 | 25.8 | 是(SD) | 低 (1×A100, 4h) | **原生兼容** | 是 |
| ADD (SDXL-Turbo) | 1-4 | 28.1 | 是 | 高 (8×A100, 数天) | 有限 | 是 |
| DMD2 | 1-4 | 24.2 | 是 | 中高 | 有限 | 是 |
| Consistency Training | 1-2 | 26.0 | 否 | 很高 (从头训练) | 需重训练 | 是 |

### 选择指南

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 社区模型 + 风格 LoRA | LCM-LoRA | 训练快、可叠加已有 LoRA |
| 极致单步速度 | SDXL-Turbo / DMD2 | 1步生成，DMD2 质量更好 |
| 最佳 2-4 步质量 | LCM 或 DMD2 | Huber loss / 分布匹配 |
| 无训练资源 | 下载社区预蒸馏权重 | LCM-LoRA / SDXL-Turbo 公开可用 |
| 自定义模型蒸馏 | Progressive Distillation | 最通用，适配性最强 |

蒸馏方法在 1-4 步区间的质量远优于同步数的 ODE 求解器，但仍不及 20 步 ODE 求解器的质量。选择取决于延迟预算。

## 小结

| 要点 | 说明 |
|------|------|
| 核心原理 | 用训练换推理——预先训练学生模型压缩采样步数 |
| 最实用方法 | LCM-LoRA（训练简单、社区兼容、4步生成） |
| 最极致速度 | ADD/SDXL-Turbo（1步生成，但质量有损） |
| 最新前沿 | DMD2（分布匹配，单步质量最佳） |
| 与求解器关系 | 互补而非替代——蒸馏解决 <10 步的问题 |

---

> **下一篇**：[模型量化](./04-quantization.md) — 从另一个维度降低推理开销
