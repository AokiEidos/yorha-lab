---
title: VLA 从入门到精通（三）：强化学习快速入门 — Policy Gradient、PPO 与机器人控制
hidden: True
date: 2026-04-15 17:04:00+08:00
draft: False
tags: ['VLA', '强化学习', '机器人']
toc: True
---
# VLA 从入门到精通（三）：强化学习快速入门 — Policy Gradient、PPO 与机器人控制

> **前置知识**：了解深度学习基础（MLP、反向传播）。本文不需要强化学习背景知识，从零开始介绍 RL 的核心概念，重点是让读者能理解 π₀ 等 VLA 工作中的 RL 部分。术语首次出现时会有 🧠 注。

---

## 0. 核心主旨 (Core Gist/Theme)

**强化学习（Reinforcement Learning，RL）** 是 VLA 训练的重要范式之一，尤其是 π₀ 这类使用扩散策略的 VLA。本文的目的是建立理解 VLA 所需的最少 RL 知识：

- **RL 的基本框架**：Agent、Environment、State、Action、Reward
- **Policy Gradient**：直接优化动作策略的神经网络
- **PPO（Proximal Policy Optimization）**：当前最流行的 RL 算法，π₀ 的训练就用 PPO
- **模仿学习（Imitation Learning）**：VLA 如何从示教数据中学习

---

## 1. RL 的基本框架：Agent-Environment 交互

### 1.1 核心要素

强化学习的基本框架是 **Agent（智能体）** 和 **Environment（环境）** 的交互：

```
Agent → 动作 a_t → Environment → 观察 o_{t+1} + 奖励 r_{t+1} → Agent
         ↑状态 s_t ←                              ↓
         └─────────── Observation ←────────────────┘
```

**各要素解释：**

| 要素 | 含义 | 例子 |
|------|------|------|
| **Agent** | 决策者（神经网络） | VLA 模型 |
| **Environment** | Agent 所在的世界 | 机器人/仿真环境 |
| **State $s_t$** | Agent 看到的当前状态 | 相机图像 |
| **Action $a_t$** | Agent 执行的动作 | 关节角度 |
| **Reward $r_t$** | 环境的反馈信号 | 任务完成=+1，失败=-1 |
| **Trajectory $\tau$** | 一整个回合的 (s,a,r) 序列 | $s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T$ |

> **🧠 回合（Episode）是什么？**
>
> 一个"回合"是从任务开始到任务结束的全过程。例如：机器人从桌上拿起积木放到碗里——从机械臂初始位置开始，到任务完成（或失败）结束，这就是一个 episode。
>
> RL 的目标是：让 Agent 学会一种策略，使它在**多个回合**中的累积奖励最大化。

### 1.2 目标：最大化累积奖励

RL 的目标是最大化**期望累积回报（Expected Return）**：

$$J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \gamma^t r_t \right]$$

其中 $\theta$ 是 Agent（策略网络）的参数，$\gamma \in (0,1)$ 是**折扣因子（discount factor）**，$T$ 是回合长度。

> **🧠 为什么需要折扣因子 $\gamma$？**
>
> $\gamma$ 控制 Agent 对"未来奖励"的重视程度：
>
> - $\gamma \to 0$：只关心立即奖励（近视）
> - $\gamma \to 1$：平等对待立即和未来奖励（远视）
>
> 实际中 $\gamma$ 通常取 0.99——这意味着 100 步后的奖励折扣到约 $0.37$ 的重要性。这确保 Agent 既有远见，又不会因为远期的不确定奖励而过度行动。

---

## 2. Policy Gradient：直接优化策略

### 2.1 什么是 Policy？

**策略（Policy）** $\pi_\theta(a_t|s_t)$ 是 Agent 的"大脑"——给定当前状态 $s_t$，输出应该执行什么动作 $a_t$ 的概率分布。

- **确定性策略（Deterministic）**：$\pi_\theta(s) = a$（给定状态输出唯一动作）
- **随机策略（Stochastic）**：$\pi_\theta(a|s)$（给定状态输出动作的概率分布）

VLA 通常使用随机策略，因为动作输出是连续的（比如关节角度是实数），确定性策略无法表达动作的不确定性。

### 2.2 Policy Gradient 目标函数

直接对 $J(\theta)$ 求梯度（策略梯度定理）：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ 是**回报（Return）**，即从时间 $t$ 开始的累积折扣奖励。

> **🧠 为什么梯度里有 $\log \pi_\theta(a_t|s_t)$？**
>
> $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 的方向，指向"增加在状态 $s_t$ 下执行动作 $a_t$ 的概率"。
>
> 如果 $G_t$ 是正的（这个动作最终带来了高奖励），梯度就往增加 $\pi_\theta(a_t|s_t)$ 的方向走；
> 如果 $G_t$ 是负的（这个动作最终导致低奖励），梯度就往减少 $\pi_\theta(a_t|s_t)$ 的方向走。
>
> 这就是"reward signal 指导策略更新"的核心数学表达。

### 2.3 REINFORCE 算法

最简单的 Policy Gradient 算法是 **REINFORCE**：

```
1. 采样：从当前策略 πθ 采集 N 条轨迹 {τ_i}
2. 估计：对每条轨迹计算回报 G_t^i
3. 更新：θ ← θ + α * Σ_t ∇_θ log πθ(a_t^i|s_t^i) * G_t^i
4. 重复
```

REINFORCE 的问题是**方差高**——每次采样的轨迹可能有很大随机性，导致梯度估计不稳定。

---

## 3. PPO：当前最主流的 RL 算法

### 3.1 重要性采样与策略比率

PPO（Proximal Policy Optimization）的核心创新是**限制策略更新的步幅**，防止"灾难性的策略崩溃"。

**策略比率（Importance Sampling Ratio）：**

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

这是新策略和旧策略在同一个状态-动作对上概率的比值。PPO 限制 $r_t(\theta)$ 不能离 1 太远。

### 3.2 PPO 剪切目标函数

PPO 的目标函数是**裁剪的代理损失（Clipped Surrogate Loss）**：

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \cdot A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t \right) \right]$$

其中：
- $A_t$ 是**优势函数（Advantage）**，衡量在状态 $s_t$ 下执行 $a_t$ 比平均水平好多少
- $\epsilon$ 通常取 0.2，即限制策略比率在 $[0.8, 1.2]$ 区间
- $\text{clip}(x, 1-\epsilon, 1+\epsilon)$ 把 $x$ 限制在指定区间内

> **🧠 为什么需要 clip？**
>
> 假设某条轨迹产生了极高的正向回报 $A_t \gg 0$。
>
> - 如果没有 clip：$r_t \cdot A_t$ 会非常大，策略会**急剧**向这个方向调整——可能导致过拟合到这一条轨迹
> - 加上 clip 后：即使 $r_t$ 很大，$\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t$ 最多只有 $(1+\epsilon) \cdot A_t$，更新幅度被限制
>
> 这就像"每次只走一小步"，而不是"一次跳到最优"——保证了训练稳定性。

### 3.3 PPO 在机器人学习中的应用

π₀ 的训练同时使用了**模仿学习（BC）**和 **PPO**：
- 示教数据用 Behavioral Cloning（行为克隆）快速学习基础策略
- PPO 用于进一步微调，提升泛化能力和鲁棒性

> **🧠 Behavioral Cloning vs PPO**
>
> - **Behavioral Cloning（BC，行为克隆）**：直接监督学习，模仿示教数据中的动作 $(s_t, a_t)$ 对
>   - 优点：简单高效，示教数据利用率高
>   - 缺点：无法自主探索新策略，遇到分布外状态会失败
>
> - **PPO**：通过试错探索优化策略，不需要完美的示教
>   - 优点：能发现示教之外的新策略，对分布外状态更鲁棒
>   - 缺点：需要大量环境交互（采样成本高），对稀疏奖励问题效果差
>
> 实际中通常先用 BC 预热（快速学到基础策略），再用 PPO 微调（探索改进）。

---

## 4. 机器人控制中的 Reward 设计

### 4.1 奖励塑形（Reward Shaping）

**奖励函数的设计**对 RL 的成功至关重要。好的奖励函数应该：

- **稀疏奖励（sparse）**：只在任务完成时给奖励（如"把积木放到碗里 +1"）
- **稠密奖励（dense）**：每一步都给出反馈（如"接近目标 +0.1"）

稠密奖励训练更快，但设计难度大——不合理的稠密奖励可能导致 Agent 找到"作弊"的方式（如重复同一动作以获得更多小奖励）。

### 4.2 VLA 的奖励函数设计

在 VLA 训练中，奖励通常来自：
- **任务成功/失败信号**：成功=+1，失败=-1
- **人类反馈（Human Feedback）**：人类对动作打分（用于 RLHF）
- **视觉对齐**：生成的动作序列与示教轨迹的相似度

> **🧠 为什么 VLA 主要用模仿学习而不是端到端 RL？**
>
> 主要原因是**样本效率（Sample Efficiency）**。
>
> RL 需要在环境中大量试错才能学到有效策略。对于真实机器人，每次试错可能耗时几分钟到几小时，而且失败可能损坏硬件。在 10 万条示教数据上，BC 可以直接训练；但用 RL 训练同样的数据量，可能需要上百万条交互。
>
> 所以 VLA 通常用 BC 学基础策略（高效利用示教数据），用少量 RL 微调（探索改进）。

---

## 5. 关键术语解释

| 术语 | 解释 |
|------|------|
| **Policy $\pi_\theta$** | Agent 的决策策略，给定状态输出动作概率 |
| **Trajectory $\tau$** | 一整个回合的状态-动作-奖励序列 |
| **Return $G_t$** | 从时间 $t$ 开始的累积折扣奖励 |
| **折扣因子 $\gamma$** | 未来奖励的重要性折扣（通常 0.99） |
| **Policy Gradient** | 直接对策略参数求梯度来优化目标 |
| **REINFORCE** | 基础的策略梯度算法，方差高 |
| **PPO（Proximal Policy Optimization）** | 限制策略更新步幅的策略优化算法 |
| **重要性采样比率 $r_t(\theta)$** | 新旧策略在同一点概率的比值 |
| **优势函数 $A_t$** | 衡量动作相对平均水平的优势程度 |
| **Behavioral Cloning（BC）** | 行为克隆，通过监督学习模仿示教动作 |
| **奖励塑形（Reward Shaping）** | 设计奖励函数以引导学习方向 |
| **样本效率（Sample Efficiency）** | 需要多少样本才能学到有效策略 |

---

## 6. 下一步预告

有了 RL 基础，下一篇将分析 **OpenVLA 和 Octo 的技术细节**：
- 开源 VLA 基座模型如何使用 LoRA 微调
- Octo 的多机器人形态支持是如何实现的
- 从预训练到部署的完整流程

---

*参考文献：*
1. Mnih et al., **REINFORCE: Policy Gradient Methods** (2012)
2. Schulman et al., **Proximal Policy Optimization Algorithms** (PPO, 2017)
3. Ouyang et al., **Training Language Models to Follow Instructions with Human Feedback** (RLHF, 2022)