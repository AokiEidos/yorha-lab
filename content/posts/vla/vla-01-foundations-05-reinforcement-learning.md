---
title: "强化学习基础"
date: 2026-04-20T16:49:32.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 强化学习基础

> 🔰 入门 → ⚙️ 进阶 | 前置知识：[模仿学习基础](./04-imitation-learning.md)
> 交叉引用：[VLA 未来展望](../07-foundation-models/06-future.md)、[观测空间与动作空间](./03-observation-action-space.md)

## RL 在机器人学习中的角色

**强化学习（Reinforcement Learning, RL）** 让智能体通过与环境交互、接收奖励信号来学习最优策略。虽然当前 VLA 主要用模仿学习训练（参见[模仿学习基础](./04-imitation-learning.md)），但 RL 在微调、对齐和超越演示质量方面有不可替代的潜力。

**核心区别**：
- 模仿学习（IL）："告诉我怎么做" → 从人类演示中学习
- 强化学习（RL）："告诉我什么是好的" → 从奖励信号中自己探索学习

## MDP 框架：完整数学形式

**马尔可夫决策过程（Markov Decision Process, MDP）** 是 RL 的数学框架：

$$\text{MDP} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$$

| 符号 | 名称 | 含义 | 机器人操作示例 |
|------|------|------|-------------|
| $\mathcal{S}$ | 状态空间 | 环境的所有可能状态 | RGB 图像 + 关节角度 + 物体位姿 |
| $\mathcal{A}$ | 动作空间 | 智能体可执行的所有动作 | 7-DoF 笛卡尔增量 + 夹爪 |
| $P(s'\|s,a)$ | 转移函数 | 执行动作后环境如何变化 | 物理引擎模拟 / 真实世界物理 |
| $R(s,a,s')$ | 奖励函数 | 执行动作获得的即时奖励 | 抓取成功 +1，碰撞 -1 |
| $\gamma \in [0,1)$ | 折扣因子 | 未来奖励的衰减系数 | 通常 0.95-0.99 |
| $\rho_0$ | 初始状态分布 | 环境重置时的状态分布 | 物体随机放置在桌面 |

### 核心目标：最大化期望累积回报

**策略（Policy）** $\pi(a|s)$：在状态 $s$ 下选择动作 $a$ 的概率分布。

RL 的目标是找到最优策略 $\pi^*$，使期望累积折扣回报最大：

$$\pi^* = \arg\max_\pi J(\pi) = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t, s_{t+1}) \right]$$

其中轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 由策略 $\pi$ 与环境交互生成。

### 价值函数

**状态价值函数** $V^\pi(s)$：从状态 $s$ 开始，遵循策略 $\pi$ 的期望累积回报：

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s \right]$$

**动作价值函数** $Q^\pi(s, a)$：在状态 $s$ 执行动作 $a$ 后，遵循策略 $\pi$ 的期望累积回报：

$$Q^\pi(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim P} [V^\pi(s')]$$

### 贝尔曼方程

$$V^\pi(s) = \mathbb{E}_{a \sim \pi} \left[ R(s, a) + \gamma \mathbb{E}_{s' \sim P}[V^\pi(s')] \right]$$

这是 RL 算法的理论基础——当前状态的价值等于即时奖励加上下一状态价值的折扣期望。

## 策略梯度基础：REINFORCE

⚙️ 进阶

**策略梯度（Policy Gradient）** 方法直接优化参数化策略 $\pi_\theta$，是现代深度 RL 的基础。

### REINFORCE 算法

**策略梯度定理**：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k$ 是从 $t$ 时刻开始的回报。

```python
# REINFORCE 算法伪代码
def REINFORCE(policy_net, env, n_episodes=1000, lr=1e-3):
    optimizer = Adam(policy_net.parameters(), lr=lr)

    for episode in range(n_episodes):
        # 1. 采样一条轨迹
        states, actions, rewards = [], [], []
        state = env.reset()

        while not done:
            action_dist = policy_net(state)        # π_θ(a|s)
            action = action_dist.sample()           # 采样动作
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # 2. 计算回报 G_t
        returns = compute_returns(rewards, gamma=0.99)

        # 3. 计算策略梯度
        loss = 0
        for t in range(len(states)):
            log_prob = action_dist.log_prob(actions[t])
            loss -= log_prob * (returns[t] - baseline)  # baseline 减少方差

        # 4. 更新策略
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**REINFORCE 的问题**：方差极大，需要大量采样才能得到稳定的梯度估计。实践中通常使用更先进的 PPO / SAC 等算法。

## 奖励工程：机器人操作的核心难题

⚙️ 进阶

RL 需要奖励信号来指导学习，但为机器人设计好的奖励函数极其困难，被称为**奖励工程（Reward Engineering）**。

### 稀疏奖励 vs 稠密奖励

```
任务: 将红色方块放入蓝色盒子

稀疏奖励 (Sparse):                    稠密奖励 (Dense):
  成功放入: +1                          距离奖励: -||block_pos - box_pos||
  其他所有: 0                           抓取奖励: +0.5 (接触方块时)
                                       抬起奖励: +0.3 (方块离开桌面)
  优点: 不引入人为偏差                    放入奖励: +1.0 (方块在盒子内)
  缺点: 信号极稀疏，几乎无法学习          碰撞惩罚: -0.1 (撞到桌子)
                                       优点: 梯度信号密集，学习快
                                       缺点: 需要手工设计，可能有偏
```

### 具体奖励函数示例

**任务 1：抓取操作**

```python
def grasp_reward(state, action, next_state):
    """抓取任务的分阶段稠密奖励"""
    gripper_pos = next_state["ee_position"]    # 末端执行器位置
    object_pos = next_state["object_position"]  # 目标物体位置

    # 阶段 1: 接近物体
    dist = np.linalg.norm(gripper_pos - object_pos)
    reaching_reward = -dist * 2.0  # 越近奖励越高

    # 阶段 2: 抓取
    is_grasped = next_state["object_grasped"]
    grasp_reward = 5.0 if is_grasped else 0.0

    # 阶段 3: 抬起
    if is_grasped:
        lift_height = next_state["object_height"] - 0.05  # 5cm 以上算抬起
        lift_reward = max(0, lift_height) * 10.0
    else:
        lift_reward = 0.0

    # 安全惩罚
    contact_force = np.linalg.norm(next_state["contact_force"])
    safety_penalty = -0.5 if contact_force > 50.0 else 0.0  # >50N 惩罚

    return reaching_reward + grasp_reward + lift_reward + safety_penalty
```

**任务 2：精密插入（Peg-in-Hole）**

多阶段奖励设计：**对准奖励**（轴与孔的角度对齐 × 2.0）+ **居中奖励**（XY 距离 5mm 内 exp 衰减 × 3.0）+ **插入深度奖励**（深度 × 50.0）+ **侧向力惩罚**（横向力 × -0.1）。关键设计：惩罚过大侧向力以防止卡死/损坏零件。

### 奖励黑客（Reward Hacking）

模型找到获取高奖励但不符合期望的"捷径"：

| 任务 | 设计奖励 | Reward Hacking 行为 | 根因 |
|------|---------|-------------------|------|
| 抓取物体 | 末端距物体距离减小 | 手快速接近但不抓取 | 缺少"成功抓取"条件 |
| 推物到目标 | 物体距目标距离减小 | 大力推飞物体（瞬间过目标） | 缺少速度约束 |
| 走路 | 前进距离最大化 | 摔倒后向前滑行 | 缺少站立约束 |
| 清洁桌面 | 桌面上物体数量减少 | 把物体推到桌子外（地上） | "桌面清洁"≠"物体放好" |

## Sim-to-Real RL：仿真中训练，真机中部署

⚙️ 进阶

真机上做 RL 太慢（1 步需要 0.1-1 秒）且不安全。解决方案：在**仿真器**中高速训练，然后迁移到真机。

### PPO in Isaac Gym / Isaac Sim

Sim-to-Real RL 管线三步走：

1. **GPU 并行仿真训练**：Isaac Gym 支持 4096 个并行环境（20K+ 样本/秒），PPO 训练约 100M 步、2-8 小时达到仿真成功率 >95%
2. **域随机化缩小 Gap**：物理参数（摩擦 ±30%、质量 ±20%）、视觉参数（纹理/光照/颜色随机）、执行/传感噪声（±5-10%）
3. **真机部署**：零样本迁移或少样本微调（10-50 次交互），典型性能衰减：仿真 95% → 真机 60-80%

### Isaac Gym 训练性能

| 任务 | 并行环境数 | 训练步数 | GPU 时间 | 仿真成功率 | 真机成功率 |
|------|-----------|---------|---------|-----------|-----------|
| 抓取 (简单) | 4096 | 10M | ~1 hr (A100) | 98% | 85% |
| 抓取 (随机物体) | 4096 | 50M | ~4 hr | 90% | 65% |
| 灵巧手翻转 | 8192 | 200M | ~12 hr | 80% | 50% |
| 双臂协作 | 2048 | 100M | ~8 hr | 75% | 45% |

## RLHF for Robots：人类反馈的机器人强化学习

🔬 前沿

借鉴 LLM 领域的 RLHF（Reinforcement Learning from Human Feedback），为 VLA 设计人类反馈对齐机制。

### 核心流程

```
第 1 步: VLA 策略 π_ref 执行任务
         生成多条轨迹 {τ₁, τ₂, ..., τₖ}

第 2 步: 人类标注员观看视频
         对轨迹进行偏好排序:
         τ₃ ≻ τ₁ ≻ τ₅ ≻ τ₂ ≻ τ₄
         (更安全、更高效、更符合意图的排在前面)

第 3 步: 训练奖励模型 R_φ(τ)
         Bradley-Terry 模型:
         P(τ_i ≻ τ_j) = σ(R_φ(τ_i) - R_φ(τ_j))

第 4 步: 用 PPO/DPO 微调 VLA
         max_π E_{τ~π} [R_φ(τ)] - β · KL(π || π_ref)
         其中 KL 约束防止策略偏离太远
```

### 机器人 RLHF 的特殊挑战

| 挑战 | LLM RLHF | 机器人 RLHF |
|------|----------|------------|
| **标注速度** | 1 条回答 ~10 秒 | 1 条轨迹视频 ~60 秒 |
| **评判维度** | 主要是内容质量 | 安全性 + 效率 + 美观性 + 力控质量 |
| **偏好一致性** | 较高 | 较低（不同标注员对"好"的理解不同） |
| **安全关键性** | 低（文字不伤人） | 高（物理动作可伤人） |
| **在线采样** | 快（毫秒级生成） | 慢（分钟级真机执行） |

## IL vs RL vs 混合方法对比

| 维度 | 模仿学习 (IL) | 强化学习 (RL) | 混合 (IL + RL) |
|------|------------|-------------|---------------|
| **数据来源** | 人类演示 | 环境交互 | 演示 + 交互 |
| **是否需要奖励** | 否 | 是 | 部分需要 |
| **样本效率** | 高 (50-200 demo) | 低 (~100K+ 交互) | 中 |
| **性能上限** | 受限于演示质量 | 可超越演示 | 可超越演示 |
| **安全性** | 安全 (只用演示) | 有风险 (需探索) | 预训练安全、微调有限风险 |
| **训练难度** | 低 | 高 (奖励设计+调参) | 中 |
| **泛化能力** | 取决于演示多样性 | 取决于奖励泛化性 | 较好 |
| **VLA 中的角色** | **主训练范式** | 微调/对齐 | 未来趋势 |

### 为什么 VLA 以 IL 为主、RL 为辅

1. **VLA 的核心优势来自预训练**：互联网 VLM 知识只能通过自监督/对比学习获取，RL 信号无法替代
2. **大模型 RL 训练不稳定**：3B-55B 参数的模型从零用 RL 训练几乎不可行（梯度方差太大）
3. **演示数据易获取**：遥操作（每次 2-5 分钟）比设计奖励函数（每个任务数天调试）更实用
4. **安全考量**：RL 探索可能损坏机器人或环境

### 混合方法：IL 预训练 + RL 微调

```
阶段 1: IL 预训练 (获得基础能力)
  数据: 大规模遥操作演示
  方法: 行为克隆 (BC)
  结果: 能完成 80% 的常规任务

阶段 2: RL 微调 (超越演示)
  环境: 仿真器 (Isaac Gym) + 少量真机
  奖励: 任务成功 + 安全约束 + 人类偏好
  方法: PPO + KL 约束 (防止遗忘 IL 知识)
  结果: 性能提升到 90%+，且更安全

具体案例 - 开瓶盖:
  IL 阶段: 学会基本抓握和旋转动作 → 成功率 60%
  RL 阶段: 在仿真中优化力度和角度 → 成功率 85%
  关键: RL 发现了"先松后紧"的策略，超越了人类演示的固定模式
```

## 在线适应场景

RL 在部署后在线适应新环境中有独特价值。例如 VLA 部署到新工厂（光照/物体/桌面不同），初始成功率仅 60%。通过在线 RL 适应：Day 1 安全区域 50 次探索（→70%）、Day 2 加入人类偏好反馈（→78%）、Day 3 持续自我改进（→83%）。关键约束：严格的安全边界（力矩/速度限制）、KL 约束防止灾难性遗忘、人工监督前 100 次交互。

## 小结

| 概念 | 要点 |
|------|------|
| MDP | RL 的数学框架：$(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$ |
| 策略梯度 | REINFORCE：$\nabla J = \mathbb{E}[\nabla \log\pi \cdot G_t]$，高方差需要 baseline |
| 奖励工程 | 稀疏奖励难学习，稠密奖励有偏差，需要多阶段设计 |
| Reward Hacking | 模型利用奖励漏洞获取高分但不完成任务 |
| Sim-to-Real | Isaac Gym 高速并行训练 + 域随机化缩小迁移差距 |
| RLHF for Robots | 借鉴 LLM RLHF，但面临标注慢、多维评判、安全关键等挑战 |
| IL vs RL | VLA 以 IL 为主（预训练+BC），RL 为辅（微调+对齐+在线适应） |

---

> **下一篇**：[从 VLM 到 VLA 的演化](../02-architecture/01-vlm-to-vla.md) — 进入模块二，理解 VLA 的核心架构创新。
