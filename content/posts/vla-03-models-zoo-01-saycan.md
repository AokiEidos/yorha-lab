---
title: "SayCan 与早期 LLM+机器人"
date: 2026-04-20T16:28:00.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# SayCan 与早期 LLM+机器人

> 🔰 入门 | 前置知识：[从 VLM 到 VLA](../02-architecture/01-vlm-to-vla.md)

## VLA 之前的范式

在 RT-2 开创 VLA 范式之前，将大模型与机器人结合的主流方案是**"LLM 规划 + 低层技能"（LLM Planning + Low-level Skills）**——LLM 负责理解指令和分解任务，预训练的底层策略负责执行具体动作。这一范式的核心假设是：语言模型虽然不具备身体，但它拥有关于世界的常识知识，可以用来指导机器人的行为。

**SayCan**（Google, 2022, "Do As I Can, Not As I Say"）是这一范式的代表性工作，发表于 *Robotics: Science and Systems (RSS) 2022*，在当时 Google 的 Everyday Robots 上进行了大规模验证。

## SayCan 框架全景

```
 用户指令: "我打翻了可乐，能帮我清理一下吗"
                │
                ▼
 ┌─────────────────────────────────────────────────┐
 │         LLM (PaLM 540B / FLAN / PaLM-SayCan)   │
 │                                                 │
 │   Prompt: "Robot can perform the following       │
 │   actions: [找到海绵, 拿起海绵, 找到可乐,        │
 │   拿起可乐, 去垃圾桶, 放下物体, ...]            │
 │                                                 │
 │   User says: 我打翻了可乐...                     │
 │   Best next action:"                            │
 │                                                 │
 │   → 为每个候选动作计算 LLM 对数概率 p(l_i | s)   │
 └─────────────────────┬───────────────────────────┘
                       │  候选动作 + LLM 分数
                       ▼
 ┌─────────────────────────────────────────────────┐
 │     可行性评估（Affordance / Value Function）    │
 │                                                 │
 │  对每个候选动作 l_i:                              │
 │    V(l_i, s_t) = 该技能在当前状态下的成功概率     │
 │    (由 RL 训练的价值函数提供)                     │
 │                                                 │
 │  → 输出: affordance score a(l_i)                │
 └─────────────────────┬───────────────────────────┘
                       │
                       ▼
 ┌─────────────────────────────────────────────────┐
 │           综合排序 & 技能执行                     │
 │                                                 │
 │  score(l_i) = p(l_i | s) × a(l_i)              │
 │  选择: l* = argmax score(l_i)                   │
 │  执行: 调用对应的底层 RL 策略                     │
 │  完成后: 更新状态描述 s ← s + "已完成 l*"        │
 │  循环直到: LLM 输出 "done" 或达到最大步数         │
 └─────────────────────────────────────────────────┘
```

## 可行性评分公式（Affordance Scoring）

SayCan 的核心数学机制是将 LLM 的语义评分与物理世界的可行性评分相乘：

$$\pi_{\text{SayCan}}(l_i | s, i) = p_{\text{LLM}}(l_i | i, s) \cdot \text{Affordance}(l_i, s)$$

其中：
- $l_i$ 是第 $i$ 个候选动作的语言描述（如"找到海绵"）
- $s$ 是当前状态的文本描述（由历史动作和观测构成）
- $i$ 是用户的原始指令
- $p_{\text{LLM}}(l_i | i, s)$ 是 LLM 在给定指令和状态下认为 $l_i$ 是合理下一步的概率
- $\text{Affordance}(l_i, s)$ 由底层策略的**价值函数（Value Function）** 给出，代表当前物理环境中成功执行 $l_i$ 的概率

**为什么用乘法而不是加法？** 乘法确保两个条件必须同时满足——即使 LLM 认为某动作在语义上完美（$p=1.0$），如果物理上不可行（$a=0$），综合得分也为 0。这避免了 LLM 的**幻觉（Hallucination）** 导致机器人尝试不可能的动作。

## 完整任务分解示例（6 步长horizon任务）

```
用户指令: "我打翻了可乐在桌子上，能帮我清理一下，然后拿一瓶水给我？"

Step 1: LLM 规划
  候选动作           LLM分数   可行性   综合
  "找到海绵"         0.35     0.90    0.315 ← 选中
  "拿起可乐"         0.25     0.70    0.175
  "找到水瓶"         0.15     0.60    0.090
  "去垃圾桶"         0.10     0.85    0.085
  ...
  → 执行 "找到海绵" → 成功

Step 2: 更新状态 s = "已找到海绵"
  候选动作           LLM分数   可行性   综合
  "拿起海绵"         0.60     0.85    0.510 ← 选中
  "拿起可乐"         0.15     0.70    0.105
  ...
  → 执行 "拿起海绵" → 成功

Step 3: 更新状态 s = "已找到海绵, 已拿起海绵"
  候选动作           LLM分数   可行性   综合
  "擦桌子"           0.70     0.75    0.525 ← 选中
  "放下海绵"         0.10     0.90    0.090
  ...
  → 执行 "擦桌子" → 成功

Step 4: 更新状态 s = "..., 已擦桌子"
  候选动作           LLM分数   可行性   综合
  "放下海绵"         0.45     0.85    0.383 ← 选中
  "找到水瓶"         0.30     0.60    0.180
  ...
  → 执行 "放下海绵" → 成功

Step 5: 更新状态 s = "..., 已放下海绵"
  候选动作           LLM分数   可行性   综合
  "找到水瓶"         0.55     0.65    0.358 ← 选中
  "done"             0.20     1.00    0.200
  ...
  → 执行 "找到水瓶" → 成功

Step 6: 更新状态 s = "..., 已找到水瓶"
  候选动作           LLM分数   可行性   综合
  "拿起水瓶"         0.50     0.80    0.400 ← 选中
  → 执行 "拿起水瓶" → 成功

Step 7: LLM 输出 "done" 的分数最高 → 任务完成
```

## 内循环伪代码

```python
# SayCan 执行循环
def saycan_execute(instruction: str, skill_library: dict, llm, max_steps=10):
    """
    instruction: 用户自然语言指令
    skill_library: {技能名称: (rl_policy, value_function)} 预训练的底层技能
    llm: 大语言模型 (PaLM 540B)
    """
    state_description = ""
    history = []
    
    for step in range(max_steps):
        # 1. 构造 LLM prompt
        skill_names = list(skill_library.keys()) + ["done"]
        prompt = build_prompt(instruction, state_description, skill_names)
        
        # 2. 获取 LLM 对每个候选技能的对数概率
        llm_scores = {}
        for skill_name in skill_names:
            llm_scores[skill_name] = llm.score_completion(
                prompt, completion=skill_name  # 返回 p(skill_name | prompt)
            )
        
        # 3. 获取价值函数的可行性评分
        affordance_scores = {}
        current_obs = robot.get_observation()  # RGB 图像 + 机器人状态
        for skill_name in skill_names:
            if skill_name == "done":
                affordance_scores[skill_name] = 1.0  # "done" 总是可行的
            else:
                _, value_fn = skill_library[skill_name]
                affordance_scores[skill_name] = value_fn(current_obs)
        
        # 4. 综合评分 (乘法)
        combined = {
            name: llm_scores[name] * affordance_scores[name]
            for name in skill_names
        }
        
        # 5. 选择最优技能
        best_skill = max(combined, key=combined.get)
        
        if best_skill == "done":
            print(f"Task completed in {step} steps")
            return True
        
        # 6. 执行底层 RL 策略
        rl_policy, _ = skill_library[best_skill]
        success = execute_skill(rl_policy, robot, timeout=30)
        
        # 7. 更新状态描述
        status = "成功" if success else "失败"
        state_description += f"\n已执行: {best_skill} ({status})"
        history.append((best_skill, success))
    
    return False  # 超过最大步数
```

## 底层技能库详细设计

SayCan 的技能库包含 **551 个预训练技能**，每个技能是独立训练的强化学习策略：

| 技能类别 | 示例 | 数量 | RL 算法 | 训练方式 |
|---------|------|------|---------|---------|
| 导航技能 | "去厨房台面"、"去垃圾桶" | ~70 | BC+RL | 真机 |
| 拾取技能 | "拿起海绵"、"拿起可乐" | ~200 | RT-1 风格 BC | 真机遥操作 |
| 放置技能 | "放到桌子上"、"放入垃圾桶" | ~150 | BC+RL | 真机 |
| 其他交互 | "打开抽屉"、"擦桌子" | ~130 | RL | 真机 |

每个技能的训练需要 **~2000-5000 条演示轨迹** 和 **数天到数周** 的 RL 训练。价值函数从同一 RL 训练过程中获得，作为成功概率的近似。

## SayCan 的实验结果

在 Google Everyday Robots 平台上的 101 个长horizon 厨房任务测试：

| 方法 | 规划成功率 | 执行成功率 |
|------|-----------|-----------|
| LLM 直接规划（无 affordance） | 80.4% | 13.5% |
| SayCan（LLM + affordance） | **93.1%** | **74.0%** |
| 人类设计的规划 | 100% | 77.0% |

关键发现：LLM 单独规划时"说得头头是道"但不考虑物理可行性，执行成功率极低。加入 affordance 后规划合理性和执行成功率都大幅提升。

## 与同期方法的对比

| 维度 | SayCan | Code-as-Policies (CaP) | ProgPrompt | Inner Monologue |
|------|--------|------------------------|------------|-----------------|
| **发布** | Google 2022 | Google 2022 | NVIDIA 2022 | Google 2022 |
| **LLM 输出** | 技能选择概率 | Python 代码 | Python 程序 | 自然语言推理 |
| **接地方式** | 价值函数 affordance | API 函数调用 | 预定义函数 | 环境反馈 |
| **技能组合** | 串行选择 | 代码逻辑组合 | 程序结构组合 | 串行 + 反思 |
| **需要 RL 训练** | 是（每个技能） | 否 | 否 | 是 |
| **灵活性** | 低（固定技能集） | 高（代码生成） | 高（程序生成） | 中 |
| **物理接地** | 强（量化可行性） | 弱（无可行性检查） | 弱 | 中（反馈修正） |
| **长horizon** | 好（贪心逐步） | 好（一次性生成） | 好 | 好（可纠错） |
| **可扩展性** | 差（需预训练技能） | 好（仅需 API） | 好 | 中 |

### 对比分析

**SayCan vs Code-as-Policies**：CaP 让 LLM 直接生成 Python 控制代码（如 `pick("apple"); place_on("plate")`），灵活性远高于 SayCan 的固定技能选择。但 CaP 缺乏物理可行性评估——生成的代码在语法上正确但物理上可能不可行。SayCan 的 affordance 机制在物理接地上更可靠。

**SayCan vs ProgPrompt**：ProgPrompt 类似 CaP 但更强调结构化程序生成（带条件判断和循环）。它的优势是能处理条件分支任务，而 SayCan 只能贪心串行选择。

**SayCan vs Inner Monologue**：Inner Monologue 引入环境反馈（如场景描述、执行成功/失败信号）来让 LLM "反思"和纠正计划。这弥补了 SayCan 缺乏闭环反馈的缺陷——SayCan 在某步失败后只能简单标记失败，不会调整策略。

## 优势与致命局限

### 优势

1. **LLM 常识推理**：PaLM 540B 拥有丰富的世界知识，能将模糊指令（"清理一下"）映射为合理的动作序列
2. **物理接地**：价值函数提供量化的可行性评估，避免 LLM 的幻觉问题
3. **模块化设计**：LLM、价值函数、底层策略三者独立，可分别升级
4. **长horizon规划**：贪心逐步选择机制天然支持长序列任务

### 致命局限（催生 VLA 的动机）

1. **技能库瓶颈**：551 个技能看似很多，但远不能覆盖真实世界的任务多样性。新增一个技能需要：收集演示数据（数千条） → RL 训练（数天） → 调试部署。这导致系统的能力被技能库的大小严格限制
2. **两阶段割裂**：LLM 不直接"看"环境——它只基于文本描述推理。当物理场景复杂时，文本描述不足以捕捉关键细节（如物体的精确位姿）
3. **无法端到端优化**：LLM 和底层策略分别训练。LLM 选择的技能可能在语义上合理但执行上次优，而这个信号无法传回 LLM 进行优化
4. **贪心规划短视**：每步只选当前最优动作，缺乏全局规划能力。在需要"先退一步再进两步"的任务上会失败
5. **Affordance 估计不准**：价值函数在 out-of-distribution 状态下的估计可能严重偏差，导致综合评分失真

### 失败模式分析

```
失败模式 1: 技能库缺失
  指令: "用微波炉加热食物"
  问题: 技能库没有"打开微波炉门"技能 → 无法执行
  
失败模式 2: Affordance 估计偏差
  状态: 海绵在抽屉里（关着）
  "拿起海绵" affordance = 0.6 (高估，因为价值函数没见过抽屉关着的情况)
  → 执行失败

失败模式 3: 贪心短视
  指令: "把红色方块放到蓝色方块上"
  当前: 蓝色方块在红色方块上
  贪心选择: "拿起红色方块" (但蓝色在上面，应该先移开蓝色)
  → 卡住

失败模式 4: 文本描述信息丢失
  指令: "拿起最大的苹果"
  LLM 只知道"桌子上有苹果" → 无法区分大小
  → 随机选择
```

## 从 SayCan 到 VLA 的演进逻辑

SayCan 的每个局限都指向一个解决方向：

| SayCan 的局限 | VLA 的解决方式 |
|--------------|--------------|
| 固定技能库 | 端到端学习任意动作，无需预定义技能 |
| LLM 不看环境 | VLM 直接处理图像输入 |
| 无法端到端优化 | 单一模型端到端训练 |
| 贪心短视 | Action Chunking 预测多步 |
| Affordance 不准 | 动作直接由模型生成，无需外部评估 |

**VLA 的出发点**：能否让一个模型同时完成"看 → 理解 → 规划 → 执行"，而不需要预定义的技能库？这正是 RT-1/RT-2 回答的问题。

## 小结

| 概念 | 要点 |
|------|------|
| SayCan | LLM 语义评分 x 价值函数可行性评分，贪心选择技能 |
| 可行性评分 | $\pi(l) = p_{\text{LLM}}(l) \cdot \text{Affordance}(l)$，乘法确保双重约束 |
| 底层技能 | 551 个预训练 RL 策略，每个需数千条演示 + 数天训练 |
| 同期方法 | CaP（代码生成）、ProgPrompt（程序生成）、Inner Monologue（反思修正） |
| 核心局限 | 技能库固定、两阶段割裂、不可端到端优化、贪心短视 |
| 催生 VLA | 追求端到端的"看 → 理解 → 规划 → 行动"一体化 |

---

> **下一篇**：[RT-1 详解](./02-rt1.md)
