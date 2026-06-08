# TODO 4.2 `UnicycleEnv` 完整对话记录

记录日期：2026-06-08

## 记录范围

本文按时间顺序整理当前对话窗口中，用户与 Codex 围绕 `TODO.md` 第 4.2 节、Unicycle 奖励、安全修正和 zero-shot 测试进行的全部教学问答。

- 保留用户的完整问题和 Codex 的完整正式答复。
- 不包含系统提示、开发者提示、命令执行过程和工具输出。
- 用户提供的仓库操作规则已经保存在项目根目录的 [`AGENTS.md`](./AGENTS.md)，此处不重复粘贴。
- 本次“把对话生成 Markdown 文档”的请求记录在文末。
- 本次整理没有修改任何项目源码。

---

## 第一轮：讲解 TODO 4.2

### 用户

> 不修改任何代码，讲解一下 TODO.md 中的 4.2。

### Codex

## 4.2 `UnicycleEnv` 讲解

核心代码位于 [`envs/unicycle_env.py`](./envs/unicycle_env.py)。

一句话概括：这是一个二维平面独轮车环境。SAC 决定“前进速度和转向速度”，环境更新机器人位置；CBF 安全层位于环境外部，负责在动作进入环境前避障。

### 4.2.1 状态、观测和动作

内部真实状态是：

```text
state = [x, y, theta]
```

- `x, y`：机器人在世界坐标系中的位置。
- `theta`：机器人朝向，单位为弧度。

策略网络收到的观测是 7 维：

```text
obs = [
    x,
    y,
    cos(theta),
    sin(theta),
    goal_compass_x,
    goal_compass_y,
    exp(-dist_goal)
]
```

使用 `cos(theta), sin(theta)` 可以避免角度从 `pi` 跳到 `-pi` 时产生数值不连续。

动作是：

```text
action = [v, omega]
```

- `v`：沿机器人朝向的线速度。
- `omega`：角速度。

策略动作空间为 `[-1, 1]^2`。代码还定义了供安全层使用的 `safe_action_space=[-2.5,2.5]^2`，但 `step()` 最终仍会把收到的动作裁剪到 `[-1,1]`。

特别注意：**7 维观测中没有障碍物位置**。策略主要学习到达目标，障碍物由外部的 RCBF-QP 安全层处理。

### 4.2.2 时间步与 episode 长度

```python
self.dt = 0.02
self.max_episode_steps = 1000
```

每次 `step()` 代表 `0.02` 秒，最多运行：

```text
1000 * 0.02 = 20 秒
```

到达目标或运行满 1000 步都会令 `done=True`。

### 4.2.3 默认障碍物

`obs_config='default'` 时创建 5 个半径为 `0.6` 的圆形障碍物：

```text
( 0.0,  0.0)
(-1.5,  1.5)
(-1.5, -1.5)
( 1.5, -1.5)
( 1.5,  1.5)
```

默认起点是 `(-2.5,-2.5)`，目标是 `(2.5,2.5)`，目标判定半径为 `0.3`。

### 4.2.4 `test/random/none`

`obs_config='test'` 使用固定的测试障碍物：

- 一个平移后的正方形；
- 一个位于 `(1.5,1.5)`、半径 `0.6` 的圆；
- 一个三角形。

除此之外，代码全部进入：

```python
else:
    self.get_random_hazard_locations(6, 0.6)
```

所以实际行为是：

| `obs_config` | 实际障碍物 |
|---|---|
| `default` | 5 个固定圆形 |
| `test` | 固定圆形和多边形 |
| `random` | 随机障碍物 |
| `none` | **随机障碍物** |
| 其他任意字符串 | 随机障碍物 |

因此，当前 Unicycle 实现里的 `none` 不表示“无障碍”。而且字符串判断区分大小写，`Default` 也会进入随机分支。

### 4.2.5 `step` 与 `_step`

公开接口 `step()` 负责：

1. 把动作裁剪到 `[-1,1]`；
2. 调用 `_step()` 更新内部状态；
3. 把 3 维真实状态转换成 7 维观测；
4. 返回 `(next_obs, reward, done, info)`。

内部 `_step()` 负责真正的环境计算：

- 更新动力学；
- 添加扰动；
- 计算 reward；
- 判断是否结束；
- 计算安全 cost。

可以理解为：

```text
step：对外包装
_step：内部物理和任务逻辑
```

### 4.2.6 独轮车动力学

代码使用控制仿射系统：

```text
state_dot = f(state) + g(state) action
```

这里 `f(state)=0`，而：

```text
g(state) =
[cos(theta)  0]
[sin(theta)  0]
[    0       1]
```

代入 `action=[v,omega]`：

```text
x_dot     = v cos(theta)
y_dot     = v sin(theta)
theta_dot = omega
```

使用前向欧拉法离散化：

```text
x     <- x     + dt * v cos(theta)
y     <- y     + dt * v sin(theta)
theta <- theta + dt * omega
```

这就是代码中的：

```python
self.state += self.dt * (f + g @ action)
```

### 4.2.7 扰动项

名义动力学更新后，代码又执行：

```python
self.state -= dt * 0.1 * g @ [cos(theta), 0]
```

展开后大约是：

```text
x     <- x - dt * 0.1 cos^2(theta)
y     <- y - dt * 0.1 sin(theta)cos(theta)
theta 不变
```

可以把它理解为一个确定性的、与朝向相关的未知动力学扰动，供 GP 学习。

虽然类中定义了 `disturb_mean` 和 `disturb_covar`，但随机扰动采样代码已被注释，所以当前环境实际使用的是上面这个**确定性扰动**。

### 4.2.8 Reward

每一步首先计算：

```text
reward = 上一步到目标的距离 - 当前到目标的距离
```

因此：

- 接近目标：正奖励；
- 远离目标：负奖励；
- 距离不变：奖励约为 0。

进入目标半径后，再增加：

```text
reward += 1.0
```

然后结束 episode。障碍物碰撞不会直接扣 reward，也不会直接终止 episode。

### 4.2.9 Cost

仅在 `obs_config='default'` 时，环境检查机器人是否进入圆形障碍物：

```text
(x-hx)^2 + (y-hy)^2 < radius^2
```

每进入一个障碍物，当前步增加 `0.1 cost`。

这里机器人被当作一个点，只检查机器人中心；渲染出来的机器人半径没有参与碰撞计算。`cost` 与 `reward` 是两个独立指标，代码注释虽然写着“Include constraint cost in reward”，实际上并没有把 cost 加入 reward。

### 4.2.10 Goal compass 与距离编码

`get_obs()` 会把世界坐标系中的目标方向旋转到机器人自身坐标系：

```text
goal_compass_x：目标在机器人前后方向的位置
goal_compass_y：目标在机器人左右方向的位置
```

然后把方向向量近似归一化，使它主要表达“方向”，不表达距离。

距离单独编码为：

```text
exp(-dist_goal)
```

它的特点是：

- 越接近目标，数值越接近 `1`；
- 越远离目标，数值越接近 `0`；
- 数值始终有界，便于神经网络处理。

### 4.2.11 随机障碍物生成

`get_random_hazard_locations()` 尝试生成 6 个障碍物：

1. 在大约 `[-2.4,2.4]^2` 中随机选择中心。
2. 随机选择圆、正方形或三角形。
3. 尺寸在基础半径的 `80%` 至 `120%` 之间变化。
4. 障碍物中心之间必须保持一定距离。
5. 不能太靠近目标。
6. 不能太靠近四个候选初始状态。
7. 每个障碍物最多尝试放置 100 次。

如果某个障碍物尝试 100 次仍无法放下，就会跳过。因此最终数量可能少于 6。函数文档说会返回位置数组，但实际没有 `return`，而是直接修改 `self.hazards`。

### 4.2.12 `none` 的实际问题

当前判断只有：

```python
if obs_config == 'default':
elif obs_config == 'test':
else:
```

所以 `none` 没有专门分支。它和 `random` 完全一样，会生成随机障碍物。这是当前实现与名称含义不一致的地方。

### 4.2.13 非默认配置没有 cost

`info['cost']` 只在下面的条件中创建：

```python
if self.obs_config == 'default':
```

所以 `test`、`random` 和错误落入随机分支的 `none` 都不会报告碰撞 cost。主训练代码通过 `next_info.get('cost', 0)` 读取，因此不会报错，但会把缺失的 cost 当成 0。

这意味着：在随机或测试障碍物中，即使机器人穿过障碍物，现有统计也可能显示 `cost=0`。因此不能用该字段直接评价 zero-shot 测试时的真实碰撞情况。

整个环境的数据流可以归纳为：

```text
7维 obs -> SAC 输出 [v,omega]
       -> 可选 RCBF-QP 修正
       -> UnicycleEnv.step()
       -> 更新 3维 state
       -> 计算任务 reward 和可选 safety cost
       -> 重新生成 7维 next_obs
```

本次仅阅读和讲解，没有修改任何文件。

---

## 第二轮：Reward、修正量与 actor 学习

### 用户

> 我现在想知道，它的 reward 是如何设置的，我要在哪里看，它是将修正量作为惩罚直接加在 reward 上还是怎么操作，来让 agent 的输出贴近修正后的值？

### Codex

## 直接结论

当前 Unicycle 代码中：

- reward **没有**加入 CBF 修正量惩罚；
- 没有 `-lambda ||u_S||^2`；
- 没有 `-lambda ||u_safe-u_RL||^2`；
- 碰撞 cost 也没有加进 reward。

### Reward 在哪里看

在 [`envs/unicycle_env.py`](./envs/unicycle_env.py)：

```python
dist_goal = self._goal_dist()
reward = self.last_goal_dist - dist_goal
self.last_goal_dist = dist_goal

if self.goal_met():
    reward += self.reward_goal  # 1.0
```

数学上是：

```text
reward = 上一步到目标的距离 - 当前到目标的距离
```

所以：

- 靠近目标：正奖励；
- 远离目标：负奖励；
- 到达目标：额外奖励 `+1`。

它只鼓励机器人接近并到达目标。

碰撞 cost 在同一文件后面单独计算：

```python
info['cost'] += 0.1 * collision
```

训练循环也是分别统计：

```python
episode_reward += reward
episode_cost += next_info.get('cost', 0)
```

所以 `cost` 只是统计指标，没有进入 SAC 的 reward。

## 动作与修正量

动作流程是：

```text
策略输出：             u_RL
CBF-QP 输出修正量：    u_S
最终动作：             u_safe = u_RL + u_S
环境执行：             env.step(u_safe)
```

对应 [`rcbf_sac/sac_cbf.py`](./rcbf_sac/sac_cbf.py)：

```python
final_action = self.get_safe_action(...)
cbf_action = final_action - action
```

这里：

```text
action       = u_RL
cbf_action   = u_S
final_action = u_safe
```

传给环境的 `action` 已经是 `u_safe`。

## 谁在让两个动作接近

这里容易把方向理解反。

代码首先做的是：

```text
让修正后的 u_safe 尽量接近策略输出 u_RL
```

这是 QP 自己的目标，不是 reward 的目标。

Unicycle 的 QP 目标矩阵位于 [`rcbf_sac/diff_cbf_qp.py`](./rcbf_sac/diff_cbf_qp.py)：

```python
P = diag([1, 1e-2, 1e5])
```

QP 大致最小化：

```text
Delta_v^2 + 0.01 Delta_omega^2 + 100000 slack^2
```

其中：

```text
u_S = [Delta_v, Delta_omega]
```

因此，QP 在满足安全约束的前提下，寻找尽可能小的修正。角速度修正的权重更小，所以相对更允许通过转向避障。

## `full` 模式怎样影响 actor

`full` 模式不是给 reward 加惩罚，而是把 actor 输出送进**可微分 QP**：

```python
pi = policy(state)             # u_RL
pi = get_safe_action(..., pi)  # u_safe
Q = critic(state, pi)
```

Actor loss 仍然是标准 SAC 形式：

```python
policy_loss = (alpha * log_pi - Q(state, u_safe)).mean()
```

梯度路径是：

```text
actor参数
  -> u_RL
  -> 可微QP
  -> u_safe
  -> Q(s, u_safe)
  -> actor loss
```

也就是：

```text
d(loss)/d(actor)
=
d(loss)/d(u_safe)
* d(u_safe)/d(u_RL)
* d(u_RL)/d(actor)
```

因此 actor 学到的是：

> 我的动作经过安全层修正之后，能不能取得较高的任务回报？

但必须严格说明：**这里没有直接要求 `u_RL` 约等于 `u_safe` 的 MSE loss，所以不能保证修正量逐渐趋近于零。**它只是让策略训练时“知道”安全层的影响。

## 不同模式的区别

| 模式 | 环境执行 | Actor loss 经过 QP | 是否直接惩罚修正量 |
|---|---|---:|---:|
| `off` | `u_RL` | 否 | 否 |
| `baseline` | `u_safe` | 否 | 否 |
| `full` | `u_safe` | 是 | 否 |
| `mod` | 交互时 `u_safe` | Unicycle 更新时基本只裁剪动作 | 否 |

`baseline` 会在 replay buffer 中保存原始动作：

```python
memory.push(obs, action - cbf_action, reward, ...)
```

所以 baseline 更可能依赖安全层，并不明确训练 actor 模仿修正动作。

## Compensator 是另一种机制

只有额外的 compensator 使用了明确的监督学习 MSE：

```python
comp_actor_loss = MSE(compensator(obs), target_correction)
```

见 [`rcbf_sac/compensator.py`](./rcbf_sac/compensator.py)。

但这是训练一个单独的修正网络来近似 QP 修正量，不是训练 SAC actor 输出 `u_safe`。

所以最准确的总结是：

> Reward 只奖励接近目标；QP 负责以最小改动保证安全；`full` 模式通过可微 QP 把安全层影响传回 actor，但没有直接惩罚修正量，也没有直接让 actor 模仿 `u_safe`。

---

## 第三轮：代码来源、矩阵乘法、Reward 作用和 Zero-shot 碰撞

### 用户

> 4.2.3 默认障碍物  
> `obs_config='default'` 时创建 5 个半径为 0.6 的圆形障碍物：
>
> ```text
> ( 0.0,  0.0)
> (-1.5,  1.5)
> (-1.5, -1.5)
> ( 1.5, -1.5)
> ( 1.5,  1.5)
> ```
>
> 默认起点是 `(-2.5,-2.5)`，目标是 `(2.5,2.5)`，目标判定半径为 `0.3`。
>
> 你这个 4.2.3 的答案是哪里来的，我在 `unicycle_env.py` 中没看到设置默认障碍物的代码。  
> 同样的我也没看到 4.2.4 答案中的相关代码。
>
> 4.2.5 中的 `self.state += self.dt * (f + g @ action)` 中的 `@` 是什么运算符？
>
> 4.2.8 的 reward 是训练 actor 用的还是训练 critic 用的？
>
> 既然 4.2.13 中说到，cost 字段无法评价 zero-shot 测试时的真实碰撞情况，那么 zero-shot 的碰撞情况是如何被反映的？

### Codex

你可以在 `UnicycleEnv.__init__()` 中找到障碍物设置，它不在 `step()` 中。

## 1. 默认障碍物代码

看 [`envs/unicycle_env.py`](./envs/unicycle_env.py)：

```python
self.obs_config = obs_config
self.hazards = []

if obs_config == 'default':
    self.hazards.append({
        'type': 'circle',
        'radius': 0.6,
        'location': 1.5 * np.array([0., 0.])
    })
    self.hazards.append({
        'type': 'circle',
        'radius': 0.6,
        'location': 1.5 * np.array([-1., 1.])
    })
    ...
```

位置来自数组乘法：

```text
1.5 * [ 0,  0] = [ 0.0,  0.0]
1.5 * [-1,  1] = [-1.5,  1.5]
1.5 * [-1, -1] = [-1.5, -1.5]
1.5 * [ 1, -1] = [ 1.5, -1.5]
1.5 * [ 1,  1] = [ 1.5,  1.5]
```

每一个字典中都写了：

```python
'type': 'circle'
'radius': 0.6
```

所以得出“5 个半径为 0.6 的圆形障碍物”。

默认起点来自：

```python
self.initial_state = np.array([
    [-2.5, -2.5, 0.0],
    ...
])
```

当 `rand_init=False` 时，`reset()` 选择第一个状态：

```python
self.state = np.copy(self.initial_state[0])
```

目标和目标半径则是：

```python
self.goal_pos = np.array([2.5, 2.5])
self.goal_size = 0.3
```

## 2. `test/random/none` 的代码

紧接着默认障碍物分支：

```python
elif obs_config == 'test':
    # 添加固定的正方形
    ...
    # 添加固定圆形
    ...
    # 添加固定三角形
    ...
else:
    n_hazards = 6
    hazard_radius = 0.6
    self.get_random_hazard_locations(n_hazards, hazard_radius)
```

这里没有：

```python
elif obs_config == 'random'
elif obs_config == 'none'
```

因此实际判断是：

```text
default -> 默认固定障碍物
test    -> 测试固定障碍物
其他值  -> 随机障碍物
```

所以 `random` 和 `none` 都进入 `else`。

具体生成圆形、正方形和三角形的代码在 `get_random_hazard_locations()`。

## 3. `@` 是什么运算符

`@` 是 Python/NumPy 的**矩阵乘法运算符**。

这里：

```python
self.get_g(self.state) @ action
```

`g` 是一个 `3x2` 矩阵：

```text
g =
[cos(theta)   0]
[sin(theta)   0]
[    0        1]
```

动作是长度为 2 的向量：

```text
action = [v, omega]
```

矩阵乘法结果是：

```text
g @ action
=
[v cos(theta),
 v sin(theta),
 omega]
```

所以：

```python
self.state += self.dt * (
    self.get_f(self.state)
    + self.get_g(self.state) @ action
)
```

相当于：

```text
x     += dt * v cos(theta)
y     += dt * v sin(theta)
theta += dt * omega
```

`@` 与 `*` 不同：

```python
A @ B  # 矩阵乘法
A * B  # 对应位置逐元素相乘
```

## 4. Reward 训练 actor 还是 critic

答案是：

> reward 直接训练 critic，间接训练 actor。

Replay buffer 会保存 reward：

```python
memory.push(obs, action, reward, next_obs, ...)
```

更新 critic 时，reward 直接出现在目标 Q 值中：

```python
next_q_value = reward_batch + gamma * next_value
```

然后 critic 学习：

```python
qf1_loss = MSE(Q1(state, action), next_q_value)
qf2_loss = MSE(Q2(state, action), next_q_value)
```

Actor loss 中没有直接出现 reward：

```python
policy_loss = (
    alpha * log_pi
    - min_qf_pi
).mean()
```

但是 `min_qf_pi` 来自 critic，而 critic 学习的是累计 reward，因此关系是：

```text
reward
  -> 训练 critic 的 Q 值
  -> actor 寻找 Q 值更高的动作
  -> reward 间接训练 actor
```

在 `full` 模式下，actor 通过安全动作接受评价：

```text
actor输出 u_RL
-> 可微 QP 得到 u_safe
-> critic 计算 Q(state, u_safe)
-> 梯度经过 QP 返回 actor
```

## 5. Zero-shot 如何反映碰撞

严格来说：

> 当前代码的 zero-shot 测试结果没有定量统计碰撞。

`test()` 只统计：

```python
episode_rewards
dones                    # 是否到达目标
policy_timings
```

最终打印：

```text
mean_reward
std_reward
mean_completion
policy_mean_wct
```

这里的 `completion` 计算是：

```python
done and env.episode_step < env.max_episode_steps
```

对于 Unicycle：

- 到达目标会提前 `done=True`；
- 碰撞不会 `done=True`；
- 碰撞也不会降低 reward；
- 所以 completion 是任务完成率，不是安全率。

随机障碍物仍然会提供给 CBF-QP。测试时有：

```python
agent.cbf_layer.env = env
```

安全层从 `env.hazards` 读取圆形和多边形障碍物，并构造避障约束。

因此当前情况是：

```text
CBF-QP 会尝试避障
渲染画面可以肉眼观察碰撞
但 test() 不计算碰撞次数或碰撞率
```

所以不能根据 `mean_reward` 或 `mean_completion` 推断“没有碰撞”。甚至也不能只根据 QP 成功求解断定严格安全，因为 QP 中允许使用 slack 放松安全约束。当前 zero-shot 实现主要评价任务迁移能力，没有完整评价真实碰撞情况。

---

## 第四轮：生成对话文档

### 用户

> 把我们这个窗口中的完整对话，做成 `.md` 文档，放在项目根目录下。

### Codex

已将本窗口中围绕 TODO 4.2 和 `UnicycleEnv` 的完整教学对话整理到当前文档：

```text
UnicycleEnv_4.2_完整对话.md
```

文档位于项目根目录，没有修改任何项目源码。

---

## 第五轮：SAC Loss 与 Unicycle Reward 的关系

### 用户

> SAC 中的 loss 和 `unicycle_env` 中的 reward 有没有关系，如果有的话，是什么关系？

### Codex

有关系，而且关系非常直接。简单说：

> `unicycle_env.py` 负责产生 reward；SAC 利用 reward 训练 critic，再由 critic 间接指导 actor。

## 1. Reward 如何产生

在 [`envs/unicycle_env.py`](./envs/unicycle_env.py)：

```python
dist_goal = self._goal_dist()
reward = self.last_goal_dist - dist_goal

if self.goal_met():
    reward += self.reward_goal
```

即：

```text
r_t = 上一步距离 - 当前距离 + 到达目标奖励
```

例如：

```text
上一步距离目标 5.0
当前距离目标   4.8
reward = 5.0 - 4.8 = 0.2
```

如果远离目标：

```text
上一步距离目标 5.0
当前距离目标   5.1
reward = 5.0 - 5.1 = -0.1
```

## 2. Reward 进入 Replay Buffer

环境返回 reward：

```python
next_obs, reward, done, next_info = env.step(action)
```

然后在 [`main.py`](./main.py) 中存入 replay buffer：

```python
memory.push(
    obs,
    action,
    reward,
    next_obs,
    mask,
    ...
)
```

数据流是：

```text
UnicycleEnv 计算 reward
        ↓
Replay Buffer 保存 reward
        ↓
SAC 随机抽取 reward_batch
        ↓
计算 loss
```

## 3. Reward 直接影响 Critic Loss

SAC 从 replay buffer 中取出：

```python
reward_batch
```

然后计算目标 Q 值，见 [`rcbf_sac/sac_cbf.py`](./rcbf_sac/sac_cbf.py)：

```python
next_q_value = reward_batch + \
    mask_batch * self.gamma * min_qf_next_target
```

数学形式大致是：

```text
目标Q值 y =
当前 reward
+ gamma × 下一状态的预计价值
```

Critic loss 是：

```python
qf1_loss = MSE(qf1, next_q_value)
qf2_loss = MSE(qf2, next_q_value)
```

即：

```text
Critic Loss =
(Q网络预测值 - reward构造的目标Q值)²
```

所以 reward **直接参与 critic loss 的计算**。

如果动作持续靠近目标，reward 较高，critic 就会逐渐认为：

```text
这个状态下的这个动作具有较高 Q 值
```

## 4. Reward 间接影响 Actor Loss

Actor loss 在 [`rcbf_sac/sac_cbf.py`](./rcbf_sac/sac_cbf.py)：

```python
policy_loss = (
    self.alpha * log_pi
    - min_qf_pi
).mean()
```

数学上可以写成：

```text
Actor Loss = alpha × log pi(a|s) - Q(s,a)
```

Actor 希望最小化这个 loss。因为里面有 `-Q(s,a)`，所以它会倾向于选择让 Q 值更大的动作。

而 Q 值是 critic 根据 reward 学到的，因此：

```text
Unicycle reward
    ↓
训练 critic 的 Q 值
    ↓
actor 选择 Q 值更高的动作
    ↓
actor 学会向目标移动
```

因此：

- reward **直接训练 critic**；
- reward **通过 critic 间接训练 actor**；
- actor loss 中没有直接写 `reward`。

## 5. 三类 Loss 与 Reward 的关系

| SAC loss | 与 reward 的关系 |
|---|---|
| `qf1_loss` | 直接使用 reward |
| `qf2_loss` | 直接使用 reward |
| `policy_loss` | 通过 critic 的 Q 值间接使用 reward |
| `alpha_loss` | 调节策略熵，与环境 reward 没有直接关系 |

## 6. Full 模式中的关系

`cbf_mode='full'` 时，actor 输出先经过可微 CBF-QP：

```python
pi = self.policy.sample(state_batch)
pi = self.get_safe_action(state_batch, pi, ...)
qf1_pi, qf2_pi = self.critic(state_batch, pi)
```

此时数据链是：

```text
actor 输出 u_RL
    ↓
CBF-QP 得到 u_safe
    ↓
环境执行 u_safe
    ↓
UnicycleEnv 产生 reward
    ↓
critic 学习 Q(state, u_safe)
    ↓
actor loss 通过 critic 和可微 QP 更新 actor
```

因此在 `full` 模式下，actor 学习的是：

> 哪种策略输出经过安全层之后，能够获得更高的任务 reward。

## 7. Cost 不参与 SAC Loss

Unicycle 还计算：

```python
info['cost']
```

但训练代码只是单独统计：

```python
episode_cost += next_info.get('cost', 0)
```

它没有放入：

```python
next_q_value
policy_loss
```

所以当前代码中：

```text
reward → SAC loss → 更新 actor/critic
cost   → 只做统计，不更新 actor/critic
```

最核心的关系可以记成：

```text
reward 定义“什么行为是好的”
critic 根据 reward 学习评价动作
actor 根据 critic 的评价学习选择动作
```
