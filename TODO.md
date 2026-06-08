# Mod-RL-RCBF 复现代码完整教学方案与进度 TODO

> 目标：把这份复现代码从“能运行”推进到“能解释、能定位、能修改、能复现实验”。
>
> 使用规则：每完成一次教学，只把对应条目从 `- [ ]` 改成 `- [x]`，并在该条目后追加简短完成记录，例如：`✅ 2026-06-08 已讲：main.py 训练主循环`。
>
> 阅读约定：原有条目主要描述论文目标、算法设计意图和应掌握的知识。若当前仓库实现与论文表述存在差异，以新增的“当前代码说明 / 已知实现问题 / 验证注意”为准；学习时应同时记录“论文希望实现什么”和“当前代码实际做了什么”，不要把二者混为一谈。
> 请注意：用户的水平很低，python语法只会和C部分重叠的那部分（比如赋值，传参，函数等概念），以前也没有过科研经历，现在正在入门，所以讲解的时候要详细，不要泛泛而谈。
---

## 0. 进度总览

- [x] 0.0 建立本教学方案与进度文件。✅ 当前已完成
- [x] 0.1 确认你当前使用的项目根目录、Python/Conda 环境、PyTorch 设备、依赖版本。✅ 2026-06-08 已讲：项目路径、Conda/Python、CUDA 设备与关键依赖版本
- [ ] 0.2 跑通一个最小 smoke test，确认代码入口、参数名、输出目录都正常。
- [ ] 0.3 建立“论文公式 ↔ 代码文件 ↔ 运行现象”的对应表。
- [ ] 0.4 完成整份代码的逐模块阅读。
- [ ] 0.5 完成一次单环境完整训练流程复盘。
- [ ] 0.6 完成一次 CBF/RCBF-QP 安全层逐行复盘。
- [ ] 0.7 完成一次 SAC 损失函数与代码实现逐行复盘。
- [ ] 0.8 完成一次 model-based rollout 与 GP 扰动学习逐行复盘。
- [ ] 0.9 完成一次 `off / baseline / full / mod / compensator` 五种模式差异复盘。(实际代码只有四种cbf_mod，compensator是仅允许搭配model-free baseline的附加选项)
- [ ] 0.10 完成论文实验命令、指标、结果文件、图表生成的复现实操。
- [ ] 0.11 能独立回答组会/论文复现中最常见的代码级问题。
- [ ] 0.12 能独立修改一个环境、reward、CBF 或训练参数，并解释影响。
- [ ] 0.13 完成一次“论文设计意图 ↔ 当前代码行为 ↔ 已知实现问题”的差异审计。
- [ ] 0.14 建立可重复的实验统计协议，并区分“代码跑通”和“论文结果复现”。

---

## 1. 项目结构总览

### 1.1 文件树认知

- [x] 1.1.1 理解根目录文件：`README.md`、`main.py`、`build_env.py`、`plot_utils.py`。✅ 2026-06-08 已讲：根目录四个文件的职责与入口关系
- [x] 1.1.2 理解环境目录：`envs/unicycle_env.py`、`envs/simulated_cars_env.py`、`envs/pvtol_env.py`。✅ 2026-06-08 已讲：三个 Gym 环境及渲染辅助文件
- [x] 1.1.3 理解算法目录：`rcbf_sac/` 中的 SAC、CBF-QP、GP、模型回放、补偿器等模块。✅ 2026-06-08 已讲：SAC、安全层、动力学/GP、模型回放与补偿器的模块分工
- [x] 1.1.4 明确哪些文件是“主线必须读”，哪些文件是“辅助/绘图/渲染”。✅ 2026-06-08 已讲：主线阅读顺序与辅助文件分类
- [x] 1.1.5 明确 README 中历史命令与当前代码参数的差异：当前入口参数是 `--env_name`，不是 README 里的 `--env`。✅ 2026-06-08 已讲：README 的 `--env` 已失效，当前应使用 `--env_name`

### 1.2 项目一句话架构

- [x] 1.2.1 讲清楚：`main.py` 创建环境、创建 agent、创建动力学模型，然后进入训练/测试。✅ 2026-06-08 已讲：入口对象创建顺序和 train/test 分流
- [x] 1.2.2 讲清楚：环境负责状态转移和 reward；SAC agent 负责策略学习；RCBF layer 负责把动作变安全；GP/DynamicsModel 负责估计扰动与生成 model rollout。✅ 2026-06-08 已讲：环境、agent、安全层与动力学模型的职责边界
- [x] 1.2.3 画出代码数据流：`obs -> policy -> u_RL -> CBF-QP -> u_safe -> env.step -> replay buffer -> SAC update`。✅ 2026-06-08 已讲：真实交互、经验回放和可选模型回放的数据流

---

## 2. 环境与最小运行

### 2.1 依赖与设备

- [ ] 2.1.1 检查依赖：`torch`、`numpy`、`gym`、`qpth`、`gpytorch`、`quadprog`、`matplotlib`、`tqdm`、`comet_ml`。
- [ ] 2.1.2 明确 GPU/CPU 切换逻辑：只有加 `--cuda` 才会走 CUDA；否则默认 CPU。
- [ ] 2.1.3 明确 `--device_num` 如何选择 GPU。
- [ ] 2.1.4 确认当前环境是否能 import 全部模块。

### 2.2 最小 smoke test

- [ ] 2.2.1 运行最短 Unicycle 测试命令，确认 `main.py` 可以启动。
- [ ] 2.2.2 解释输出中的 `Episode ... eps_rew ... eps_cost ...`。
- [ ] 2.2.3 解释 `--max_episodes`、`--batch_size`、`--start_steps`、`--seed` 的实际作用。
- [ ] 2.2.4 确认模型保存目录 `output/<Env>-runX` 的生成规则。
- [ ] 2.2.5 明确该命令不是纯启动检查：默认 `--eval True` 会在第 0 个 episode 后额外运行 5 个评估 episode。
- [ ] 2.2.6 明确单个 episode 不会保存 actor/critic；当前代码仅在大于 0 且能被 20 整除的 episode 编号保存中间模型。
- [ ] 2.2.7 分开执行“CLI/import 快速检查”“单 episode 训练检查”“checkpoint 保存/加载检查”，不要用一个命令替代全部验收。

推荐最小命令：

```bash
python main.py --env_name Unicycle --cbf_mode off --max_episodes 1 --batch_size 256 --start_steps 5000 --seed 12345
```

验证注意：

- 该命令会完整运行一个最长 1000 步的 Unicycle episode，并可能额外进行评估，因此“最小”是指最少训练 episode，不代表只运行几步。
- 当前 `argparse` 对多个布尔参数使用 `type=bool`；例如字符串形式的 `--eval False` 仍可能被解析为 `True`，应在教学中单独验证。
- 只检查入口和参数时，优先先运行 `python main.py --help` 与依赖 import 检查，再决定是否执行完整 episode。

---

## 3. `main.py` 主流程教学

### 3.1 参数系统

- [ ] 3.1.1 逐项解释环境参数：`--env_name`、`--obs_config`、`--rand_init`。
- [ ] 3.1.2 逐项解释训练参数：`--mode`、`--output`、`--resume`、`--eval`。
- [ ] 3.1.3 逐项解释 SAC 参数：`--gamma`、`--tau`、`--lr`、`--alpha`、`--automatic_entropy_tuning`。
- [ ] 3.1.4 逐项解释采样参数：`--batch_size`、`--max_episodes`、`--updates_per_step`、`--start_steps`、`--replay_size`。
- [ ] 3.1.5 逐项解释 CBF/GP 参数：`--gp_model_size`、`--gp_max_episodes`、`--k_d`、`--gamma_b`、`--l_p`。
- [ ] 3.1.6 逐项解释 model-based 参数：`--model_based`、`--real_ratio`、`--k_horizon`、`--rollout_batch_size`。
- [ ] 3.1.7 逐项解释安全模式参数：`--cbf_mode`、`--use_comp`、`--comp_rate`、`--comp_train_episodes`、`--comp_update_episode`。
- [ ] 3.1.8 核对帮助文本、默认值和实际行为是否一致，例如 `start_steps`、自动熵调节和环境选项的 help 文本。
        main.py (line 276) 的帮助文本只写了 Unicycle 和 SimulatedCars，但实际上还支持 Pvtol。
        main.py (line 302) 默认自动调节熵是 True，帮助文本却写 False。
        main.py (line 314) 的实际 start_steps=5000，帮助文本却写 10000。
        另外，cbf_mode 没有设置 choices，所以拼写错误不会在参数解析阶段被拦截。

- [ ] 3.1.9 讲清 `type=bool` 的 argparse 陷阱涉及 `--rand_init`、`--eval`、`--automatic_entropy_tuning`、`--use_comp`。
        涉及 rand_init (line 278)、eval (line 291)、automatic_entropy_tuning (line 302)、use_comp (line 343)。
        argparse 的 type=bool 实际执行 bool("False")，而任何非空字符串都为真。因此：
        --eval False
        很可能仍然得到 args.eval == True。更可靠的设计应使用 store_true、store_false 或专门的字符串转布尔函数。

### 3.2 训练循环

- [ ] 3.2.1 讲清 `train(agent, env, dynamics_model, args, experiment=None)` 的输入输出。
- [ ] 3.2.2 讲清真实 replay buffer `memory` 和模型 replay buffer `memory_model` 的区别。
- [ ] 3.2.3 讲清 episode reset 后 `obs, info = env.reset()` 的含义。
- [ ] 3.2.4 讲清每一步如何从 obs 得到内部 state：`dynamics_model.get_state(obs)`。
- [ ] 3.2.5 讲清何时生成 model rollout：`args.model_based and episode_steps % 5 == 0 ...`。
- [ ] 3.2.6 讲清何时开始更新网络：`len(memory) + len(memory_model) * args.model_based > args.batch_size`。
- [ ] 3.2.7 讲清 action 采样：warmup 随机动作 vs policy 输出动作。
- [ ] 3.2.8 讲清 safe action：`agent.select_action(... safe_action=args.cbf_mode!='off')`。
- [ ] 3.2.9 讲清 `env.step(action)` 返回的 `next_obs, reward, done, next_info`。
- [ ] 3.2.10 讲清 `mask` 为什么把时间上限导致的 done 排除。
- [ ] 3.2.11 讲清不同 `cbf_mode` 存进 replay buffer 的 action 为什么不同。
- [ ] 3.2.12 讲清 GP 扰动学习数据如何加入：`dynamics_model.append_transition(...)`。
- [ ] 3.2.13 讲清模型保存：每 20 个 episode 保存 agent 和 disturbance models。
- [ ] 3.2.14 讲清 evaluation：每 5 个 episode 评估一次平均 reward/cost。
- [ ] 3.2.15 核对 compensator rollout 环形缓冲区更新逻辑，特别是缓冲区填满后的索引赋值代码。

### 3.3 测试流程

- [ ] 3.3.1 讲清 `test(agent, dynamics_model, args, visualize=True, debug=True)` 的流程。
- [ ] 3.3.2 讲清 `agent.load_weights` 与 `dynamics_model.load_disturbance_models`。
- [ ] 3.3.3 讲清测试时是否还经过 CBF 层：取决于 `args.cbf_mode != 'off'`。
- [ ] 3.3.4 讲清 zero-shot transfer 测试中 `obs_config=random` 的含义。
- [ ] 3.3.5 讲清 `dones.append(done and env.episode_step < env.max_episode_steps)` 如何统计 completion。
- [ ] 3.3.6 明确测试会同时加载 actor、critic 和 GP 文件；仅有 actor/critic checkpoint 时测试仍可能失败。
- [ ] 3.3.7 核对默认 `resume` 路径与训练输出编号是否一致：当前训练首次通常创建 `run1`，默认测试路径却指向 `run0`。

---

## 4. 环境代码教学

### 4.1 `build_env.py`

- [ ] 4.1.1 讲清 `build_env(env_name, obs_config='default', rand_init=False)` 是环境工厂。
- [ ] 4.1.2 讲清支持的三个环境：`Unicycle`、`SimulatedCars`、`Pvtol`。
- [ ] 4.1.3 讲清新增环境时应修改哪里。

### 4.2 `UnicycleEnv`

- [ ] 4.2.1 讲清状态、观测、动作维度：state 是 `[x, y, theta]`，obs 是 7 维，action 是 2 维。
- [ ] 4.2.2 讲清 `dt=0.02`、`max_episode_steps=1000`。
- [ ] 4.2.3 讲清默认障碍物配置 `obs_config='default'`。
- [ ] 4.2.4 讲清 `obs_config='test' / random / none` 相关逻辑。
- [ ] 4.2.5 讲清 `step` 与 `_step` 的分工。
- [ ] 4.2.6 讲清动力学：`self.state += dt * (f + g @ action)`。
- [ ] 4.2.7 讲清扰动项：`- dt * 0.1 * g @ [cos(theta), 0]`。
- [ ] 4.2.8 讲清 reward：上一时刻到目标距离减当前距离，到达目标额外奖励。
- [ ] 4.2.9 讲清 cost：进入障碍物半径范围时增加 cost。
- [ ] 4.2.10 讲清 `get_obs` 中 goal compass 与 `exp(-dist_goal)`。
- [ ] 4.2.11 讲清 `get_random_hazard_locations` 如何生成随机障碍物。
- [ ] 4.2.12 当前代码说明：除 `default` 和 `test` 外的所有 `obs_config` 都进入随机障碍物分支，因此 `obs_config='none'` 在 Unicycle 中并不表示无障碍。
- [ ] 4.2.13 当前代码说明：Unicycle 仅在 `obs_config='default'` 时写入 `info['cost']`，随机/测试障碍配置下不能直接依赖该字段统计碰撞。

### 4.3 `SimulatedCarsEnv`

- [ ] 4.3.1 讲清五车跟驰环境的 state、obs、action。
- [ ] 4.3.2 讲清 agent 只控制第 4 辆车。
- [ ] 4.3.3 讲清 reward 为什么是控制能量惩罚。
- [ ] 4.3.4 讲清 cost 与碰撞/安全距离的关系。
- [ ] 4.3.5 讲清 `reset` 如何初始化车队。
- [ ] 4.3.6 讲清该环境为什么比 Unicycle 更体现 differentiable CBF 的优势。
- [ ] 4.3.7 当前代码说明：SimulatedCars 的碰撞 cost 以 `-0.1` 累加，符号与 Unicycle/PVTOL 的正 cost 不一致，跨环境比较前必须统一指标解释。

### 4.4 `PvtolEnv`

- [ ] 4.4.1 讲清 PVTOL state、obs、action。
- [ ] 4.4.2 讲清 thrust、角速度、扰动项的建模。
- [ ] 4.4.3 讲清安全操作员 constraint 相关 `cbf_info`。
- [ ] 4.4.4 讲清 PVTOL 中为什么涉及 higher relative degree / cascaded RCBF。
- [ ] 4.4.5 讲清 PVTOL zero-shot transfer 的测试逻辑。
- [ ] 4.4.6 当前代码说明：审计 `DynamicsModel.get_obs` 中 `Pvtol`/`PVTOL` 名称和状态索引，确认 PVTOL model rollout 路径是否可执行。

---

## 5. SAC 基础实现教学

### 5.1 网络结构：`rcbf_sac/model.py`

- [ ] 5.1.1 讲清 `QNetwork` 的输入是 `(state, action)`，输出两个 Q 值。
- [ ] 5.1.2 讲清为什么有 twin critics：缓解 Q 过估计。
- [ ] 5.1.3 讲清 `GaussianPolicy.forward` 输出 mean 和 log_std。
- [ ] 5.1.4 讲清 `GaussianPolicy.sample` 的 reparameterization trick。
- [ ] 5.1.5 讲清 tanh squash 与 action rescale。
- [ ] 5.1.6 讲清 `DeterministicPolicy` 在本项目中的备用作用。

### 5.2 SAC agent：`rcbf_sac/sac_cbf.py`

- [ ] 5.2.1 讲清 `RCBF_SAC.__init__` 如何创建 critic、target critic、policy、CBF layer、compensator。
- [ ] 5.2.2 讲清 `select_action` 的四种情况：warmup、train sampling、evaluate deterministic、safe_action。
- [ ] 5.2.3 讲清 `update_parameters` 的 batch 采样逻辑。
- [ ] 5.2.4 讲清 critic target：`r + gamma * (min(Q_target) - alpha log pi)`。
- [ ] 5.2.5 讲清 critic loss：两个 MSE。
- [ ] 5.2.6 讲清 policy loss：`alpha * log_pi - min_qf_pi`。
- [ ] 5.2.7 讲清自动熵温度 `alpha` 的更新。
- [ ] 5.2.8 讲清 target network soft update。
- [ ] 5.2.9 讲清保存/加载 actor、critic 的文件。

### 5.3 Replay Buffer：`rcbf_sac/replay_memory.py`

- [ ] 5.3.1 讲清 buffer 存储元组：`state, action, reward, next_state, mask, t, next_t, cbf_info, next_cbf_info`。
- [ ] 5.3.2 讲清 circular buffer 的 position 更新。
- [ ] 5.3.3 讲清 `sample` 如何随机采样 batch。
- [ ] 5.3.4 讲清为什么要保存 `cbf_info` 和 `next_cbf_info`。

---

## 6. RCBF/CBF-QP 安全层教学

### 6.1 论文公式到代码

- [ ] 6.1.1 讲清论文中的 `u_RL`、`u_S`、`u* = u_RL + u_S` 在代码中的对应变量。
- [ ] 6.1.2 讲清 `cbf_mode='off'`：不使用安全层。
- [ ] 6.1.3 讲清 `cbf_mode='baseline'`：采样时用安全层，但训练 loss 不对安全层反传。
- [ ] 6.1.4 讲清 `cbf_mode='full'`：训练时 actor/critic 目标显式使用 safe action。
- [ ] 6.1.5 讲清 `cbf_mode='mod'`：constraint-agnostic / modular task learning。
- [ ] 6.1.6 讲清 `use_comp=True`：用神经网络 compensator 学 CBF-QP 输出。
- [ ] 6.1.7 当前代码说明：`cbf_mode` 只有 `off / baseline / full / mod` 四种；compensator 不是第五种 `cbf_mode`，而是 baseline 上的附加选项。
- [ ] 6.1.8 当前代码说明：`mod` 同时影响 critic target 和 actor loss。对 Unicycle/SimulatedCars，训练更新中的 modular safe action 基本退化为动作裁剪；对 PVTOL，仍保留场地边界、姿态和推力约束，但移除障碍物与 safety operator 相关约束。

### 6.2 `diff_cbf_qp.py`

- [ ] 6.2.1 讲清 `CBFQPLayer.__init__` 中环境、gamma_b、k_d、l_p 的作用。
- [ ] 6.2.2 讲清 `get_safe_action` 如何批量求 safe action。
- [ ] 6.2.3 讲清 `solve_qp` 如何调用 `qpth.qp.QPFunction`。
- [ ] 6.2.4 讲清 `cbf_layer` 中 QP 标准形式：目标函数矩阵与约束矩阵。
- [ ] 6.2.5 讲清 `get_cbf_qp_constraints` 在不同环境下如何生成 CBF 约束。
- [ ] 6.2.6 讲清 circle obstacle 的 barrier function。
- [ ] 6.2.7 讲清 polygon obstacle 的 segment distance barrier。
- [ ] 6.2.8 讲清 SimulatedCars 的 cascaded CBF 约束。
- [ ] 6.2.9 讲清 PVTOL 的 cascaded CBF 与 safety operator 约束。
- [ ] 6.2.10 讲清 slack variable 的作用：QP 不可行时软化约束。
- [ ] 6.2.11 讲清 action bounds 如何加入 QP。
- [ ] 6.2.12 讲清为什么 qpth 版本是“可微 QP 层”。
- [ ] 6.2.13 区分“理论上满足硬安全约束”和“当前带 slack QP 的实际约束违反”，记录 slack 与约束残差。
- [ ] 6.2.14 当前代码说明：`k_d` 被构造函数保存，但当前可微 QP 约束中未实际使用；不能仅凭命令行参数存在就认定置信系数已生效。

### 6.3 `cbf_qp.py`

- [ ] 6.3.1 讲清 `CascadeCBFLayer` 与 `CBFQPLayer` 的区别。
- [ ] 6.3.2 讲清 `quadprog.solve_qp` 非 PyTorch 可微链路的限制。
- [ ] 6.3.3 讲清这个文件在当前主流程中是否被直接调用。

---

## 7. GP 扰动学习与动力学模型教学

### 7.1 `rcbf_sac/dynamics.py`

- [ ] 7.1.1 讲清 `DynamicsModel.__init__` 如何根据 env 建立模型。
- [ ] 7.1.2 讲清 `get_state(obs)`：obs 到系统真实 state 的映射。
- [ ] 7.1.3 讲清 `get_obs(state)`：state 到 policy observation 的映射。
- [ ] 7.1.4 讲清 `get_dynamics`：返回先验动力学 `f(x)`、`g(x)`。
- [ ] 7.1.5 讲清 `append_transition` 如何构造 GP 训练数据。
- [ ] 7.1.6 讲清 `fit_gp_model` 何时训练 GP。
- [ ] 7.1.7 讲清 `predict_disturbance` 如何输出均值与标准差。
- [ ] 7.1.8 讲清 `predict_next_state` 如何用于 model rollout。
- [ ] 7.1.9 讲清 GP 模型保存/加载。

### 7.2 `rcbf_sac/gp_model.py`

- [ ] 7.2.1 讲清 `BaseGPy` 的 GP mean/covar 模块。
- [ ] 7.2.2 讲清为什么每个状态维度训练一个 GP。
- [ ] 7.2.3 讲清 `GPyDisturbanceEstimator.train` 的训练流程。
- [ ] 7.2.4 讲清 `GPyDisturbanceEstimator.predict` 的输出含义。
- [ ] 7.2.5 讲清论文中的扰动集合 `D(x)` 如何由 GP 均值和方差给出。

---

## 8. Model-Based Rollout 教学

### 8.1 `rcbf_sac/generate_rollouts.py`

- [ ] 8.1.1 讲清何时触发 model rollout。
- [ ] 8.1.2 讲清 rollout 从真实 replay buffer 中采样初始状态。
- [ ] 8.1.3 讲清 policy/CBF 如何给出 synthetic action。
- [ ] 8.1.4 讲清 DynamicsModel 如何预测 next state。
- [ ] 8.1.5 讲清 Unicycle synthetic reward 如何计算。
- [ ] 8.1.6 讲清 SimulatedCars synthetic reward 如何计算。
- [ ] 8.1.7 讲清 synthetic transition 如何进入 `memory_model`。
- [ ] 8.1.8 讲清 `real_ratio` 如何控制真实数据和模型数据的比例。
- [ ] 8.1.9 讲清 model rollout 为什么通常只用短 horizon。
- [ ] 8.1.10 当前代码说明：`generate_model_rollouts.py` 只实现 Unicycle 和 SimulatedCars reward/done，PVTOL 会进入不支持环境的异常分支。
- [ ] 8.1.11 当前代码说明：核对 Unicycle synthetic reward 的到达目标奖励是否被重复累加。
- [ ] 8.1.12 当前代码说明：当 `k_horizon > 1` 时核对传给 `predict_next_state` 的是否为更新后的 `t_batch_`，避免每一步重复使用初始时间。
- [ ] 8.1.13 当前代码说明：核对 synthetic transition 是否需要保存 `cbf_info/next_cbf_info`，以及 PVTOL safety operator 等动态约束信息如何传播。

---

## 9. Compensator baseline 教学

### 9.1 `rcbf_sac/compensator.py`

- [ ] 9.1.1 讲清 compensator 的目标：学习 CBF-QP 输出的修正量。
- [ ] 9.1.2 讲清 `CompensatorModel` 的网络结构。
- [ ] 9.1.3 讲清 `Compensator.__call__` 如何输出修正动作。
- [ ] 9.1.4 讲清 compensator 训练数据来自 `episode_rollout['u_safe']` 与 `episode_rollout['u_comp']`。
- [ ] 9.1.5 讲清为什么 `use_comp` 只能配合 `cbf_mode='baseline'` 和 model-free。
- [ ] 9.1.6 讲清 compensator 与 differentiable CBF layer 的本质区别。
- [ ] 9.1.7 明确 compensator 是实验方法组合，不是独立的 `cbf_mode` 枚举值。
- [ ] 9.1.8 审计 compensator 数据缓冲区填满后的替换逻辑，确认长于 50 个 rollout 的训练不会因索引代码失败。

---

## 10. 四类/五类方法对比教学

- [ ] 10.1 对比 SAC：`cbf_mode=off`，没有安全层。
- [ ] 10.2 对比 Baseline：采样时有 RCBF-QP，loss 不显式考虑 safe action。
- [ ] 10.3 对比 Baseline + Compensator：用 NN 近似 RCBF-QP 修正。
- [ ] 10.4 对比 MF SAC-RCBF：不用 synthetic rollout，但用 differentiable CBF 影响训练。
- [ ] 10.5 对比 MB SAC-RCBF：用 differentiable CBF + GP synthetic rollout。
- [ ] 10.6 对比 Constraint-Agnostic SAC-RCBF：训练安全，但 cost-to-go 忽略下一步安全层影响。
- [ ] 10.7 能用一句话说明每种方法“安全层参与采样吗、参与 critic target 吗、参与 actor loss 吗、用不用模型 rollout”。
- [ ] 10.8 在保留论文方法定义的同时，另画一张“当前代码实际分支表”，标出 `select_action`、critic target、actor loss、replay action 和 modular 约束的真实行为。

---

## 11. 论文实验复现教学

### 11.1 实验 1：Sample Efficiency

- [ ] 11.1.1 复现实验 1.1：Unicycle Baseline。
- [ ] 11.1.2 复现实验 1.1：Unicycle Baseline + Compensator。
- [ ] 11.1.3 复现实验 1.1：Unicycle MF SAC-RCBF。
- [ ] 11.1.4 复现实验 1.1：Unicycle MB SAC-RCBF。
- [ ] 11.1.5 复现实验 1.2：SimulatedCars Baseline。
- [ ] 11.1.6 复现实验 1.2：SimulatedCars Baseline + Compensator。
- [ ] 11.1.7 复现实验 1.2：SimulatedCars MF SAC-RCBF。
- [ ] 11.1.8 复现实验 1.2：SimulatedCars MB SAC-RCBF。
- [ ] 11.1.9 解释 Fig.2/Fig.3 中 reward 曲线与 sample efficiency 的含义。

### 11.2 实验 2：Constraint-Agnostic / Zero-Shot Transfer

- [ ] 11.2.1 复现 Unicycle SAC upper bound：无障碍、无安全层。
- [ ] 11.2.2 复现 Unicycle SAC-RCBF：`cbf_mode=full`。
- [ ] 11.2.3 复现 Unicycle Constraint-Agnostic SAC-RCBF：`cbf_mode=mod`。
- [ ] 11.2.4 复现 Unicycle Baseline。
- [ ] 11.2.5 运行 Unicycle zero-shot transfer：`--mode test --obs_config random`。
- [ ] 11.2.6 生成并解释 Unicycle value function heatmap。
- [ ] 11.2.7 复现 PVTOL upper bound、baseline、full、mod。
- [ ] 11.2.8 运行 PVTOL zero-shot transfer。
- [ ] 11.2.9 解释 Table I：Mean Ep. Rew.、Std Ep. Rew.、Mean Comp.。
- [ ] 11.2.10 解释为什么安全层保证安全不等于一定完成任务。

### 11.3 严格复现实验统计协议

- [ ] 11.3.1 区分三个层级：代码可启动、单次实验可完成、统计结果可复现论文。
- [ ] 11.3.2 为每种方法明确随机种子列表和重复次数，不能只报告单个 `seed=12345`。
- [ ] 11.3.3 固定环境配置、训练 episode 数、warmup、评估频率、评估 episode 数和硬件/软件版本。
- [ ] 11.3.4 明确 checkpoint 选择规则：最终 checkpoint、固定 episode checkpoint，或按验证指标选择；禁止测试后挑选最优结果。
- [ ] 11.3.5 明确训练曲线的横轴、滑动平均窗口、置信区间或标准差计算方式。
- [ ] 11.3.6 对 reward、cost、completion、碰撞/越界次数、QP slack 和运行时间分别定义统计口径。
- [ ] 11.3.7 保存每个 seed 的原始 episode 数据、参数快照、commit/diff 状态和最终聚合结果。
- [ ] 11.3.8 为 Fig.2、Fig.3、Table I 分别建立“论文指标 ↔ 本地输出 ↔ 聚合脚本/步骤”的对应表。
- [ ] 11.3.9 在结论中分别报告“趋势一致”“数值接近”“未复现”，避免仅凭曲线外观宣称复现成功。
- [ ] 11.3.10 记录 GPU/CPU、训练墙钟时间、QP 平均耗时和失败次数，使效率比较可解释。

---

## 12. 代码级调试与验证教学

### 12.1 常见报错

- [ ] 12.1.1 处理 `ModuleNotFoundError: comet_ml`。
- [ ] 12.1.2 处理 `ModuleNotFoundError: qpth/gpytorch/quadprog`。
- [ ] 12.1.3 处理 Gym 版本兼容问题。
- [ ] 12.1.4 处理 CUDA 不可用或显存不足。
- [ ] 12.1.5 处理 Pyglet/render 相关问题。
- [ ] 12.1.6 处理 `--env` 与 `--env_name` 参数不一致问题。
- [ ] 12.1.7 处理 `bool` 参数如 `--rand_init True` 的 argparse 陷阱。
- [ ] 12.1.8 处理训练首次输出通常为 `run1`、默认测试却查找 `run0` 的路径不一致。
- [ ] 12.1.9 处理只有 actor/critic、没有 GP 文件时测试加载失败的问题。
- [ ] 12.1.10 处理 PVTOL model rollout 当前不受支持的问题。

### 12.2 可观测调试点

- [ ] 12.2.1 在 `main.py` 打印 obs/action/cbf_action/reward/cost。
- [ ] 12.2.2 在 `sac_cbf.py` 打印 policy action 与 safe action 的差异。
- [ ] 12.2.3 在 `diff_cbf_qp.py` 打印 QP 约束矩阵和解。
- [ ] 12.2.4 在 `dynamics.py` 打印 GP 扰动均值/方差。
- [ ] 12.2.5 在 `generate_rollouts.py` 打印 synthetic transition。
- [ ] 12.2.6 验证 CBF 值是否始终非负或至少 cost 不增加。
- [ ] 12.2.7 对 12.2.6 增加严格解释：CBF 非负或累计 cost 变化只能作为观察量，不能单独证明安全，尤其当前 QP 允许 slack。
- [ ] 12.2.8 记录每一步最小 barrier 值、QP 约束残差、slack、实际碰撞/越界事件和 action bound 违反次数。
- [ ] 12.2.9 同时记录 `u_RL`、QP 修正量 `u_S`、compensator 修正量和最终环境动作，避免把不同动作变量混淆。
- [ ] 12.2.10 统一不同环境的 cost 符号和含义后再聚合比较。

### 12.3 当前代码已知问题审计

- [ ] 12.3.1 审计 `get_output_folder` 的首次 run 编号与默认 `resume` 路径不一致。
- [ ] 12.3.2 审计所有 `type=bool` 参数，验证字符串 `False` 的实际解析结果。
- [ ] 12.3.3 审计 PVTOL 的 `DynamicsModel.get_obs` 环境名称大小写和状态索引。
- [ ] 12.3.4 审计 Unicycle synthetic reward 中 goal reward 的重复累加。
- [ ] 12.3.5 审计多步 model rollout 的时间变量更新。
- [ ] 12.3.6 审计 model rollout 的环境支持范围和 `cbf_info` 传播。
- [ ] 12.3.7 审计 Unicycle `obs_config='none'` 的真实行为。
- [ ] 12.3.8 审计不同环境 cost 的符号、缺省值和统计可比性。
- [ ] 12.3.9 审计 `k_d` 是否真正参与当前可微 RCBF 约束。
- [ ] 12.3.10 审计 compensator rollout 缓冲区填满后的替换代码。
- [ ] 12.3.11 对每个问题记录状态：仅影响解释、影响单一实验、阻断实验、已修复但尚未验证。

---

## 13. 修改能力教学

### 13.1 修改环境

- [ ] 13.1.1 修改 Unicycle 初始位置并解释影响。
- [ ] 13.1.2 修改 goal 位置并解释 observation 中 goal compass 的变化。
- [ ] 13.1.3 修改障碍物数量/位置/半径。
- [ ] 13.1.4 修改 reward 结构。
- [ ] 13.1.5 修改 episode 长度。

### 13.2 修改安全层

- [ ] 13.2.1 修改安全半径 `delta`。
- [ ] 13.2.2 修改 barrier function 的 class-K 系数 `gamma_b`。
- [ ] 13.2.3 修改 QP slack 惩罚权重。
- [ ] 13.2.4 修改 action bounds。
- [ ] 13.2.5 新增一个简单 CBF 约束。

### 13.3 修改算法训练

- [ ] 13.3.1 修改 batch size、learning rate、hidden size。
- [ ] 13.3.2 修改 start_steps 和 updates_per_step。
- [ ] 13.3.3 修改 model rollout horizon。
- [ ] 13.3.4 修改 real_ratio。
- [ ] 13.3.5 修改 GP 最大训练 episode。

---

## 14. 最终验收问题清单

完成整份代码学习后，应能独立回答以下问题：

- [ ] 14.1 `action`、`cbf_action`、`u_RL`、`u_S`、`u_safe` 分别在哪里出现？
- [ ] 14.2 为什么 baseline 的 replay buffer 中存的是 `action-cbf_action`？
- [ ] 14.3 为什么 full/mod 模式下 replay buffer 中存的是 safe action？
- [ ] 14.4 `cbf_mode=mod` 到底在哪里把下一步的安全层影响去掉？
- [ ] 14.5 actor loss 中的 safe action 梯度是如何穿过 QP 层的？
- [ ] 14.6 critic target 中的 `next_state_action` 在不同模式下有什么区别？
- [ ] 14.7 GP 学的是动力学本身，还是 nominal dynamics 之外的 disturbance？
- [ ] 14.8 model-based rollout 生成的数据和真实数据如何混合训练？
- [ ] 14.9 reward 里有没有直接惩罚 CBF 修正量？
- [ ] 14.10 safety cost 和 reward 是不是同一个东西？
- [ ] 14.11 为什么安全训练不等于高 completion？
- [ ] 14.12 为什么 Constraint-Agnostic 方法更适合 zero-shot transfer？
- [ ] 14.13 如果想复现 Table I，应该训练哪些模型、测试哪些配置？
- [ ] 14.14 如果想把这个方法迁移到新机器人，需要改哪些文件？
- [ ] 14.15 哪些结论来自论文设计，哪些结论能由当前代码直接验证？
- [ ] 14.16 为什么带 slack 的 QP 不能仅凭“求解成功”就宣称严格安全？
- [ ] 14.17 当前代码中哪些问题会阻断 PVTOL/model-based/compensator 实验？
- [ ] 14.18 怎样用多 seed、固定 checkpoint 规则和原始数据证明结果具有可重复性？

---

## 15. 推荐教学顺序

1. 第 0 讲：运行环境与 smoke test。
2. 第 1 讲：项目总览和主数据流。
3. 第 2 讲：`main.py` 训练循环。
4. 第 3 讲：Unicycle 环境。
5. 第 4 讲：SAC 网络和损失函数。
6. 第 5 讲：安全层入口与 `cbf_mode`。
7. 第 6 讲：`diff_cbf_qp.py` 的 QP 细节。
8. 第 7 讲：GP 扰动学习与 DynamicsModel。
9. 第 8 讲：model-based rollout。
10. 第 9 讲：baseline/compensator/full/mod 对比。
11. 第 10 讲：论文实验复现命令与结果解释。
12. 第 11 讲：调试、修改和最终验收。
13. 补充专题 A：当前代码已知问题审计，区分论文意图与实际实现。
14. 补充专题 B：严格复现实验统计协议。

---

## 16. 关键代码定位表

| 主题 | 主要文件 | 重点对象/函数 |
|---|---|---|
| 程序入口 | `main.py` | `train`、`test`、argparse |
| 环境选择 | `build_env.py` | `build_env` |
| Unicycle 环境 | `envs/unicycle_env.py` | `UnicycleEnv.step/reset/get_obs/_get_dynamics` |
| Cars 环境 | `envs/simulated_cars_env.py` | `SimulatedCarsEnv.step/_get_reward/_get_cost` |
| PVTOL 环境 | `envs/pvtol_env.py` | `PvtolEnv.step/get_obs` |
| SAC agent | `rcbf_sac/sac_cbf.py` | `RCBF_SAC.select_action/update_parameters/get_safe_action` |
| Actor/Critic | `rcbf_sac/model.py` | `GaussianPolicy`、`QNetwork` |
| 可微 CBF-QP | `rcbf_sac/diff_cbf_qp.py` | `CBFQPLayer.get_safe_action/solve_qp/get_cbf_qp_constraints` |
| 非可微 QP 参考 | `rcbf_sac/cbf_qp.py` | `CascadeCBFLayer` |
| GP 与动力学 | `rcbf_sac/dynamics.py` | `DynamicsModel.predict_disturbance/predict_next_state` |
| GP 模型 | `rcbf_sac/gp_model.py` | `GPyDisturbanceEstimator` |
| 模型 rollout | `rcbf_sac/generate_rollouts.py` | `generate_model_rollouts` |
| Replay buffer | `rcbf_sac/replay_memory.py` | `ReplayMemory.push/sample` |
| Compensator | `rcbf_sac/compensator.py` | `Compensator.train` |
| 画 value heatmap | `plot_utils.py` | `plot_value_function` |

---

## 17. 本项目的核心理解主线

这份代码不是普通 SAC，而是：

```text
SAC policy 输出 u_RL
        ↓
RCBF-QP 根据动力学、障碍物/安全约束、GP 扰动估计，求最小修正 u_S
        ↓
环境执行 u_safe = u_RL + u_S
        ↓
真实 transition 进入 replay buffer
        ↓
根据 cbf_mode 决定 SAC 的 critic target / actor loss 是否显式经过安全层
        ↓
可选：用 GP + prior dynamics 生成 synthetic rollout，提高 sample efficiency
```

当前代码阅读时，还必须在这条理想主线旁维护一条“实现审计线”：

```text
论文/README 描述
        ↓
当前参数和条件分支
        ↓
当前实际存储、训练、测试行为
        ↓
已知缺陷与实验支持范围
        ↓
验证证据和可复现结论
```

学习时必须始终追踪 4 条线：

- [ ] 状态线：`obs -> state -> next_state -> next_obs`。
- [ ] 动作线：`u_RL -> u_S -> u_safe -> env.step`。
- [ ] 学习线：`replay buffer -> critic loss -> actor loss -> target update`。
- [ ] 安全线：`CBF/RCBF constraint -> QP -> safe action -> cost/safety`。

并额外追踪一条当前仓库实现审计线：

- [ ] 审计线：`论文意图 -> 当前实现 -> 已知问题 -> 验证结果`。
