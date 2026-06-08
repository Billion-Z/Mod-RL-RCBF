# main.py 不是算法细节文件，它是“把所有模块接起来跑实验”的入口文件。

# import comet_ml at the top of your file
from comet_ml import Experiment

import argparse
import time
import torch
import numpy as np

from rcbf_sac.generate_rollouts import generate_model_rollouts
from rcbf_sac.sac_cbf import RCBF_SAC
from rcbf_sac.replay_memory import ReplayMemory
from rcbf_sac.dynamics import DynamicsModel
from build_env import *
import os

from rcbf_sac.utils import prGreen, get_output_folder, prYellow

# 训练流程
# agent：选择动作、执行 SAC 更新、保存权重。
# env：执行状态转移并计算 reward/cost。
# dynamics_model：提取内部 state、学习扰动、生成模型预测。
# args：全部运行参数。
# experiment：可选 Comet 日志对象。
def train(agent, env, dynamics_model, args, experiment=None):

    # Load the weight if we're continuing training
    if hasattr(args, 'load_agent'):
        agent.load_weights(args.resume)

    # Memory
    # 创建真实经验池 memory,保存真实环境产生的 transition。
    memory = ReplayMemory(args.replay_size, args.seed)
    # 创建模型经验池 memory_model,保存 DynamicsModel 生成的虚拟 transition。
    memory_model = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    if args.use_comp:
        compensator_rollouts = []
        comp_buffer_idx = 0

    # 对每一个episode
    for i_episode in range(args.max_episodes):
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        done = False

        # obs 是策略网络输入，info 保存环境附加信息，例如 PVTOL 的 cbf_info
        obs, info = env.reset()

        # Saving rollout here to train compensator
        if args.use_comp:
            episode_rollout = dict()
            episode_rollout['obs'] = np.zeros((0, env.observation_space.shape[0]))
            episode_rollout['u_safe'] = np.zeros((0, env.action_space.shape[0]))
            episode_rollout['u_comp'] = np.zeros((0, env.action_space.shape[0]))

        # 如果这一局还没结束
        while not done:
            if episode_steps % 10 == 0:
                prYellow('Episode {} - step {} - eps_rew {} - eps_cost {}'.format(i_episode, episode_steps, episode_reward, episode_cost))
            
            # 环境 observation 是给神经网络使用的表示，state 是动力学方程和 CBF 使用的物理状态，两者不一定相同。
            state = dynamics_model.get_state(obs)

            # 生成 model rollout
            # 如果启用 model_based,且当前 episode 步数能被 5 整除，真实经验数量超过 GP 历史容量的三分之一。
            if args.model_based and episode_steps % 5 == 0 and len(memory) > dynamics_model.max_history_count / 3:
                # 用动力学模型生成一些虚拟经验
                memory_model = generate_model_rollouts(env, memory_model, memory, agent, dynamics_model,
                                                       k_horizon=args.k_horizon,
                                                       batch_size=min(len(memory), 5 * args.rollout_batch_size),
                                                    #    训练初期使用随机动作生成 rollout；warmup 结束后使用 policy 动作。
                                                       warmup=args.start_steps > total_numsteps)

            # 何时更新网络
            # If using model-based RL then we only need to have enough data for the real portion of the replay buffer
            # 经验总量必须大于 batch_size。
            if len(memory) + len(memory_model) * args.model_based > args.batch_size:

                #更新 agent 的神经网络
                # Number of updates per step in environment
                # 表示每执行一个环境 step，重复更新网络 updates_per_step 次。
                for i in range(args.updates_per_step):

                    # Update parameters of all the networks
                    if args.model_based:
                        # Pick the ratio of data to be sampled from the real vs model buffers
                        real_ratio = max(min(args.real_ratio, len(memory) / args.batch_size),
                                         1 - len(memory_model) / args.batch_size)
                        # Update parameters of all the networks
#                         Model-based 调用，会同时传入：
                            # memory_model
                            # real_ratio
                            # real_ratio=0.3 表示理想情况下，一个 batch 约 30% 来自真实经验，70% 来自模型经验。代码还会根据两个 buffer 的实际大小调整该比例。
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                             args.batch_size,
                                                                                                             updates,
                                                                                                             dynamics_model,
                                                                                                             memory_model,
                                                                                                             real_ratio)
                    else:
                        # Model-free 调用
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates,
                                                                                                         dynamics_model)

                    if experiment:
                        experiment.log_metric('loss/critic_1', critic_1_loss, updates)
                        experiment.log_metric('loss/critic_2', critic_2_loss, step=updates)
                        experiment.log_metric('loss/policy', policy_loss, step=updates)
                        experiment.log_metric('loss/entropy_loss', ent_loss, step=updates)
                        experiment.log_metric('entropy_temperature/alpha', alpha, step=updates)
                    updates += 1

            # Sample action from policy
            # Warmup 随机动作与 Policy 动作
            if args.use_comp:
                # warmup=args.start_steps > total_numsteps, 如果前者大于后者，则还处于热身阶段，此阶段会随机采样动作，而不是policy神经网络更具当前obs产生动作。
                # 动作的具体选择位于 sac_cbf.py (line 73)。
                action, comp_action, cbf_action = agent.select_action(obs, dynamics_model,
                                                                      warmup=args.start_steps > total_numsteps, safe_action=args.cbf_mode!='off', cbf_info=info.get('cbf_info', None))
            else:
                action, cbf_action = agent.select_action(obs, dynamics_model,
                                             warmup=args.start_steps > total_numsteps, safe_action=args.cbf_mode!='off', cbf_info=info.get('cbf_info', None))  # Sample action from policy

#             action：真正发送给环境的动作。
            # next_obs：执行动作后的新观测。
            # reward：这一步获得的任务奖励。
            # done：当前 episode 是否结束。
            # next_info：额外信息，例如碰撞 cost、CBF 信息。
            next_obs, reward, done, next_info = env.step(action)  # Step
            if 'cost_exception' in next_info:
                prYellow('Cost exception occured.')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += next_info.get('cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#             这是 Python 的条件表达式，相当于：
            # if episode_steps == env.max_episode_steps:
            #     mask = 1
            # else:
            #     mask = float(not done)
#             done 可能有两种原因：
            # 智能体真正到达目标，任务自然终止。
            # 智能体没有完成任务，但已经达到最大步数，程序强制停止。
#           SAC 计算目标 Q 值时大致使用：
            # 目标值 = reward + mask × gamma × 下一个状态价值
            # 真正终止时 mask=0，不再计算后续价值。
            # 时间上限只是程序截断，并不表示物理任务真正结束，所以设置 mask=1，仍然允许估计后续价值。
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            if args.use_comp:  # action is (rl_action + cbf_action + comp_action)
                memory.push(obs, action-cbf_action-comp_action, reward, next_obs, mask, t=episode_steps * env.dt, next_t=(episode_steps+1) * env.dt, cbf_info=info.get('cbf_info', None), next_cbf_info=next_info.get('cbf_info', None))  # Append transition to memory
            elif args.cbf_mode == 'baseline':  # action is (rl_action + cbf_action)
                memory.push(obs, action-cbf_action, reward, next_obs, mask, t=episode_steps * env.dt, next_t=(episode_steps+1) * env.dt, cbf_info=info.get('cbf_info', None), next_cbf_info=next_info.get('cbf_info', None))  # Append transition to memory
            else:
                memory.push(obs, action, reward, next_obs, mask, t=episode_steps * env.dt, next_t=(episode_steps+1) * env.dt, cbf_info=info.get('cbf_info', None), next_cbf_info=next_info.get('cbf_info', None))  # Append transition to memory

            # 3.2.12 收集 GP 扰动数据
            # Update state and store transition for GP model learning
            # 这会把神经网络使用的 observation 转换为动力学使用的物理 state。
            next_state = dynamics_model.get_state(next_obs)
            # 表示每隔两步收集一次，并且只在训练前 gp_max_episodes 个 episode 收集。
            if episode_steps % 2 == 0 and i_episode < args.gp_max_episodes:  # Stop learning the dynamics after a while to stabilize learning
                # TODO: Clean up line below, specifically (t_batch)
                dynamics_model.append_transition(state, action, next_state, t_batch=np.array([episode_steps*env.dt]))

            # append comp rollout with step before updating
            if args.use_comp:
                episode_rollout['obs'] = np.vstack((episode_rollout['obs'], obs))
                episode_rollout['u_safe'] = np.vstack((episode_rollout['u_safe'], cbf_action))
                episode_rollout['u_comp'] = np.vstack((episode_rollout['u_comp'], comp_action))

            obs = next_obs
            info = next_info

        # Train compensator
        # 3.2.15 Compensator 缓冲区问题
        if args.use_comp and i_episode < args.comp_train_episodes:
            if comp_buffer_idx < 50:  # TODO: Turn the 50 into an arg
                # 设计目标是最多保存最近 50 个 episode rollout。前 50 个应该追加没错
                compensator_rollouts.append(episode_rollout)
            else:
#                 填满后应该覆盖旧位置：
                # compensator_rollouts[comp_buffer_idx] = episode_rollout
                # 作者也在这个地方写了一个TODO
                comp_buffer_idx[comp_buffer_idx] = episode_rollout

            # comp_buffer_idx = (comp_buffer_idx + 1) % 50
            # 使它始终位于 0～49，所以前面的：
            # if comp_buffer_idx < 50:
            # 永远为真，错误的 else 实际上永远进不去。结果是列表会不断追加，并没有真正形成长度为 50 的环形缓冲区。
            comp_buffer_idx = (comp_buffer_idx + 1) % 50
            if i_episode % args.comp_update_episode == 0:
                agent.update_parameters_compensator(compensator_rollouts)

        # [optional] save intermediate model
        # 3.2.13 保存模型
        if i_episode > 0 and i_episode % 20 == 0:
            #  保存的内容，前者保存 actor、critic 和可选 compensator；后者保存 GP 权重及训练数据。
            # 每次保存使用同一文件名，因此后一次保存会覆盖前一次，而不是为每个 episode 单独创建 checkpoint。
            agent.save_model(args.output)
            dynamics_model.save_disturbance_models(args.output)

        if experiment:
            # Comet.ml logging
            experiment.log_metric('reward/train', episode_reward, step=i_episode)
            experiment.log_metric('cost/train', episode_cost, step=i_episode)
        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                             round(episode_reward, 2), round(episode_cost, 2)))

        # Evaluation
        # 3.2.14 训练期间 Evaluation
        if i_episode % 5 == 0 and args.eval is True:
#             评估过程不会：
            # 把经验写入 Replay Buffer；
            # 更新 actor 或 critic；
            # 增加 total_numsteps。
            # 它只统计当前策略的平均 reward 和 cost。
            print('Size of replay buffers: real : {}, \t\t model : {}'.format(len(memory), len(memory_model)))
            avg_reward = 0.
            avg_cost = 0.
            # 每次评估固定运行 5 个 episode：
            episodes = 5
            for _ in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_cost = 0
                done = False
                while not done:
                    # evaluate=True，这会让 Gaussian policy 使用均值动作，而不是随机采样动作，因此评估结果更稳定。
                    action = agent.select_action(obs, dynamics_model, evaluate=True, safe_action=args.cbf_mode!='off')[0]  # Sample action from policy
                    next_obs, reward, done, next_info = env.step(action)
                    episode_reward += reward
                    episode_cost += next_info.get('cost', 0)
                    obs = next_obs
                    info = next_info

                avg_reward += episode_reward
                avg_cost += episode_cost
            avg_reward /= episodes
            avg_cost /= episodes
            if experiment:
                experiment.log_metric('avg_reward/test', avg_reward, step=i_episode)
                experiment.log_metric('avg_cost/test', avg_cost, step=i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Cost: {}".format(episodes, round(avg_reward, 2), round(avg_cost, 2)))
            print("----------------------------------------")

# 测试流程
# 参数含义：
# agent：已经创建好的 SAC 智能体。
# dynamics_model：动力学与 GP 扰动模型。
# args：命令行参数。
# visualize：是否显示环境画面。
# debug：是否打印每局测试结果。
# 这个函数没有创建新的神经网络。

# 确定 checkpoint 目录
# → 加载 actor、critic 和 GP
# → 封装测试策略 policy()
# → 重复执行多个测试 episode
# → 每一步选择动作并执行 env.step()
# → 统计 reward、完成率和动作计算时间
# → 返回平均 reward
def test(agent, dynamics_model, args, visualize=True, debug=True):

    model_path = args.resume
    safe_action = args.cbf_mode != 'off'

#     agent.load_weights(model_path)
    # 具体实现位于 sac_cbf.py (line 224)。
    # 它加载两个文件：
    # actor.pkl
    # critic.pkl
    # 其中：
    # actor.pkl：策略网络参数，负责根据 observation 产生动作。
    # critic.pkl：Q 网络参数，负责评价状态和动作。
    agent.load_weights(model_path)
    dynamics_model.load_disturbance_models(model_path)

    def policy(observation):
        return agent.select_action(observation, dynamics_model, safe_action=safe_action, evaluate=True)[0]

    if visualize and 'Unicycle' in model_path:
        from plot_utils import plot_value_function
        plot_value_function(build_env(args.env_name), agent, dynamics_model, save_path=model_path, safe_action=False)

    episode_rewards = []
    dones = []

    for episode in range(args.validate_episodes):

        env = build_env(args.env_name, obs_config=args.obs_config, rand_init=args.rand_init)
        if agent.cbf_layer:
            agent.cbf_layer.env = env

        # reset at the start of episode
        observation, info = env.reset()
        episode_steps = 0
        episode_reward = 0.
        assert observation is not None

        # Time policy
        policy_timings = []

        # start episode
        done = False
        while not done:
            # basic operation, action ,reward, blablabla ...
            policy_start_time = time.time()
            action = policy(observation)
            policy_timings.append(time.time() - policy_start_time)
            if visualize:
                env.render(mode='human')

            observation, reward, done, info = env.step(action)

            # update
            episode_reward += reward
            episode_steps += 1

        episode_rewards.append(episode_reward)
        dones.append(done and env.episode_step < env.max_episode_steps)

        if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}, mean_reward:{}, std_reward:{}, mean_completion:{}, policy_mean_wct={}'.format(episode, episode_reward, np.mean(episode_rewards), np.std(episode_rewards), np.mean(dones), np.mean(policy_timings)))

        env.close()

    if debug:
        prYellow('[Evaluate]: mean_reward:{}, std_reward:{}, mean_completion:{}'.format(np.mean(episode_rewards), np.std(episode_rewards), np.mean(dones)))

    return np.mean(episode_rewards)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # 如果当前文件是直接运行的，命令行参数args

    # --env_name：选择环境。实际支持 Unicycle、SimulatedCars、Pvtol；帮助文本遗漏了 Pvtol。
    parser.add_argument('--env_name', default="Unicycle", help='Options are Unicycle or SimulatedCars.')
    # --obs_config：控制障碍物配置，主要影响 Unicycle 和 Pvtol，随后传给 build_env()。
    parser.add_argument('--obs_config', default="default", help='How to generate obstacles for Unicycle env.')
    # --rand_init：是否随机选择初始状态。它使用 type=bool，存在解析陷阱。
    parser.add_argument('--rand_init', type=bool, default=False, help='How to generate obstacles for Unicycle env.')
    # Comet ML
    # --log_comet：命令中出现该开关时启用 Comet.ml 实验记录；未提供时不联网记录。
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    # --comet_key：Comet.ml API key，仅在 log_comet=True 时使用。
    parser.add_argument('--comet_key', default='', help='Comet API key')
    # --comet_workspace：Comet.ml 工作区名称，仅在 log_comet=True 时使用。
    parser.add_argument('--comet_workspace', default='', help='Comet workspace')
    # --comet_project_name：Comet.ml 项目名后缀，程序会结合环境名生成完整项目名。
    parser.add_argument('--comet_project_name', default='', help='Comet project Name')
    # SAC Args
    # --mode：选择 train 或 test；默认进入训练流程。
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    # --visualize：仅测试流程使用；出现该开关时调用 env.render() 显示环境。
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only available test mode')
    # --output：训练输出的父目录；实际运行目录会自动命名为 output/<Env>-runN。
    parser.add_argument('--output', default='output', type=str, help='')
    # --policy：策略网络类型；Gaussian 是标准 SAC，Deterministic 是备用确定性策略。
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    # --eval：是否每 5 个训练 episode 额外评估 5 局；type=bool 使 "--eval False" 仍可能得到 True。
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 5 episode (default: True)')
    # --gamma：奖励折扣因子，越接近 1 越重视长期回报。
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    # --tau：target critic 的软更新系数，每次只把 critic 的一小部分参数变化同步过去。
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    # --lr：critic、policy 以及自动熵温度优化器使用的学习率。
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    # --alpha：SAC 熵正则项权重；自动熵调节开启后，该值会在训练中被学习值替代。
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    # --automatic_entropy_tuning：是否自动学习 alpha；type=bool 有字符串 False 被解析为 True 的风险。
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    # --seed：环境、动作空间、NumPy、PyTorch 和动力学模型的随机种子。
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
    # --batch_size：每次更新 SAC 网络时从 replay buffer 采样的 transition 数量。
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    # --max_episodes：训练循环最多执行的 episode 数量。
    parser.add_argument('--max_episodes', type=int, default=400, metavar='N',
                        help='maximum number of episodes (default: 400)')
    # --hidden_size：policy 和 twin critic 中隐藏层的神经元数量。
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    # --updates_per_step：每执行一个真实环境 step，最多进行多少次网络参数更新。
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    # --start_steps：训练初期使用环境随机动作而不是 policy 动作的总步数；实际默认值为 5000。
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    # --target_update_interval：每隔多少次 SAC update 对 target critic 做一次软更新。
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    # --replay_size：真实 replay buffer 和模型 replay buffer 各自允许保存的最大 transition 数量。
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    # --cuda：出现该开关时使用 CUDA；未提供时即使有 GPU 也使用 CPU。
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    # --device_num：启用 CUDA 后选择的 GPU 编号，例如 0 表示第一张 GPU。
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')

    # 如果你想测试已有模型，就要：python main.py --mode test --resume 1
    # --resume：需要加载的运行目录；数字 1 会转换为 output/<Env>-run1，default 则指向 run0。
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # --validate_episodes：test() 中执行的测试 episode 数量。
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    # --validate_steps：计划用于限制测试步数，但当前 test() 没有读取该参数。
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # CBF, Dynamics, Env Args
    # --gp_model_size：GP 扰动模型最多保留的真实状态转移样本数量。
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    # --gp_max_episodes：只在此前若干 episode 收集 GP 数据，之后停止更新扰动历史。
    parser.add_argument('--gp_max_episodes', default=100, type=int, help='gp max train episodes.')
    # --k_d：设计上表示 GP 不确定性的置信倍数；当前可微 QP 保存了它但未实际用于约束。
    parser.add_argument('--k_d', default=3.0, type=float)
    # --gamma_b：CBF 约束中 class-K 函数的增益，影响安全约束修正强度。
    parser.add_argument('--gamma_b', default=20, type=float)
    # --l_p：Unicycle 的前视点距离，用于把运动学转换为适合构造 CBF 的输出动力学。
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Look-ahead distance for unicycle dynamics output.")
    # Model Based RL
    # --model_based：出现该开关时启用 DynamicsModel 生成的虚拟 rollout，并混合真实/模型经验训练。
    parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
    # --real_ratio：model-based 训练 batch 中真实 replay buffer 样本所占的目标比例。
    parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
    # --k_horizon：每条模型 rollout 使用动力学模型向前预测的步数。
    parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
    # --rollout_batch_size：模型 rollout 的基础初始状态批量大小；主循环实际最多取其 5 倍样本。
    parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial states batch to rollout from.')
    # Modular Task Learning
    # --cbf_mode：off 不使用安全层；baseline 只在执行时修正；full 在损失中使用安全动作；mod 做模块化任务学习。
    parser.add_argument('--cbf_mode', default='mod', help="Options are `off`, `baseline`, `full`, `mod`.")
    # Compensator
    # --use_comp：是否启用神经网络 compensator；type=bool 有解析陷阱，且仅允许 model-free baseline。
    parser.add_argument('--use_comp', type=bool, default=False, help='If the compensator is to be used.')
    # --comp_rate：compensator 优化器的学习率。
    parser.add_argument('--comp_rate', default=0.005, type=float, help='Compensator learning rate')
    # --comp_train_episodes：只在前多少个 episode 内收集 rollout 并训练 compensator。
    parser.add_argument('--comp_train_episodes', default=200, type=int, help='Number of initial episodes to train compensator for.')
    # --comp_update_episode：每隔多少个 episode 调用一次 compensator 参数更新。
    parser.add_argument('--comp_update_episode', default=50, type=int, help='Modulo for compensator updates')
    args = parser.parse_args()

    # 处理 resume 路径
    if args.resume == 'default':
        args.resume = os.getcwd() + '/output/{}-run0'.format(args.env_name)
    elif args.resume.isnumeric():
        args.resume = os.getcwd() + '/output/{}-run{}'.format(args.env_name, args.resume)
        args.load_agent = True

    # 如果用 cuda:
    if args.cuda:
        torch.cuda.set_device(args.device_num)

    # ******最重要的三个********
    # Environment  env = 创建环境
    # 你给它一个动作，它推进一步，返回新状态、奖励、是否结束。
    # next_obs, reward, done, info = env.step(action)
    env = build_env(args.env_name, args.obs_config, args.rand_init)

    # Agent agent = 创建强化学习智能体
    # 它负责“根据当前观测选择动作”，也负责“根据经验更新神经网络”。
    # action = agent.select_action(...)
    # agent.update_parameters(...)
    agent = RCBF_SAC(env.observation_space.shape[0], env.action_space, env, args)

    # dynamics_model = 创建动力学模型
    # 它负责理解系统状态和扰动。这个项目里它和 GP、RCBF 安全层关系很大。
    # state = dynamics_model.get_state(obs)
    # dynamics_model.append_transition(...)
    dynamics_model = DynamicsModel(env, args)

    # Random Seed 设置随机种子
    if args.seed > 0:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dynamics_model.seed(args.seed)

     # 如果 args.mode == "train":
    if args.mode == 'train':
        if args.use_comp and (args.model_based or args.cbf_mode != "baseline"):
            raise Exception('Compensator can only be used with model free RL and baseline CBF.')

        # 创建输出目录
        args.output = get_output_folder(args.output, args.env_name)
        # 如果需要 comet 日志:
        if args.log_comet:
            # 创建日志实验
            import random
            project_name = 'rl-rcbf-' + args.comet_project_name.lower() + '-' + args.env_name.lower()
            experiment_name = 'comp_' if args.use_comp else ''
            experiment_name += args.cbf_mode
            experiment_name += 'MB_' if args.model_based else '_'
            experiment_name += args.output[args.output.index('run') + 3:]  # str(random.randint(0, 1000))
            prYellow('Logging experiment on comet.ml!')
            # Create an experiment with your api key
            experiment = Experiment(
                api_key=args.comet_key,
                project_name=project_name,
                workspace=args.comet_workspace,
            )
            experiment.set_name(experiment_name)
            # Log args on comet.ml
            experiment.log_parameters(vars(args))
            experiment_tags = [str(args.batch_size) + '_batch',
                               str(args.updates_per_step) + '_step_updates',
                               args.cbf_mode]
            if args.model_based:
                experiment_tags.append('MB')
            if args.use_comp:
                experiment_tags.append('use_comp')
            print('Comet tags: {}'.format(experiment_tags))
            experiment.add_tags(experiment_tags)
        else:
            experiment = None
        train(agent, env, dynamics_model, args, experiment)
    # 否则如果 args.mode == "test":
    elif args.mode == 'test':
        test(agent, dynamics_model, args, visualize=args.visualize, debug=True)

    # env.close()
