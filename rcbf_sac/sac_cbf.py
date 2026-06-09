import torch
import torch.nn.functional as F
from torch.optim import Adam
from rcbf_sac.utils import soft_update, hard_update
from rcbf_sac.model import GaussianPolicy, QNetwork, DeterministicPolicy
from rcbf_sac.diff_cbf_qp import CBFQPLayer
from rcbf_sac.utils import to_tensor
from rcbf_sac.compensator import Compensator
import numpy as np


"""
class RCBF_SAC(object):
近似 C++：
class RCBF_SAC : public object

Python 中几乎所有东西都是对象。这里的 object 是最基础的父类。
"""
class RCBF_SAC(object):

    """
    struct RCBF_SAC {
        float gamma;
        float tau;
        float alpha;
        QNetwork *critic;
        QNetwork *critic_target;
        Policy *policy;
    };

    """

    """
    RCBF_SAC(
        int num_inputs,             //	observation 的维数，Unicycle 中是 7
        gym::Box& action_space,     // 	动作的范围、维数、随机采样方法
        GymEnvironment& env,        // 	可能是 UnicycleEnv、SimulatedCarsEnv 或 PvtolEnv
        argparse::Namespace& args   // 	命令行参数集合，如 args.gamma、args.cuda
    );

    Python 不在函数声明处检查这些类型。类型是调用函数时才确定的。
    例如创建 agent 的地方是：
    agent = RCBF_SAC(
        env.observation_space.shape[0],
        env.action_space,
        env,
        args
    )
    见 main.py (line 491)。

    Unicycle 中这相当于：
    RCBF_SAC(7, 一个Box对象, 一个UnicycleEnv对象, 一个Namespace对象)

    对象.成员变量
    对象.成员函数(...)
    是否在调用函数，看后面有没有圆括号


    """

    """
    __init__ 相当于 C++ 构造函数。创建 RCBF_SAC(...) 时会自动调用它。

    Python 成员变量不需要提前声明。第一次执行：
    self.gamma = ...
    就创建了成员变量 gamma。
    """
    def __init__(self, num_inputs, action_space, env, args):
        
        """
        近似 C++：
        this->gamma = args.gamma;
        this->tau = args.tau;
        this->alpha = args.alpha;
        默认值定义在 main.py (line 382)。
        """
        self.gamma = args.gamma  # 未来奖励折扣
        self.tau = args.tau      # target critic 软更新比例
        self.alpha = args.alpha  # 熵温度

        # 这里的policy_type 是字符串 str，通常内容是："Gaussian"
        self.policy_type = args.policy

        # 这是 int，默认值是 1。
        self.target_update_interval = args.target_update_interval

        # 这是 bool，即 True 或 False
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        # 这里没有复制动作空间，只是让 self.action_space 也指向传进来的那个 Box 对象，类似保存一个 C++ 引用或指针。
        self.action_space = action_space

        # 值A if 条件 else 值B, 类似于C++的args.cuda ? "cuda" : "cpu"
        self.device = torch.device("cuda" if args.cuda else "cpu")

        """
        critic 内部包含两个 Q 网络 Q1、Q2，结构见 model.py (line 34)。
        从内向外拆开。

        Unicycle 动作是：[v, omega]
        所以：action_space.shape == (2,)，其中(2,) 是一个只有一个元素的元组 tuple。逗号表示它是元组。
        action_space.shape[0]读取元组的第 0 个元素，结果是整数 2。
        """
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        # critic_target 与 critic 结构相同，但更新更缓慢。
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        # 把普通 critic 的参数完整复制过去。
        hard_update(self.critic_target, self.critic)

        # Alter to Learn Task Modularly without safety considerations
        self.cbf_mode = args.cbf_mode

#         如果使用默认的 Gaussian policy，还会创建：
        # policy：actor 神经网络。
        # policy_optim：actor 优化器。
        # log_alpha、alpha_optim：自动调节探索强度。
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            # 它根据状态输出一个高斯分布，再从分布中采样动作，具体见 model.py (line 94)。
#             返回三个结果：
            # action, log_prob, mean_action = self.policy.sample(state)
            # action：随机采样动作，用于训练和探索
            # log_prob：这个动作的对数概率，用于计算熵
            # mean_action：均值动作，用于稳定评估
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)

            # Adam 是参数修改器：可以暂时把它理解成：根据梯度修改 policy 内部的权重。
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # CBF layer
        # 当 cbf_mode != 'off' 时创建可微 CBF-QP 层；
        self.env = env
        self.cbf_layer = None
        if self.cbf_mode != 'off':
            self.cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)

        # compensator
        # 当 use_comp=True 时额外创建 compensator。
        # Compensator 是一个小神经网络，试图提前预测 CBF 会怎样修正动作。当前代码只允许它与 model-free baseline 搭配，限制见 main.py (line 507)。
        if args.use_comp:
            self.compensator = Compensator(num_inputs, action_space.shape[0], action_space.low, action_space.high, args)
        else:
            self.compensator = None

    def select_action(self, state, dynamics_model, evaluate=False, warmup=False, safe_action=True, cbf_info=None):

        state = to_tensor(state, torch.FloatTensor, self.device)
        if cbf_info:
            cbf_info = to_tensor(cbf_info, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)
            if cbf_info:
                cbf_info = cbf_info.unsqueeze(0)
        if warmup:
            batch_size = state.shape[0]
            action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
            for i in range(batch_size):
                # action_space.sample() 从动作允许范围内随机产生动作。
                action[i] = torch.from_numpy(self.action_space.sample()).to(self.device)
        else:
            if evaluate is False:

                # 由策略神经网络根据当前观测产生动作。
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)

        if self.compensator:
            action_comp = self.compensator(state)
            action += action_comp

        if safe_action:
            final_action = self.get_safe_action(state, action, dynamics_model, cbf_info_batch=cbf_info)
            cbf_action = final_action - action
        else:
            final_action = action
            cbf_action = torch.zeros_like(final_action)

        if not self.compensator:
            if expand_dim:
                return final_action.detach().cpu().numpy()[0], cbf_action.detach().cpu().numpy()[0]
            return final_action.detach().cpu().numpy(), cbf_action.detach().cpu().numpy()
        else:
            action_comp = action_comp.detach().cpu().numpy()[0] if expand_dim else action_comp.detach().cpu().numpy()
            cbf_action = cbf_action.detach().cpu().numpy()[0] if expand_dim else cbf_action.detach().cpu().numpy()
            final_action = final_action.detach().cpu().numpy()[0] if expand_dim else final_action.detach().cpu().numpy()
            return final_action, action_comp, cbf_action

    def update_parameters(self, memory, batch_size, updates, dynamics_model, memory_model=None, real_ratio=None):
        """

        Parameters
        ----------
        memory : ReplayMemory
        batch_size : int
        updates : int
        dynamics_model : GP Dynamics' Disturbance model D(x) in x_dot = f(x) + g(x)u + D(x)
        memory_model : ReplayMemory, optional
                If not none, perform model-based RL.
        real_ratio : float, optional
                If performing model-based RL, then real_ratio*batch_size are sampled from the real buffer, and the rest
                is sampled from the model buffer.

        Returns
        -------

        """


        # Model-based vs regular RL
        if memory_model and real_ratio:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch, cbf_info_batch, next_cbf_info_batch = memory.sample(
                batch_size=int(real_ratio * batch_size))
            state_batch_m, action_batch_m, reward_batch_m, next_state_batch_m, mask_batch_m, t_batch_m, next_t_batch_m, cbf_info_batch_m, next_cbf_info_batch_m = memory_model.sample(
                batch_size=int((1 - real_ratio) * batch_size))
            state_batch = np.vstack((state_batch, state_batch_m))
            action_batch = np.vstack((action_batch, action_batch_m))
            reward_batch = np.hstack((reward_batch, reward_batch_m))
            next_state_batch = np.vstack((next_state_batch, next_state_batch_m))
            mask_batch = np.hstack((mask_batch, mask_batch_m))
            if cbf_info_batch is not None and cbf_info_batch[0] is not None:
                cbf_info_batch = np.hstack((cbf_info_batch, cbf_info_batch_m))
                next_cbf_info_batch = np.hstack((next_cbf_info_batch, next_cbf_info_batch_m))
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch, cbf_info_batch, next_cbf_info_batch = memory.sample(batch_size=batch_size)


        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        if cbf_info_batch is not None and cbf_info_batch[0] is not None:
            cbf_info_batch = torch.FloatTensor(cbf_info_batch).to(self.device)
            next_cbf_info_batch = torch.FloatTensor(next_cbf_info_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if self.cbf_mode == 'full' or self.cbf_mode == 'mod':
                next_state_action = self.get_safe_action(next_state_batch, next_state_action, dynamics_model, modular=self.cbf_mode == 'mod', cbf_info_batch=next_cbf_info_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute Actions and log probabilities
        pi, log_pi, _ = self.policy.sample(state_batch)
        # Compute safe action using Differentiable CBF-QP
        if self.cbf_mode == 'full' or self.cbf_mode == 'mod':
            pi = self.get_safe_action(state_batch, pi, dynamics_model, modular=self.cbf_mode == 'mod', cbf_info_batch=cbf_info_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For Comet.ml logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For Comet.ml logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def update_parameters_compensator(self, comp_rollouts):

        if self.compensator:
            self.compensator.train(comp_rollouts)

    # Save model parameters
    def save_model(self, output):
        print('Saving models in {}'.format(output))
        torch.save(
            self.policy.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        if self.compensator:
            self.compensator.save_model(output)

    # Load model parameters
    def load_weights(self, output):
        if output is None: return
        print('Loading models from {}'.format(output))

        self.policy.load_state_dict(
            torch.load('{}/actor.pkl'.format(output), map_location=self.device)
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output), map_location=self.device)
        )

        if self.compensator:
            self.compensator.load_weights(output)

    def get_safe_action(self, obs_batch, action_batch, dynamics_model, modular=False, cbf_info_batch=None):
        """Given a nominal action, returns a minimally-altered safe action to take.

        Parameters
        ----------
        obs_batch : torch.tensor
        action_batch : torch.tensor
        dynamics_model : DynamicsModel

        Returns
        -------
        safe_action_batch : torch.tensor
            Safe actions to be taken (cbf_action + action).
        """
        state_batch = dynamics_model.get_state(obs_batch)
        mean_pred_batch, sigma_pred_batch = dynamics_model.predict_disturbance(state_batch)

        safe_action_batch = self.cbf_layer.get_safe_action(state_batch, action_batch, mean_pred_batch, sigma_pred_batch, modular=modular, cbf_info_batch=cbf_info_batch)

        return safe_action_batch



