import numpy as np
import gym
from gym import spaces
from envs.utils import to_pixel
from rcbf_sac.utils import get_polygon_normals
# UnicycleEnv 继承于 Gym 的环境基类。
class UnicycleEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self, obs_config='default', rand_init=False):

        super(UnicycleEnv, self).__init__()

        self.dynamics_mode = 'Unicycle'
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        # 动作空间，action = [v, omega]，[线速度, 角速度]，shape=(2,)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,))
        self.bds = np.array([[-3., -3.], [3., 3.]])

        # 每次 step() 代表 0.02 秒
        self.dt = 0.02
        self.max_episode_steps = 1000
        self.reward_goal = 1.0
        self.goal_size = 0.3
        # Initialize Env
        self.state = None
        self.episode_step = 0
#         state = [x, y, theta]
        # x：机器人横坐标。
        # y：机器人纵坐标。
        # theta：机器人朝向角，单位是弧度。
#         [-2.5, -2.5, 0.0]
        # 表示机器人位于 (-2.5, -2.5)，朝向角为 0，也就是朝 x 轴正方向。    
        self.initial_state = np.array([[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [-2.5, 0.0, 0.0], [2.5, -2.5, np.pi/2]])
        self.goal_pos = np.array([2.5, 2.5])
        self.rand_init = rand_init  # Random Initial State

        self.reset()

        # Get Dynamics
        self.get_f, self.get_g = self._get_dynamics()
        # Disturbance
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20

        # Build Hazards
        self.obs_config = obs_config
        self.hazards = []
        if obs_config == 'default':  # default
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([0., 0.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([-1., 1.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([-1., -1.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., -1.])})
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., 1.])})
        elif obs_config == 'test':
            # self.build_hazards(obs_config)
            self.hazards.append({'type': 'polygon', 'vertices': 0.6*np.array([[-1., -1.], [1., -1], [1., 1.], [-1., 1.]])})
            self.hazards[-1]['vertices'][:, 0] += 0.5
            self.hazards[-1]['vertices'][:, 1] -= 0.5
            self.hazards.append({'type': 'circle', 'radius': 0.6, 'location': 1.5*np.array([1., 1.])})
            self.hazards.append(
                {'type': 'polygon', 'vertices': np.array([[0.9, 0.9], [2.1, 2.1], [2.1, 0.9]])})
        # none 没有专门分支。它和 random 完全一样，会生成随机障碍物。这是当前实现与名称含义不一致的地方。
        else:
            n_hazards = 6
            hazard_radius = 0.6
            self.get_random_hazard_locations(n_hazards, hazard_radius)

        # Viewer
        self.viewer = None


#     公开接口 step() 负责：
    # 把动作裁剪到 [-1,1]；
    # 调用 _step() 更新内部状态；
    # 把 3 维真实状态转换成 7 维观测；
    # 返回 (next_obs, reward, done, info)。
    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------

        策略网络收到的观测:,使用 cos(theta), sin(theta) 可以避免角度从 π 跳到 -π 时产生数值不连续。
        7 维观测中没有障碍物位置。策略主要学习到达目标，障碍物由外部的 RCBF-QP 安全层处理。
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self._step(action)
        return self.get_obs(), reward, done, info

#   内部 _step() 负责真正的环境计算：
    # 更新动力学；
    # 添加扰动；
    # 计算 reward；
    # 判断是否结束；
    # 计算安全 cost。
    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent.
        reward : float
            Reward collected during this transition.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        # 动力学系统的动态
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        # 系统扰动的动态
#         展开后大约是：
        # x     ← x - dt · 0.1 cos²(theta)
        # y     ← y - dt · 0.1 sin(theta)cos(theta)
        # theta 不变
        # 可以把它理解为一个确定性的、与朝向相关的未知动力学扰动，供 GP 学习。
        # 虽然类中定义了 disturb_mean 和 disturb_covar，但随机扰动采样代码已被注释，所以当前环境实际使用的是上面这个确定性扰动。
        self.state -= self.dt * 0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0])  #* np.random.multivariate_normal(self.disturb_mean, self.disturb_covar, 1).squeeze()

        self.episode_step += 1

        info = dict()

        dist_goal = self._goal_dist()

        # reward = 上一步到目标的距离 - 当前到目标的距离
        reward = (self.last_goal_dist - dist_goal)  # -1e-3 * dist_goal
        self.last_goal_dist = dist_goal
        # Check if goal is met
#         进入目标半径后，再增加：
        # reward += 1.0
        # 然后结束 episode。障碍物碰撞不会直接扣 reward，也不会直接终止 episode。
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps

        # Include constraint cost in reward (only during training, i.e. obs_config=='default')
        # 仅在 obs_config='default' 时，环境检查机器人是否进入圆形障碍物：(x-hx)² + (y-hy)² < radius²
        # 这里机器人被当作一个点，只检查机器人中心；渲染出来的机器人半径没有参与碰撞计算。
        # cost 与 reward 是两个独立指标，代码注释虽然写着“Include constraint cost in reward”，实际上并没有把 cost 加入 reward。
        if self.obs_config == 'default':
#             info['cost'] 只在下面的条件中创建：
            # if self.obs_config == 'default':
            # 所以 test、random 和错误落入随机分支的 none 都不会报告碰撞 cost。主训练代码通过 next_info.get('cost', 0) 读取，因此不会报错，但会把缺失的 cost 当成 0。

            # 这意味着：在随机或测试障碍物中，即使机器人穿过障碍物，现有统计也可能显示 cost=0。因此不能用该字段直接评价 zero-shot 测试时的真实碰撞情况。
            # 整个环境的数据流可以归纳为：
            # 7维 obs → SAC 输出 [v,omega]
            #        → 可选 RCBF-QP 修正
            #        → UnicycleEnv.step()
            #        → 更新 3维 state
            #        → 计算任务 reward 和可选 safety cost
            #        → 重新生成 7维 next_obs
            info['cost'] = 0
            for hazard in self.hazards:
                if hazard['type'] == 'circle': # They should all be circles if 'default'
                    info['cost'] += 0.1 * (np.sum((self.state[:2] - hazard['location']) ** 2) < hazard['radius'] ** 2)
        return self.state, reward, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """

        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.episode_step = 0

        # Re-initialize state
        if self.rand_init:
            self.state = np.copy(self.initial_state[np.random.randint(self.initial_state.shape[0])])
        else:
            self.state = np.copy(self.initial_state[0])

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        return self.get_obs(), dict()

    def render(self, mode='human', close=False):
        """Render the environment to the screen

        Parameters
        ----------
        mode : str
        close : bool

        Returns
        -------

        """

        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step, self.state, self._goal_dist(), theta_error))

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from envs import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards)):
                if self.hazards[i]['type'] == 'circle':
                    obstacles.append(pyglet_rendering.make_circle(radius=to_pixel(self.hazards[i]['radius'], shift=0), filled=True))
                    obs_trans = pyglet_rendering.Transform(translation=(to_pixel(self.hazards[i]['location'][0], shift=screen_width/2), to_pixel(self.hazards[i]['location'][1], shift=screen_height/2)))
                    obstacles[i].set_color(1.0, 0.0, 0.0)
                    obstacles[i].add_attr(obs_trans)
                elif self.hazards[i]['type'] == 'polygon':
                    obstacles.append(pyglet_rendering.make_polygon(to_pixel(self.hazards[i]['vertices'], shift=[screen_width/2, screen_height/2]), filled=True))
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(0.1, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(to_pixel(self.goal_pos[0], shift=screen_width/2), to_pixel(self.goal_pos[1], shift=screen_height/2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_circle(radius=to_pixel(0.1), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(15.0, 0.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

        if self.state is None:
            return None

        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2))
        self.robot_trans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    #  该函数会把世界坐标系中的目标方向旋转到机器人自身坐标系：
        # goal_compass_x：目标在机器人前后方向的位置
        # goal_compass_y：目标在机器人左右方向的位置
    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        # 然后把方向向量近似归一化，使它主要表达“方向”，不表达距离。
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal
#         距离单独编码为：
        # exp(-dist_goal)
        # 它的特点是：
        # 越接近目标，数值越接近 1；
        # 越远离目标，数值越接近 0；
        # 数值始终有界，便于神经网络处理。
        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

    def _get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            return f_x

        def get_g(state):
            theta = state[2]
            g_x = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [            0, 1.0]])
            return g_x

        return get_f, get_g

    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # 随机障碍物生成
    # get_random_hazard_locations (line 315) 尝试生成 6 个障碍物：
    # 在大约 [-2.4,2.4]^2 中随机选择中心。
    # 随机选择圆、正方形或三角形。
    # 尺寸在基础半径的 80%～120% 之间变化。
    # 障碍物中心之间必须保持一定距离。
    # 不能太靠近目标。
    # 不能太靠近四个候选初始状态。
    # 每个障碍物最多尝试放置 100 次。
    # 如果某个障碍物尝试 100 次仍无法放下，就会跳过。因此最终数量可能少于 6。函数文档说会返回位置数组，但实际没有 return，而是直接修改 self.hazards。
    def get_random_hazard_locations(self, n_hazards: int, hazard_radius: float):
        """

        Parameters
        ----------
        n_hazards : int
            Number of hazards to create
        hazard_radius : float
            Radius of hazards

        Returns
        -------
        hazards_locs : ndarray
            Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
        """

        # Create buffer with boundaries
        buffered_bds = np.copy(self.bds)
        buffered_bds[0] = buffered_bds[0] + hazard_radius
        buffered_bds[1] -= hazard_radius

        hazards = []
        hazards_centers = np.zeros((n_hazards, 2))
        n = 0  # Number of hazards actually placed
        for i in range(n_hazards):
            successfully_placed = False
            iter = 0
            hazard_type = np.random.randint(3)  # 0-> Circle 1->Square 2->Triangle
            radius = hazard_radius * (1-0.2*2.0*(np.random.random() - 0.5))
            while not successfully_placed and iter < 100:
                hazards_centers[n] = (buffered_bds[1] - buffered_bds[0]) * np.random.random(2) + buffered_bds[0]
                successfully_placed = np.all(np.linalg.norm(hazards_centers[:n] - hazards_centers[[n]], axis=1) > 3.5*hazard_radius)
                successfully_placed = np.logical_and(successfully_placed, np.linalg.norm(self.goal_pos - hazards_centers[n]) > 2.0*hazard_radius)
                successfully_placed = np.logical_and(successfully_placed, np.all(np.linalg.norm(self.initial_state[:, :2] - hazards_centers[[n]], axis=1) > 2.0*hazard_radius))
                iter += 1
            if not successfully_placed:
                continue
            if hazard_type == 0:  # Circle
                hazards.append({'type': 'circle', 'location': hazards_centers[n], 'radius': radius})
            elif hazard_type == 1:  # Square
                hazards.append({'type': 'polygon', 'vertices': np.array(
                    [[-radius, -radius], [-radius, radius], [radius, radius], [radius, -radius]])})
                hazards[-1]['vertices'] += hazards_centers[n]
            else:  # Triangle
                hazards.append({'type': 'polygon', 'vertices': np.array(
                    [[-radius, -radius], [-radius, radius], [radius, radius], [radius, -radius]])})
                # Pick a vertex and delete it
                idx = np.random.randint(4)
                hazards[-1]['vertices'] = np.delete(hazards[-1]['vertices'], idx, axis=0)
                hazards[-1]['vertices'] += hazards_centers[n]
            n += 1

        self.hazards = hazards


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch
    from rcbf_sac.utils import to_tensor, to_numpy
    from rcbf_sac.cbf_qp import CascadeCBFLayer
    from rcbf_sac.diff_cbf_qp import CBFQPLayer
    from rcbf_sac.dynamics import DynamicsModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="SafetyGym", help='Either SafetyGym or Unicycle.')
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=50, type=float)
    parser.add_argument('--l_p', default=0.03, type=float, help="Look-ahead distance for unicycle dynamics output.")
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    parser.add_argument('--diff_qp', action='store_true', dest='diff_qp', help="Use differentiable QP layer.")
    args = parser.parse_args()

    if args.diff_qp:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    env = UnicycleEnv()
    dynamics_model = DynamicsModel(env, args)
    if args.diff_qp:
        cbf_wrapper = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
    else:
        cbf_wrapper = CascadeCBFLayer(env, gamma_b=args.gamma_b, k_d=args.k_d)


    def simple_controller(env, state, goal):
        goal_xy = goal[:2]
        goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
        v = 4.0 * goal_dist
        relative_theta = 1.0 * np.arctan2(goal_xy[1], goal_xy[0])
        omega = 5.0 * relative_theta
        return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)

    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_step = 0

    while not done:
        # Take Action and get next state
        # random_action = env.action_space.sample()
        state = dynamics_model.get_state(obs)
        random_action = simple_controller(env, state, obs[-3:])
        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
        if args.diff_qp:
            state = to_tensor(state, torch.FloatTensor, 'cpu')
            random_action = to_tensor(random_action, torch.FloatTensor, 'cpu')
            disturb_mean = to_tensor(disturb_mean, torch.FloatTensor, 'cpu')
            disturb_std = to_tensor(disturb_std, torch.FloatTensor, 'cpu')
        action_safe = cbf_wrapper.get_safe_action(state, random_action, disturb_mean, disturb_std)
        if args.diff_qp:
            action_safe = to_numpy(action_safe)
        obs, reward, done, info = env.step(action_safe)
        env.render()
        plt.pause(0.01)
        episode_reward += reward
        episode_step += 1
        print('step {} \tepisode_reward = {}'.format(episode_step, episode_reward))
    plt.show()

