from . import RLAgent
import os
import gym
import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from collections import deque
import torch.nn.functional as F
from torch.autograd import Variable
from torch import distributions as pyd
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator

#from utils.gan import FCGAN

@Registry.register_model('stepsac')
class StepSAC(RLAgent):
    def __init__(self, world, rank):
        super(StepSAC, self).__init__(world, world.intersection_ids[rank])
        # world information
        self.world = world
        self.sub_agents = 1  # n/1个agent
        self.rank = rank
        self.n_intersections = len(world.id2intersection)
        self.num_agent = int(len(self.world.intersections) / self.sub_agents)

        # base parameters
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.reward_scale = Registry.mapping['model_mapping']['setting'].param['reward_scale']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.c_lr = Registry.mapping['model_mapping']['setting'].param['c_lr']
        self.a_lr = Registry.mapping['model_mapping']['setting'].param['a_lr']
        self.log_std_bounds = Registry.mapping['model_mapping']['setting'].param['log_std_bounds']

        # data analysis
        self._batchwise = batchwise()

        # create GAN、critic and n actor
        # self.Goal_GAN = GoalGAN()       #TODO

        self.StepCritic = StepMAACCritic(self.ob_generator.ob_length,
                                         self.action_space.n,
                                         self.c_lr)
        self.StepActor = []
        for n in range(self.num_agent):
            agent = StepMAACagent(self.world,
                                  self.rank + n,                #agents rank编号
                                  self.log_std_bounds)
            self.StepActor.append(agent)

    def __repr__(self):
        return self.StepCritic.__repr__() + '\n' + self.StepActor.__repr__()

    def create_agents(self, num_actors):
        self.StepCritic.create_critic()
        for i in range(num_actors):
            self.StepActor[i].create_actor()

    def update_critic(self, samples, discount, not_done=None):
        obs, next_obs, rewards, actions = self._batchwise.batch(samples)
        target_next_act_list = []
        next_log_prob = []
        with torch.no_grad():
            for i in range(0, len(samples)):
                dist = self.StepActor.actor_model[i].target_actor_model(next_obs[i])
                next_action = dist.rsample()
                log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

                next_log_prob.append(log_prob)
                target_next_act_list.append(next_action)

        target_Q1, target_Q2 = self.StepCritic.target_critic_model(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2)
        target_V -= self.alpha.detach() * log_prob  # TODO
        target_Q = self.reward_scale_factor * rewards + \
                   (discount * target_V * not_done.unsqueeze(1))

        Q1, Q2 = self.StepCritic.critic_model(obs, actions)
        critic_loss = 1.0 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        # optimize encoder and critic
        self.StepCritic.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.StepCritic.critic_model.parameters(), self.grad_clip)
        self.StepCritic.critic_optimizer.step()

        return critic_loss.clone().detach().numpy()

    def update_agents(self, samples):
        obs, next_obs, rewards, actions = self._batchwise.batch(samples)
        dist = self.StepActor.actor_model(next_obs[self.rank])
        act_curr = dist.rsample()
        log_prob = dist.log_prob(act_curr).sum(-1, keepdim=True)
        # TODO

    def step(self, obs, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            obs: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        pass

class StepMAACCritic(nn.Module):
    def __init__(self, ob_length, action_length, c_lr):
        super(StepMAACCritic, self).__init__()

        # create critic
        self.ob_length = ob_length
        self.action_length = action_length
        self.c_lr = c_lr
        self._batchwise = batchwise()  # 解析replay-buffer
        self.critic_model = None
        self.target_critic_model = None
        self.critic_optimizer = None
        self.MSELoss = nn.MSELoss()

    def create_critic(self):
        self.critic_model = SACCritic(self.ob_length, self.action_length, 256)
        self.target_critic_model = SACCritic(self.ob_length, self.action_length, 256)

        self.sync_critic()  # copy critic network parameters to target critic network
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=self.c_lr, weight_decay=1e-3)

    def sync_critic(self):
        critic_weights = self.critic_model.state_dict()
        self.target_critic_model.load_state_dict(critic_weights)

    def save_critic_model(self, e):
        path_q = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_critic')
        if not os.path.exists(path_q):
            os.makedirs(path_q)
        model_q_name = os.path.join(path_q, f'{e}_{self.rank}.pt')
        torch.save(self.critic_model.state_dict(), model_q_name)

    def update_target_critic_network(self):
        soft_update(self.target_critic_model, self.critic_model, self.tau)


def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class StepMAACagent(nn.Module):
    def __init__(self, world, rank, log_std_bounds):
        super(StepMAACagent, self).__init__()
        # replay buffer
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.rank = rank
        self.id = world.intersection_ids
        # base
        self.log_std_bounds = log_std_bounds
        # create actor model
        self.actor_model = None
        self.target_actor_model = None
        self.actor_optimizer = None
        self.criterion = None  # 平方差

        # param
        self.a_lr = Registry.mapping['model_mapping']['setting'].param['a_lr']
        self.eps_end = Registry.mapping['model_mapping']['setting'].param['eps_end']
        self.eps_start = Registry.mapping['model_mapping']['setting'].param['eps_start']
        self.eps_decay = Registry.mapping['model_mapping']['setting'].param['eps_decay']

        # get generator for each SACagent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

    def __call__(self, ob, *args, **kwargs):
        return self.actor_model(ob)

    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)

    def create_actor(self):
        self.actor_model = self._build_model(self.ob_generator.ob_length, self.action_space.n, self.log_std_bounds)
        self.target_actor_model = self._build_model(self.ob_generator.ob_length, self.action_space.n, self.log_std_bounds)
        self.sync_actor()
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=self.a_lr, weight_decay=1e-3)
        self.criterion = nn.MSELoss(reduction='mean')

    def _build_model(self, input_dim, output_dim, log_std_bounds):
        model = SACActor(input_dim, 256, output_dim, log_std_bounds=log_std_bounds)  # TODO
        return model

    def sync_actor(self):
        actor_weights = self.actor_model.state_dict()
        self.target_actor_model.load_state_dict(actor_weights)

    def get_delay(self):
        pass

    def get_queue(self):
        pass

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):  # 可根据自己的情况修改reward TODO reward需要处理
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards))
        return rewards

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

    '''yjl 4.5 探索与利用'''
    def get_action(self, ob, steps_done, test=False):
        sample = random.random()
        eps_threshold = max(self.eps_end,
                            self.eps_start * (1 - steps_done / self.eps_decay) +
                            self.eps_end * (steps_done / self.eps_decay))
        observation = torch.tensor(ob, dtype=torch.float32)     # TODO 加入goal
        if sample > eps_threshold:
            with torch.no_grad():
                actions_o = self.actor_model(observation)
        else:
            actions_o = self.sample()   #随机策略
        actions_prob = self.G_softmax(actions_o)  # 噪声
        actions_m = torch.argmax(actions_prob, dim=1)  #
        prob = actions_m.clone().detach().numpy()
        return prob

    def get_action_prob(self, ob, phase):
        feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        # actions = self.actor_model(observation, train=False)
        with torch.no_grad():
            actions = self.actor_model(observation)

        actions_prob = self.G_softmax(actions)
        return actions_prob

    def G_softmax(self, p):
        u = torch.rand(self.action_space.n)
        prob = F.softmax((p - torch.log(-torch.log(u)) / 1), dim=1)
        return prob

    def sample(self):
        return np.random.randint(0, self.action_space.n)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions_prob, rewards, obs, cur_phase, done)))

    def sample_data(self):
        sample_index = random.sample(range(len(self.replay_buffer)), self.batch_size)
        data = np.array(list(self.replay_buffer), dtype=object)[sample_index]
        return data

    def update_target_network(self):
        soft_update(self.target_actor_model, self.actor_model, self.tau)

    def save_model(self, e):
        path_p = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_actor')
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        model_p_name = os.path.join(path_p, f'{e}_{self.rank}.pt')
        torch.save(self.actor_model.state_dict(), model_p_name)


class batchwise(object):
    def __init__(self):
        super(batchwise, self).__init__()

    def batch(self, samples):
        all_state_t = []
        all_state_tp = []
        all_rewards = []
        all_actions = []
        for items in samples:
            obs_t = np.concatenate([item[1][0] for item in items])
            obs_tp = np.concatenate([item[1][4] for item in items])
            # 转tensor
            state_t = torch.tensor(obs_t, dtype=torch.float32)
            state_tp = torch.tensor(obs_tp, dtype=torch.float32)
            rewards = torch.tensor(np.array([item[1][3] for item in items]), dtype=torch.float32)
            actions = torch.cat([item[1][2] for item in items], dim=0)

            all_state_t.append(state_t)
            all_state_tp.append(state_tp)
            all_rewards.append(rewards)
            all_actions.append(actions)

        return all_state_t, all_state_tp, all_rewards, all_actions

class SACActor(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, log_std_bounds, dropout_rate=0.0):
        '''
        input_dim : 输入维度
        hid_dim: 隐藏层
        output_dim: 输出维度
        log_std_bounds: log_std的截断范围
        dropout_rate ： 神经元随机失效概率，用来做数据增强
        '''
        super(SACActor, self).__init__()
        self.log_std_bounds = log_std_bounds
        self.policy = nn.Sequential(nn.Linear(input_dim, hid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hid_dim, hid_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hid_dim, output_dim))
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, continuous=False):
        if continuous:      # 连续动作空间
            mu, log_std = self.policy(x).chunk(2, dim=1)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = torch.clip(log_std, log_std_min, log_std_max)  # TODO 截断的范围
            # 输入张量的e指数
            std_pred = log_std.exp()
            dist = torch.distributions.normal.Normal(mu, std_pred)

            normal_sample = dist.rsample()  # rsample()是重参数化采样
            log_prob = dist.log_prob(normal_sample)
            action = torch.tanh(normal_sample)
            # 计算tanh_normal分布的对数概率密度
            log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
            action = action * self.action_bound  # self.action_bound 为动作空间最大值
            return action, log_prob
        else:               # 离散动作空间
            out = self.policy(x)
            out = self.dropout(out)
            # 策略网络的输出修改为在离散动作空间上的 softmax 分布
            return out

class SACCritic(nn.Module):
    # 当我们从state学习时，Repr_dim将会是 - 1, feature_dim将会是state_dim
    def __init__(self, obs_dim, action_shape, hidden_dim):
        super(SACCritic, self).__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + action_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + action_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        # self.apply(weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2

# ***************************************************************************
def weight_init(m):
    if isinstance(m, nn.Linear):
        # TODO: Changed initialization to xavier_uniform_
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class GoalGAN(object):
    def __init__(self, obs_size, eva_size, noise_size, g_lr, d_lr):
        super(GoalGAN, self).__init__()
        self.gan = FCGAN(obs_size, eva_size, noise_size, g_lr, d_lr)
        self.obs_size = obs_size
        self.eva_size = eva_size
        self.noise_size = noise_size



