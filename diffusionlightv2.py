import torch
from . import RLAgent
import os
import gym
import json
import jsonpath
import random
import numpy as np
from utils.diffusion_utils import MLP
from utils.diffusionrl import *
from utils.base_transformer import *
from utils.base_transformer import SelfAttention
from common.registry import Registry
from torch.nn import functional as F
from torch.distributions import Categorical
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator


@Registry.register_model('diffusionlightv2')
class Diffusionlightv2Agent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        # world params
        self.world = world
        self.rank = rank
        self.n_intersections = len(world.id2intersection)
        self.sub_agents = len(self.world.intersections)     # 创建一个大的模型
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter = self.world.id2intersection[self.inter_id]
        self.device = torch.device("cpu")

        """ keep network track """
        self.name = 'diffusionlight'
        self.actions_buffer = []
        self.actions_pro_buffer = []
        self.obs_buffer = []
        self.last_obs_buffer = []
        self.reward_buffer = []

        """ base param """
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        self.gamma = Registry.mapping['model_mapping']['setting'].param["gamma"]
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.buffer_size = Registry.mapping['model_mapping']['setting'].param['buffer_size']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        """ diffusion network param """
        self.batch_sample_action = Registry.mapping['model_mapping']['setting'].param['batch_sample_action']
        self.ob_length = Registry.mapping['model_mapping']['setting'].param['ob_length']
        self.beta_schedule = Registry.mapping['model_mapping']['setting'].param['beta_schedule']
        self.n_timesteps = Registry.mapping['model_mapping']['setting'].param['n_timesteps']
        self.max_action = Registry.mapping['model_mapping']['setting'].param['max_action']
        self.eta = Registry.mapping['model_mapping']['setting'].param['eta']
        self.loss_type = Registry.mapping['model_mapping']['setting'].param['loss_type']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']

        # create generators
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators,
               key=lambda x: x[0])  # now generator's order is according to its index in graph 现在生成器的顺序是根据其在图中的索引
        self.ob_generator = observation_generators

        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["pressure"], in_only=True, average='all',
                                                 negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, node_obj, ['phase'], targets=['cur_phase'],
                                                       negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        queues = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"], in_only=True,
                                                 negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        delays = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_delay"], in_only=True, average="all",
                                                 negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        self.action_space = gym.spaces.Discrete(len(self.inter.phases))
        # 创建 buffer
        self.buffer = replay_buffer(self.sub_agents,
                                    self.buffer_size,
                                    self.device,
                                    self.ob_length,
                                    self.action_space.n,
                                    )  # 创建replay_buffer

        self.policy = self.build_model()

    def reset(self):
        """
        重置信息，包括 ob_generator, phase_generator, reward_generator, queue,delay等
        """
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["pressure"], in_only=True, average='all',
                                                 negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, node_obj, ['phase'], targets=['cur_phase'],
                                                       negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        queues = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"], in_only=True,
                                                 negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        delays = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_delay"], in_only=True, average="all",
                                                 negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    def build_model(self):
        policy = DiffusionPolicy(self.buffer,
                                 self.sub_agents,       # n_agents
                                 self.ob_length,        # 观测长度
                                 self.action_space.n,   # 动作空间长度
                                 self.beta_schedule,    # β 产生的方式
                                 self.n_timesteps,      # 扩散的times
                                 self.max_action,
                                 self.eta,              # q_loss 在actor loss 更新中的权重
                                 self.loss_type,        # loss 类型
                                 self.learning_rate,    # 学习率
                                 self.batch_size,       # 抽样的batch size
                                 self.gamma,            # 折扣因子
                                 self.grad_clip,        # 梯度的最大范数
                                 self.tau,              # 更新target speed
                                 )
        return policy

    def save_model(self, e):
        """
        保存一回合的模型参数
        """
        path_p = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_actor')
        path_q = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_critic')
        path_f = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_a_critic')
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        if not os.path.exists(path_q):
            os.makedirs(path_q)
        if not os.path.exists(path_f):
            os.makedirs(path_f)
        model_p_name = os.path.join(path_p, f'{e}_{self.rank}.pt')
        model_q_name = os.path.join(path_q, f'{e}_{self.rank}.pt')
        model_f_name = os.path.join(path_f, f'{e}_{self.rank}.pt')
        # for i in range(self.n_intersections):
        #     torch.save(self.policy..policy.state_dict(), model_p_name)
        torch.save(self.policy.actor.state_dict(), model_f_name)
        torch.save(self.policy.critic.state_dict(), model_q_name)

    def get_ob(self):
        """
        从环境中获取观测
        """
        # sub_agents * lane_nums,
        x_obs = []
        for i in range(len(self.ob_generator)):
            x_obs.append(self.ob_generator[i][1].generate())
        length = set([len(i) for i in x_obs])
        if len(length) == 1:
            x_obs = np.array(x_obs, dtype=np.float32)
        else:
            x_obs = [np.expand_dims(x, axis=0) for x in x_obs]
        return x_obs

    def get_phase(self):
        """
        从环境中获取交叉口的当前相位
        """
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_reward(self):
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards))
        return rewards

    # 分散执行
    def get_action(self, obs, phase, test=False):

        _, actions = self.policy.sample_action(obs, self.batch_sample_action, test)

        return actions

    def get_queue(self):
        """
        获取交叉口的队列长度
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape) == 2 else 0)
        return queue

    def get_delay(self):
        """
        获取交叉口的延迟
        """
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay))
        return delay  # [intersections,]

    def sample(self):
        size = (1, self.n_intersections, self.action_space.n)
        action_pro = torch.randn(size)      # 生成(0,1)的正太分布
        action = torch.argmax(F.softmax(action_pro, dim=2), dim=2).reshape(1, self.n_intersections, 1)

        return action

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.buffer.append_track(last_obs, actions, obs, rewards, done)

    def train(self):
        loss = self.policy.train()
        return loss

    def update_target_network(self):
        self.policy.update_target_model()

class replay_buffer(object):
    def __init__(self, n_agent, buffer_size, device, obs_dim, action_dim):
        self.buffer_size = buffer_size
        self.device = device
        self.n_intersection = n_agent

        self.last_obs = torch.zeros((buffer_size, n_agent, obs_dim)).float()
        self.actions = torch.zeros(buffer_size, n_agent).float()
        self.actions_pro = torch.zeros(buffer_size, n_agent, action_dim).float()        # 动作对应的概率值actions_pro(16,8)
        self.obs = torch.zeros(buffer_size, n_agent, obs_dim).float()
        self.rewards = torch.zeros(buffer_size, n_agent).float()                        # view(-1,1)
        self.done = torch.zeros(buffer_size).float()

        self.pointer = 0
        self.episode = 0

    def sample_track(self, batch_size):
        idx = np.random.randint(0, self.buffer_size, size=(batch_size,))
        return (
            self.last_obs[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.actions_pro[idx].to(self.device),
            self.obs[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.done[idx].to(self.device),
        )

    def append_track(self, last_obs, actions, obs, rewards, done):
        move = len(actions.shape)
        if not done:
            dones = np.array(0.0)
        if self.pointer + move >= self.buffer_size:
            self.pointer = 0
            self.episode += 1
        self.last_obs[self.pointer] = torch.from_numpy(last_obs).float()    # last_obs(16,12)
        self.obs[self.pointer] = torch.from_numpy(obs).float()              # obs(16,12)
        self.actions[self.pointer] = torch.from_numpy(actions).flatten()    # actions(1,16,1) -- actions(16,)
        self.rewards[self.pointer] = torch.from_numpy(rewards).flatten()    # rewards(16,)
        self.done[self.pointer] = torch.from_numpy(dones)

        self.pointer += move

    def append_pro(self, a_pro):
        self.actions_pro[self.pointer] = a_pro

def sync_network(model, target_model):
    weights = model.state_dict()
    target_model.load_state_dict(weights)

class DiffusionPolicy:
    def __init__(self, buffer, n_agent, obs_dim, action_dim, beta_schedule, n_timesteps, max_action, eta, loss_type,
                 lr, batch_size, discount, grad_norm, tau,
                 clip_denoised=True,
                 predict_epsilon=True):
        super(DiffusionPolicy, self).__init__()
        self.device = torch.device("cpu")
        self.buffer = buffer
        self.n_agents = n_agent
        # 这里的MLP相当于DDPM图像生成中的Unet网络
        self.model = MLPv2(obs_dim=obs_dim, action_dim=action_dim, device=self.device)
        # self.noise_model = shared_MLP(n_agent,)
        self.noise_model = Encoderv2(obs_dim, n_block=2, n_embd=64, action_dim=action_dim, n_head=4, n_agent=n_agent)
        self.noise_optimizer = torch.optim.Adam(self.noise_model.parameters(), lr=lr)
        self.shape = (batch_size, action_dim)
        self.actor = Dis_Diffusion(
                    n_agent=n_agent,  # 剩下的都为**kwargs
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    model=self.model,
                    actor=self.noise_model,
                    betas_s=beta_schedule,
                    batch_size=batch_size,
                    n_timesteps=n_timesteps,
                    max_action=max_action,
                    eta=eta,
                    loss_type=loss_type,
                    clip_denoised=clip_denoised,
                    predict_epsilon=predict_epsilon).to(self.device)

        self.target_actor = Dis_Diffusion(
                    n_agent=n_agent,  # 剩下的都为**kwargs
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    model=self.model,
                    actor=self.noise_model,
                    betas_s=beta_schedule,
                    batch_size=batch_size,
                    n_timesteps=n_timesteps,
                    max_action=max_action,
                    eta=eta,
                    loss_type=loss_type,
                    clip_denoised=clip_denoised,
                    predict_epsilon=predict_epsilon).to(self.device)
        # self.actor = Actor(obs_dim, action_dim, action_dim)
        # self.target_actor = Actor(obs_dim, action_dim, action_dim)
        self.is_diff = True

        sync_network(self.actor, self.target_actor)
        # self.noise_model = noise_MLP(self.n_agents, batch_size, obs_dim)  # n_agents, batch, obs_dim,
        # self.actor_param = list(self.actor.parameters()) + list(self.noise_model.parameters())
        # self.actor_optimizer = torch.optim.Adam(self.actor_param, lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=64).to(self.device)
        self.target_critic = Dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=64).to(self.device)
        sync_network(self.critic, self.target_critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.loss_fn = Losses[loss_type]()

        self.tau = tau
        self.grad_norm = grad_norm
        self.eta = eta
        self.n_agents = n_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.beta_schedule = beta_schedule
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self):
        last_obs, actions, actions_pro, obs, rewards, done = self.buffer.sample_track(self.batch_size)
        # x = self.noise_model(obs)
        # noise_loss = - F.mse_loss(torch.var(x), torch.tensor(1, dtype=torch.float32))
        # self.noise_optimizer.zero_grad()
        # noise_loss.backward()
        # if self.grad_norm > 0:
        #     nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        # self.noise_optimizer.step()

        current_q1, current_q2 = self.critic(last_obs, actions_pro)     #(64,16)
        new_actions = []
        new_actions_log_pi = []
        with torch.no_grad():
            # noise = self.noise_model(obs)   #noise(256,8)
            noise = torch.randn(self.shape)   #noise(256,8)
            for i in range(0, self.n_agents):
                # noise = torch.randn(self.shape)  # noise(256,8)
                new_action, new_action_log_pi = self.target_actor.sample(obs[:, i, :], noise)
                new_actions.append(new_action)
                new_actions_log_pi.append(new_action_log_pi)
        new_actions = torch.stack(new_actions, dim=1)               #(256,16,8)
        new_actions_log_pi = torch.stack(new_actions_log_pi, dim=1)     #(256,16,1)

        target_q1, target_q2 = self.target_critic(obs, new_actions)     #(64,16)
        target_q = torch.min(target_q1, target_q2)

        target_q = target_q - (self.alpha.detach() * new_actions * new_actions_log_pi).mean(-1)  #(64,16)
        # rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)     #标准化
        target_q = (rewards + self.discount * target_q).detach()  # 多维
        critic_loss = 0.5 * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        """ Policy Training """
        # last_noise = self.noise_model(last_obs)
        last_noise = torch.randn(self.shape)
        if self.is_diff:
            bcs_loss = []
            last_actions = []
            last_actions_log_pi = []
            for i in range(0, self.n_agents):
                # last_noise = torch.randn(self.shape)

                bc_loss = self.actor.loss(actions_pro[:, i, :], last_noise, last_obs[:, i, :])  #
                bcs_loss.append(bc_loss)
                last_action, last_action_log_pi = self.actor.sample(last_obs[:, i, :], last_noise)
                last_actions.append(last_action)
                last_actions_log_pi.append(last_action_log_pi)
            bcs_loss = torch.stack(bcs_loss, dim=0)
            last_actions = torch.stack(last_actions, dim=1)
            last_actions_log_pi = torch.stack(last_actions_log_pi, dim=1)

            q1_new_action, q2_new_action = self.critic(last_obs, last_actions)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bcs_loss.mean() + self.eta * q_loss + (self.alpha.detach() * last_actions_log_pi * last_actions).detach().mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
        else:
            last_actions = []
            last_actions_log_pi = []
            for i in range(0, self.n_agents):
                last_action, last_action_log_pi = self.actor.sample(last_obs[:, i, :], last_noise)
                last_actions.append(last_action)
                last_actions_log_pi.append(last_action_log_pi)
            last_actions = torch.stack(last_actions, dim=1)
            last_actions_log_pi = torch.stack(last_actions_log_pi, dim=1)
            q1_new_action, q2_new_action = self.critic(last_obs, last_actions)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

        return critic_loss.clone().detach().numpy()

    def sample_action(self, obs, batch, test=False):
        obs = torch.FloatTensor(obs.reshape(-1, self.n_agents, self.obs_dim))     # obs(1,16,12)
        obs_rpt = torch.repeat_interleave(obs, repeats=batch, dim=0)     #(batch, n_agent, obs_dim)
        actions_pro = []
        with torch.no_grad():
            # noise = self.noise_model(obs_rpt)       #(1,8)
            noise_x = self.noise_model(obs_rpt)       #(1,8)
            noise = torch.randn_like(noise_x)
            for i in range(0, self.n_agents):
                # noise = torch.randn_like(noise_x)
                action_pro, _ = self.actor.sample(obs_rpt[:, i, :], noise)
                """ yjl 7.8 """
                # action_pro = action_pro + noise * 1e-6
                actions_pro.append(action_pro)
        # TODO
        actions_pro = torch.stack(actions_pro, dim=1)   #actions_pro(1,16,8)
        if test is False:  # train 增加掩码
            mask = torch.randint_like(actions_pro, low=0, high=2).clamp(0, 1)
            self.buffer.append_pro(actions_pro)
            actions_pro = actions_pro * mask  # (batch,n_agents,actions)
            actions = torch.argmax(F.softmax(actions_pro, dim=2), dim=2).squeeze()  #(16, )
            # actions = torch.multinomial(F.softmax(actions_pro.squeeze(), dim=1), 1)
        else:
            actions = torch.argmax(F.softmax(actions_pro, dim=2), dim=2).squeeze()

        return actions_pro, actions.numpy()

    def update_target_model(self):
        polyak = 1.0 - self.tau
        for t_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)
        for t_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)

"""返回重复项"""
def return_duplicate(datas):
    """
    param: datas: batch_size 的 值索引
    output： 重复次数最多的值索引
    """
    """
    unique: tuple(2)
    unique_datas: 去除重复数据 按从小到大排列
    dup_idx：数据的重复次数
    """
    unique_datas, dup_idx = torch.unique(datas, return_counts=True)
    max_idx = np.max(dup_idx.numpy())   #idx = 5 max=7
    max_idx_times = 0
    max_datas = []
    for i in range(len(dup_idx.numpy())):
        if dup_idx[i] == max_idx:
            max_idx_times += 1
            max_datas.append(unique_datas[i])

    max_random_idx = np.random.randint(0, max_idx_times, size=(1,))     #在最大的几个数中挑选最大的
    idx = max_datas[int(max_random_idx)].numpy()

    return idx