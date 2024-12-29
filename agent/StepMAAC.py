from . import RLAgent
import os
import gym
import torch
import random
from itertools import chain
import numpy as np
import torch.nn as nn
from agent import utils
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
from torch.autograd import Variable
from common.registry import Registry
from torch.nn.utils import clip_grad_norm_
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator

@Registry.register_model('smcritic')
class Maaccritic(RLAgent):
    def __init__(self, world, rank):
        super(Maaccritic, self).__init__(world, world.intersection_ids[rank])
        # word information
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
        self.attention_heads = Registry.mapping['model_mapping']['setting'].param['attention_heads']

        # get generator for each SACagent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

        self.ob_length = self.ob_generator.ob_length

        self._batchwise = batchwise()   # 解析replay-buffer
        self.critic_model = None
        self.target_critic_model = None
        self.critic_optimizer = None
        self.MSELoss = nn.MSELoss()

    def create_critic(self):
        self.critic_model = SAC_Attention_Critic(self.num_agent, self.ob_length, self.action_space.n,
                                                attend_heads=self.attention_heads)
        self.target_critic_model = SAC_Attention_Critic(self.num_agent, self.ob_length, self.action_space.n,
                                                         attend_heads=self.attention_heads)
        self.sync_critic()  # copy critic network parameters to target critic network
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=self.c_lr, weight_decay=1e-3)

    def __repr__(self):
        return self.critic_model.__repr__()

    def sync_critic(self):
        critic_weights = self.critic_model.state_dict()
        self.target_critic_model.load_state_dict(critic_weights)

    def update_critic(self, agents, samples, soft=False):
        # sample : 16个交叉路口 [256, 12] -- [batch size, observation]
        b_t, b_tp, rewards, actions = self._batchwise.batch(samples, self.action_space.n)
        target_act_next_list = []
        next_log_pi = []
        for i in range(0, len(samples)):
            target_act_next = agents[i].target_actor_model(b_tp[i], train=False)

            target_act_next_probs = F.softmax(target_act_next, dim=1)
            int_act, act = categorical_sample(target_act_next_probs)

            log_pi = F.log_softmax(target_act_next, dim=1).gather(1, int_act)
            next_log_pi.append(log_pi)
            target_act_next_list.append(act)

        target_q_next = self.target_critic_model(b_tp, target_act_next_list, train=False)
        #  running_mean should contain 13 elements not 20
        q_next = self.critic_model(b_t, actions, regularize=True, train=True)
        #pq_list, regs = batch_q_next(q_next)
        # q_next([pq]     q loss
        #        [regs])  attention logits
        q_loss = 0.
        for i, target_qn, log_pi, (pq, regs) in zip(range(len(samples)), target_q_next, next_log_pi, q_next):
            target_q = (rewards[i].view(-1, 1) + self.gamma * target_qn)        # TODO view
                        # *(1 - dones[i].view(-1, 1)))25571831597222
            if soft:    # TODO mean
                target_q -= log_pi / self.reward_scale
            q_loss = q_loss + self.MSELoss(pq, target_q.detach())   # TODO
            for reg in regs:
                q_loss += reg   # regularizing attention

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_model.scale_shared_grads()
        clip_grad_norm_(self.critic_model.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        return q_loss.clone().detach().numpy()

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

@Registry.register_model('stepmaac')
class Maacagent(RLAgent):
    def __init__(self, world, rank):
        super(Maacagent, self).__init__(world, world.intersection_ids[rank])
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1  # n/1个agent
        self.rank = rank
        self.n_intersections = len(world.id2intersection)
        self.num_agent = int(len(self.world.intersections) / self.sub_agents)

        # *************************************************************************************** #
        # get generator for each SACagent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))
        # 观测长度
        self.ob_length = self.ob_generator.ob_length

        # *************************************************************************************** #
        # base parameter
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']

        self.a_lr = Registry.mapping['model_mapping']['setting'].param['a_lr']
        # self.n_layers = Registry.mapping['model_mapping']['setting'].param['n_layers']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['a_lr']
        self.attention_heads = Registry.mapping['model_mapping']['setting'].param['attention_heads']

        # *************************************************************************************** #
        # network  MASAC CTDE 结合了COMO和MADDPG 集中式critic 和 n个actor
        self._batchwise = batchwise()
        self.actor_model = None
        self.target_actor_model = None
        self.actor_optimizer = None
        self.criterion = None  # 平方差

    def __call__(self, ob, *args, **kwargs):
        return self.actor_model(ob, train=False)

    def __repr__(self):
        return self.actor_model.__repr__()

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
        self.actor_model = self._build_model(self.ob_length, self.action_space.n)
        self.target_actor_model = self._build_model(self.ob_length, self.action_space.n)
        self.sync_actor()
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=self.a_lr, weight_decay=1e-3)
        self.criterion = nn.MSELoss(reduction='mean')

    def _build_model(self, input_dim, output_dim):
        model = SAC_Actor(input_dim, output_dim)
        return model

    def sync_actor(self):
        actor_weights = self.actor_model.state_dict()
        self.target_actor_model.load_state_dict(actor_weights)

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs
    """
        tsc_trainer 调用：
            self.env.step 调用
        当环境初始化时，reward 为0 
    """
    def get_reward(self):  # 可根据自己的情况修改reward
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards))
        return rewards

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        if not test:  # True
            if np.random.rand() <= self.epsilon:
                return self.sample()
        #feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
        feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions_o = self.actor_model(observation, train=False)  # train=False:没有梯度
        actions_prob = self.G_softmax(actions_o)   #噪声
        actions_m = torch.argmax(actions_prob, dim=1)  #
        prob = actions_m.clone().detach().numpy()
        return prob

    def get_action_prob(self, ob, phase):
        feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.actor_model(observation, train=False)
        actions_prob = self.G_softmax(actions)
        return actions_prob

    def G_softmax(self, p):
        mv = torch.full((8,), 0)
        # TODO 参数
        noise = Ornstein_Uhlenbeck_Noise(theta=0.2, sigma=0.1, dt=1e-2, mean_value=mv)
        prob = F.softmax((p + noise()), dim=1)
        prob = torch.as_tensor(prob.clone().detach(), dtype=torch.float32)
        return prob

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions_prob, rewards, obs, cur_phase, done)))

    def sample_data(self):
        sample_index = random.sample(range(len(self.replay_buffer)), self.batch_size)
        data = np.array(list(self.replay_buffer), dtype=object)[sample_index]
        return data

    def update_agents(self, samples):
        b_t, b_tp, rewards, actions = self._batchwise.batch(samples, self.action_space.n)
        act_curr = self.actor_model(b_tp[self.rank], train=True)
        act_probs = F.softmax(act_curr, dim=1)  # 归一为动作概率
        int_act, act = categorical_sample(act_probs)
        log_pi = F.log_softmax(act_curr, dim=1).gather(1, int_act)
        regs = torch.as_tensor(act_curr ** 2).mean().clone().detach()

        data = (b_t, act_curr, act_probs, log_pi, regs)
        return data

    def update_target_network(self):
        soft_update(self.target_actor_model, self.actor_model, self.tau)

    def save_model(self, e):
        path_p = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_actor')
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        model_p_name = os.path.join(path_p, f'{e}_{self.rank}.pt')
        torch.save(self.actor_model.state_dict(), model_p_name)

# OU noise
class Ornstein_Uhlenbeck_Noise:
    def __init__(self, theta, sigma, dt, mean_value):
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mean_value = mean_value  # 平均值
        self._x = torch.zeros(*mean_value.shape)

    def __call__(self):
        x = self._x + \
            self.theta * (self.mean_value - self._x) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean_value.shape)
        self._x = x
        return x

class batchwise(object):
    def __init__(self):
        super(batchwise, self).__init__()

    def batch(self, samples, action_space_n):
        all_obs_t = []
        all_obs_tp = []
        all_state_t = []
        all_state_tp = []
        all_rewards = []
        all_actions = []
        all_done = []
        for items in samples:
            obs_t = np.concatenate([item[1][0] for item in items])
            obs_tp = np.concatenate([item[1][4] for item in items])
            # done = np.concatenate(item[1][6] for item in items)
            # 相位
            # phase_t = np.concatenate([utils.idx2onehot(item[1][1], action_space_n) for item in items])
            # phase_tp = np.concatenate([utils.idx2onehot(item[1][5], action_space_n) for item in items])
            # feature_t = np.concatenate([obs_t, phase_t], axis=1)
            # feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
            feature_t = obs_t
            feature_tp = obs_tp
            # 转tensor
            state_t = torch.tensor(feature_t, dtype=torch.float32)
            state_tp = torch.tensor(feature_tp, dtype=torch.float32)
            rewards = torch.tensor(np.array([item[1][3] for item in items]), dtype=torch.float32)
            # rewards = torch.tensor(np.concatenate([item[1][3] for item in samples])[:, np.newaxis],
            #                       dtype=torch.float32)  # TODO: BETTER WA
            #actions = torch.tensor(np.array([item[1][2] for item in items]), dtype=torch.long)
            actions = torch.cat([item[1][2] for item in items], dim=0)

            all_obs_t.append(obs_t)
            all_obs_tp.append(obs_tp)
            all_state_t.append(state_t)
            all_state_tp.append(state_tp)
            all_rewards.append(rewards)
            all_actions.append(actions)
            #all_done.append(done)

        return all_state_t, all_state_tp, all_rewards, all_actions #, all_done

# TODO test
def categorical_sample(probs):
    # 返回tensor值大的索引 back:tensor([[1]])
    int_acs = torch.multinomial(probs, 1)
    # 把softmax概率分布转换成onehot0-1分布
    acs = Variable(torch.FloatTensor(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

class SAC_Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SAC_Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.nonlin = F.leaky_relu

    def _forward(self, x):
        x = self.nonlin(self.fc1(x))
        x = self.nonlin(self.fc2(x))
        out = self.fc3(x)
        return out

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)

# TODO 将叠加的Transformer 看成一种串行Resnet结构，那么输出的多层是否也能跟FPN(特征金字塔)一样，通过反卷积进行特征融合
class SAC_Attention_Critic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, agents_num, o_sizes, a_sizes, hidden_dim=32, attend_heads=1):
        super(SAC_Attention_Critic, self).__init__()
        """
        Inputs:
            agents_num ：agent的数量
            o_sizes ： observation-dim
            a_sizes ： action-dim
            hidden_dim (int): Number of hidden dimensions
            attend_heads (int): mutil-head (use a number that hidden_dim is divisible by)
        """
        self.agents_n = agents_num
        self.o_sizes = o_sizes
        self.outsizes = a_sizes
        self.insizes = o_sizes + a_sizes  # input_dim = state_dim + action_dim
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        # iterate over agents
        for i in range(agents_num):
            # encoder
            encoder = nn.Sequential()
            encoder.add_module('enc_bn', nn.BatchNorm1d(self.insizes, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(self.insizes, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            # critic
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.outsizes))
            self.critics.append(critic)

            # state_encoder
            state_encoder = nn.Sequential()
            state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(self.o_sizes, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(self.o_sizes, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

            attend_dim = hidden_dim // attend_heads
            self.key_extractors = nn.ModuleList()
            self.selector_extractors = nn.ModuleList()
            self.value_extractors = nn.ModuleList()
            for j in range(attend_heads):
                self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
                self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
                self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU()))

            self.shared_modules = [self.key_extractors, self.selector_extractors,
                                   self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        chain : 接收多个可迭代对象作为参数，将它们『连接』起来，作为一个新的迭代器返回
        """
        return chain(*[m.parameters() for m in self.shared_modules])  # TODO

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.agents_n)  # agents num

    def forward(self, obs, actions, return_all_q=False, regularize=False, train=False, logger=None):
        """
        Inputs:
            inputs (list of PyTorch Matrices): Inputs to each agents' encoder (batch of obs + ac)
            agents (int): indices of agents to return Q for
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        # extract state-action encoding for each agent
        inps = [torch.cat((o, a), dim=1) for o, a in zip(obs, actions)]
        # obs-act encoding
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # act encoding
        s_encodings = [self.state_encoders[i](obs[i]) for i in range(self.agents_n)]
        # K=V self-attention
        all_head_keys = [[k(sa_encoding) for sa_encoding in sa_encodings] for k in self.key_extractors]   # K
        all_head_values = [[v(sa_encoding) for sa_encoding in sa_encodings] for v in self.value_extractors]  # V
        all_head_selectors = [[s(s_encoding) for s_encoding in s_encodings] for s in self.selector_extractors]  # S

        other_all_values = [[] for _ in range(self.agents_n)]
        all_attend_logits = [[] for _ in range(self.agents_n)]
        all_attend_probs = [[] for _ in range(self.agents_n)]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            for i, selector in zip(range(self.agents_n), curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != i]
                values = [v for j, v in enumerate(curr_head_values) if j != i]
                # TODO test
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits.detach())  # yjl TODO
                all_attend_probs[i].append(attend_weights.detach())

        # calculate Q per agent
        q_list = []
        for i in range(self.agents_n):
            agent_rets = []
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean())
                                for probs in all_attend_probs[i]]
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            # self.critics - ModuleList
            all_q = self.critics[i](critic_in)
            int_acs = actions[i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            agent_rets.append(q)
            if train:   # TODO 防止梯度的 change yjl
                if regularize:
                    attend_mag_reg = 1e-3 * sum((logit ** 2).mean() for logit in all_attend_logits[i])
                    regs = (attend_mag_reg,)
                    agent_rets.append(regs)
                    q_list.append(agent_rets)
                if return_all_q:
                    agent_rets.append(all_q)
                    q_list.append(agent_rets)
            else:
                q_list.append(q)
            if logger is not None:
                logger.add_scalars('agent %i/attention' % i,
                                   dict(('head %i_entropy' % h_i, ent) for h_i, ent in enumerate(head_entropies)))
        return q_list




