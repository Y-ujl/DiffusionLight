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
from utils.base_transformer import SelfAttention
from common.registry import Registry
from torch.nn import functional as F
from torch.distributions import Categorical
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator


@Registry.register_model('diffusionlight')
class DiffusionlightAgent(RLAgent):
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
        self.sample_type = Registry.mapping['model_mapping']['setting'].param['sample_type']
        self.timestep_respacing = Registry.mapping['model_mapping']['setting'].param['timestep_respacing']
        self.max_q_backup = Registry.mapping['model_mapping']['setting'].param['max_q_backup']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']

        """offline diffusion train param"""
        self.diffusion_weight = Registry.mapping['model_mapping']['setting'].param['diffusion_weight']
        self.diffusion_model = Registry.mapping['model_mapping']['setting'].param['diffusion_model']
        self.diff_iters = Registry.mapping['model_mapping']['setting'].param['diff_iters']
        self.diff_steps = Registry.mapping['model_mapping']['setting'].param['diff_steps']
        self.load_offline_data = Registry.mapping['model_mapping']['setting'].param['load_offline_data']
        self.path = Registry.mapping['model_mapping']['setting'].param['path']
        self.json_name = Registry.mapping['model_mapping']['setting'].param['json_name']

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
                                    self.load_offline_data,
                                    self.path,
                                    self.json_name)  # 创建replay_buffer

        # self.buffer = Prioritized_Replay_Buffer(self.sub_agents,
        #                                         self.buffer_size,
        #                                         self.device,
        #                                         self.ob_length,
        #                                         self.action_space.n,
        #                                         self.load_offline_data,
        #                                         self.path,
        #                                         self.json_name)  # 创建replay_buffer
        # section 4: 创建模型、目标模型和其他
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

    def build_model(self):#TODO
        # buffer, n_agent, obs_dim, action_dim, beta_schedule, n_timesteps, max_action, eta, loss_type,
        # sample_type, timestep_respacing, d_iters, d_steps, lr, batch_size,
        # max_q_backup, discount, grad_norm, tau,
        policy = DiffusionPolicy(self.buffer,
                                 self.sub_agents,       # n_agents
                                 self.ob_length,        # 观测长度
                                 self.action_space.n,   # 动作空间长度
                                 self.beta_schedule,    # β 产生的方式
                                 self.n_timesteps,      # 扩散的times
                                 self.max_action,
                                 self.eta,              # q_loss 在actor loss 更新中的权重
                                 self.loss_type,        # loss 类型
                                 self.sample_type,
                                 self.timestep_respacing,
                                 self.diff_iters,       # 迭代次数
                                 self.diff_steps,       # 迭代steps
                                 self.learning_rate,    # 学习率
                                 self.batch_size,       # 抽样的batch size
                                 self.max_q_backup,     # 是否对action、obs复制 默认为False
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
        path_q_a = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_a_critic')
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        if not os.path.exists(path_q):
            os.makedirs(path_q)
        model_p_name = os.path.join(path_p, f'{e}_{self.rank}.pt')
        model_q_name = os.path.join(path_q, f'{e}_{self.rank}.pt')
        model_qa_name = os.path.join(path_q_a, f'{e}_{self.rank}.pt')
        if self.policy.q_type == 'brother':
            if not os.path.exists(path_q_a):
                os.makedirs(path_q_a)
            torch.save(self.policy.actor.state_dict(), model_p_name)
            torch.save(self.policy.b_critic.state_dict(), model_q_name)
            torch.save(self.policy.b_a_critic.state_dict(), model_qa_name)
        else:
            torch.save(self.policy.actor.state_dict(), model_p_name)
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

    # TODO
    def get_reward(self):
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards))
        return rewards

    def get_action(self, obs, phase, test=False):
        if test is False:
            act_pro, actions = self.policy.sample_action(obs, self.batch_sample_action, test)
            self.remember_time(0, [], [], [], [], act_pro, is_pro=True)
        else:
            act_pro, actions = self.policy.sample_action(obs, self.batch_sample_action, test)

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

    # TODO how to offline train
    def sample(self):
        if self.diffusion_weight:
            if os.path.exists("./model_weight/%s" % self.diffusion_model):
                # 直接返回actions
                pass
            else:
                self.policy.offline_train()     #没有权重文件，就离线训练
        else:   # 训练offline数据, 大概为数据样本量的2倍，满足采样定律
            size = (1, self.n_intersections, self.action_space.n)
            action_pro = torch.randn(size)      # 生成(0,1)的正太分布
            # action_pro.clamp_(-self.max_action, self.max_action)
            self.remember_time(0, [], [], [], [], action_pro, is_pro=True)
            self.buffer.append_pro(action_pro)
            action = torch.argmax(F.softmax(action_pro, dim=2), dim=2).reshape(1, self.n_intersections, 1)

        return action

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.buffer.append_track(last_obs, actions, obs, rewards, done)

    def remember_time(self, episode, action, last_obs, obs, reward, action_pro, is_pro=False):
        if not os.path.exists('./analysis_data/%s/%s' % (self.name, 'actions')):
            os.makedirs('./analysis_data/%s/%s' % (self.name, 'actions'))
        if not os.path.exists('./analysis_data/%s/%s' % (self.name, 'obs')):
            os.makedirs('./analysis_data/%s/%s' % (self.name, 'obs'))
        if not os.path.exists('./analysis_data/%s/%s' % (self.name, 'last_obs')):
            os.makedirs('./analysis_data/%s/%s' % (self.name, 'last_obs'))
        if not os.path.exists('./analysis_data/%s/%s' % (self.name, 'rewards')):
            os.makedirs('./analysis_data/%s/%s' % (self.name, 'rewards'))
        if not os.path.exists('./analysis_data/%s/%s' % (self.name, 'actions_pro')):
            os.makedirs('./analysis_data/%s/%s' % (self.name, 'actions_pro'))

        if is_pro is False:
            action_dimt = {"action": action.tolist(), }
            for i in range(len(obs)):
                obs[i] = obs[i].tolist()
            for i in range(len(last_obs)):
                last_obs[i] = last_obs[i].tolist()
            last_obs_dimt = {"last_obs": last_obs}
            obs_dimt = {"obs": obs}
            reward_dimt = {"reward": reward.tolist()}

            self.actions_buffer.append(action_dimt)
            self.obs_buffer.append(obs_dimt)
            self.last_obs_buffer.append(last_obs_dimt)
            self.reward_buffer.append(reward_dimt)
        else:
            action_pro_dimt = {"action_pro": action_pro.tolist(), }
            self.actions_pro_buffer.append(action_pro_dimt)

        if len(self.actions_buffer) == 360:
            if episode > 200:
                with open('./analysis_data/%s/%s/%s%s%d.json' % (self.name, 'actions', self.name, '_actions_', episode,),
                          'w', encoding='utf8') as f1:
                    json.dump(self.actions_buffer, f1)
                    self.actions_buffer = []
                with open('./analysis_data/%s/%s/%s%s%d.json' % (self.name, 'obs', self.name, '_obs_', episode,), 'w',
                          encoding='utf8') as f2:
                    json.dump(self.obs_buffer, f2)
                    self.obs_buffer = []
                with open('./analysis_data/%s/%s/%s%s%d.json' % (self.name, 'last_obs', self.name, '_last_obs_', episode,),
                          'w',
                          encoding='utf8') as f3:
                    json.dump(self.last_obs_buffer, f3)
                    self.last_obs_buffer = []
                with open('./analysis_data/%s/%s/%s%s%d.json' % (self.name, 'rewards', self.name, '_reward_', episode,), 'w',
                          encoding='utf8') as f4:
                    json.dump(self.reward_buffer, f4)
                    self.reward_buffer = []
                with open('./analysis_data/%s/%s/%s%s%d.json' % (self.name, 'actions_pro', self.name, '_actions_pro_', episode,), 'w',
                          encoding='utf8') as f5:
                    json.dump(self.actions_pro_buffer, f5)
                    self.actions_pro_buffer = []
            else:
                self.actions_buffer = []
                self.obs_buffer = []
                self.last_obs_buffer = []
                self.reward_buffer = []
                self.actions_pro_buffer = []

    def train(self, e, episodes):
        loss = self.policy.train()
        """ yjl 5.28 在每轮sample、train完再随机 """
        self.policy.actor.random_spaced(e, episodes, self.n_timesteps, fun='sample')
        self.policy.target_actor.random_spaced(e, episodes, self.n_timesteps, fun='sample')
        return loss

    def update_target_network(self):
        self.policy.update_target_model()

#TODO
class replay_buffer(object):
    def __init__(self, n_agent, buffer_size, device, obs_dim, action_dim, load_offline_data, path, json_name):
        self.buffer_size = buffer_size
        self.device = device
        self.n_intersection = n_agent

        self.last_obs = torch.zeros((buffer_size, n_agent, obs_dim)).float()
        self.actions = torch.zeros(buffer_size, n_agent).float()
        self.actions_pro = torch.zeros(buffer_size, n_agent, action_dim).float()  # 动作对应的概率值actions_pro(16,8)
        self.obs = torch.zeros(buffer_size, n_agent, obs_dim).float()
        self.rewards = torch.zeros(buffer_size, n_agent).float()  # view(-1,1)
        self.done = torch.zeros(buffer_size).float()

        if load_offline_data:
            move = self.load_offline_track(path, json_name)
            self.pointer = move
        else:
            self.pointer = 0

    def load_offline_track(self, path, json_name):
        with open(path + '%s.json' % json_name, 'r') as f:
            datas = json.load(f)
            last_obs = jsonpath.jsonpath(datas, '$..last_obs')
            length = len(last_obs)
            self.last_obs[:length] = torch.squeeze(torch.Tensor(last_obs))
            self.actions[:length] = torch.squeeze(torch.Tensor(jsonpath.jsonpath(datas, '$..action')))
            #act = torch.Tensor(jsonpath.jsonpath(datas, '$..action_pro'))
            act_pro = torch.softmax(torch.Tensor(jsonpath.jsonpath(datas, '$..action_pro')), dim=2)
            self.actions_pro[:length] = act_pro
            self.obs[:length] = torch.squeeze(torch.Tensor(jsonpath.jsonpath(datas, '$..obs')))
            self.rewards[:length] = torch.squeeze(torch.Tensor(jsonpath.jsonpath(datas, '$..reward')))
            # self.done[:length] = torch.zeros(self.buffer_size).float()

        return length

    def sample_track(self, batch_size):
        weights = np.zeros(1)
        tree_idxs = np.ones(1)
        idx = np.random.randint(0, self.buffer_size, size=(batch_size,))
        return (
            self.last_obs[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.actions_pro[idx].to(self.device),
            self.obs[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.done[idx].to(self.device),
            weights,
            tree_idxs        #yjl 5.28
        )

    def append_track(self, last_obs, actions, obs, rewards, done):
        move = len(actions.shape)
        if not done:
            dones = np.array(0.0)
        if self.pointer + move >= self.buffer_size:
            self.pointer = 0
        self.last_obs[self.pointer] = torch.from_numpy(last_obs).float()    # last_obs(16,12)
        self.obs[self.pointer] = torch.from_numpy(obs).float()              # obs(16,12)
        self.actions[self.pointer] = torch.from_numpy(actions).flatten()    # actions(1,16,1) -- actions(16,)
        self.rewards[self.pointer] = torch.from_numpy(rewards).flatten()    # rewards(16,)
        self.done[self.pointer] = torch.from_numpy(dones)

        self.pointer += move

    def append_pro(self, a_pro):
        self.actions_pro[self.pointer] = a_pro

    def update_priorities(self, data_idxs, priorities):
        pass

class Prioritized_Replay_Buffer(object):
    def __init__(self, n_agent, buffer_size, device, obs_dim, action_dim,
                 load_offline_data, path, json_name, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)
        self.device = device
        # PER params
        self.eps = eps              # 最小优先级，防止零概率
        self.alpha = alpha          # 确定使用多少优先级，α = 0对应于统一情况
        self.beta = beta            # 确定重要性抽样校正量，b = 1完全补偿非均匀概率
        self.max_priority = eps     # 优先为新样本，init为eps

        # transition: last_obs, actions, actions_pro, obs, rewards, done
        self.last_obs = torch.empty(buffer_size, n_agent, obs_dim, dtype=torch.float)
        self.actions = torch.empty(buffer_size, n_agent, dtype=torch.float)
        self.actions_pro = torch.empty(buffer_size, n_agent, action_dim, dtype=torch.float)
        self.obs = torch.zeros(buffer_size, n_agent, obs_dim)
        self.rewards = torch.empty(buffer_size, n_agent, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def load_offline_track(self, path, json_name):
        pass

    def sample_track(self, batch_size):
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        """
        要对大小为k的minibatch进行抽样，将范围[0,p_total]平均划分为k个范围。
        下一步，从每个范围中均匀采样一个值。最后是相应的转换
        从树中检索这些采样值中的每个值。(按比例优先级)
        """
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # Sample_idx是缓冲区中的示例索引，需要进一步对实际转换进行采样
            # Tree_idx是树中样本的索引，需要进一步更新优先级
            tree_idx, priority, idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(idx)

        # 具体地说，我们定义采样转移i的概率为 P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total
        """
        随机更新期望值的估计依赖于相应的更新到与期望相同的分布。优先重播引入偏见，因为它改变了这一点
        以不受控制的方式分布，因此改变了估计的解决方案
        收敛到(即使策略和状态分布是固定的)。我们可以使用
        mportance-sampling (IS) weights w_i = (1/N * 1/P(i))^β，完全补偿了非均匀性
        概率P(i)如果β = 1。这些权重可以通过w_i * δ_i折叠到Q-learning更新中
        代替δ_i(因此这是加权is，而不是普通is，参见例如Mahmood等人，2014)
        出于稳定性的考虑，我们总是将权重归一化为1/max wi，以便它们只缩放
        """
        weights = (self.real_size * probs) ** -self.beta
        """
        每当使用重要性抽样时，所有权重w_i都被缩放
        so max_i = 1;我们发现这在实践中效果更好，因为它保留了所有的权重
        在一个合理的范围内，避免了极大的更新的可能性。(附录B.2.1按比例排序)
        """
        weights = weights / weights.max()

        return (
            self.last_obs[sample_idxs].to(self.device),
            self.actions[sample_idxs].to(self.device),
            self.actions_pro[sample_idxs].to(self.device),
            self.obs[sample_idxs].to(self.device),
            self.rewards[sample_idxs].to(self.device),
            self.done[sample_idxs].to(self.device),
            weights,
            tree_idxs  # yjl 5.28
        )

    def append_track(self, last_obs, actions, obs, rewards, done):
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)
        # store transition in the buffer
        self.last_obs[self.count] = torch.as_tensor(last_obs)
        self.obs[self.count] = torch.as_tensor(obs)
        self.actions[self.count] = torch.as_tensor(actions).flatten()
        self.rewards[self.count] = torch.as_tensor(rewards)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    """ 
    append_pro 在每次get_actions 时调用 此时还没有训练 
    但是设置了learning_start, 前面没有训练但是此时也需要更新
    """
    def append_pro(self, a_pro):
        # store transition index with maximum priority in sum tree
        # self.tree.add(self.max_priority, self.count)
        # store transition in the buffer
        self.actions_pro[self.count] = torch.as_tensor(a_pro)

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = ((priority + self.eps) ** self.alpha).mean()
            priority = np.clip(priority, 1e-8, 2)
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.done = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.done[:]


def sync_network(model, target_model):
    weights = model.state_dict()
    target_model.load_state_dict(weights)

class DiffusionPolicy:
    def __init__(self, buffer, n_agent, obs_dim, action_dim, beta_schedule, n_timesteps, max_action, eta, loss_type,
                 sample_type, timestep_respacing, d_iters, d_steps, lr, batch_size,
                 max_q_backup, discount, grad_norm, tau,
                 clip_denoised=True,
                 predict_epsilon=True,
                 rescale_timesteps=False    #TODO 使用加速算法需要改成True
                 ):
        super(DiffusionPolicy, self).__init__()
        self.last_q_vale_lock = 0.0
        self.q_value_lock = 0.0       # 这里限制估计过高的q_value, 在sample_action() 限制挑选过高的 action

        self.device = torch.device("cpu")
        self.buffer = buffer
        self.n_agents = n_agent
        # 这里的MLP相当于DDPM图像生成中的Unet网络
        # TODO 在MLP中加attention
        self.model = MLP(n_agents=n_agent, obs_dim=obs_dim, action_dim=action_dim, device=self.device)

        """使用原始diffusion代码"""
        self.actor = Diffusion(
                    n_agent=n_agent,  # 剩下的都为**kwargs
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    model=self.model,
                    betas_s=beta_schedule,
                    n_timesteps=n_timesteps,
                    max_action=max_action,
                    eta=eta,
                    loss_type=loss_type,
                    sample_type=sample_type,
                    timestep_respacing=timestep_respacing,
                    clip_denoised=clip_denoised,
                    predict_epsilon=predict_epsilon,
                    rescale_timesteps=rescale_timesteps).to(self.device)
        self.target_actor = Diffusion(
                    n_agent=n_agent,  # 剩下的都为**kwargs
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    model=self.model,
                    betas_s=beta_schedule,
                    n_timesteps=n_timesteps,
                    max_action=max_action,
                    eta=eta,
                    loss_type=loss_type,
                    sample_type=sample_type,
                    timestep_respacing=timestep_respacing,
                    clip_denoised=clip_denoised,
                    predict_epsilon=predict_epsilon,
                    rescale_timesteps=rescale_timesteps).to(self.device)
        """使用加速算法"""
        # self.actor = create_diffusion(
        #     n_agent=n_agent,  # 剩下的都为**kwargs
        #     obs_dim=obs_dim,
        #     action_dim=action_dim,
        #     model=self.model,
        #     betas_schedule=beta_schedule,
        #     n_timesteps=n_timesteps,
        #     max_action=max_action,
        #     eta=eta,
        #     loss_type=loss_type,
        #     sample_type=sample_type,
        #     timestep_respacing=timestep_respacing,
        #     clip_denoised=clip_denoised,
        #     predict_epsilon=predict_epsilon,
        #     rescale_timesteps=rescale_timesteps
        # )
        # self.target_actor = create_diffusion(
        #     n_agent=n_agent,  # 剩下的都为**kwargs
        #     obs_dim=obs_dim,
        #     action_dim=action_dim,
        #     model=self.model,
        #     betas_schedule=beta_schedule,
        #     n_timesteps=n_timesteps,
        #     max_action=max_action,
        #     eta=eta,
        #     loss_type=loss_type,
        #     sample_type=sample_type,
        #     timestep_respacing=timestep_respacing,
        #     clip_denoised=clip_denoised,
        #     predict_epsilon=predict_epsilon,
        #     rescale_timesteps=rescale_timesteps
        # )
        sync_network(self.actor, self.target_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        """ 6.6 yjl """
        self.q_type = 'normal'  # normal lstm shapley atten dueling brother
        if self.q_type == 'normal':
            self.critic = Dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            self.target_critic = Dqn(self.n_agents, (obs_dim+action_dim), hidden_dim=256).to(self.device)
            sync_network(self.critic, self.target_critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        elif self.q_type == 'lstm':
            self.critic = Drqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            self.target_critic = Drqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            sync_network(self.critic, self.target_critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        elif self.q_type == 'shapley':
            self.critic = Shapley_Q(self.n_agents, obs_dim, action_dim, sample_size=64, hidden_dim=512).to(self.device)
            self.target_critic = Shapley_Q(self.n_agents, obs_dim, action_dim, sample_size=64, hidden_dim=512).to(self.device)
            sync_network(self.critic, self.target_critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        elif self.q_type == 'atten':
            self.critic = attention_dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            self.target_critic = attention_dqn(self.n_agents, (obs_dim+action_dim), hidden_dim=256).to(self.device)
            sync_network(self.critic, self.target_critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        elif self.q_type == 'dueling':
            self.critic = Dueling_Dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            self.target_critic = Dueling_Dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            sync_network(self.critic, self.target_critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        elif self.q_type == 'brother':
            self.b_critic = brother_dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            self.b_target_critic = brother_dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            self.b_a_critic = brother_atten_dqn(self.n_agents, (obs_dim+action_dim), hidden_dim=256).to(self.device)
            self.b_a_target_critic = brother_atten_dqn(self.n_agents, (obs_dim + action_dim), hidden_dim=256).to(self.device)
            sync_network(self.b_critic, self.b_target_critic)
            sync_network(self.b_a_critic, self.b_a_target_critic)
            self.b_critic_optimizer = torch.optim.Adam(self.b_critic.parameters(), lr=3e-4)
            self.b_a_critic_optimizer = torch.optim.Adam(self.b_a_critic.parameters(), lr=3e-4)

        self.init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.tau = tau
        self.grad_norm = grad_norm
        self.eta = eta
        self.n_agents = n_agent
        self.d_iters = d_iters
        self.d_steps = d_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.beta_schedule = beta_schedule
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size        #256
        self.max_q_backup = max_q_backup    #False
        self.sample_type = sample_type

        self.q_lock = torch.tensor(0.0)
        self.last_q_lock = self.q_lock
        self.last_action = torch.zeros(1, 16)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # TODO
    def train(self):
        last_obs, actions, actions_pro, obs, rewards, done, weights, tree_idxs = self.buffer.sample_track(self.batch_size)
        """ Q Training """
        """ 联合的q值  normal lstm shapley atten dueling brother """
        if self.q_type in ('normal', 'shapley', 'atten', 'dueling'):
            current_q1, current_q2 = self.critic(last_obs, actions_pro)  # last_obs(256,16,12)  actions(256,16,1)
        elif self.q_type == 'lstm':
            h, c = self.critic.init_hidden_state(self.batch_size, training=True)
            current_q1, current_q2 = self.critic(last_obs, actions_pro, h, c)
        elif self.q_type == 'brother':
            current_q1 = self.b_critic(last_obs, actions_pro)
            current_q2 = self.b_a_critic(last_obs, actions_pro)

        with torch.no_grad():
            next_actions, next_action_log_pis = self.target_actor.sample(obs)   #(256,16,8)
            # next_actions = torch.argmax(F.softmax(next_actions, dim=2), dim=2).reshape(self.batch_size, self.n_agents, 1)   #(256,16,1)
            if self.q_type in ('normal', 'shapley', 'atten', 'dueling'):
                target_q1, target_q2 = self.target_critic(obs, next_actions)
            elif self.q_type == 'lstm':
                target_h, target_c = self.target_critic.init_hidden_state(self.batch_size, training=True)
                target_q1, target_q2 = self.target_critic(last_obs, actions_pro, target_h, target_c)
            elif self.q_type == 'brother':
                target_q1 = self.b_target_critic(obs, next_actions)
                target_q2 = self.b_a_target_critic(obs, next_actions)

        target_q = torch.min(target_q1, target_q2)
        """ yjl 6.2 熵值 torch.log(action + z) """
        if self.q_type in ('normal', 'shapley', 'atten', 'dueling', 'lstm'):
            target_q = target_q - self.alpha.detach() * next_action_log_pis
            # rewards = (rewards - rewards.mean())/(rewards.std() + 1e-8)     #标准化
            target_q = (rewards + self.discount * target_q).detach()     #多维
            """ 更新replay buffer 的权值 """
            td_error = torch.abs(torch.min(current_q1, current_q2) - target_q).detach()
            self.buffer.update_priorities(tree_idxs, td_error.numpy())      # 这里没有更新
            critic_loss = 0.5 * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))
        elif self.q_type == 'brother':
            target_q1 = target_q1 - self.alpha.detach() * next_action_log_pis
            target_q1 = (rewards + self.discount * target_q1).detach()  # 多维
            target_q2 = target_q2 - self.alpha.detach() * next_action_log_pis
            target_q2 = (rewards + self.discount * target_q2).detach()  # 多维
            critic_loss = 0.5 * (F.mse_loss(current_q1, target_q1) + F.mse_loss(current_q2, target_q2))

        if self.q_type in ('normal', 'shapley', 'atten', 'dueling', 'lstm'):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()
        elif self.q_type == 'brother':
            self.b_critic_optimizer.zero_grad()
            self.b_a_critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.b_critic.parameters(), max_norm=self.grad_norm,norm_type=2)
                nn.utils.clip_grad_norm_(self.b_a_critic.parameters(), max_norm=self.grad_norm,norm_type=2)
            self.b_critic_optimizer.step()
            self.b_a_critic_optimizer.step()

        """ Policy Training """
        bc_loss = self.actor.loss(actions_pro, last_obs)        # actions对应的pro: (256,16,8) last_obs(256,16,12)
        new_action, new_action_log_pis = self.actor.sample(last_obs)   # forward - sample新的action
        if self.q_type in ('normal', 'shapley', 'atten', 'dueling'):
            q1_new_action, q2_new_action = self.critic(last_obs, new_action)
        elif self.q_type == 'lstm':
            new_h, new_c = self.critic.init_hidden_state(self.batch_size, training=True)
            q1_new_action, q2_new_action = self.critic(last_obs, actions_pro, new_h, new_c)
        elif self.q_type == 'brother':
            q1_new_action = self.b_critic(last_obs, new_action)
            q2_new_action = self.b_a_critic(last_obs, new_action)

        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        # actor_loss = bc_loss + self.eta * q_loss
        """ yjl 6.2 熵值 torch.log(action + z) """
        actor_loss = bc_loss + self.eta * q_loss + (self.alpha.detach() * new_action_log_pis).detach().mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ yjl 6.2 熵值 torch.log(action + z) """
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-new_action_log_pis - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return critic_loss.clone().detach().numpy()

    def offline_train(self):
        pass

    def sample_action(self, obs, batch, test=False):
        """
        param : obs ndarray (16,12)  全局观测
        """
        obs = torch.FloatTensor(obs.reshape(-1, self.n_agents, self.obs_dim))     # obs(1,16,12)
        obs_rpt = torch.repeat_interleave(obs, repeats=batch, dim=0)              # repeat_interleave dim=0  在列向复制batch份
        with torch.no_grad():
            if self.sample_type == "DDPM":
                action_pro, _ = self.actor.sample(obs_rpt)                             # (10,16,8)
            elif self.sample_type == "DDIM":
                action_pro = self.actor.ddim_sample(obs_rpt)                           # (10,16,8)

            if self.q_type in ('normal', 'shapley', 'atten', 'dueling'):
                q_value = self.target_critic.q_min(obs_rpt, action_pro).squeeze()     # obs_rpt(10,16,12) action(10,16,8) q_value(10,16,1)
            elif self.q_type == 'lstm':
                h, c = self.critic.init_hidden_state(batch, training=True)
                q_value = self.target_critic.q_min(obs_rpt, action_pro, h, c).squeeze()
            elif self.q_type == 'brother':
                q1_value = self.b_target_critic(obs_rpt, action_pro).squeeze()
                q2_value = self.b_a_target_critic(obs_rpt, action_pro).squeeze()
                q_value = torch.min(q1_value, q2_value)

            if batch > 1:
                action = torch.argmax(F.softmax(action_pro, dim=2), dim=2).squeeze()  # (10,16,1)
                # multinomial: 多项式概率分布中采样的num_samples索引
                # idxa = torch.argmax(F.softmax(q_value, dim=0), dim=1)
                idxm = torch.multinomial(F.softmax(q_value, dim=0), 1)  # q_value(17,16)  idx(1)
                idx = return_duplicate(idxm)  # idx(1) 从重复的数据里挑
                self.buffer.append_pro(action_pro[idx])
                action = action[idx].cpu().data.numpy().flatten()
                action_pro = action_pro[idx].cpu().unsqueeze(0)
            else:
                if test is False:   #train 增加掩码
                    mask = torch.randint_like(action_pro, low=0, high=3).clamp(0, 1)
                    action_pro = action_pro * mask
                    action = torch.argmax(F.softmax(action_pro, dim=2), dim=2).squeeze()  # (10,16,1)
                else:
                    action = torch.argmax(F.softmax(action_pro, dim=2), dim=2).squeeze()
                self.buffer.append_pro(action_pro)
                self.last_action = action.unsqueeze(dim=0)
        return action_pro, action.cpu().data.numpy().flatten()

    def update_target_model(self):
        polyak = 1.0 - self.tau
        if self.q_type in ('normal', 'shapley', 'atten', 'dueling', 'lstm'):
            for t_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)
            for t_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)
        if self.q_type == 'brother':
            for t_param, param in zip(self.b_target_critic.parameters(), self.b_critic.parameters()):
                t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)
            for t_param, param in zip(self.b_a_target_critic.parameters(), self.b_a_critic.parameters()):
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

