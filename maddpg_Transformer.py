from . import RLAgent
from common.registry import Registry
from collections import deque
import random
import os

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from agent import utils

import gym
import numpy as np

from torch import nn
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


# q_loss 为 0
@Registry.register_model('T_maddpg')
class Maddpgagent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        # # self.replay_buffer deque-tuple
        self.replay_buffer = deque(maxlen=self.buffer_size)
        # self.capacity = Registry.mapping['model_mapping']['setting'].param['capacity']
        # self.replay_buffer = PER_Buffer(capacity=int(self.capacity))

        self.world = world
        self.sub_agents = 1
        self.rank = rank
        self.n_intersections = len(world.id2intersection)
        self.agents = None

        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']

        # get generator for each DQNAgent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length

        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        """
            time: 23.3.2 第一次添加
        """
        self.noise_rate = Registry.mapping['model_mapping']['setting'].param['noise_rate']
        self.best_epoch = 0
        # param
        self.local_q_learn = Registry.mapping['model_mapping']['setting'].param['local_q_learn']
        self.action = 0
        self.last_action = 0
        self.q_length = 0

        # net works
        self.q_model = None
        self.target_q_model = None
        self.p_model = None
        self.target_p_model = None

        self.criterion = None
        self.q_optimizer = None
        self.p_optimizer = None

    def link_agents(self, agents):
        self.agents = agents
        if not self.local_q_learn:  # False
            full_action = 0
            full_observe = 0
            for ag in self.agents:  # 共享 information
                full_action += ag.action_space.n
                full_observe += ag.ob_length
            self.q_length = full_observe + full_action
        else:
            self.q_length = self.ob_length + self.action_space.n

        self.q_model = self._build_model(self.q_length, 1)
        self.target_q_model = self._build_model(self.q_length, 1)

        self.p_model = self._build_model(self.ob_length, self.action_space.n)
        self.target_p_model = self._build_model(self.ob_length, self.action_space.n)
        self.sync_network()

        self.criterion = nn.MSELoss(reduction='mean')
        self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=self.learning_rate)
        self.p_optimizer = optim.Adam(self.p_model.parameters(), lr=self.learning_rate)

    def __repr__(self):
        return self.p_model.__repr__() + '\n' + self.q_model.__repr__()

    def reset(self):
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action = 0
        self.last_action = 0

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards))
        # 列表对比(对应位置是否相同) True or False 自动转换为 0 和 1
        # rewards 设计
        rewards = rewards + (self.action == self.last_action) * 2

        if type(rewards) == np.float64:
            rewards = np.array(rewards, dtype=np.float64)[np.newaxis]
        self.last_action = self.action
        return rewards

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions_o = self.p_model(observation, train=False)

        # actions = torch.argmax(actions_o, dim=1)
        actions_prob = self.G_softmax(actions_o)

        actions = torch.argmax(actions_prob, dim=1)
        actions = actions.clone().detach().numpy()
        self.last_action = self.action
        self.action = actions
        return actions

    def get_action_prob(self, ob, phase):
        if self.phase:
            if self.one_hot:
                feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.p_model(observation, train=False)
        actions_prob = self.G_softmax(actions)
        return actions_prob

    def sample(self):
        # TODO self.action_space.n   self.action_sapce
        # random.randint(low,high,size) 有多少agent size有多大
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    """
            type: gaussian noise 高斯噪声
            time: 23.3.2 15.53
            function: 增加agent动作探索空间
    """

    def Gaussian_noise(self, noise_rate, action_space):
        noise = noise_rate * np.random.randn(*action_space)
        return noise

    """
        Gaussian-softmax
    """

    # def G_softmax(self, p):
    #     u = torch.rand(self.action_space.n)
    #     prob = F.softmax((p - torch.log(-torch.log(u))/1), dim=1)
    #     #prob = F.softmax(p, dim=1)
    #     return prob

    def G_softmax(self, p):
        mv = torch.full((8,), 0)
        noise = Ornstein_Uhlenbeck_Noise(theta=0.1, sigma=0.1, dt=1e-2, mean_value=mv)
        prob = F.softmax((p + noise()), dim=1)
        prob = torch.tensor(prob.clone().detach(), dtype=torch.float32)
        return prob

    """
        Ornstein-Uhlenbeck noise-softmax
    """

    # def OU_softmax(self, p):
    #     prob = F.softmax(p)
    #     return prob

    def _batchwise(self, samples):
        obs_t = np.concatenate([item[1][0] for item in samples])
        obs_tp = np.concatenate([item[1][4] for item in samples])
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n) for item in samples])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n) for item in samples])
            else:
                phase_t = np.concatenate([item[1][1] for item in samples])
                phase_tp = np.concatenate([item[1][5] for item in samples])
            feature_t = np.concatenate([obs_t, phase_t], axis=1)
            feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        # list [array{[]}、array.....
        t = [item[1][3] for item in samples]
        rewards = torch.tensor(np.concatenate([item[1][3] for item in samples])[:, np.newaxis],
                               dtype=torch.float32)  # TODO: BETTER WA

        # TODO: reshape
        actions_prob = torch.cat([item[1][2] for item in samples], dim=0)  # why is prob
        return state_t, state_tp, rewards, actions_prob

    def train(self):
        b_t_list = []
        b_tp_list = []
        rewards_list = []
        action_list = []
        target_q = 0.0
        tree_indexs = []    # 抽样的叶子indexs
        IS_Weights = []     # SumTree 更新权重
        sample_index = random.sample(range(len(self.replay_buffer)), self.batch_size)
        for ag in self.agents:
            # ndarray(256, 2)
            samples = np.array(list(ag.replay_buffer), dtype=object)[sample_index]
            #samples = ag.replay_buffer.sample(self.batch_size)
            # samples[0] : indices  samples[1]: np_IS_weights samples[2]:datas
            b_t, b_tp, rewards, actions = ag._batchwise(samples)
            #tree_indexs.append(samples[0])
            #IS_Weights.append(samples[1])
            b_t_list.append(b_t)
            b_tp_list.append(b_tp)

            rewards_list.append(rewards)
            #action_list.append(actions)
            action_list.append(actions)  # actions(256, 8)
        target_act_next_list = []
        for i, ag in enumerate(self.agents):
            target_act_next = ag.target_p_model(b_tp_list[i], train=False)
            target_act_next = ag.G_softmax(target_act_next)
            target_act_next_list.append(target_act_next)
        full_b_t = torch.cat(b_t_list, dim=1)
        full_b_tp = torch.cat(b_tp_list, dim=1)
        full_action_tp = torch.cat(target_act_next_list, dim=1)
        full_action_t = torch.cat(action_list, dim=1)
        # combine b_t and corresponding full_actions

        if self.local_q_learn:
            q_input_target = torch.cat((b_tp_list[self.rank], full_action_tp[self.rank]), dim=1)
            q_input = torch.cat((b_t_list[self.rank], full_action_t), dim=1)
        else:
            q_input_target = torch.cat((full_b_tp, full_action_tp), dim=1)
            q_input = torch.cat((full_b_t, full_action_t), dim=1)

        # q_input_target (256, 320) torch.float64  torch.float32
        target_q_next = self.target_q_model(q_input_target, train=False)
        target_q += rewards_list[self.rank] + self.gamma * target_q_next
        q = self.q_model(q_input, train=True)
        """
            更新SunTree 23.3.5
        """
        # diff_q = q - target_q   # TODO may need sure ? 差值
        # np_error = torch.abs(diff_q).detach().numpy()
        # self.replay_buffer.batch_update(tree_indexs, np_error)

        # update q network
        q_reg = torch.mean(torch.square(q))
        q_loss = self.criterion(q, target_q)  # TODO
        """
            huber_weight 更新 
        """
        #q_loss = cal_huber_weight(diff_q, IS_Weights)
        loss_of_q = q_loss + q_reg * 1e-3

        self.q_optimizer.zero_grad()
        loss_of_q.backward()
        clip_grad_norm_(self.q_model.parameters(), self.grad_clip)
        self.q_optimizer.step()

        # update p network
        p = self.p_model.forward(b_t_list[self.rank], train=True)
        p_prob = self.G_softmax(p)
        p_reg = torch.mean(torch.square(p))
        if self.local_q_learn:
            pq_input = torch.cat((b_t_list[self.rank], p_prob), dim=1)
        else:
            action_list[self.rank] = p_prob
            full_action_t_q = torch.cat(action_list, dim=1)
            pq_input = torch.cat((full_b_t.detach(), full_action_t_q), dim=1)
            # pq_input = torch.cat((full_b_t, full_action_t_q), dim=1)

        # todo: test here
        p_loss = torch.mul(-1, torch.mean(self.q_model(pq_input, train=True)))
        loss_of_p = p_loss + p_reg * 1e-3

        self.p_optimizer.zero_grad()
        loss_of_p.backward()
        clip_grad_norm_(self.p_model.parameters(), self.grad_clip)

        self.p_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss_of_q.clone().detach().numpy()

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions_prob, rewards, obs, cur_phase)))

    def _build_model(self, input_dim, output_dim):
        model = DQNNet(input_dim, output_dim)
        return model

    def _build_actor(self, input_dim, output_dim):
        model = Actor(input_dim, output_dim)
        return model

    def update_target_network(self):
        polyak = 1.0 - self.tau
        for t_param, param in zip(self.target_q_model.parameters(), self.q_model.parameters()):
            t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)
        for t_param, param in zip(self.target_p_model.parameters(), self.p_model.parameters()):
            t_param.data.copy_(t_param.data * polyak + (1 - polyak) * param.data)

    def sync_network(self):
        p_weights = self.p_model.state_dict()
        self.target_p_model.load_state_dict(p_weights)
        q_weights = self.q_model.state_dict()
        self.target_q_model.load_state_dict(q_weights)

    def load_model(self, e):
        model_p_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                    'model_p', f'{e}_{self.rank}.pt')
        model_q_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                    'model_q', f'{e}_{self.rank}.pt')
        self.model_q = self._build_model(self.q_length, 1)
        self.model_p = self._build_model(self.ob_length, self.action_space.n)
        self.model_q.load_state_dict(torch.load(model_q_name))
        self.model_p.load_state_dict(torch.load(model_p_name))
        self.sync_network()

    def save_model(self, e):
        path_p = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_p')
        path_q = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model_q')
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        if not os.path.exists(path_q):
            os.makedirs(path_q)
        model_p_name = os.path.join(path_p, f'{e}_{self.rank}.pt')
        model_q_name = os.path.join(path_q, f'{e}_{self.rank}.pt')
        torch.save(self.p_model.state_dict(), model_p_name)
        torch.save(self.q_model.state_dict(), model_q_name)

    # 没有使用
    def load_best_model(self, ):
        model_p_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                    'model_p', f'{self.best_epoch}_{self.rank}.pt')
        model_q_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                    'model_q', f'{self.best_epoch}_{self.rank}.pt')
        self.q_model.load_state_dict(torch.load(model_q_name))
        self.p_model.load_state_dict(torch.load(model_p_name))
        self.sync_network()


class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.tanh(self.dense_3(x))
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


# {Tensor:(8, )} u = tensor([0.7895, 0.5494, 0.8615, 0.8921, 0.1390, 0.5416, 0.9220, 0.3262])
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


class SumTree(object):
    data_pointer = 0
    stored_data = 0

    def __init__(self, capacity):
        self.capacity = capacity
        # Generate the tree with all nodes values = 0
        # 记住，我们是在一个二进制节点(每个节点最多有2个子节点)所以2倍大小的叶(容量)- 1(根节点)
        # 父节点= capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        #self.tree[0:] = 0.0001  #初始化，防止为零
        """ tree:
          0
         / \
        0   0
       / \ / \
      0  0 0  0  [Size: capacity]这一行存放优先级分数,最后一层需要是2的n次方
      """
        self.data = np.zeros(capacity, dtype=object)
        #self.data[0:] = 0.001  error: zero-size array to reduction operation maximum which has no identity
        # self.data = deque(maxlen=capacity)    #deque 将会修改每条经验的索引，导致sumtree失效

    # 这里在sumtree叶子中添加优先级分数，并在数据中添加经验
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        # 从左到右填满叶子
        """ tree:
               0
              / \
             0   0
            / \ / \
  tree_index  0 0  0    capacity = 4  eg. tree_index=3  self.data_pointer=0
      """
        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        self.stored_data += 1

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority, train=False):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # 然后通过树传播更改
        #if tree_index.any == 0:     # how to do
        # TODO 'int' object has no attribute 'any'!!!
        if train is not False:
            if tree_index.any() == 0:
                print('error')
            while bool(tree_index.all()):  # 此方法比参考代码中的递归循环更快
                """
                  0         如果我们在叶索引6处,则更新优先级分数
                 / \        然后我们需要更新索引2节点
                1   2       那么tree_index = (tree_index - 1) // 2
               / \ / \      Tree_index = (6-1)//2
              3  4 5 [6]    Tree_index = 2(因为//舍入结果)           
                """
                tree_index = (tree_index - 1) // 2
                self.tree[tree_index] += change
        else:
            if tree_index == 0:
                print('error')
            while tree_index != 0:
                tree_index = (tree_index - 1) // 2
                self.tree[tree_index] += change


    # 这里我们得到leaf_index, 该叶的优先级值和与该索引相关的经验
    def get_leaf(self, v):
        """
        树结构和数组存储:
        树指数:
           0         -> storing priority sum
          / \
         1   2
        / \ / \
       3  4 5  6    -> storing priority for experiences
      Array type for storing:  [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # 向下搜索，总是搜索优先级更高的节点
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def get_leafs(self, values):
        indices = []
        priorities = []
        datas = []
        for v in values:
            idx, prior, data = self.get_leaf(v)
            if 'int' in str(type(data)):
                print("\nWARNING: No obs leaf: data:{}  v:{}  idx:{}  priority:{}".format(
                    data, v, idx, prior))
                continue
            indices.append(idx)
            priorities.append(prior)
            datas.append(data)
        return np.array(indices), np.array(priorities), datas

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node

    def pri(self):
        print(self.tree)

# 原代码 每个网络计算一次， Maddpg 需要其他网络的参数，这样SumTree和weight的更新会不会有问题
def cal_huber_weight(difference, weights=None, d=1):
    batch_loss1 = (difference.abs() < 1).float() * (0.5 * (difference ** 2))
    batch_loss2 = (difference.abs() >= 1).float() * (d * difference.abs() - 0.5 * d)
    batch_loss = batch_loss1 + batch_loss2
    if weights is not None:
        # TODO 因为这里我传入的weights是16个智能体的所有(maddgp Critic 需要所有的agent的信息), weight(list16)(256,1) batch_loss(256,1)
        weights_batch_loss = weights * batch_loss
    else:
        weights_batch_loss = batch_loss
    weights_loss = weights_batch_loss.mean()
    return weights_loss

class PER_Buffer:
    def __init__(self, capacity):
        super().__init__()
        # PER_E: 使用超参数来避免某些经历被采取的概率为0
        self.PER_E = 0.01
        # PER_A: 使用这个超参数来在只获取具有高优先级的exp和随机抽样之间进行权衡
        self.PER_A = 0.6
        self.PER_B = 0.1  # 重要性抽样，从初始值递增到1
        self.PER_B_increment = 0.001  # PER_B 每次增量

        self.minimal_priority = 1  # 防止优先级为0，导致经验值无法被选中
        """
        记住,我们的树是由一个树组成的,它包含了各叶的优先级分数
        还有一个数据数组
        不使用deque,因为这意味着在每个时间步,我们的经验改变一个索引。
        我们更喜欢使用一个简单的数组, 并在内存满时进行覆盖。
        """
        self.Priority_tree = SumTree(capacity)

    """
    在我们的树中存储一个新的经验
    每个新的经验都有一个max_priority的分数(当我们使用这个exp来训练我们的DDQN时,它会得到改善)
    """

    def store(self, key, last_obs, last_phase, actions_prob, rewards, obs, cur_phase):
        experience = (key, (last_obs, last_phase, actions_prob, rewards, obs, cur_phase))

        # Find the max priority
        max_priority = np.max(self.Priority_tree.tree[-self.Priority_tree.capacity:])  # 从最左子叶开始
        """
            如果最大优先级 = 0,我们不能把优先级 = 0,因为这个exp将永远没有机会被选中
            所以使用小的优先级代替 0
        """
        if max_priority == 0:
            max_priority = self.minimal_priority

        self.Priority_tree.add(max_priority, experience)  # set the max p for new p

    """
    首先,要对n个大小的minibatch进行抽样,范围[0,priority_total]为/,分为n个范围。
    在每个范围内统一采样一个值
    我们在sumtree中搜索,从优先级分数对应的样本值中检索经验。
    然后,我们计算每个小批量元素的ISWeights
    """

    def sample(self, n):

        # 计算优先级段,在这里，正如论文中解释的那样，我们将Range[0, ptotal]划分为n个Range
        priority_segment = self.Priority_tree.total_priority / n
        # 在这里，我们每次对一个新的小批量进行采样时都增加PER_B
        self.PER_B = np.minimum(1., self.PER_B + self.PER_B_increment)

        values = []
        for i in range(n):
            """
                一个值是从每个范围中统一采样的
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            values.append(value)
            """
                检索与每个值对应的经验
            """
        indices, prioritys, datas = self.Priority_tree.get_leafs(values)
        # 不需要pow(p, alpha)，因为我们已经做了AD更新
        sampling_probs = prioritys / self.Priority_tree.total_priority
        N = self.Priority_tree.stored_data
        weights = np.power(N * sampling_probs, -self.PER_B)
        max_weight = weights.max()

        np_IS_weights = weights / max_weight
        np_IS_weights = np_IS_weights.reshape((-1, 1))

        out_IS_weights = torch.from_numpy(np_IS_weights).float().to("cuda:0" if torch.cuda.is_available() else "cpu")
        return indices, out_IS_weights, datas

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        """
        assumes abs_errors is > 0
        """
        abs_errors += self.PER_E  # add eps to avoid zero
        clipped_errors = np.minimum(abs_errors, self.minimal_priority)  # upper bound 1.
        ps = np.power(clipped_errors, self.PER_A)

        for ti, p in zip(tree_idx, ps):
            self.Priority_tree.update(ti, p, train=True)

    def __len__(self):
        return min(self.Priority_tree.stored_data, self.Priority_tree.capacity)
