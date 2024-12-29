import math
import os
from collections import deque

from . import RLAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import gym
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from . import matlight_utils as U # No module named 'matlight_utils'
from agent import utils
from pfrl import replay_buffers

@Registry.register_model('matlight')
class MATLightAgent(RLAgent):

    def __init__(self, world, rank):
        """
        根据输入的参数和来自 Registry 的配置信息来定义模型结构
        """
        super().__init__(world, world.intersection_ids[rank])
        # print(Registry.mapping)
        # section 1: 获取配置
        self.world = world
        self.rank = rank
        self.sub_agents = len(self.world.intersections)
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter = self.world.id2intersection[self.inter_id]

        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.gae_lambda = Registry.mapping['model_mapping']['setting'].param['gae_lambda']
        # self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        # self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.hidden_size = Registry.mapping['trainer_mapping']['setting'].param['hidden_size']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.n_rollout_threads = Registry.mapping['trainer_mapping']['setting'].param['n_rollout_threads']
        self.recurrent_N = Registry.mapping['trainer_mapping']['setting'].param['recurrent_N']

        self.replay_buffer = U.SharedReplayBuffer(self.sub_agents, self.action_space, self.buffer_size, self.gamma, self.gae_lambda, self.n_rollout_threads, self.hidden_size, self.recurrent_N)
        # self.replay_buffer = replay_buffers.ReplayBuffer(360)
        # self.replay_buffer = deque(maxlen=360)

        self.device = torch.device("cpu")
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.clip_param = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.ppo_epoch = Registry.mapping['trainer_mapping']['setting'].param['epoch']
        self.num_mini_batch = Registry.mapping['model_mapping']['setting'].param['mini_batch']
        self.data_chunk_length = 10
        self.value_loss_coef = 1
        self.entropy_coef = Registry.mapping['trainer_mapping']['setting'].param['entropy_coef']
        self.max_grad_norm = 10.0
        self.huber_delta = 10.0
        self._use_recurrent_policy = False
        self._use_naive_recurrent = False
        self._use_max_grad_norm = True
        self._use_clipped_value_loss = True
        self._use_huber_loss = False
        self._use_valuenorm = True
        self._use_value_active_masks = False
        self._use_policy_active_masks = False
        self.dec_actor = Registry.mapping['model_mapping']['setting'].param['dec_actor']
        self.n_block = Registry.mapping['model_mapping']['setting'].param['n_block']
        self.n_embd = Registry.mapping['model_mapping']['setting'].param['n_embd']      # 64
        self.n_head = Registry.mapping['model_mapping']['setting'].param['n_head']
        self.encode_state = Registry.mapping['model_mapping']['setting'].param['encode_state']
        self.share_actor = Registry.mapping['model_mapping']['setting'].param['share_actor']
        if self._use_valuenorm:
            self.value_normalizer = U.ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self.data = None

        # section 2: 为每个智能体创建生成器
        # get generators for MATLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph 现在生成器的顺序是根据其在图中的索引
        self.ob_generator = observation_generators

        #  get reward generator
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["pressure"], in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, node_obj, ['phase'], targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        #  get queue generator
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"], in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        #  get delay generator
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_delay"], in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        # section 3: 设置动作空间和观测长度
        self.action_space = gym.spaces.Discrete(len(self.inter.phases))
        # if self.phase:
        #     if self.one_hot:
        #         self.ob_length = self.ob_generator.ob_length + len(self.inter.phases)
        #     else:
        #         self.ob_length = self.ob_generator.ob_length + 1
        # else:
        #     self.ob_length = self.ob_generator.ob_length

        # section 4: 创建模型、目标模型和其他
        self.model = self.build_model()
        # self.target_model = self.build_model()
        # self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.9, centered=False, eps=1e-7)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-7, weight_decay=0)
        # self.optimizer = self.model.optimizer

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
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["pressure"], in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, node_obj, ['phase'], targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        queues = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_waiting_count"], in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        delays = []
        for inter in self.world.intersections:
            node_id = inter.id
            node_idx = self.world.id2idx[node_id]
            node_obj = self.world.id2intersection[node_id]
            tmp_generator = LaneVehicleGenerator(self.world, node_obj, ["lane_delay"], in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        self.replay_buffer = U.SharedReplayBuffer(self.sub_agents, self.action_space, self.buffer_size, self.gamma,
                                                  self.gae_lambda, self.n_rollout_threads, self.hidden_size,
                                                  self.recurrent_N)
    def __repr__(self):
        """
        返回 self.model 结构
        """
        return self.model.__repr__()

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
            x_obs = [np.expand_dims(x,axis=0) for x in x_obs]
        return x_obs

    def get_reward(self):
        """
        从环境中获取奖励
        """
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards))
        return rewards

    def get_phase(self):
        """
        从环境中获取交叉口的当前相位
        """
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

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

    def get_action(self, ob, phase, test=False):
        """
        根据特征生成动作
        """
        """ DQN 2022.11.18 需要更改 """
        # if not test:
        #     if np.random.rand() <= self.epsilon:
        #         return self.sample()
        # if self.phase:
        #     if self.one_hot:
        #         feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
        #     else:
        #         feature = np.concatenate([ob, phase], axis=1)
        # else:
        #     feature = ob
        # observation = torch.tensor(feature, dtype=torch.float32)
        # actions = self.model(observation, train=False)
        # actions = actions.clone().detach().numpy()
        # return np.argmax(actions, axis=1)
        # print("get action's ob: ", ob)
        # print("get action's phase: ", phase)
        # 调用 matlight 获取动作
        if not test:
            # 训练阶段
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.model.get_actions(np.concatenate(ob),
                                                                                                 np.concatenate(ob),
                                                                                                 np.concatenate(self.replay_buffer.rnn_states[0]),
                                                                                                 np.concatenate(self.replay_buffer.rnn_states_critic[0]),
                                                                                                 np.concatenate(self.replay_buffer.masks[0]),
                                                                                                 )
            values = np.array(np.split(self._t2n(value), 1))
            actions = np.array(np.split(self._t2n(action), 1))
            action_log_probs = np.array(np.split(self._t2n(action_log_prob), 1))
            rnn_states = np.array(np.split(self._t2n(rnn_state), 1))
            rnn_states_critic = np.array(np.split(self._t2n(rnn_state_critic), 1))
            self.data = values, actions, action_log_probs, rnn_states, rnn_states_critic
            actionlist = []
            for i in range(len(actions[0])):
                actionlist.append(int(actions[0][i][0]))  # + 1
            actions = np.array(actionlist)
            return actions
        else:
            # 评估阶段
            eval_rnn_states = np.zeros((1, self.sub_agents, 1, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, self.sub_agents, 1), dtype=np.float32)
            eval_actions, eval_rnn_states = self.model.act(np.concatenate(ob),
                                                            np.concatenate(ob),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks),
                                                            deterministic=True)
            eval_actions = np.array(np.split(self._t2n(eval_actions), 1))
            actionlist = []
            for i in range(len(eval_actions[0])):
                actionlist.append(int(eval_actions[0][i][0]))
            actions = np.array(actionlist)
            return actions

    def _t2n(self, x):
        return x.detach().cpu().numpy()

    def sample(self):
        """
        随机采样动作
        """
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        """
        将当前步骤信息放入重播缓冲区，用于稍后训练智能体
        """
        """ 2022.11.23 存的过程有问题 """
        # print("==========START==============")
        # print("last_obs: ", last_obs)
        # print("last_phase: ", last_phase)
        # print("actions: ", actions)
        # print("actions_prob: ", actions_prob)
        # print("rewards: ", rewards)
        # print("obs: ", obs)
        # print("cur_phase: ", cur_phase)
        # print("done: ", done)
        # print("key: ", key)
        # print("===========END==============")
        value_preds, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.data
        # rnn_states = np.zeros((1, self.num_agent, self.dic_agent_conf["RECURRENT_N"], self.dic_agent_conf["HIDDEN_SIZE"]), dtype=np.float32)
        # rnn_states_critic = np.zeros((1, self.num_agent, 1, self.buffer.rnn_states_critic.shape[-1]), dtype=np.float32)
        masks = np.ones((1, self.sub_agents, 1), dtype=np.float32)
        active_masks = np.ones((1, self.sub_agents, 1), dtype=np.float32)
        bad_masks = np.array([[[1.0] for agent_id in range(self.sub_agents)]])
        rewards = np.asarray(rewards)
        rewards = rewards.reshape(self.sub_agents, 1)

        # self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))
        self.replay_buffer.insert(obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
                                  value_preds, rewards, masks, bad_masks, active_masks)
        # self.replay_buffer.append((key, (obs, cur_phase, rewards, done)))
        # self.do_observe(obs, cur_phase, rewards, done)
        # print("===============")

    # def do_observe(self, ob, phase, reward, done):
    #     if self.phase:
    #         if self.one_hot:
    #             obs = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
    #         else:
    #             obs = np.concatenate([ob, phase], axis=1)
    #     else:
    #         obs = ob
    #     obs = torch.tensor(obs, dtype=torch.float32)
    #     # self.agent.observe(obs, reward, done, False) # 重要，他们使用的pfrl里的基本深度强化学习算法

    def build_model(self):
        """
        创建模型（网络）
        """
        model = TransformerPolicy(self.action_space, self.sub_agents, self.learning_rate, self.n_block, self.n_embd, self.n_head, self.encode_state, self.dec_actor, self.share_actor)
        return model

    def update_target_network(self):
        """
        更新目标网络的参数
        """
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def train(self):
        """
        训练智能体，优化智能体产生的动作
        """
        # # take batch-sized samples from the replay buffer randomly
        # # 从重放缓冲区随机抽取批量大小的样本
        # samples = random.sample(self.replay_buffer, self.batch_size)
        # # convert samples into corresponding formats
        # # 将样本转换为相应的格式
        # b_t, b_tp, rewards, actions = self._batchwise(samples)
        # # put the next_feature into target model
        # # 将 next_feature 放入目标模型
        # out = self.target_model(b_tp, train=False)
        # target = rewards + self.gamma * torch.max(out, dim=1)[0]
        # # put the current_feature into target model
        # # 将 current_feature 放入目标模型
        # target_f = self.model(b_t, train=False)
        # for i, action in enumerate(actions):
        #     target_f[i][action] = target[i]
        # loss = self.criterion(self.model(b_t, train=True), target_f)
        # self.optimizer.zero_grad()
        # loss.backward()
        # clip_grad_norm_(self.model.parameters(), self.grad_clip)
        # self.optimizer.step()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        # return loss.clone().detach().numpy()
        self.compute()
        self.prep_training()
        train_infos = self.matlight_train(self.replay_buffer)
        self.replay_buffer.after_update()
        # return train_infos["loss"].clone().detach().numpy()
        return np.array(train_infos["loss"])

    def load_model(self, e):
        """
        加载回合的模型参数
        """
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model', f'{e}_{self.rank}.pt')
        self.model = self.build_model()
        self.model.transformer.load_state_dict(torch.load(model_name))
        # self.target_model = self.build_model()
        # self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        """
        保存一回合的模型参数
        """
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.model.transformer.state_dict(), model_name)

    def compute(self):
        """
        计算回报
        """
        self.prep_rollout()
        next_values = self.model.get_values(np.concatenate(self.replay_buffer.share_obs[-1]),
                                            np.concatenate(self.replay_buffer.obs[-1]),
                                            np.concatenate(self.replay_buffer.rnn_states_critic[-1]),
                                            np.concatenate(self.replay_buffer.masks[-1]))
        next_values = np.array(np.split(self._t2n(next_values), 1))
        self.replay_buffer.compute_returns(next_values, self.value_normalizer)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
            计算价值函数损失。
        """
        # clamp() 将张量压缩至区间[min, max]
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        if self._use_huber_loss: # 2022.9.23
            value_loss_clipped = U.huber_loss(error_clipped, self.huber_delta)
            value_loss_original = U.huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = U.mse_loss(error_clipped)
            value_loss_original = U.mse_loss(error_original)
        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks: # 2022.9.23
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        return value_loss

    def ppo_update(self, sample):
        """
            更新 actor 和 critic 网络
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample
        old_action_log_probs_batch = U.check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = U.check(adv_targ).to(**self.tpdv)
        value_preds_batch = U.check(value_preds_batch).to(**self.tpdv)
        return_batch = U.check(return_batch).to(**self.tpdv)
        active_masks_batch = U.check(active_masks_batch).to(**self.tpdv)

        # 重塑以对所有步骤进行一次前向传递
        values, action_log_probs, dist_entropy = self.model.evaluate_actions(share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, masks_batch, available_actions_batch, active_masks_batch)
        # actor 更新
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        if self._use_policy_active_masks: # 2022.9.23
            policy_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic 更新
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.model.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.model.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = U.get_gard_norm(self.model.transformer.parameters())
        self.model.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights, loss

    def matlight_train(self, buffer):
        """
            使用小批量 GD 进行训练更新。
        """
        # 2022.10.5 batch advantage normalization
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['loss'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, loss = self.ppo_update(sample)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['loss'] += loss.item()
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.model.train()

    def prep_rollout(self):
        self.model.eval()

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads 所有头的键、查询、值预测
        self.key = U.init_(nn.Linear(n_embd, n_embd))
        self.query = U.init_(nn.Linear(n_embd, n_embd))
        self.value = U.init_(nn.Linear(n_embd, n_embd))
        # output projection 输出投影
        self.proj = U.init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence 因果掩码，以确保仅将注意力应用于输入序列的左侧
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()  # 1,16,64    # 360 64 16 360为采样个数、64为线性映射 16为智能体个数
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 计算批处理中所有标头的查询、关键字和值，并将标头向前移动到批处理
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # self.att_bp = F.softmax(att, dim=-1)
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side
        # output projection
        y = self.proj(y)
        return y

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(U.init_(nn.Linear(n_embd, 1 * n_embd), activate=True), nn.GELU(), U.init_(nn.Linear(1 * n_embd, n_embd))) # 2022.10.7 GELU Tanh

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(U.init_(nn.Linear(n_embd, 1 * n_embd), activate=True), nn.GELU(), U.init_(nn.Linear(1 * n_embd, n_embd))) # 2022.10.7 GELU Tanh

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim), U.init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU()) # 2022.10.7 GELU Tanh
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim), U.init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU()) # 2022.10.7 GELU Tanh
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(U.init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd), U.init_(nn.Linear(n_embd, 1))) # 2022.10.7 GELU Tanh

    def forward(self, state, obs):
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings
        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)
        return v_loc, rep
        #return rep

class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        if action_type != 'Discrete': # list
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
        if self.dec_actor:  #dec_actor: False
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim), U.init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd), U.init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd), U.init_(nn.Linear(n_embd, action_dim))) # 2022.10.7 GELU
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim), U.init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd), U.init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd), U.init_(nn.Linear(n_embd, action_dim))) # 2022.10.7 GELU
                    self.mlp.append(actor)
        else:
            if action_type == 'Discrete': # list
                self.action_encoder = nn.Sequential(U.init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU()) # 2022.10.5 GELU()
            else:
                self.action_encoder = nn.Sequential(U.init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU()) # 2022.10.7 GELU
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim), U.init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU()) # 2022.10.5 GELU()
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(U.init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd), U.init_(nn.Linear(n_embd, action_dim))) # 2022.10.5 GELU()

    def zero_std(self, device):
        if self.action_type != 'Discrete': # list
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, obs_rep, obs):
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)
        return logit

class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent, n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"), action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()
        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent, self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete': # list
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 12), dtype=np.float32) # 2022.9.18 这里也有一个维度
        state = U.check(state).to(**self.tpdv)
        obs = U.check(obs).to(**self.tpdv)
        action = U.check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = U.check(available_actions).to(**self.tpdv)
        batch_size = np.shape(state)[0]

        v_loc, obs_rep = self.encoder(state, obs)

        if self.action_type == 'Discrete': # list
            action = action.long()
            action_log, entropy = U.discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size, self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = U.continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size, self.n_agent, self.action_dim, self.tpdv)
        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 12), dtype=np.float32) # 2022.9.18 这里还有个维度
        state = U.check(state).to(**self.tpdv)
        obs = U.check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = U.check(available_actions).to(**self.tpdv)
        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete": # Discrete 离散的，因为我们没有用MultiDiscrete类，而是list
            output_action, output_action_log = U.discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size, self.n_agent, self.action_dim, self.tpdv, available_actions, deterministic)
        else:
            output_action, output_action_log = U.continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size, self.n_agent, self.action_dim, self.tpdv, deterministic)
        return output_action, output_action_log, v_loc

    def get_values(self, state, obs):
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 12), dtype=np.float32) # 2022.9.18 这里也有一个维度
        state = U.check(state).to(**self.tpdv)
        obs = U.check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot

class TransformerPolicy:
    """
        MAPPO 策略类。 包装 actor 和 critic 网络来计算动作和价值函数预测。
    """
    def __init__(self, act_space, num_agents, lr, n_block, n_embd, n_head, encode_state, dec_actor, share_actor, device=torch.device("cpu")):
        self.device = device
        self.lr = float(lr)
        self.opti_eps = 1e-5
        self.weight_decay = 0
        self._use_policy_active_masks = False
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = 12 # U.get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = 12 # U.get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        # print("obs_dim: ", self.obs_dim)
        # print("share_obs_dim: ", self.share_obs_dim)
        # print("act_dim: ", self.act_dim)

        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.transformer = MultiAgentTransformer(self.share_obs_dim, self.obs_dim, self.act_dim, num_agents,
                                                 n_block=n_block, n_embd=n_embd, n_head=n_head,
                                                 encode_state=encode_state, device=device,
                                                 action_type=self.action_type, dec_actor=dec_actor,
                                                 share_actor=share_actor)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
            衰减actor和critic的学习率。
        """
        U.update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
            计算给定输入的动作和价值函数预测。
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)
        actions, action_log_probs, values = self.transformer.get_actions(cent_obs, obs, available_actions, deterministic)
        actions = actions.view(-1, self.act_num)    # actions 已经转换为int
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        rnn_states_actor = U.check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = U.check(rnn_states_critic).to(**self.tpdv)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, obs, rnn_states_critic, masks):
        """
            获取价值函数预测。
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        values = self.transformer.get_values(cent_obs, obs)
        values = values.view(-1, 1)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks, available_actions=None, active_masks=None):
        """
            获取 动作日志概率 / 熵 和 值函数预测 以进行 actor 更新。
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        # --> encoder
        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)
        if self._use_policy_active_masks: # and active_masks is not None: # 2022.9.23
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()
        return values, action_log_probs, entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True):
        """
            使用给定的输入计算动作。
        """
        # this function is just a wrapper for compatibility
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, rnn_states_actor, _ = self.get_actions(cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()