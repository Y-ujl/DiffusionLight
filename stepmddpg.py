from . import RLAgent
import os
import math
import torch
import random
import numpy as np
from torch import nn
from agent import utils
import torch.optim as optim
from common.registry import Registry
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from utils.sac import SAC
from utils.gan import TSCGAN, noise_generator
from utils.gan_param import param_
from utils.base_transformer import Encoder
from utils.replay_buffer_gan import replay_buffer, gan_sampler

@Registry.register_model('stepmddpg')
class StepMDDPGAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        # world params
        self.world = world
        self.rank = rank
        self.n_intersections = len(world.id2intersection)
        self.sub_agents = len(self.world.intersections)     # 创建一个大的模型

        # base param / world generator
        self.agent_management = param_(self.world, self.rank, self.sub_agents)
        self.agent_management.create()

        self.gan_module = gan_sampler()      # TODO
        self.replay_buffer = replay_buffer(self.agent_management, self.gan_module)
        self.encoder = Encoder(self.agent_management.phase_ob_length,
                               self.agent_management.n_block,
                               self.agent_management.n_embd,
                               self.agent_management.n_head,
                               self.agent_management.n_agent)

        self.gan = TSCGAN(self.agent_management.phase_dim,
                          self.agent_management.phase_dim,
                          self.agent_management.phase_ob_length,
                          self.agent_management.gan_nembd,
                          self.agent_management.gan_glr,
                          self.agent_management.gan_dlr)

        self.agents = SAC()

    def __repr__(self):
        return self.agents.__repr__()

    def reset(self):
        self.agent_management.reset_world_generator(self.world, self.rank, generator_type='unite')

    def sample_goal_phase(self):
        pass

    def get_phase(self):
        phase = []
        for i in range(len(self.agent_management.phase_generator)):
            phase.append(self.agent_management.phase_generator[i][1].generate())
        phase = np.concatenate(phase, dtype=np.int8)
        achieved_goal_phase = phase                     #TODO 修改
        # step1 generator noise
        noise = noise_generator(self.agent_management, noise_type='Ram_distribution') #noise(8,16,20)
        # step2 Encoder
        noise, _ = self.encoder(noise)  #noise(8,16,64)
        # step3 goal_phase
        # TODO 输出的目标值不太正确 应为[0,7]中的整数
        goal_phase = self.gan.get_goal_phase(noise)  # noise(8,16,64)TODO

        return {
            'observation_phase': phase.copy(),
            'achieved_goal_phase': achieved_goal_phase.copy(),
            'desired_goal_phase': goal_phase,
        }

    # def _is_success(self, achieved_goal, desired_goal):
    #     d = goal_phase_dis(achieved_goal, desired_goal)
    #     return (d < self.distance_threshold).astype(np.float32)

    def get_ob(self):
        """
        环境中获取N交叉路口的观测值
        obs： 各路口车辆 lane_count 只计算in方向
        """
        obs = []
        for i in range(len(self.agent_management.ob_generator)):
            obs.append(self.agent_management.ob_generator[i][1].generate())
        length = set([len(i) for i in obs])
        if len(length) == 1:
            obs = np.array(obs, dtype=np.float32)
        else:
            obs = [np.expand_dims(x,axis=0) for x in obs]
        return obs

    def get_reward(self):
        pass

    def get_action(self, ob, phase, test=False):
        pass

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def G_softmax(self, p):
        u = torch.rand(self.action_space.n)
        prob = F.softmax((p - torch.log(-torch.log(u)) / 1), dim=1)
        # prob = F.softmax(p, dim=1)
        return prob

    def _batchwise(self, samples):
        pass

    def train(self):
        pass

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions_prob, rewards, obs, cur_phase)))

    def update_target_network(self):
        pass

    def sync_network(self):
        pass

    def load_model(self, e):
        pass

    def save_model(self, e):
        pass

    # 没有使用
    def load_best_model(self, ):
        pass

