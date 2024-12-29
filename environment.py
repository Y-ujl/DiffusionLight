import gym
import numpy as np
from common.registry import Registry

class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    def __init__(self, world, agents, metric):
        """
        :param world: one world object to interact with agents. Support multi world
        objects in different TSCEnvs.
        :param agents: single agents, each control all intersections. Or multi agents,
        each control one intersection.
        actions is a list of actions, agents is a list of agents.
        :param metric: metrics to evaluate policy.
        """
        self.world = world
        self.eng = self.world.eng
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepsac':
            self.n_agents = len(agents)
        else:
            self.n_agents = len(agents) * agents[0].sub_agents
        # test agents number == intersection number
        assert len(world.intersection_ids) == self.n_agents
        self.agents = agents
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepsac':
            action_dims = [agent.action_space.n for agent in agents]
        else:
            action_dims = [agent.action_space.n * agent.sub_agents for agent in agents]
        # total action space of all agents.
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.metric = metric

    def step(self, actions):
        """
        :param actions: keep action as N_agents * 1
        """
        if not actions.shape:
            assert (self.n_agents == 1)
            actions = actions[np.newaxis]
        else:
            assert len(actions) == self.n_agents
        self.world.step(actions)

        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]
            # obs = np.expand_dims(np.array(obs),axis=1)
            rewards = [agent.get_reward() for agent in self.agents]
            # rewards = np.expand_dims(np.array(rewards),axis=1)
        else:
            obs = [self.agents[0].get_ob()]
            rewards = [self.agents[0].get_reward()]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]  # [agent, sub_agent==1, feature]
            # obs = np.expand_dims(np.array(obs),axis=1)
        else:
            obs = [self.agents[0].get_ob()]  # [agent==1, sub_agent, feature]
        return obs

# 相似性计算 余弦相似性计算
def goal_similarity(goal_a, goal_b):
    goal_a = np.mat(goal_a)
    goal_b = np.mat(goal_b)
    num = float(goal_a * goal_b.T)
    denom = np.linalg.norm(goal_a) * np.linalg.norm(goal_b)
    cos = num / denom
    similarity = 0.5 + 0.5 * cos
    return similarity

'''当前phase = 上一次action'''
class TSCGoalEnv(gym.GoalEnv):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, 用于计算评价指标
    """
    def __init__(self, world, agents, metric, similarity_threshold):
        self.world = world
        self.n_agents = len(agents) * agents[0].sub_agents
        assert len(world.intersection_ids) == self.n_agents
        self.agents = agents

        self.min_real_travel_time = 0.0
        self.similarity_threshold = similarity_threshold        # 差一个很小值即认为认为成功

        # TODO yjl 4.15add
        action_dims = [agent.action_space.n for agent in agents]
        self.goal = [agent.sample_goal_phase() for agent in agents]  # TODO 由GAN网络生成 获取目标相位
        self.phase = self.get_agents_phase()
        # 根据phase shape 修改
        self.phase_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(0, 8, shape=self.phase['achieved_goal_phase'].shape,
                                        dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=self.phase['achieved_goal_phase'].shape,
                                         dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf, shape=self.phase['observation_phase'].shape, dtype='float32'),
        ))

        # total action space of all agents.
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.metric = metric

    def step(self, actions):
        if not actions.shape:
            assert (self.n_agents == 1)
            actions = actions[np.newaxis]
        else:
            assert len(actions) == self.n_agents
        self.world.step(actions)

        phase = self.get_agents_phase()
        obs = self.get_agents_obs()
        rewards = self.compute_reward()

        dones = [False] * self.n_agents
        info = {
            'is_success': self._is_success(phase['achieved_goal'], self.goal),  # TODO self.goal 在什么时候更新
        }
        return phase, obs, rewards, dones, info

    def reset(self):
        self.world.reset()
        obs = self.get_agents_obs()
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):    # TODO 修改跟phase goal 有关
        if not len(self.agents) == 1:
            rewards = [agent.get_reward() for agent in self.agents]
        else:
            rewards = [self.agents[0].get_reward()]
        return rewards

    def get_agents_phase(self):
        if not len(self.agents) == 1:
            phase = [agent.get_phase() for agent in self.agents]
        else:
            phase = [self.agents[0].get_phase()]
        return phase

    def get_agents_obs(self):
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]
        else:
            obs = [self.agents[0].get_obs()]
        return obs

    def _is_success(self, achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        achieved_goal = achieved_goal.flatten()     # 将ndarray(16,1) 转化成(16)
        desired_goal = desired_goal.flatten()
        d = 1 - goal_similarity(achieved_goal, desired_goal)
        return (d < self.similarity_threshold).astype(np.float32)
