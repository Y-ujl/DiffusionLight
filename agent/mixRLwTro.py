import gym
from . import BaseAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import numpy as np

@Registry.register_model('mixRlwTro')
class mixRlwTro(BaseAgent):
    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world         #World object
        self.rank = rank
        self.model = None

        # 获取每个mixRlwTro的生成器
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id] #Intersection object
        """
            "lane_count": get number of running vehicles on each lane. 
            "lane_waiting_count": get number of waiting vehicles(speed less than 0.1m/s) on each lane. 
            "lane_waiting_time_count": get the sum of waiting time of vehicles on the lane since their last action. 
            "lane_delay": the delay of each lane: 1 - lane_avg_speed/speed_limit. 每个车道的延迟1 - lane_avg_speed/speed_limit
        """
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'],
                                                    in_only=True, average=None)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                    in_only=True, average='all', negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_waiting_count"],
                                                    in_only=True, negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_delay"],
                                                    in_only=True, negative=False)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter_obj, ["phase"],
                                                    targets=["cur_phase"], negative=False)
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        # 不同的数据集具有相同的t_fixed
        self.t_mix = Registry.mapping['model_mapping']['setting'].param['t_mix']

    # 当输出某个实例化对象时，其调用的就是该对象的 __repr__() 方法，输出的是该方法的返回值
    def __repr__(self):
        return 'mixRLwTro Agent has no Network model'
        #return self.model.__repr__()

    def reset(self):
        """
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        """
        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'],
                                                                        in_only=True, average=None)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                                        in_only=True, average='all', negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_waiting_count"],
                                                                        in_only=True, negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_delay"],
                                                                        in_only=True, negative=False)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                                        targets=["cur_phase"], negative=False)
    def get_ob(self):
        """
        Get observation from environment.
        :param: None
        :return x_obs: observation generated by ob_generator
        """
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        """
        Get reward from environment.
        :param: None
        :return rewards: rewards
        generated by reward_generator
        """
        rewards = []
        rewards.append(self.reward_generator.generate())
        # squeeze从array中删除长度为1的轴
        # array: create an array
        rewards = np.squeeze(np.array(rewards)) * 12   #TODO baseagent*12 rlagent*12 ？
        return rewards

    def get_phase(self):
        """
        get_phase阶段
        Get current phase of intersection(s) from environment.

        :param: None
        :return phase: current phase generated by phase_generator
        """
        phase = []
        phase.append(self.phase_generator.generate())
        # phase = np.concatenate(phase, dtype=np.int8)
        # concatenate:沿着现有的轴连接数组序列
        # astype:数组的副本，转换为指定类型
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=True):
        """
        get_action
        Generate action.

        :param ob: observation
        :param phase: current phase
        :param test: boolean, decide whether is test process
        :return action: action in the next order
        """
        # phases: just index of self.inter_obj.phases, not green light index
        # 阶段:只是self.inter_obj.阶段的索引，而不是绿灯索引
        #
        assert self.inter_obj.current_phase == phase[-1]
        if self.inter_obj.current_phase_time < self.t_mix:
            return self.inter_obj.current_phase
        else:
            return (self.inter_obj.current_phase+1) % len(self.inter_obj.phases)

    def get_queue(self):
        """
        get_queue
        Get queue length of intersection.

        :param: None
        :return: total queue length
        """
        queue = []
        queue.append(self.queue.generate())
        queue = np.sum(np.squeeze(np.array(queue)))
        return queue

    def get_delay(self):
        """
        get_delay
        Get delay of intersection.

        :param: None
        :return: total delay
        """
        delay = []
        delay.append(self.delay.generate())
        delay = np.sum(np.squeeze(np.array(delay)))
        return delay