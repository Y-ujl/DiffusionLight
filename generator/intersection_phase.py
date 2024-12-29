# from .base import BaseGenerator

class IntersectionPhaseGenerator():
    '''
    Generate state or reward based on statistics of intersection phases.
    根据交叉口阶段的统计数据生成状态或奖励
    :param world: World object      世界的对象
    :param I: Intersection object   十字路口对象
    :param fns: list of statistics to get, "phase" is needed for result "cur_phase"
                要获取的统计信息列表，结果"cur_phase"需要"phase"
    :param targets: list of results to return, currently support "cur_phase": current phase of the intersection (not before yellow phase)
                    返回的结果列表，目前支持"cur_phase":交集的当前相位(不是在黄色相位之前)
             See section 4.2 of the intelliLight paper[Hua Wei et al, KDD'18] for more detailed description on these targets.
             有关这些目标的更详细描述，请参见intelligence论文[Hua Wei et al, KDD'18]的4.2节
    :param negative: boolean, whether return negative values (mostly for Reward)
                    布尔值，是否返回负值(主要用于Reward)
    :param time_interval: use to calculate  用于计算
    '''

    def __init__(self, world, I, fns=("phase"), targets=("cur_phase"), negative=False):
        self.world = world
        self.I = I

        # get cur phase of the intersection
        self.phase = I.current_phase

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns
        self.targets = targets

        self.negative = negative


    def generate(self):
        '''
        generate
        Generate current phase based on current simulation state.
        
        :param: None
        :return ret: result based on current phase
        '''
        ret = [self.I.current_phase]

        if self.negative:
            ret = ret * (-1)

        return ret


if __name__ == "__main__":
    from world.world_cityflow import World

    world = World("examples/configs.json", thread_num=1)
    laneVehicle = IntersectionPhaseGenerator(world, world.intersections[0],
                                               ["vehicle_trajectory", "lane_vehicles", "vehicle_distance"],
                                               ["passed_time_count", "passed_count", "vehicle_map"])
    for _ in range(1, 301):
        world.step([_ % 3])
        ret = laneVehicle.generate()

        if _ % 10 != 0:
            continue
        print(ret)


