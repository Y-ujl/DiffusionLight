import gym
from collections import deque
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator

class param_(object):
    def __init__(self, world, rank, n_agent):
        super(param_, self).__init__()
        # base param
        self.gamma = None
        self.grad_clip = None
        self.epsilon_decay = None
        self.epsilon_min = None
        self.epsilon = None
        self.learning_rate = None
        self.vehicle_max = None
        self.batch_size = None
        self.tau = None
        self.buffer_size = None
        self.phase = None
        self.one_hot = None
        self.max_timesteps = None
        self.latent_dim = None

        # generator param
        self.ob_generator = None
        self.phase_generator = None
        self.pressure_generator = None
        self.action_space = None
        self.reward_generator = None
        self.queue = None
        self.delay = None

        # network param
        self.ob_length = None
        self.phase_dim = None
        self.phase_ob_length = None

        # Encoder param
        self.n_block = None
        self.n_embd = None
        self.n_head = None
        self.n_agent = n_agent  # agent 的数量

        # TSCGAN param
        self.gan_nembd = None
        self.gan_glr = None
        self.gan_dlr = None

        # world
        self.world = world
        self.rank = rank
        inter_id = world.intersection_ids[rank]
        self.inter = world.id2intersection[inter_id]

    def create(self):
        self.load_base_param()
        self.create_world_generator(self.world, self.rank, generator_type='unite')

    def load_base_param(self):
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        '''base params'''
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        # self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        # self.max_timesteps = Registry.mapping['model_mapping']['setting'].param['max_timesteps']
        self.latent_dim = Registry.mapping['model_mapping']['setting'].param['latent_dim']

        '''Encoder param'''
        self.ob_length = Registry.mapping['model_mapping']['setting'].param['ob_length']
        self.phase_dim = Registry.mapping['model_mapping']['setting'].param['phase_dim']
        self.n_block = Registry.mapping['model_mapping']['setting'].param['n_block']
        self.n_embd = Registry.mapping['model_mapping']['setting'].param['n_embd']
        self.n_head = Registry.mapping['model_mapping']['setting'].param['n_head']
        '''GAN param'''
        self.gan_nembd = Registry.mapping['model_mapping']['setting'].param['gan_nembd']
        self.gan_glr = Registry.mapping['model_mapping']['setting'].param['gan_glr']
        self.gan_dlr = Registry.mapping['model_mapping']['setting'].param['gan_dlr']

    def create_world_generator(self, world, rank, generator_type='disperse'):
        """
        type: unite : 创建联合generator 适合一次创建多个generator
              disperse : 分散创建generator 适合一次创建一次generator
        """
        if generator_type == 'disperse':
            self.ob_generator = LaneVehicleGenerator(world, self.inter, ['lane_count'], in_only=True, average=None)
            self.phase_generator = IntersectionPhaseGenerator(world, self.inter, ["phase"],
                                                              targets=["cur_phase"], negative=False)
            '''计算平均压力值作为GAN的评价指标'''
            # self.pressure_generator = LaneVehicleGenerator(world, self.inter, ["pressure"],
            #                                               average="all", negative=False)

            self.reward_generator = LaneVehicleGenerator(world,  self.inter, ["lane_waiting_count"],
                                                          in_only=True, average='all', negative=True)
            self.action_space = gym.spaces.Discrete(len(self.inter.phases))
        elif generator_type == 'unite':
            observation_generators = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ['lane_count'], in_only=True, average=None)
                observation_generators.append((node_idx, tmp_generator))
            sorted(observation_generators,
                   key=lambda x: x[0])  # now generator's order is according to its index in graph 现在生成器的顺序是根据其在图中的索引
            self.ob_generator = observation_generators

            #  get reward generator
            rewarding_generators = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ["pressure"], in_only=True, average='all',
                                                     negative=True)
                rewarding_generators.append((node_idx, tmp_generator))
            sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
            self.reward_generator = rewarding_generators

            #  get phase generator
            phasing_generators = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = IntersectionPhaseGenerator(world, node_obj, ['phase'], targets=['cur_phase'],
                                                           negative=False)
                phasing_generators.append((node_idx, tmp_generator))
            sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
            self.phase_generator = phasing_generators

            #  get queue generator
            queues = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ["lane_waiting_count"], in_only=True,
                                                     negative=False)
                queues.append((node_idx, tmp_generator))
            sorted(queues, key=lambda x: x[0])
            self.queue = queues

            #  get delay generator
            delays = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ["lane_delay"], in_only=True, average="all",
                                                     negative=False)
                delays.append((node_idx, tmp_generator))
            sorted(delays, key=lambda x: x[0])
            self.delay = delays
            self.action_space = gym.spaces.Discrete(len(self.inter.phases))

        if self.phase:
            if self.one_hot:
                self.phase_ob_length = self.ob_length + self.phase_dim
            else:
                self.phase_ob_length = self.ob_generator.ob_length + 1
        else:
            self.phase_ob_length = self.ob_generator.ob_length

    def reset_world_generator(self, world, rank, generator_type='disperse'):
        if generator_type =='disperse':
            inter_id = world.intersection_ids[rank]
            inter_obj = world.id2intersection[inter_id]
            self.inter = inter_obj
            self.ob_generator = LaneVehicleGenerator(world, inter_obj, ['lane_count'], in_only=True, average=None)
            self.phase_generator = IntersectionPhaseGenerator(world, inter_obj, ["phase"],
                                                              targets=["cur_phase"], negative=False)
            # pressure
            self.pressure_generator = LaneVehicleGenerator(world, inter_obj, ["pressure"],
                                                           average="all", negative=False)

            # self.reward_generator = LaneVehicleGenerator(world, inter_obj, ["lane_waiting_count"],
            #                                              in_only=True, average='all', negative=True)
        elif generator_type == 'unite' :
            observation_generators = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ['lane_count'], in_only=True, average=None)
                observation_generators.append((node_idx, tmp_generator))
            sorted(observation_generators,
                   key=lambda x: x[0])  # now generator's order is according to its index in graph
            self.ob_generator = observation_generators

            rewarding_generators = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ["pressure"], in_only=True, average='all',
                                                     negative=True)
                rewarding_generators.append((node_idx, tmp_generator))
            sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
            self.reward_generator = rewarding_generators

            phasing_generators = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = IntersectionPhaseGenerator(world, node_obj, ['phase'], targets=['cur_phase'],
                                                           negative=False)
                phasing_generators.append((node_idx, tmp_generator))
            sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
            self.phase_generator = phasing_generators

            queues = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ["lane_waiting_count"], in_only=True,
                                                     negative=False)
                queues.append((node_idx, tmp_generator))
            sorted(queues, key=lambda x: x[0])
            self.queue = queues

            delays = []
            for inter in world.intersections:
                node_id = inter.id
                node_idx = world.id2idx[node_id]
                node_obj = world.id2intersection[node_id]
                tmp_generator = LaneVehicleGenerator(world, node_obj, ["lane_delay"], in_only=True, average="all",
                                                     negative=False)
                delays.append((node_idx, tmp_generator))
            sorted(delays, key=lambda x: x[0])
            self.delay = delays

