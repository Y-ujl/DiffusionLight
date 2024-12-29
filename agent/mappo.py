from . import RLAgent
import torch
import torch.nn as nn
from common.registry import Registry

@Registry.register_model('mappo')
class MappoAgent(RLAgent):
    def __init__(self, world, rank):
        """
        根据输入的参数和来自 Registry 的配置信息来定义模型结构
        """
        super().__init__(world, world.intersection_ids[rank])
        # world parma
        self.world = world
        self.rank = rank
        self.sub_agents = len(self.world.intersections)
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter = self.world.id2intersection[self.inter_id]

class R_Actor(nn.Module):
    """
        Actor network class for MAPPO. Outputs actions given observations.
        :param args: (argparse.Namespace) arguments containing relevant model information.
        :param obs_space: (gym.Space) observation space.
        :param action_space: (gym.Space) action space.
        :param device: (torch.device) specifies the device to run on (cpu/gpu).
        """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)



