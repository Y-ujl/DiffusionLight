import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from utils.base_transformer import SelfAttention
from torch.nn import init


def load_dataset(file):
    # load_hdf5
    pass

class data_sampler:
    def __init__(self, data, reward_tune='no'):
        self.obs = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_obs = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()

        # self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()
        # TODO


def extract(a, t, x_shape):
    a = a[None, :]  # 原本a为(256,)
    b, *_ = t.shape  # b(10,)  *_(16)
    a = torch.repeat_interleave(a, repeats=b, dim=0)  # a(10, 5)
    out = a.gather(-1, t)  # gather :沿dim指定的轴聚集值
    # o = out.reshape(b, _[0], 1)
    # return out.reshape(b, (_, *((1,) * (len(x_shape) - 1))))    #(10,16,1)
    return out.reshape(b, _[0], 1)


def extractv2(a, t, x_shape):
    b, *_ = t.shape  # t(256,) b-256  *_:[]
    out = a.gather(-1, t)  # a(5,)  t(256,)
    # o = out.reshape(b, *((1,) * (len(x_shape) - 1)))  # (256,1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#
class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()

        return weighted_loss


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#
def get_betas(beta_schedule, timesteps):
    if beta_schedule == "linear":
        return linear_beta_schedule(timesteps)
    elif beta_schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    elif beta_schedule == "vp":
        return vp_beta_schedule(timesteps)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

# 正弦位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, n_agnet, dim):
        super().__init__()
        self.n_agent = n_agnet
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # math.log(10000)=9.210340371976184 : 默认以e为底
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -emb)  # torch.arange(half_dim, device=device) = [0,1,2,3,4,5,6,7]
        # x1 = x[:, :, None]          #x(10,16) x1 (10,16,1)
        # e = emb_rpt[None, :, :]     #e(1,16,8)
        # emb = x[:, :, None] * emb[None, :, :]
        # emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb[None, :]
        emb_rpt = torch.repeat_interleave(emb, repeats=self.n_agent, dim=0)  # TODO 代验证，这里相当于复制了16份位置编码
        emb_rpt = x[:, :, None] * emb_rpt[None, :, :]
        emb_rpt = torch.cat((emb_rpt.sin(), emb_rpt.cos()), dim=-1)  # (10,16,16)
        return emb_rpt

class SinusoidalPosEmbv2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = torch.device("cpu")
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        """
        x (256,) [[4,4,...4]...[]]
        x1 (256,1) = x[:, None] 相当于在数组中多加一个维度
        """
        emb = x[:, None] * emb[None, :]     # (256,8)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)    #(256,16)
        return emb

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 n_agents,
                 obs_dim,
                 action_dim,
                 device,
                 t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(n_agents, t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = obs_dim + action_dim + t_dim     #36
        # self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
        #                                nn.Mish(),
        #                                nn.Linear(256, 256),
        #                                nn.Mish(),
        #                                nn.Linear(256, 256),
        #                                nn.Mish())

        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.mid_linear1 = nn.Linear(input_dim, 256)
        self.attention_layer = MobileViTv2Attention(256)

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, obs):
        t = self.time_mlp(time)
        x = torch.cat([obs, x, t], dim=2)  # x(10,16,8) t(10,16,16) obs(10,16,12)
        x = self.mid_layer(x)           # x(10,16,36)

        x = self.final_layer(x)
        return x

class MLPv2(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 device,
                 t_dim=16,
                 id_dim=16):
        super(MLPv2, self).__init__()
        self.device = device

        self.id_mlp = nn.Sequential(
            SinusoidalPosEmbv2(id_dim),
            nn.Linear(id_dim, id_dim * 2),
            nn.Mish(),
            nn.Linear(id_dim * 2, id_dim),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbv2(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = obs_dim + action_dim + t_dim     #36
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.mid_linear1 = nn.Linear(input_dim, 256)
        self.attention_layer = MobileViTv2Attention(256)

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, obs):
        t = self.time_mlp(time)
        x = torch.cat([obs, x, t], dim=1)  # x(64,8) t(64，) obs(64,12)
        x = self.mid_layer(x)           # x(10,16,36)

        x = self.final_layer(x)
        return x

class shared_MLP(nn.Module):
    def __init__(self, n_agents,):
        super(shared_MLP, self).__init__()
        # 将obs[bacth, n_agent, obs_dim] --> obs[batch, obs_dim, n_agent]
        input_dim = n_agents
        self.frond_linear = nn.Linear(input_dim, 64)     #n_agent -->64
        self.mid_linear = nn.Linear(64, 64)
        self.final_layer = nn.Linear(64, 1)               # n_agent -->64

    def forward(self, obs):
        """ obs 为全局观测 """
        obs = obs.transpose(1, 2)
        obs = F.relu(self.frond_linear(obs))
        obs = F.relu(self.mid_linear(obs))
        shared_obs = F.relu(self.final_layer(obs)).transpose(1, 2)

        return shared_obs   #[batch, 1, obs_dim]

class noise_MLP(nn.Module):
    def __init__(self, n_agents, obs_dim):
        super(noise_MLP, self).__init__()
        """ 对id进行编码 """
        agent_dim = n_agents

        self.share_agent = nn.Sequential(nn.Linear(agent_dim, 64),
                                         nn.Mish(),
                                         nn.Linear(64, 64),
                                         nn.Mish(),
                                         nn.Linear(64, 1),)

        self.share_obs = nn.Sequential(nn.Linear(obs_dim, 64),
                                         nn.Mish(),
                                         nn.Linear(64, 64),
                                         nn.Mish(),
                                         nn.Linear(64, 8),)

    def forward(self, obs):
        x = obs.transpose(1, 2)  #(1,16,12)-->(1,12,16)
        x = self.share_agent(x).squeeze(-1)     #(1,12)
        x = self.share_obs(x)   #(1,8)

        return x

# ---------------------------------- respace ----------------------------------#
# -----------------------------------------------------------------------------#
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]
    """
    eg: num_timesteps = 100     section_counts = "10, 15, 20"
    相当与将100 分成份33,33,33,第一个33steps缩短为10steps,第二个33steps压缩为15steps,第三个33steps压缩为20steps
    """
    size_per = num_timesteps // len(section_counts)  # 将num_timesteps 均分为 3 份
    extra = num_timesteps % len(section_counts)  # 未整除的部分
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):  # i=0,1,2  section_count=10,15,20
        size = size_per + (1 if i < extra else 0)  # 这里会将extra加在前i项上 eg：33,33,33--> 34,33,33
        if size < section_count:  # 只能进行
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1  # 重新调整的步长
        else:
            frac_stride = (size - 1) / (section_count - 1)  # (34-1)/(10-1)=round(3.6666) = 4
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):  # i=0 section_count=10
            taken_steps.append(start_idx + round(cur_idx))  # round():数字的四舍五入
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size  # size =34,33,33
    return set(all_steps)  # set() 无序不重复元素集，可进行关系测试，删除重复数据


# ---------------------------------Shapley--------------------------------------------#
""" 采样 联盟 """
def sample_grandcoalitions(n_agent, batch_size, sample_size):
    """
    seq_set shape = (n,n)    n = n_agent
    [[1,0,0...0]
     [1,1,0,...0]
     [1,1,1....1]]
    """
    seq_set = torch.tril(torch.ones(n_agent, n_agent), diagonal=0, out=None)  # 下三角矩阵
    """
    grand_coalitions_pos : 值是随机的multinomial()  # shape = (b*n_s, n)  eg(256*64, 16)
    """
    grand_coalitions_pos = torch.multinomial(torch.ones(batch_size * sample_size, n_agent) / n_agent, n_agent, replacement=False)
    """
    tensor.scatter_(dim, index, src)
    将张量src中的所有值写入索引张量中指定索引处的self中。
    对于src中的每个值，对于dimension != dim,其输出索引由其在src中的索引指定
                    对于 dimension = dim,索引中对应的值，
    """
    individual_map = torch.zeros(batch_size*sample_size*n_agent, n_agent)                           #(256*64*16,16)
    individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)                    #(256*64*16,16)
    individual_map = individual_map.contiguous().view(batch_size, sample_size, n_agent, n_agent)    #(256,64,16,16)
    # 子联盟
    subcoalition_map = torch.matmul(individual_map, seq_set)    #(256,64,16,16)

    # 从torche grand_coalitions_pos构造torche大联盟(按agent_idx顺序) (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
    offset = (torch.arange(batch_size * sample_size) * n_agent).reshape(-1, 1)  #(256*64)*16 -- (16384)        .reshape(-1, 1) 从[16384]--[16384,1]
    grand_coalitions_pos_alter = grand_coalitions_pos + offset                  #(16384,16)
    grand_coalitions = torch.zeros_like(grand_coalitions_pos_alter.flatten())   #(16384*16) = (262144) grand_coalitions_pos -- agent_idx
    grand_coalitions[grand_coalitions_pos_alter.flatten()] = torch.arange(batch_size * sample_size * n_agent)   #
    grand_coalitions = grand_coalitions.reshape(batch_size * sample_size, n_agent) - offset     #(16384,16)

    grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size * sample_size, n_agent, n_agent).contiguous().view(batch_size,
                                                                                                sample_size,
                                                                                                n_agent,
                                                                                                n_agent)  # shape = (b, n_s, n, n)

    return subcoalition_map, grand_coalitions

""" Separable self-attention """
class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

        self.fc_o = nn.Linear(d_model, d_model) #输出

        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)    #(bs,nq,1)
        weight_i = torch.softmax(i, dim=2)  #bs,nq,1  归一
        context_score = weight_i * self.fc_k(input)     #bs,nq,d_model

        context_vector = torch.sum(context_score, dim=2, keepdim=True)  #bs,1,d_model
        v = self.fc_v(input) * context_vector   #bs,nq,d_model
        out = F.relu(self.fc_o(v))  #bs,nq,d_model

        return out

"""
    input=torch.randn(50,49,512)
    ea = ExternalAttention(d_model=512,S=8)
    output=ea(input)
"""
class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.k = nn.Linear(d_model, S, bias=False)
        self.v = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.k(queries)      #bs,n,S
        attn = self.softmax(attn)   #bs,n,S
        attn = attn/torch.sum(attn, dim=2, keepdim=True)     #bs,n,S
        out = self.v(attn)          #bs,n,d_model

        return out

"""
    input=torch.randn(b,n,s)
    eg. resatt = ResidualAttention(channel=512,num_class=1000,la=0.2)
"""
class ResidualAttention(nn.Module):

    def __init__(self, input_dim, hid, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Linear(input_dim, hid, bias=False)

    def forward(self, x):
        b, n, s = x.shape
        y_raw = self.fc(x)                      #b, n, s
        y_avg = torch.mean(y_raw, dim=2)        #b, n, s
        y_max = torch.max(y_raw, dim=2)[0]      #b,num_class
        score = y_avg + self.la * y_max
        return score


# -----------------------------Prioritized Sampling------------------------------
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)   #子节点
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
