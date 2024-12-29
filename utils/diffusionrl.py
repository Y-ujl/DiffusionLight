import numpy as np
import torch

from agent.utils import *
from utils.diffusion_utils import *
from utils.diffusion_utils import WeightedLoss, space_timesteps, sample_grandcoalitions
from utils.diffusion_utils import get_betas


class Dqn(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim):
        super(Dqn, self).__init__()
        """ yjl 5-22"""
        # self.attention_layer = SelfAttention(input_dim, n_head=1, n_agent=n_agent, masked=False)     #TODO
        self.attention_layer = MobileViTv2Attention(input_dim)  # TODO

        self.q1_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        # x = torch.cat([obs, action], dim=-1).flatten(start_dim=1)    #obs(10,16,12) action(10,16,1) -- x(10,16,13) - (10, 13*16) 联合flatten 那么在sample_action中， 不能产生唯一的q_value
        x = torch.cat([obs, action], dim=-1)  # 如果flatten 难以训练 如果不flatten
        return torch.squeeze(self.q1_model(x)), torch.squeeze(self.q2_model(x))

    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        min_q = torch.squeeze(torch.min(q1, q2))  # 这里输出的是原先的min q值
        # clip
        return min_q


class brother_dqn(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim):
        super(brother_dqn, self).__init__()
        self.hidden_dim = hidden_dim
        self.q1_linear1 = nn.Linear(input_dim, hidden_dim)
        self.q1_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.q1_linear1(x))
        x = F.relu(self.q1_linear2(x))
        x = self.q1_linear3(x).squeeze()

        return x


class brother_atten_dqn(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim):
        super(brother_atten_dqn, self).__init__()
        self.hidden_dim = hidden_dim
        self.q1_linear1 = nn.Linear(input_dim, hidden_dim)
        self.q1_atten = MobileViTv2Attention(hidden_dim)
        self.q_atten = SelfAttention(hidden_dim, n_head=1, n_agent=n_agent, masked=True)
        self.q1_linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.q1_linear1(x))
        x = self.q_atten(x)
        x = self.q1_linear2(x).squeeze()

        return x


class Drqn(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim):
        super(Drqn, self).__init__()
        self.hidden_dim = hidden_dim
        self.q1_linear1 = nn.Linear(input_dim, hidden_dim)
        self.q1_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.q1_linear2 = nn.Linear(hidden_dim, 1)

        self.q2_linear1 = nn.Linear(input_dim, hidden_dim)
        self.q2_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.q2_linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action, h, c):
        # x = torch.cat([obs, action], dim=-1).flatten(start_dim=1)    #obs(10,16,12) action(10,16,1) -- x(10,16,13) - (10, 13*16) 联合flatten 那么在sample_action中， 不能产生唯一的q_value
        x = torch.cat([obs, action], dim=-1)  # 如果flatten 难以训练 如果不flatten
        x1 = F.relu(self.q1_linear1(x))
        x1, (new_h, new_c) = self.q1_lstm(x1, (h, c))
        x1 = self.q1_linear2(x1).squeeze()

        x2 = F.relu(self.q2_linear1(x))
        x2, (new_h, new_c) = self.q2_lstm(x2, (h, c))
        x2 = self.q2_linear2(x2).squeeze()

        return x1, x2

    def q_min(self, obs, action, h, c):
        q1, q2 = self.forward(obs, action, h, c)
        min_q = torch.squeeze(torch.min(q1, q2))  # 这里输出的是原先的min q值
        # clip
        return min_q

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_dim]), torch.zeros([1, batch_size, self.hidden_dim])
        else:
            return torch.zeros([1, 1, self.hidden_dim]), torch.zeros([1, 1, self.hidden_dim])


class attention_dqn(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim):
        super(attention_dqn, self).__init__()
        self.hidden_dim = hidden_dim
        self.q1_linear1 = nn.Linear(input_dim, hidden_dim)
        self.q1_atten = MobileViTv2Attention(hidden_dim)
        self.q1_linear2 = nn.Linear(hidden_dim, 1)

        self.q2_linear1 = nn.Linear(input_dim, hidden_dim)
        self.q2_atten = MobileViTv2Attention(hidden_dim)
        self.q2_linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x1 = F.relu(self.q1_linear1(x))
        x1 = self.q1_atten(x1)
        x1 = self.q1_linear2(x1).squeeze()

        x2 = F.relu(self.q2_linear1(x))
        x2 = self.q1_atten(x2)
        x2 = self.q2_linear2(x2).squeeze()

        return x1, x2

    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        min_q = torch.squeeze(torch.min(q1, q2))  # 这里输出的是原先的min q值
        # clip
        return min_q


class Dueling_Dqn(nn.Module):
    def __init__(self, n_agent, input_dim, hidden_dim):
        super(Dueling_Dqn, self).__init__()

        self.q1_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.fc1_val = nn.Linear(hidden_dim, 1)
        self.fc1_adv = nn.Linear(hidden_dim, 8)

        self.q2_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim))

        self.fc2_val = nn.Linear(hidden_dim, 1)
        self.fc2_adv = nn.Linear(hidden_dim, 8)

        self.relu = nn.ReLU()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x1 = self.relu(self.q1_model(x))
        adv1 = self.fc1_adv(x1)
        val1 = self.fc1_val(x1).expand(-1, -1, 8)
        q1 = (val1 + adv1 - adv1.mean(2).unsqueeze(2).expand(-1, -1, 8)).mean(2)

        x2 = F.relu(self.q2_model(x))
        adv2 = self.fc2_adv(x2)
        val2 = self.fc2_val(x2)
        q2 = (val2 + adv2 - adv2.mean(2).unsqueeze(2).expand(-1, -1, 8)).mean(2)

        return q1, q2  # (17,16)

    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        min_q = torch.squeeze(torch.min(q1, q2))  # 这里输出的是原先的min q值
        # clip
        return min_q


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),  # 一个自正则化的非单调神经激活函数
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Shapley_Q(nn.Module):
    def __init__(self, n_agent, state_dim, action_dim, sample_size, hidden_dim=256):
        super(Shapley_Q, self).__init__()
        self.n_agent = n_agent
        self.sample_size = sample_size
        self.obs_dim = state_dim
        self.action_dim = action_dim
        input_dim = (state_dim + action_dim) * n_agent
        self.q1_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Mish(),  # 一个自正则化的非单调神经激活函数
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        batch_size = state.size(0)
        # shape = (b, n_s, n, n)
        subcoalition_map, grand_coalitions = sample_grandcoalitions(self.n_agent, batch_size, self.sample_size)
        # shape = (b, n_s, n, n, a)
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_agent,
                                                                 self.n_agent,
                                                                 self.action_dim)
        # action(batch_size, n_agent, act_dim)(256,16,8)
        # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        act = action.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_agent, self.n_agent,
                                                      self.action_dim).gather(3, grand_coalitions)
        # shape = (b, n_s, n, n, 1)
        act_map = subcoalition_map.unsqueeze(-1).float()
        act = act * act_map
        # shape = (b, n_s, n, n*a)  a为act_dim
        act = act.contiguous().view(batch_size, self.sample_size, self.n_agent, -1)

        # shape = (b, n, o) -> (b, 1, n, o) -> (b, 1, 1, n, o) -> (b, n_s, n, n, o)
        obs = state.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_agent, self.n_agent,
                                                     self.obs_dim)
        # shape = (b, n_s, n, n, o) -> (b, n_s, n, n*o)    #
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_agent, self.n_agent * self.obs_dim)
        x = torch.cat((obs, act), dim=-1)
        shapley_q1 = []
        for i in range(self.n_agent):
            q = self.q1_model(x[:, :, i, :])
            shapley_q1.append(q)
        shapley_q1 = torch.squeeze(torch.stack(shapley_q1, dim=2).mean(dim=1))
        shapley_q2 = shapley_q1

        return shapley_q1, shapley_q2  # 输出为shapley value(边际贡献值)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.max_action = 1
        # input_dim = input_dim + id_dim
        input_dim = input_dim
        self.dense_1 = nn.Linear(input_dim, 64)
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, output_dim)

    def forward(self, obs):
        action_pro, log_action_pro = self.sample(obs)
        return action_pro, log_action_pro

    def sample(self, obs):
        """
        param : noise (64, 8) 为全局信息得到的噪声
        param : obs (64, 12)
        """
        # x = torch.cat([obs, noise], dim=1)
        x = F.relu(self.dense_1(obs))
        x = F.relu(self.dense_2(x))
        noise = self.dense_3(x)

        # n = torch.randn_like(action) * 1e-8
        # log_action_probs = torch.log(torch.clamp(action + n, 1e-8, 100)).mean(dim=-1, keepdim=True)
        return noise

class Actors(nn.Module):
    def __init__(self, n_agents, input_dim, output_dim):
        super(Actors, self).__init__()
        self.n_agents = n_agents
        self.agent = Actor(input_dim, output_dim)

    def forward(self, obs):
        n_batch = obs.shape[0]
        obs = obs.reshape(-1, obs.shape[-1])
        y = self.agent.forward(obs)
        y = y.reshape(n_batch, self.n_agents, -1)

        return y


def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Diffusion(nn.Module):
    def __init__(self,
                 *,  # 留给 子类SpacedDiffusion 的use_timesteps
                 n_agent,  # 剩下的都为**kwargs
                 obs_dim,
                 action_dim,
                 model,
                 betas_s,
                 n_timesteps=35,
                 max_action=1,
                 eta=0.0,
                 loss_type='l2',
                 sample_type='DDPM',
                 timestep_respacing="",
                 clip_denoised=True,
                 predict_epsilon=True,
                 rescale_timesteps=False):
        super(Diffusion, self).__init__()
        """
        param ： n_agent             执行的个数
        param ： obs_dim             观测维度
        param ： action_dim          执行动作维度
        param :  model               MLP 用于预测值
        param ： n_timesteps         Denoise的次数
        param ： beta_schedule       产生beta的模式        
        param ： loss_type           loss的类型
        param ： sample_type         DDPM/DDIM  选择sample方式
        param ： timestep_respacing  加速间隔设置
        param ： clip_denoised       截断
        param ： predict_epsilon     预测eps值
        param :  rescale_timesteps  是否使用respace加速算法
        """
        self.n_agents = n_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model = model
        self.sample_type = sample_type

        """ yjl 6.11 局部观测的actor """
        # self.forward_actor = {}
        # for i in range(n_agent):
        #     self.forward_actor[i] = Actor(obs_dim, action_dim)

        self.joint_actor = Actor(obs_dim, action_dim)
        # self.joint_actor = actor

        # 定义随机加速 self.n_timesteps 的随机范围
        self.max_nsteps = int(n_timesteps)
        self.max_action = max_action

        self.eta = eta
        self.betas_s = betas_s
        self.n_timesteps = int(n_timesteps)  # 输入的n_timesteps
        self.timestep_respacing = timestep_respacing
        self.rescale_timesteps = rescale_timesteps
        """
        eg :  n_timesteps = 5
        betas = tensor([0.8041, 0.5412, 0.3642, 0.2451, 0.1650])
        """
        if isinstance(betas_s, str):
            betas = get_betas(beta_schedule=self.betas_s, timesteps=n_timesteps)

        alphas = 1. - betas  # α
        # self.n_timesteps = int(betas.shape[0])
        """
        cumprod : 返回维度dim中输入元素的累积乘积
        yi = x1 * x2 * ... * xi
        alphas_cumprod = tensor([0.8041, 0.4352, 0.1585, 0.0389, 0.0064])
        """
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        """
        alphas_cumprod[:-1] : tensor([0.8041, 0.4352, 0.1585, 0.0389]) 没有最后一项
        alphas_cumprod_prev = tensor([1.0000, 0.8041, 0.4352, 0.1585, 0.0389])
        """
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.zeros(1)])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.alphas_cumprod_next = alphas_cumprod_next

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = posterior_variance

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        """
        torch.clamp : 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
                | min, if x_i < min
        y_i =   | x_i, if min <= x_i <= max
                | max, if x_i > max
        """
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        self.loss_fn = Losses[loss_type]()

    def random_spaced(self, e, episodes, max_nsteps, fun='sample'):
        if fun == 'sample':
            """ yjl 5.28 随episode 减小t策略， 最开始在40， 截至在10"""
            # range_t = self.sample_range_with_e(e, episodes, max_nsteps)
            # self.n_timesteps = int(np.random.randint(range_t[0], range_t[1], size=1))
            # self.n_timesteps = int(np.random.randint(35, max_nsteps, size=1))                 #TODO 20-40之间难收敛 -- 30-40
            self.n_timesteps = max_nsteps
            betas = get_betas(beta_schedule=self.betas_s, timesteps=self.n_timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)

            alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
            alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.zeros(1)])

            self.alphas_cumprod = alphas_cumprod
            self.alphas_cumprod_prev = alphas_cumprod_prev
            self.alphas_cumprod_next = alphas_cumprod_next

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            self.posterior_variance = posterior_variance

            self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
            self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
            self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        elif fun == 'train':
            self.n_timesteps = max_nsteps

    """ 随着episode 逐步下降t的取值范围"""

    def sample_range_with_e(self, e, episodes, max_nsteps):
        range_t = np.zeros(2)
        min_nsteps = max_nsteps / 4  # 40 / 4 = 10
        delta = max_nsteps / 8  # 40 / 8 = 5
        stop_e = episodes * 0.75  # 200 * 0.75 = 150
        start_e = episodes * 0.15  # 200 * 0.15 = 30
        k = (max_nsteps - min_nsteps) / (start_e - stop_e)
        b = max_nsteps - start_e * k
        if e > start_e:  # 前三十论不下降
            if e < stop_e:
                range_t[1] = k * e + b  # eg 40
                range_t[0] = range_t[1] - delta  # eg 40 -5
            else:
                range_t[1] = k * stop_e + b  # -0.25 * 150 + 40 = 10
            return range_t
        else:
            range_t[1] = max_nsteps
            range_t[0] = range_t[1] - delta

        return range_t

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise):  # _predict_xstart_from_eps()
        '''
            if self.predict_epsilon(ε), model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:  # True
            return (
                # extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)()   x_t (10,16,8)
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_start_from_prev(self, x_t, t, prev):
        return (  # (xprev - coef2 *x_t) / coef1
                extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * prev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape)
                * x_t
        )

    def pred_eps_from_xstart(self, x_t, t, pred_x_start):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_x_start) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # def _scale_timesteps(self, t):
    #     if self.rescale_timesteps:
    #         return t.float() * (1000.0 / self.num_timesteps)
    #     return t

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, s):
        # noise = self.model(x, t, s) x t s 需要有相同的维度
        noise = model(x, t, s)  # model 反向更新时预测的noise--> noise -- model_output
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)  # pred_xstart 预测的初始值
        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)  # TODO no have max_action
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return x_recon, model_mean, posterior_variance, posterior_log_variance

    # ——————————————————————————————————————————DDPM sampling———————————————————————————————————
    # DDPM的单个的采样过程  ddpm和ddim
    # 共用：p_mean_variance
    def p_sample(self, model, x, t, s, n):  # x为动作  t timestep  s obs
        b, *_, device = *x.shape, x.device
        pred_start, model_mean, _, model_log_variance = \
            self.p_mean_variance(model,
                                 x=x,
                                 t=t,
                                 s=s)  # 获得均值和方差
        noise = n + torch.randn_like(x) * 1e-4  # 每个扩散步，一个固定的噪声+随机波动
        # 当t = 0 mask=0 提升采样
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((_.shape[1], 1) * (len(x.shape) - 2)))
        # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def get_part_action_pro(self, batch, n_agent, obs):
        a = []
        """ 局部信息生成动作latent空间 """
        # 这里添加上id
        for i in range(n_agent):  # (batch, agent_i, obs_dim)
            # a_i = self.forward_actor[i](obs[:, i, :])
            # id = torch.full((batch,), i,dtype=torch.long)
            # a_i = self.joint_actor(obs[:, i, :], id)
            a_i = self.joint_actor(obs[:, i, :])
            a.append(a_i)
        a = torch.stack(a, dim=1)  # 局部动作空间
        return a.clone().detach()  # (1,12,8)

    # DDPM循环采样过程
    def p_sample_loop(self, model, obs, shape, verbose=False, return_diffusion=False):
        """
        param : obs 观测 (10, 16, 12)
        param : shape (10, 16, 8)
        """
        device = torch.device("cpu")

        batch_size = shape[0]
        #x = torch.randn(shape, device=device)  # x:(10,16,8) 8为action_dim
        x = self.get_part_action_pro(batch_size, self.n_agents, obs)
        # x = self.joint_actor(obs)
        # x = x[:, None, :]
        # x = torch.repeat_interleave(x, repeats=self.n_agents, dim=1)
        # x = noise
        n = torch.randn(shape)
        if return_diffusion:
            diffusion = [x]
        # reversed 逆转 从4-->0
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, self.n_agents,), i, device=device,
                                   dtype=torch.long)  # timesteps(10,16,)
            x = self.p_sample(model, x, timesteps, obs, n)  # x(17,16,8) timesteps(17,16) obs(17,16,12)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def sample(self, obs):
        """
        param : obs (10,16,12)
        """
        batch_size = obs.shape[0]
        shape = (batch_size, self.n_agents, self.action_dim)  # shape (10,16,8)
        action = self.p_sample_loop(self.model, obs, shape)
        noise = torch.randn_like(action) * 1e-8
        """ 
        yjl 6.2
        log_action_probs :Logs of action probabilities, used for entropy.
        """
        log_action_probs = torch.log(torch.clamp(action + noise, 1e-8, 2)).mean(dim=-1, keepdim=True)
        # return action.clamp_(-self.max_action, self.max_action), torch.squeeze(log_action_probs)
        return action.clamp_(-self.max_action, self.max_action), log_action_probs

    # ——————————————————————————————————————————DDMP sampling———————————————————————————————————

    # ——————————————————————————————————————————DDIP sampling———————————————————————————————————

    """ yjl 5.18 加入DDIM 加速模块"""

    # 单次DDIM sampling
    def i_sample(self, model, x, t, s):
        b, *_, device = *x.shape, x.device
        pred_start, model_mean, _, model_log_variance = \
            self.p_mean_variance(model,
                                 x=x,
                                 t=t,
                                 s=s)  # 获得均值和方差

        eps = self.pred_eps_from_xstart(x, t, pred_start)
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (  # 论文公式中的超参，当self.eta = 1 DDIM与DDPM sample一致
                self.eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
                pred_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((_.shape[1], 1) * (len(x.shape) - 2)))

        # mean_pred:(17,16,8)  nonzero_mask:(17,16,1) sigma:(17,16,1) noise:(17,16,8)
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample

    def i_sample_loop(self, model, obs, shape, steps=100, return_diffusion=False):
        """
        param : obs 观测 (10, 16, 12)
        param : shape (10, 16, 8)
        """
        device = torch.device("cpu")

        batch_size = shape[0]
        x = torch.randn(shape, device=device)  # x:(10,16,8) 8为action_dim

        if return_diffusion: diffusion = [x]
        # reversed 逆转 从4-->0
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, self.n_agents,), i, device=device,
                                   dtype=torch.long)  # timesteps(10,16,)

            x = self.i_sample(model, x, timesteps, obs)  # x(17,16,8) timesteps(17,16) obs(17,16,12)

            if return_diffusion: diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def ddim_sample(self, obs):
        """
        param : obs (10,16,12)
        """
        batch_size = obs.shape[0]
        shape = (batch_size, self.n_agents, self.action_dim)  # shape (10,16,8)
        action = self.i_sample_loop(self.model, obs, shape)  # TODO
        return action.clamp_(-self.max_action, self.max_action)

    # ——————————————————————————————————————————DDIP sampling———————————————————————————————————

    # ------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            # self.sqrt_alphas_cumprod(5)  t(256,)  x_start.shape:(256,16)
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    # Loss的更新
    def p_losses(self, model, x_start, obs, t, weights=1.0):
        """ 这里的x_start是采样的action_pro """
        """ 这里更改的思路是将noise空间替换成latent空间 """
        # noise = torch.randn_like(x_start)
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # x_start(256,16,8) t(256,16) noise(256,16,8)
        batch = x_start.shape[0]
        noise = self.get_part_action_pro(batch, self.n_agents, obs)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = model(x_noisy, t, obs)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    # x:actions
    def loss(self, x, obs, weights=1.0):
        batch_size = len(x)  # 这里self.n_timesteps 不应该改变
        t = torch.randint(0, self.n_timesteps, (batch_size, self.n_agents,), device=x.device).long()
        return self.p_losses(self.model, x, obs, t, weights)

    def forward(self, obs):
        if self.sample_type == 'DDPM':
            action_pro, log_action_pro = self.sample(obs)
            return action_pro, log_action_pro
        elif self.sample_type == 'DDIM':
            return self.ddim_sample(obs)

class Dis_Diffusion(nn.Module):
    def __init__(self,
                 *,  # 留给 子类SpacedDiffusion 的use_timesteps
                 n_agent,  # 剩下的都为**kwargs
                 obs_dim,
                 action_dim,
                 model,
                 actor,
                 betas_s,
                 batch_size,
                 n_timesteps=35,
                 max_action=1,
                 eta=0.0,
                 loss_type='l2',
                 clip_denoised=True,
                 predict_epsilon=True,):
        super(Dis_Diffusion, self).__init__()
        """
        param ： n_agent             执行的个数
        param ： obs_dim             观测维度
        param ： action_dim          执行动作维度
        param :  model               MLP 用于预测值
        param ： n_timesteps         Denoise的次数
        param ： beta_schedule       产生beta的模式        
        param ： loss_type           loss的类型
        param ： sample_type         DDPM/DDIM  选择sample方式
        param ： timestep_respacing  加速间隔设置
        param ： clip_denoised       截断
        param ： predict_epsilon     预测eps值
        param :  rescale_timesteps  是否使用respace加速算法
        """
        self.n_agents = n_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.model = model  # MLP
        self.joint_model = actor

        # 定义随机加速 self.n_timesteps 的随机范围
        self.max_nsteps = int(n_timesteps)
        self.max_action = max_action

        self.eta = eta
        self.betas_s = betas_s
        self.n_timesteps = int(n_timesteps)  # 输入的n_timesteps
        """
        eg :  n_timesteps = 5
        betas = tensor([0.8041, 0.5412, 0.3642, 0.2451, 0.1650])
        """
        if isinstance(betas_s, str):
            betas = get_betas(beta_schedule=self.betas_s, timesteps=n_timesteps)

        alphas = 1. - betas  # α
        # self.n_timesteps = int(betas.shape[0])
        """
        cumprod : 返回维度dim中输入元素的累积乘积
        yi = x1 * x2 * ... * xi
        alphas_cumprod = tensor([0.8041, 0.4352, 0.1585, 0.0389, 0.0064])
        """
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        """
        alphas_cumprod[:-1] : tensor([0.8041, 0.4352, 0.1585, 0.0389]) 没有最后一项
        alphas_cumprod_prev = tensor([1.0000, 0.8041, 0.4352, 0.1585, 0.0389])
        """
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.zeros(1)])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.alphas_cumprod_next = alphas_cumprod_next

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = posterior_variance

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise):  # _predict_xstart_from_eps()
        '''
            if self.predict_epsilon(ε), model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:  # True
            return (
                # extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)()   x_t (10,16,8)
                    extractv2(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extractv2(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_start_from_prev(self, x_t, t, prev):
        return (  # (xprev - coef2 *x_t) / coef1
                extractv2(1.0 / self.posterior_mean_coef1, t, x_t.shape) * prev -
                extractv2(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape)
                * x_t
        )

    def pred_eps_from_xstart(self, x_t, t, pred_x_start):
        return (
                (extractv2(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_x_start) /
                extractv2(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extractv2(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extractv2(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extractv2(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extractv2(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, s):
        noise = model(x, t, s)  # model 反向更新时预测的noise--> noise -- model_output
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)  # pred_xstart 预测的初始值
        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)  # TODO no have max_action
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return x_recon, model_mean, posterior_variance, posterior_log_variance

    # ——————————————————————————————————————————DDPM sampling———————————————————————————————————
    # DDPM的单个的采样过程  ddpm和ddim
    # 共用：p_mean_variance
    def p_sample(self, model, x, t, s, n):  # x为动作  t timestep  s obs
        b, *_, device = *x.shape, x.device
        pred_start, model_mean, _, model_log_variance = \
            self.p_mean_variance(model,
                                 x=x,
                                 t=t,
                                 s=s)  # 获得均值和方差
        noise = n + torch.randn_like(x) * 1e-4  # 每个扩散步，一个固定的噪声+随机波动
        # TODO
        # noise = torch.randn_like(x)  # 每个扩散步，随机波动
        # 当t = 0 mask=0 提升采样
        # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((_.shape[1], 1) * (len(x.shape) - 2)))
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # (64,8) + (64,) * (0.5 * )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # DDPM循环采样过程
    def p_sample_loop(self, model, obs, noise, shape, return_diffusion=False):
        """
        param : obs 观测 (64, 12)
        param : shape (64, 8)
        """
        device = torch.device("cpu")
        x = noise

        batch_size = shape[0]
        # x = self.get_part_action_pro(batch_size, self.n_agents, obs)
        # 先不使用
        n = torch.randn(shape)
        if return_diffusion:
            diffusion = [x]
        # reversed 逆转 从4-->0
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)  # timesteps(10,16,)
            x = self.p_sample(model, x, timesteps, obs, n)  # x(17,16,8) timesteps(17,16) obs(17,16,12)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def one_hot(self, batch, n_agent, idi):
        result = torch.zeros((batch,))
        result[:, idi] = 1.0
        return result

    def get_part_action_pro(self, batch, n_agent, obs):
        a = []
        """ 局部信息生成动作latent空间 """
        # 这里添加上id
        for i in range(n_agent):  # (batch, agent_i, obs_dim)
            # a_i = self.forward_actor[i](obs[:, i, :])
            # id = torch.full((batch,), i,dtype=torch.long)
            # a_i = self.joint_actor(obs[:, i, :], id)
            a_i = self.joint_actor(obs[:, i, :])
            a.append(a_i)
        a = torch.stack(a, dim=1)  # 局部动作空间
        return a.clone().detach()  # (1,12,8)

    def sample(self, obs, noise):
        """
        param : noise (64, 8) 为全局信息得到的噪声
        param : obs (64, 12)
        """
        shape = (obs.shape[0], self.action_dim)  # shape (64, 8)
        action = self.p_sample_loop(self.model, obs, noise, shape)      # noise(1,16,12)
        n = torch.randn_like(action) * 1e-8
        """ 
        yjl 6.2
        log_action_probs :Logs of action probabilities, used for entropy.
        """
        log_action_probs = torch.log(torch.clamp(action + n, 1e-8, 100)).mean(dim=-1, keepdim=True)
        return action.clamp_(-self.max_action, self.max_action), log_action_probs

    # ------------------------------------------ training ------------------------------------------#
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            # self.sqrt_alphas_cumprod(5)  t(256,)  x_start.shape:(256,16)
                extractv2(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extractv2(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    # 前向传播 ：Loss的更新
    def p_losses(self, model, x_start, noise, obs, t, weights=1.0):
        """ 这里的x_start是采样的action_pro """
        """ 这里更改的思路是将noise空间替换成latent空间 """
        # noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      #x_start(64,8)
        x_recon = model(x_noisy, t, obs)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # 这里的noise其实就是加噪声的最后一层
            loss = self.loss_fn(x_recon, noise, weights)            #这是往前的loss
        else:
            loss = self.loss_fn(x_recon, x_start, weights)          #这是往后的loss

        return loss

    # x:actions
    def loss(self, x, noise, obs, weights=1.0):
        batch_size = len(x)  # 这里self.n_timesteps 不应该改变
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        bc_loss = self.p_losses(self.model, x, noise, obs, t, weights)
        return bc_loss

    def forward(self, raw_actions, obs):
        action_pro, log_action_pro = self.sample(raw_actions, obs)
        return action_pro, log_action_pro


# 继承Diffusion 的一个子类
class SpacedDiffusion(Diffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])  # original_num_steps=100

        base_diffusion = Diffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):  # 原先的betas
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = torch.tensor(new_betas)  # 新的betas
        super().__init__(**kwargs)

    def p_mean_variance(
            self, model, x, t, s
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), x, t, s)

    # TODO
    def p_losses(self, model, x_start, obs, t, weights=1.0):  # pylint: disable=signature-differs
        return super().p_losses(self._wrap_model(model), x_start, obs, t, weights=weights)

    def _wrap_model(self, model):  # 初始化
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, s):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (40.0 / self.original_num_steps)
        return self.model(x, new_ts, s)  # TODO 输入参数不一致


def create_diffusion(
        n_agent,  # 剩下的都为**kwargs
        obs_dim,
        action_dim,
        model,
        betas_schedule='linear',
        n_timesteps=20,
        max_action=1,
        eta=0.0,
        loss_type='l2',
        sample_type='DDPM',
        timestep_respacing="",
        clip_denoised=True,
        predict_epsilon=True,
        rescale_timesteps=False
):
    betas = get_betas(betas_schedule, n_timesteps)
    if not timestep_respacing:
        timestep_respacing = [n_timesteps]
    diffmodel = SpacedDiffusion(
        use_timesteps=space_timesteps(n_timesteps, timestep_respacing),  # steps:1000  timestep_respacing:True
        n_agent=n_agent,  # 剩下的都为**kwargs
        obs_dim=obs_dim,
        action_dim=action_dim,
        model=model,
        betas_s=betas,
        n_timesteps=n_timesteps,
        max_action=max_action,
        eta=eta,
        loss_type=loss_type,
        sample_type=sample_type,
        timestep_respacing=timestep_respacing,
        clip_denoised=clip_denoised,
        predict_epsilon=predict_epsilon,
        rescale_timesteps=rescale_timesteps
    )
    return diffmodel
