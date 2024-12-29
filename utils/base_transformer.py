import math
import torch
import torch.nn as nn
from torch.nn import functional as F

def init_module(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    # 正交初始化
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init_module(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads 所有头的键、查询、值预测
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection 输出投影
        self.proj = init_(nn.Linear(n_embd, n_embd))
        #if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence 因果掩码，以确保仅将注意力应用于输入序列的左侧
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

    def forward(self, x):
        B, L, D = x.size()      #(8,16,64) -> 8,64,16
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 计算批处理中所有标头的查询、关键字和值，并将标头向前移动到批处理
        k = self.key(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # B, D = x.size()  # (1,64) -> 1,64
        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # # 计算批处理中所有标头的查询、关键字和值，并将标头向前移动到批处理
        # k = self.key(x).view(B, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)-(1,64,1)
        # q = self.query(x).view(B, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)-(1,64,1)
        # v = self.value(x).view(B,  self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)-(1,64,1)
        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     #(1,64,64)
        # self.att_bp = F.softmax(att, dim=-1)
        # if self.masked:
        #     att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)        #(1,64,64)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs) - (1,64,1)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  #(1,64) re-assemble all head outputs side by side
        # y = y.transpose(1, 2).contiguous().view(B, D)  #(1,64) re-assemble all head outputs side by side
        # output projection
        y = self.proj(y)    #(1,64)
        return y

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
                                 nn.GELU(),
                                 init_(nn.Linear(1 * n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x))  #(1,64)
        x = self.ln2(x + self.mlp(x))   #(1,64)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, n_block, n_embd, action_dim, n_head, n_agent):
        super(Encoder, self).__init__()
        """
        param: phase_dim    相位的维度
        param: obs_dim      观测的车辆的维度
        param: n_block      Encoder叠加层数
        param: n_embd       隐藏层
        param: n_head       多头注意力
        param: n_agent      n个智能体
        param: encode_phase=True    是否对phase相位进行编码
        """
        self.input_dim = input_dim      # 输入的维度
        self.n_block = n_block          # Encoder 堆叠的层数
        self.n_embd = n_embd            # 位置编码 = 64     # TODO 输入what
        self.n_head = n_head            # 多头注意力

        # phase_embeddings
        # nn.Sequential ： 顺序模块，按照构造顺序传递
        # nn.LayerNorm ： 对小批量输入应用层规范化
        # nn.GELU() ： 应用高斯误差线性单位函数 在零处处在梯度
        self.phase_obs_encoder = nn.Sequential(nn.LayerNorm(self.input_dim),
                                         init_(nn.Linear(self.input_dim, n_embd), activate=True),
                                         nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                  nn.GELU(),
                                  nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def forward(self, inp):
        embeddings = self.phase_obs_encoder(inp)      # Emb [8,16,20]
        rep = self.blocks(self.ln(embeddings))        #
        # v_loc = self.head(rep)
        v_loc = self.head(rep)
        #$ return rep, v_loc
        return v_loc

class SelfAttentionv2(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttentionv2, self).__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads 所有头的键、查询、值预测
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection 输出投影
        self.proj = init_(nn.Linear(n_embd, n_embd))
        #if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence 因果掩码，以确保仅将注意力应用于输入序列的左侧
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

    def forward(self, x):
        B, L, D = x.size()      #(1,16,64) -> 1,64,16
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 计算批处理中所有标头的查询、关键字和值，并将标头向前移动到批处理
        k = self.key(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # （1，1，16，64）(B, nh, L, hs)
        q = self.query(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))     #(1,1，16,16)
        # self.att_bp = F.softmax(att, dim=-1)
        # if self.masked:
             # att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)        #(1,1,16,16)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs) - (1,1,16,64)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  #(1,64) re-assemble all head outputs side by side
        # output projection
        y = self.proj(y)    #(1,64)
        return y

class EncodeBlockv2(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlockv2, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttentionv2(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
                                 nn.GELU(),
                                 init_(nn.Linear(1 * n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x))  #(1,16,64)
        x = self.ln2(x + self.mlp(x))   #(1,16,64)
        return x

class Encoderv2(nn.Module):
    def __init__(self, input_dim, n_block, n_embd, action_dim, n_head, n_agent):
        super(Encoderv2, self).__init__()
        """
        param: phase_dim    相位的维度
        param: obs_dim      观测的车辆的维度
        param: n_block      Encoder叠加层数
        param: n_embd       隐藏层
        param: n_head       多头注意力
        param: n_agent      n个智能体
        param: encode_phase=True    是否对phase相位进行编码
        """
        self.input_dim = input_dim      # 输入的维度
        self.n_block = n_block          # Encoder 堆叠的层数
        self.n_embd = n_embd            # 位置编码 = 64     # TODO 输入what
        self.n_head = n_head            # 多头注意力

        # phase_embeddings
        # nn.Sequential ： 顺序模块，按照构造顺序传递
        # nn.LayerNorm ： 对小批量输入应用层规范化
        # nn.GELU() ： 应用高斯误差线性单位函数 在零处处在梯度
        self.phase_obs_encoder = nn.Sequential(nn.LayerNorm(self.input_dim),
                                         init_(nn.Linear(self.input_dim, n_embd), activate=True),
                                         nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlockv2(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                  nn.GELU(),
                                  nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

        self.final_layer = init_(nn.Linear(n_agent, 1))

    def forward(self, inp):
        embeddings = self.phase_obs_encoder(inp)      # Emb [8,16,20]
        rep = self.blocks(self.ln(embeddings))        # rep(1,16, 64)      #
        # v_loc = self.head(rep)
        v_loc = self.head(rep)                        # (1,16,8)
        v_loc = self.final_layer(v_loc.transpose(1, 2)).squeeze(-1)     #(1,8)
        #$ return rep, v_loc
        return v_loc

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
                                 nn.GELU(),
                                 init_(nn.Linear(1 * n_embd, n_embd)))

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

class Decoder(nn.Module):
    def __init__(self, phase_dim, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False, encode_phase=True):
        super(Decoder, self).__init__()
        self.phase_dim = phase_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_block = n_block
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_agent = n_agent
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.encode_phase = encode_phase

        if action_type != 'Discrete':   # 不为离散动作
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)      #TODO mean?
        if self.dec_actor:
            if self.share_actor:
                # 共用一个 mlp
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True),
                                         nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True),
                                         nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True),
                                          nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True),
                                          nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    # 每个actor都存在一个 nn.Sequential()
                    self.mlp.append(actor)
        else:
            if action_type == 'Discrete':   # list
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True),
                                                    nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True),
                                             nn.GELU())
            self.phase_obs_dim = phase_dim + obs_dim
            self.phase_obs_encoder = nn.Sequential(nn.LayerNorm(self.phase_obs_dim),
                                             init_(nn.Linear(self.phase_obs_dim, n_embd), activate=True),
                                             nn.GELU())

            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                      nn.GELU(),
                                      nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':  # list
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, intput_rep, intput):    #TODO obs_rep is mean?
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(intput)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](intput[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, intput_rep)
            logit = self.head(x)
        return logit














