import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from random import sample

# from decimal import *

cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

Boundary = 365  # 分界线   #聚类中心
train = False  # 是否训练
epochs = 50  # 迭代次数
batch = 32  # 抽样的大小
root = '../data/Mydataset/'
dataset1_name = 'good'
dataset2_name = 'bad'
loss = torch.nn.MSELoss()
latent_space = 128
n_intersection = 16
n_embd = 16  #
length = 8  # 预测长度

"""
step1 : 制作数据集
step2 : 每次抽取
"""


def load_dataset(path, dataset_name, batch_size, random=False):
    buffer = []
    if not os.path.exists(path + '%s.json' % dataset_name):
        print("file is not exist")
    if random:
        # with open(path + '%s.json' % dataset_name, 'r') as f:
        #     datas = json.load(f)
        #     phase = []
        #     random_load = random.sample(range(0, len(datas)), (len(datas)/batch_size, batch_size))
        #     for
        pass
    else:
        with open(path + '%s.json' % dataset_name, 'r') as f:
            datas = json.load(f)
            phase = []
            for d in range(len(datas)):
                phase.append(datas[d]["phase"])
                if (len(datas) - 1 - d) > batch_size:  # 判断数据集最后一段数据是否够一个batch_size
                    if d != 0 and len(phase) % batch_size == 0:
                        buffer.append(phase)
                        phase = []
                else:  # 剩余的数据不够一个batch
                    if len(phase) == batch_size:  # 只到数据的结尾才进行保存
                        buffer.append(phase)
                        phase = []
                    elif d == len(datas) - 1:
                        buffer.append(phase)
                        phase = []
    return buffer, len(buffer)


def gen_loss(x1, x2, y):
    # mx2 = mx1.round()

    c = torch.ones(size=y.shape)
    genloss = loss(x1 - x2, c)

    return genloss


def dis_loss(x1, y1, x2, y2):
    """
    y : label
    x : D/M(data [[datas]...[datas][gan(data)]...[gan(data)]]) label
    """
    z = torch.tensor(-1.)
    a = torch.ones(size=x2.shape) * z
    b = torch.ones(size=x1.shape)
    disloss = loss(y1 * x1, b) + loss(x2, a)

    return disloss


def mea_loss(x1, y1, x2, y2):
    """
    y : label
    x : D/M(data [[datas]...[datas][gan(data)]...[gan(data)]]) label
    """
    z = torch.tensor(-1.)
    a1 = torch.ones(size=x1.shape) * z
    a2 = torch.ones(size=x2.shape) * z
    b = torch.ones(size=y1.shape)
    mealoss = loss(x1, a1) + loss(x2, a2)

    return mealoss


class TSCGAN(object):
    def __init__(self, generator_output_size, discriminator_intput_size, meaner_input_size,
                 noise_size, n_embd, g_lr, d_lr):
        super(TSCGAN, self).__init__()
        # base param
        self.generator_output_size = generator_output_size  # phase
        self.noise_size = noise_size

        self.buffer = replay_buffer()
        # gan model relate
        self.TransGenerator = TransGenerator(noise_size, generator_output_size, n_embd)
        self.TransDiscriminator = TransDiscriminator(discriminator_intput_size)
        # self.TransMeaner = TransMeaner(meaner_input_size)

        self.G_Optimizer = Adam(self.TransGenerator.parameters(), lr=g_lr, weight_decay=1e-3)
        self.D_Optimizer = Adam(self.TransDiscriminator.parameters(), lr=d_lr, weight_decay=1e-3)
        #self.M_Optimizer = Adam(self.TransDiscriminator.parameters(), lr=d_lr, weight_decay=1e-3)
        #self.criterion = nn.BCELoss(reduction='mean')  # 用于测量目标概率和输入概率之间的二进制交叉熵

    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    # 获取目标相位
    def get_goal_phase(self, noise):
        goal_phase = self.TransGenerator(noise)
        return goal_phase

    # loader a batch
    # step = 900
    def train(self, n, epoch, goodloader, badloader, steps):
        gen_phase = None
        phase_dict = []
        gan_phase_argmax = []
        for e in range(epoch):
            for i in range(steps):
                # Adversarial ground truths
                valid_data = np.array(goodloader[i], dtype=np.float32)

                g_valid = torch.ones(size=(valid_data.shape[0], 1))
                g_fake = torch.zeros(size=(valid_data.shape[0], 1))

                # Configure input 配置输入
                t_valid_data = torch.tensor(valid_data)  # array转tensor

                gan.G_Optimizer.zero_grad()

                z = noise(length, n_intersection, latent_space)
                gen_phase = gan.TransGenerator(z)
                gen_phase = gen_phase.clone().detach().numpy()
                gan_phase_argmax.append(np.argmax(gen_phase, axis=2))

                """更新loss"""
                gan_valid = torch.ones(size=(gan_phase_argmax[0].shape[0], 1))
                m = gan.TransDiscriminator(gan_phase_argmax[0])
                g_loss = loss(gan.TransDiscriminator(gan_phase_argmax[0]), gan_valid)
                g_loss.backward()
                gan.G_Optimizer.step()
                if g_loss.item() < 0.1:
                    self.buffer.add_phase(gan_phase_argmax[0])

                gan.D_Optimizer.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                """
                lsgan:  a = -1  b = 1
                    Eg~Pdata(g) [square(y(D(g) - b))] + Ez~Pz[square(D(G(z)) - a)]
                """
                r = gan.TransDiscriminator(t_valid_data)
                f = gan.TransDiscriminator(gan_phase_argmax[0])
                joint_phase = self.buffer.joint_phase(gan_phase_argmax[0], batch)
                joint_fake = torch.zeros(size=(joint_phase.shape[0], 1))
                real_loss = loss(gan.TransDiscriminator(t_valid_data), g_valid)
                fake_loss = loss(gan.TransDiscriminator(joint_phase), joint_fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                gan.D_Optimizer.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] "
                    % (e, epoch, i, len(goodloader), d_loss.item(), g_loss.item())

                )
            phase_dict.append("phase")
            phase_dict.append(gan_phase_argmax[0])
        with open(os.path.join('../data/gan_output', f'phase.json'), "w") as f1:
            f1.write(str(self.buffer.phase_buffer))


class replay_buffer:
    def __init__(self):
        self.idx = 0
        self.phase_buffer = []

    def add_phase(self, phase):
        self.phase_buffer.extend(phase)
        self.idx += 1

    def len(self):
        buffer_len = len(self.phase_buffer)
        return buffer_len

    def joint_phase(self, phase, size):
        if self.len() >= size:
            j_phase = np.concatenate((self.phase_buffer[-(size-len(phase)):], phase), axis=0)
        else:
            j_phase = phase
        return j_phase

"""TODO"""


def writelog(file, e, epoch, i, steps, d_loss, g_loss):
    res = "%d/%d" % (e, epoch) + '\t' + "%d/%d" % (i, steps) + \
          '\t' + "D loss:%f" % d_loss + '\t' + "D loss:%f" % g_loss
    log_handle = open(file, "a")
    log_handle.write(res + "\n")
    log_handle.close()


class TransGenerator(nn.Module):
    def __init__(self, noise_dim, phase_dim, n_embd, slope=0.2):
        """
        param: noise_size : 输入噪声维度
        param: phase_dim :  输出数据维度
        param: n_embd :     隐藏层维度
        param: slope:       激活层斜率
        """
        super(TransGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.n_embd = n_embd  # 128

        # self.phase_embed = nn.Parameter(torch.zeros(1, self.bottom_width ** 2, n_embd))

        def block(in_dim, out_dim):
            """nn.nn.LeakyReLU(slope, inplace=True)] slope 为斜率"""
            layers = [nn.Linear(in_dim, out_dim), nn.LeakyReLU(slope, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(n_embd, n_embd),
            *block(n_embd, n_embd * 2),
            nn.Linear(n_embd * 2, phase_dim)  # 256 16
        )

    # 输入为noise
    def forward(self, z):
        # x = z + self.phase_embed                # input noise + 正太分布
        phase = self.model(z)
        return phase  # 8 * 16


class TransDiscriminator(nn.Module):
    def __init__(self, phase_dim):
        super(TransDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(phase_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, phase):
        # phase = phase.reshape(1, 16)
        phase = torch.tensor(phase, dtype=torch.float32)  # 转tensor
        validity = self.model(phase)

        return validity


'''恶语者'''


class TransMeaner(nn.Module):
    def __init__(self, phase_dim):
        super(TransMeaner, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(phase_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, phase):
        # phase = phase.reshape(1, 16)
        phase = torch.tensor(phase, dtype=torch.float32)  # 转tensor
        spurious = self.model(phase)

        return spurious


def noise_generator(agent_management, noise_type):
    # step1 noise input
    n = np.random.normal(0, 1,
                         (agent_management.phase_ob_length, agent_management.latent_dim))  # TODO 需要4和3的整倍数   #[20, 128]
    # step2 reshape()
    z = n.reshape(-1, agent_management.n_agent, agent_management.phase_ob_length)  # [8,16,20]
    z = torch.tensor(z, dtype=torch.float32)

    return n


def noise(length, n_inter, latent_dim):
    n = np.random.normal(0, 1, (n_inter, latent_dim))  # 16 * 128 = 2048
    z = n.reshape((length, n_inter, n_inter))  # 8 16 16 = 2048
    # z = torch.tensor(n)
    z = Variable(Tensor(z))
    return z


if train:
    '''step1 load dataset'''
    good_buffer, g_ = load_dataset(root, dataset1_name, batch, random=False)
    bad_buffer, b_ = load_dataset(root, dataset2_name, batch, random=False)
    gan = TSCGAN(n_intersection, n_intersection, n_intersection, n_intersection, n_embd, 0.05, 0.05)
    gan.train(n_intersection, epochs, good_buffer, bad_buffer, min(g_, b_))
