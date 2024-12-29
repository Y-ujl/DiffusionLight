import os
import random
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv, TSCGoalEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
import rl_plotter
from rl_plotter.logger import Logger

"""4.20 yjl"""
from utils.gan import TransGenerator


@Registry.register_trainer("tsc")  # TSCTrainer = register_trainer('tsc')
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    为交通信号控制任务注册 TSCTrainer
    '''
    def __init__(
            self,
            logger,
            gpu=0,
            cpu=False,
            name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        # over jump to base_trainer
        # base_trainer create() 函数(世界、智能体、评判指标、模拟器)构建完，跳回！！！！
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # replay file is only valid in cityflow now. 重播文件现在仅在 cityflow 中有效。
        # ********************************************************************


        # TODO: support SUMO and Openengine later
        # TODO: support other dataset in the future
        # jump to onfly_dataset.py -- OnFlyDataset()
        self.dataset = Registry.mapping['dataset_mapping'][
            Registry.mapping['command_mapping']['setting'].param['dataset']]\
            (os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir']))
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name 由输出dir+log_dir+文件处理程序名称的路径组成
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
                                         '_BRF.log') + '_DTL.log')
        name = Registry.mapping['model_mapping']['setting'].param['name']
        self.text_time_logger = Logger(exp_name=name, env_name="Cityflow")
        self.queue_logger = Logger(exp_name=name, env_name="Cityflow")
        self.delay_logger = Logger(exp_name=name, env_name="Cityflow")
        self.throughput_logger = Logger(exp_name=name, env_name="Cityflow")

    # 继承base_trainer create()
    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.
        创建世界，目前支持 CityFlow world、SUMO world 和 Citypb world。

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping 交通设置在世界地图中
        # 地址映射 jump to world_cityflow.py -- World()
        # Registry.mapping['world_mapping'][cityflow]
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']] \
            (self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],
             interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.
        创建度量以评估模型性能，当前支持奖励、队列长度、延迟（近似或实际）和吞吐量。

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        # jump to common.metrics 评判指标
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepsac':
            self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents.StepActor)
        else:
            self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.
        为交通信号控制任务创建智能体。

        :param: None
        :return: None
        '''
        '''yjl 4.5'''
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepsac':   #'stepsac'
            # 调用StepSAC
            self.agents = Registry.mapping['model_mapping']\
                [Registry.mapping['command_mapping']['setting'].param['agent']]\
                (self.world, 0)
            print(self.agents)
        else:
            self.agents = []
            # jump to (what in Registry.mapping)(presslight.py -- PressLightAgent())
            agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                self.world, 0)
            print(agent)
            num_agent = int(len(self.world.intersections) / agent.sub_agents)

            self.agents.append(agent)  # initialized N agents for traffic light control 已初始化用于交通灯控制的 N 个智能体
            # print(self.agents)
            for i in range(1, num_agent):
                self.agents.append(
                    Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                        self.world, i))

            # for magd agents should share information 对于 magd 智能体应共享信息
            if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
                for ag in self.agents:
                    ag.link_agents(self.agents)
                print(self.agents)
            # yjl
            # for maddpg agents should share information 对于 maddpg 智能体应共享信息
            if Registry.mapping['model_mapping']['setting'].param['name'] == 'T_maddpg':
                for ag in self.agents:
                    ag.link_agents(self.agents)
                print(self.agents)
            # yjl
            if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepmaac':
                for ag in self.agents:
                    ag.create_actor()
                print(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.
        创建与智能体通信的模拟环境。

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        # jump to environment.py -- TSCEnv()
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepsac':
            self.env = TSCEnv(self.world, self.agents.StepActor, self.metric)
        elif Registry.mapping['model_mapping']['setting'].param['name'] == 'stepmddpg':
            self.env = TSCGoalEnv(self.world, self.agents, self.metric, self.similarity_threshold)
        else:
            self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s).
        训练智能体

        :param: None
        :return: None
        '''
        global obs
        total_decision_num = 0  # 总决策数
        flush = 0

        for e in range(self.episodes):  # 回合数
            # 在每个事件的开始，环境和智能体被重置，将生成初始状态，之后它们开始交互
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)

            episode_loss = []
            i = 0
            while i < self.steps:  # 时间步数 3600
                if i % self.action_interval == 0:
                    # phase
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    # total_decision_num：每轮加360次
                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))  # 智能体针对给定的状态产生动作
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    # 并不是所有算法都有get_action_prob()方法
                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                         actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []

                    for _ in range(self.action_interval):  # 一个动作执行10s
                        obs, rewards, dones, _ = self.env.step(actions.flatten())  # 接着环境产生下一个状态，并对给定的动作进行奖励，然后进入下一个时间步
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)
                    # agent.act 和 env.step 循环将继续，直到达到最大时间步T或环境终止
                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])

                    # Store experience in replay_buffer (在replay buffer 中保存经验)
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                                    obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    """yjl 5.15"""
                    # self.agents[15].remember_time(e, actions.flatten(), last_obs.copy(), obs.copy(), rewards.copy(), [], is_pro=True)
                    # self.agents[0].remember_time(e, actions.flatten(), last_obs.copy(), obs.copy(), rewards.copy(), [], is_pro=True)
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                # print("total_decision_num: ", total_decision_num) # 360
                # if total_decision_num > self.learning_start and\
                #         total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == 0:  # TODO maybe has error
                    # 训练智能体 go to
                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # training
                    #cur_loss_q = np.stack([ag.train(e, self.episodes) for ag in self.agents])  # training
                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]
                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(), \
                          mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(),
                          self.metric.throughput())
            self.logger.info(
                "step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps, \
                                                                                              mean_loss,
                                                                                              self.metric.rewards(),
                                                                                              self.metric.queue(),
                                                                                              self.metric.delay(),
                                                                                              int(self.metric.throughput())))
            if e % self.save_rate == 0:  # 保存模型
                [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes,
                                                                             self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug(
                    "intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j], \
                                                                                    self.metric.lane_queue()[j]))
            if self.test_when_train:  # 评估
                self.train_test(e)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.
        在每次训练过程后评估模型性能。

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                # stack 将列表作为堆栈使用（后进先出）
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format( \
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards(), \
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))

        self.text_time_logger.update(score=[self.metric.real_average_travel_time()], total_steps=e)
        self.queue_logger.update(score=[self.metric.queue()], total_steps=e)
        self.delay_logger.update(score=[self.metric.delay()], total_steps=e)
        self.throughput_logger.update(score=[self.metric.throughput()], total_steps=e)

        self.writeLog("TEST", e, self.metric.real_average_travel_time(), \
                      100, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.
        测试进程。评估模型性能。

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
            self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        return self.metric

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.
        编写日志以进行记录和调试。

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" + \
              "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

    def my_train(self):
        '''
        train
        Train the agent(s).
        训练智能体

        :param: None
        :return: None
        '''
        global dones
        global last_phase
        total_decision_num = 0  # 总决策数
        flush = 0

        for e in range(self.episodes):  # 回合数
            # 在每个事件的开始，环境和智能体被重置，将生成初始状态，之后它们开始交互
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            last_phase = np.stack([ag.get_phase() for ag in self.agents])
            goal_phase = self.agents.gan.get_goal_phase()

            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)

            episode_loss = []
            i = 0
            while i < self.steps:  # 时间步数 3600
                if i % self.action_interval == 0:
                    # phase
                    pass
