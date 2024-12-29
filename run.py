import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse

# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=20, help='number of threads')  # used in cityflow cityflow 模拟的线程数
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card 选择 gpu 卡
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process") # 此运行进程中的前缀数
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")  # pytorch 后端的种子
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo', 'traci'], help="interface type") # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx', 'real'], help="method of calculating delay") # apx(approximate) or real 计算延迟的方法 近似或真实

parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")   # 要运行的任务类型
parser.add_argument('-a', '--agent', type=str, default="colight", help="agent type of agents in RL environment")   # RL 环境中的智能体类型
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow', 'sumo'], help="simulator type")  # 模拟器类型
parser.add_argument('-n', '--network', type=str, default="cityflow1x3", help="network name")    # 网络名称
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')   # 训练过程中的数据集类型

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        使用已处理的配置实例化 runner 对象，并将配置注册到 Registry 类中
        """
        self.config, self.duplicate_config = build_config(pArgs)    # --> logger/build_config
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        将配置注册到 Registry 类
        """
        interface.Command_Setting_Interface(self.config)        # --> args 命令
        interface.Logger_param_Interface(self.config)           # --> base.yml + fixedtime.yml 合成的 logger
        interface.World_param_Interface(self.config)            # --> cityflow.cfg
        if self.config['model'].get('graphic', False):          # --> base.yml >> model >> graphic #TODO why do this  default if = False
            param = Registry.mapping['world_mapping']['setting'].param      # <-- cityflow.cfg
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']          #data 文件下目录
            else:
                roadnet_path = param['road_file_addr']
            # register graphic parameters in Registry class 在 Registry 类中注册图形参数
            interface.Graph_World_Interface(roadnet_path)       # --> cityflow.cfg

        interface.Logger_path_Interface(self.config)                #output dir path eg: data/output_data/tsc/cityflow_fixedtime/cityflow1x1/test
        # make output dir if not exist 如果不存在，则生成输出目录
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)        
        interface.Trainer_param_Interface(self.config)              # base.yml + xxx.yml --> trainer
        interface.ModelAgent_param_Interface(self.config)           # base.yml + xxx.yml --> model

    def run(self):
        # print(Registry.mapping) # {'command_mapping': {'setting': <class 'common.interface.Command_Setting_Interface'>}, 'task_mapping': {'base': <class 'task.task.BaseTask'>, 'tsc': <class 'task.task.TSCTask'>}, 'dataset_mapping': {'onfly': <class 'dataset.onfly_dataset.OnFlyDataset'>}, 'model_mapping': {'base': <class 'agent.base.BaseAgent'>, 'rl': <class 'agent.rl_agent.RLAgent'>, 'maxpressure': <class 'agent.maxpressure.MaxPressureAgent'>, 'dqn': <class 'agent.dqn.DQNAgent'>, 'sotl': <class 'agent.sotl.SOTLAgent'>, 'frap': <class 'agent.frap.FRAP_DQNAgent'>, 'ppo_pfrl': <class 'agent.ppo_pfrl.IPPO_pfrl'>, 'presslight': <class 'agent.presslight.PressLightAgent'>, 'fixedtime': <class 'agent.fixedtime.FixedTimeAgent'>, 'mplight': <class 'agent.mplight.MPLightAgent'>, 'matlight': <class 'agent.matlight.MATLightAgent'>, 'setting': <class 'common.interface.ModelAgent_param_Interface'>}, 'logger_mapping': {'path': <class 'common.interface.Logger_path_Interface'>, 'setting': <class 'common.interface.Logger_param_Interface'>}, 'world_mapping': {'cityflow': <class 'world.world_cityflow.World'>, 'setting': <class 'common.interface.World_param_Interface'>}, 'trainer_mapping': {'base': <class 'trainer.base_trainer.BaseTrainer'>, 'tsc': <class 'trainer.tsc_trainer.TSCTrainer'>, 'setting': <class 'common.interface.Trainer_param_Interface'>}}
        logger = setup_logging(logging_level)
        # Registry.mapping['command_mapping']['setting'].param['task'] == 'tsc'
        """
            'Register'.register()(logger) == 'register'(' ',logger)
        """
        # jump to TSCTrainer
        """
            @Registry.register_trainer("tsc")
            Registry.mapping['trainer_mapping']['tsc'](logger)  :  TSCTrainer().__init__(logger)    
        """
        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)          # jump to TSCTrainer

        # jump to task.py -- BaseTask.__init__(self.trainer) need transmit self.trainer
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)    # jump to TSCTask
        start_time = time.time()
        self.task.run()     # 运行入口
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)  # 配置
    test.run()  # 运行


