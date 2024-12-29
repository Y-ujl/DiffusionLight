from common.utils import build_index_intersection_map
from common.registry import Registry
from utils.logger import load_config_dict, modify_config_file, get_output_file_path
import os


class Interface(object):
    def __init__(self):
        pass

# @ 装饰器 的 语法糖 ： 相当于省略 Command_Setting_Interface = register_command(setting)
@Registry.register_command('setting')
class Command_Setting_Interface(Interface):
    """
    register command line into Registry
    将命令行注册到注册表
    """
    def __init__(self, config):
        super(Command_Setting_Interface, self).__init__()
        Command_Setting_Interface.param = config['command']  #key -- object 键值对


#@Registry.register_world('setting')
@Registry.register_world('graph_setting')
class Graph_World_Interface(Interface):
    """
    convert world roadnet into graph structure
    将世界路网转换为图形结构
    """
    def __init__(self, path):
        super(Graph_World_Interface, self).__init__()
        # TODO: support other roadnet file formation other than cityflow
        Graph_World_Interface.graph = build_index_intersection_map(path)


@Registry.register_world('setting')
class World_param_Interface(Interface):
    """
    use this interface to load and modify simulator configuration of logfiles
    使用此接口加载和修改日志文件的模拟器配置
    """
    def __init__(self, config):
        super(World_param_Interface, self).__init__()
        path = os.path.join(os.getcwd(), 'configs/sim', config['command']['network'] + '.cfg')
        other_world_settings = modify_config_file(path, config)
        World_param_Interface.param = load_config_dict(path, other_world_settings)
        

@Registry.register_model('setting')
class ModelAgent_param_Interface(Interface):
    """
    set model parameters
    设置模型参数
    """
    def __init__(self, config):
        super(ModelAgent_param_Interface, self).__init__()
        param = config['model']
        ModelAgent_param_Interface.param = param


@Registry.register_logger('path')
class Logger_path_Interface(Interface):
    """"
    set output path
    设置输出路径
    """
    def __init__(self, config):
        super(Logger_path_Interface, self).__init__()
        Logger_path_Interface.path = get_output_file_path(config)


@Registry.register_logger('setting')
class Logger_param_Interface(Interface):
    """
    setup logger path for logging, replay, model, dataset
    设置日志记录、回放、模型、数据集的记录器路径
    """
    def __init__(self, config):
        super(Logger_param_Interface, self).__init__()
        param = config['logger']
        Logger_param_Interface.param = param


@Registry.register_trainer('setting')
class Trainer_param_Interface(Interface):
    """
    set trainer parameters
    设置训练器参数
    """
    def __init__(self, config):
        super(Trainer_param_Interface, self).__init__()
        param = config['trainer']
        Trainer_param_Interface.param = param
