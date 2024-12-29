# This idea is borrowed from open catalyst project
"""
Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.
注册是真相的中心来源。受Redux全球存储概念的启发，Registry维护各种信息到唯一键的映射。注册表中的特殊函数可以用作装饰器来注册不同类型的类。
"""

class Registry:
    mapping = {
        'command_mapping': {},
        'task_mapping': {},
        'dataset_mapping': {},
        'model_mapping': {},
        'logger_mapping': {},
        'world_mapping': {},
        'trainer_mapping': {}
    }

    # @staticmethod 或 @classmethod，就可以不需要实例化，直接类名.方法名()来调用
    # self : 实例方法
    # cls ：  类方法
    @classmethod
    def register_command(cls, name):
        def wrap(f):
            cls.mapping['command_mapping'][name] = f
            # 把 f 函数/变量 当作参数传递进来时， 执行f 即运行传递进来的 函数/参数
            return f
        return wrap

    @classmethod
    def register_world(cls, name):
        def wrap(f):
            cls.mapping['world_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(f):
            cls.mapping['model_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_logger(cls, name):
        def wrap(f):
            cls.mapping['logger_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_task(cls, name):
        def wrap(f):
            cls.mapping['task_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_trainer(cls, name):
        def wrap(f):
            cls.mapping['trainer_mapping'][name] = f
            return f
        return wrap

    @classmethod
    def register_dataset(cls, name):
        def wrap(f):
            cls.mapping['dataset_mapping'][name] = f
            return f
        return wrap


Registry()
