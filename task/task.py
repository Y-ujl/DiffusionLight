import logging
from common.registry import Registry


@Registry.register_task('base')
class BaseTask:
    '''
    Register BaseTask, currently support TSC task.
    注册BaseTask,当前支持TSC任务。
    '''
    def __init__(self, trainer):   #TODO is run.py.trainer ?
        self.trainer = trainer

    def run(self):
        raise NotImplementedError

    def _process_error(self, e):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
        ):
            for name, parameter in self.trainer.agents.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )


@Registry.register_task("tsc")
class TSCTask(BaseTask):
    '''
    Register Traffic Signal Control task.
    注册交通信号控制任务。
    '''
    def run(self):
        '''
        run
        Run the whole task, including training and testing.
        运行整个任务，包括训练和测试。

        :param: None
        :return: None
        '''
        """
            算法（基于梯度、基于策略等）的区别导致，有的算法训练时需要探索，所以要关注测试的值
            训练和测试，并不一定是同时进行，可以并行，可以异步
        """
        try:
            if Registry.mapping['model_mapping']['setting'].param['train_model']: # 训练
                '''yjl 4.5'''
                if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepmddpg':
                    self.trainer.my_train()
                else:
                    self.trainer.train()
            if Registry.mapping['model_mapping']['setting'].param['test_model']: # 测试
                '''yjl 4.5'''
                if Registry.mapping['model_mapping']['setting'].param['name'] == 'stepmddpg':
                    self.trainer.test_sac()
                else:
                    self.trainer.test()

        except RuntimeError as e:
            self._process_error(e)
            raise e
