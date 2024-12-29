class BaseGenerator(object):
    '''
    Generate state or reward based on current simulation state.
    根据当前模拟状态生成状态或奖励
    '''
    def generate(self):
        '''
        Different types of generators have different methods to implement it.
        不同类型的生成器有不同的实现方法，继承的方法实现
        '''
        raise NotImplementedError()
