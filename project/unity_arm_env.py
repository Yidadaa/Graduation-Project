'''
Unity Arm Env的python接口
'''

import numpy as np
import requests

class UnityArmEnv(object):
    action_bound = []
    action_dim = 3
    state_dim = 17
    last_distance = 10000

    def __init__(self):
        '''
        初始化用到的类
        '''
        pass

    def step(self, action:list):
        '''
        执行动作

        Args:
            action: 动作向量
        '''
        state = self.call_unity(self.float2str(action))
        state = self.str2float(state)
        reward = self.reward(state)
        done = state[0] > 0
        return state, reward, done

    def reset(self):
        '''
        重置环境
        '''
        state = self.call_unity('reset')
        state = self.str2float(state)
        return state

    def shutdown(self):
        '''
        结束通信
        '''
        self.call_unity('exit')

    def reward(self, state):
        '''
        回报函数
        '''
        r = -state[1] / 10

        if state[0] > 0:
            r += 10

        if state[1] < self.last_distance:
            r += 1

        self.last_distance = state[1]

        return r

    def render(self):
        '''
        渲染类

        由于使用unity渲染，所以什么也不做
        '''
        pass
    
    def call_unity(self, data:str)->str:
        '''
        与unity进行通信
        '''
        resp = requests.post('http://127.0.0.1:88', data=data)
        return resp.text

    def float2str(self, l:list)->str:
        '''
        将浮点列表转换为字符串
        '''
        return ','.join(list(map(str, l)))

    def str2float(self, string:str)->np.array:
        '''
        将字符串转换为浮点列表
        '''
        return np.array(list(map(float, string.split(','))))