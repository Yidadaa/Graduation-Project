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
    header = {
        'Connection': 'close'
    }

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
        d = state[1]

        r = 0

        if state[0] > 0 or d < 1:
            r = 0.1

        self.last_distance = d

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
        resp = requests.post('http://127.0.0.1:88', data=data, headers=self.header)
        return resp.text

    def float2str(self, l:list)->str:
        '''
        将浮点列表转换为字符串
        '''
        str_l = list(map(str, l))
        if 'nan' in str_l:
            print('Wrong: ', l)
        return ','.join(str_l)

    def str2float(self, string:str)->np.array:
        '''
        将字符串转换为浮点列表
        '''
        return np.array(list(map(float, string.split(','))))