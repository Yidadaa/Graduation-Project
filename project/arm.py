import numpy as np
import pyglet

# 全局设置
pyglet.clock.set_fps_limit(10000)

class RobotArm(object):
    '''
    机械臂环境
    仿照gym的结构，但是底层使用pyglet实现
    '''
    # 环境参数
    action_bound = [-1, 1] # 动作范围
    action_dim = 2 # 动作空间
    state_dim = 7 # 状态空间

    # 机械臂参数
    dt = 0.1 # 关节转动角速度
    arm_length = [100, 100] # 这是一个两关节机械臂

    # GUI设置
    viewer = None
    viewer_size = (400, 400) # 窗口大小
    get_point = False # 是否有鼠标点击事件
    mouse_in = np.array([False]) # 鼠标移入事件
    point_l = 15 # ???
    grab_counter = 0 # 抓住计数器

    def __init__(self):
        '''
        初始化机械臂参数
        '''
        self.arm_info = np.zeros((2, 4))
        self.arm_info[0, 0] = self.arm_length[0]
        self.arm_info[1, 0] = self.arm_length[1]
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_size) / 2

    def step(self, action):
        '''
        执行动作
        '''
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        arm_rad = self.arm_info[:, 1]
        arm_dxdy = np.array(self.arm_info[:, 0])
