import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import threading, queue


EP_MAX = 2000 # 训练次数
EP_LEN = 300
N_WORDKERS = 4 # 线程数目
GAMMA = 0.9 # 折扣系数
ACTOR_LR = 0.0001 # 行动网络的学习率
CRITIC_LR = 0.0005 # 评价网络的学习率
BATCH_SIZE = 64 # 批量数据尺寸
UPDATE_STEP = 5 # 网络更新步数
EPSILON = 0.2 # 裁剪区间

S_DIM = 4 # 状态空间维度
A_DIM = 2 # 动作空间维度
A_BOUND = 1 # 动作范围

class PPO(object):
    '''
    近端策略优化算法
    '''
    def __init__(self):
        self.sess = tf.Session()
        self.state_input = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.discout_reward_input = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        
        # 评价者
        dense = tf.layers.dense(self.state_input, 100, tf.nn.relu)
        self.value = tf.layers.dense(dense, 1) # 价值函数的近似网络
        self.advantage = self.discout_reward_input - self.value # 优势评价指标
        self.critic_loss = tf.reduce_mean(tf.square(self.advantage)) # 评价网络的loss
        self.critic_train_op = tf.train.AdamOptimizer(CRITIC_LR).minimize(self.critic_loss) # 评价网络的训练操作

        # 行动者
        pi, pi_params = self._build_pi_net('pi', True)
        old_pi, old_pi_params = self._build_pi_net('old_pi', True)

        self.action_input = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.advantage_input = tf.placeholder(tf.float32, [None, 1], 'advantage')
        pi_ratio = pi.prob(self.action_input) / (old_pi.prob(self.action_input) + 1e-5)
        surrogate = pi_ratio * self.advantage_input
        # 定义行动者的loss函数
        self.actor_loss = - tf.reduce_mean(tf.minimum(surrogate, 
            tf.clip_by_value(pi_ratio, 1 - EPSILON, 1 + EPSILON) * self.advantage_input))

        self.actor_train_op = tf.train.AdamOptimizer(ACTOR_LR).minimize(self.actor_loss)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0) # 定义生成动作的计算图操作
        self.update_old_pi_op = [old_p.assign(p) for p, old_p in zip(pi_params, old_pi_params)] # 定义备份策略的操作

        self.sess.run(tf.global_variables_initializer()) # 初始化全局参数

    def _build_pi_net(self, name:str, trainable:bool):
        '''
        构造策略网络

        Args:
            name: 计算图的名字
            trainable: 是否用于训练
        
        Returns:
            pi_norm_distribution: 策略正态分布
            pi_params: 计算图参数
        '''
        with tf.variable_scope(name):
            dense = tf.layers.dense(self.state_input, 200, tf.nn.relu, trainable=trainable)
            mu = A_BOUND * tf.layers.dense(dense, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(dense, A_DIM, tf.nn.softplus, trainable=trainable)
            pi_norm_distribution = Normal(loc=mu, scale=sigma)

        pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return pi_norm_distribution, pi_params

    def choose_action(self, state:np.array):
        '''
        生成一个动作

        Args:
            state: 当前的状态

        Returns:
            action: 当前状态做出的动作
        '''
        state = state[np.newaxis, :] # 什么意思
        action = self.sess.run(self.sample_op, { self.state_input: state })[0]
        action = np.clip(action, -2, 2)

        return action

    def get_value(self, state:np.array):
        '''
        获取当前值函数的评价值

        Args:
            state: 当前状态

        Returns:
            value: 当前值
        '''
        if state.ndim < 2:
            state = state[np.newaxis, :]
        value = self.sess.run(self.value, { self.state_input: state })[0, 0]

        return value

    def update(self, data:np.array):
        '''
        训练策略网络

        Args:
            data: 批量数据
        '''
        self.sess.run(self.update_old_pi_op) # old_pi ← pi
        state, action, reward = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
        advantage = self.sess.run(self.advantage, {
            self.state_input: state,
            self.discout_reward_input: reward
        }) # 计算优势评价值
        for i in range(UPDATE_STEP):
            self.sess.run(self.actor_train_op, {
                self.state_input: state,
                self.action_input: action,
                self.advantage_input: advantage
            }) # 优化J_PPO
            self.sess.run(self.critic_train_op, {
                self.state_input: state,
                self.discout_reward_input: reward
            }) # 优化L_BL

    def sample_data(self, batch_size:int=64):
        '''
        采样数据

        Args:
            batch_size: 数据数量
        '''
        