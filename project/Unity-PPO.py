"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import gym
from arm_env import ArmEnv
from unity_arm_env import UnityArmEnv
import json, time
from tqdm import tqdm

class Plot():
    def __init__(self):
        plt.figure(figsize=(5, 3))
        # plt.ion()

    def update(self, data):
        plt.cla()
        plt.plot(data)
        plt.savefig('./fig.jpg')

class PPO(object):

    def __init__(self, config={}):
        # 默认配置
        self.EP_MAX = 1000
        self.EP_LEN = 200
        self.GAMMA = 0.9
        self.A_LR = 0.0001
        self.C_LR = 0.0002
        self.BATCH = 32
        self.A_UPDATE_STEPS = 10
        self.C_UPDATE_STEPS = 10
        self.S_DIM, self.A_DIM = 7, 2
        self.optimization_type = 1
        self.kl_lambda = 0.5
        self.kl_target = 0.01
        self.clip_epsilon = 0.2
        self.should_norm_advantage = False # 是否归一化优势
        self.num_layers = 3 # 全连接层数
        self.hidden_units = 128 # 每个全连接层的单元数目

        # 覆盖默认配置
        for key, value in config.items():
            setattr(self, key, value)

        self.METHOD = [
            dict(name='kl_pen', kl_target=self.kl_target, lam=self.kl_lambda),
            dict(name='clip', epsilon=self.clip_epsilon),
        ][self.optimization_type]

        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            hidden = self.tfs
            for i in range(self.num_layers):
                hidden = tf.layers.dense(hidden, self.hidden_units, tf.nn.tanh)
            self.v = tf.layers.dense(hidden, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(
                self.C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(
                pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(
                p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-6)
                surr = ratio * self.tfadv
            if self.METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-self.METHOD['epsilon'], 1.+self.METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(
                self.A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, { self.tfs: s, self.tfdc_r: r })
        if self.should_norm_advantage:
            adv = (adv - adv.mean())/(adv.std() + 1e-6)

        # update actor
        if self.METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.METHOD['lam']})
                if kl > 4 * self.METHOD['kl_target']:  # this in in google's paper
                    break
            # adaptive lambda, this is in OpenAI's paper
            if kl < self.METHOD['kl_target'] / 1.5:
                self.METHOD['lam'] /= 1.5
            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 1.5
            # sometimes explode, this clipping is my solution
            self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {
                           self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})
         for _ in range(self.C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            hidden = self.tfs
            for i in range(self.num_layers):
                hidden = tf.layers.dense(
                    hidden, self.hidden_units, tf.nn.tanh, trainable=trainable)
            mu = 2 * tf.layers.dense(hidden, self.A_DIM,
                                     tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(
                hidden, self.A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -5, 5)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save(self, name):
        self.saver.save(self.sess, './model/' + name)

    def load(self, name):
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./model'))


def train(config={}, use_unity_arm=False):
    tf.reset_default_graph()

    should_random_target = 'should_random_target' in config.keys() and config['should_random_target']

    env = UnityArmEnv() if use_unity_arm else ArmEnv(mode='easy', should_random_target=should_random_target)
    config['A_DIM'] = env.action_dim
    config['S_DIM'] = env.state_dim
    ppo = PPO(config)
    all_ep_r = []
    lambdas = []

    should_render = 'should_render' in config.keys() and config['should_render']

    start = time.clock()

    plot = Plot()

    for ep in range(ppo.EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in tqdm(range(ppo.EP_LEN), desc='Training EP-' + str(ep) + '/' + str(ppo.EP_MAX) + ': '):    # in one episode
            if should_render:
                env.render()
            a = ppo.choose_action(s)
            s_, r, done = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)    # normalize reward, find to be useful
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % ppo.BATCH == 0 or t == ppo.EP_LEN-1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + ppo.GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(
                    buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                # bs = (bs - bs.mean()) / (bs.std() + 1e-6)
                # br = (br - br.mean()) / (br.std() + 1e-6)
                ppo.update(bs, ba, br)
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

        print('Current Reward: ', ep_r)
        plot.update(all_ep_r)

        if ppo.METHOD['name'] == 'kl_pen':
            lambdas.append(ppo.METHOD['lam'])

    elapsed = time.clock() - start

    print('Train with method {} done!'.format(ppo.METHOD['name']))
    print('Time elapsed {}s'.format(elapsed))

    return {
        'method': ppo.METHOD['name'],
        'ep_r': all_ep_r,
        'lambda': lambdas,
        'time': elapsed, # 耗时
        'config': config, # 当前变量
    }, ppo, env

def play():
    print('Playing...')
    env = ArmEnv()
    ppo = PPO()
    ppo.load('unity-arm')
    s = env.reset()
    while True:
        a = ppo.choose_action(s)
        s, r, done = env.step(a)
        env.render()


if __name__ == '__main__':
    play()
    configs = {
        'optimization_type': [0, 1],
        'EP_LEN': [100, 300, 500],
        'GAMMA': [0.1, 0.5, 0.9, 0.99],
        'A_LR': [0.001, 0.0001, 0.0005, 0.01],
        'C_LR': [0.002, 0.0002, 0.0005, 0.01],
        'BATCH': [16, 32, 64, 128],
        'A_UPDATE_STEPS': [10, 20, 50],
        'C_UPDATE_STEPS': [10, 20, 50],
        'kl_target': [0.1, 0.01, 0.05],
        'clip_epsilon': [0.1, 0.2, 0.5],
        'should_norm_advantage': [True, False],
        'should_random_target': [True, False]
    }
    configs = {
        'test': [1]
    }
    for attr, values in configs.items():
        for value in values:
            current_data, ppo, env = train({
                attr: value,
                'should_render': True, # 关闭GUI显示
                'optimization_type': 1,
                'EP_LEN': 2560,
                'BATCH': 128,
                'EP_MAX': 400,
                'should_random_target': True
            }, use_unity_arm=True)
            # 保存实验数据r
            with open('./data/{}__{}.json'.format(attr, value), 'w') as f:
                json.dump(current_data, f)

            ppo.save('unity-arm')
