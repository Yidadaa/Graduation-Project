from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import json
import os

rcParams['font.family'] = 'Times New Roman'
font1 = {
    'family': 'STSong',
    'size': 12
}


def picpath(filename):
    return '../thesis/pic/{}.pdf'.format(filename)


def loadjson(filename):
    with open('./data/{}'.format(filename), 'r') as fp:
        current_data = json.load(fp)
    return current_data

def loadcsvs():
    files = os.listdir('./data/csv')
    data = []
    for f in files:
        with open('./data/csv/{}'.format(f), 'r') as fp:
            raw = fp.readlines()[1:]
            current_data = list(map(lambda x: [int(x.split(',')[1]), float(x.split(',')[2])], raw))
            data.append(current_data)
    return data

def collect_data_by_group(group_names):
    data = {}
    files = os.listdir('./data')

    for name in group_names:
        data[name] = []
        for f in files:
            if name in f:
                data[name].append(loadjson(f))
    return data

def setlabel(ax):
    # ax.set_xlabel('episode')
    # ax.set_ylabel('reward')
    ax.legend(prop=font1)


def plot_all():
    files = os.listdir('./data')
    plt.figure(figsize=(800, 600))

    figs = {}

    i = 0
    for f in files:
        current_data = loadjson(f)
        name = list(current_data['config'])[0]
        if name not in figs:
            i += 1
            figs[name] = plt.subplot(3, 4, i)
            figs[name].set_xlabel('episode')
            figs[name].set_ylabel('reward')
        figs[name].plot(current_data['ep_r'], label='{}={}'.format(
            name, current_data['config'][name]))

    for ax in figs.values():
        ax.legend()

    plt.show()


def kl_vs_clip():
    fig_id = 0
    fig_size = (5, 2.5)
    kl = loadjson('optimization_type__0.json')
    clip = loadjson('optimization_type__1.json')
    fig_1 = plt.figure(fig_id, fig_size)
    fig_id += 1
    ax = fig_1.add_subplot(111)
    ax.plot(kl['ep_r'], label=r'$L_{BL}(\theta)$')
    ax.plot(clip['ep_r'], label=r'$L_{CLIP}(\theta)$')
    setlabel(ax)
    fig_1.savefig(picpath('kl-vs-clip'))

    fig_2 = plt.figure(2, fig_size)
    fig_id += 1
    ax = fig_2.add_subplot(111)
    ax.plot(kl['lambda'], label=r'$\lambda$')
    setlabel(ax)
    fig_2.savefig(picpath('kl-lambda'))

    data = collect_data_by_group(['kl_target', 'clip_epsilon', 'should_random_target', 'should_norm_advantage', 'EP_LEN', 'BATCH', 'test'])
    labels = {
        'kl_target': lambda value: r'$KL_{target}=$' + str(value),
        'clip_epsilon': lambda value:r'$\epsilon$=' + str(value),
        'should_random_target': lambda v: u'目标点随机出现' if v else u'目标点固定出现',
        'should_norm_advantage': lambda v: u'执行规范化' if v else u'没有执行规范化',
        'EP_LEN': lambda v: r'$EP_{LEN}=$' + str(v),
        'BATCH': lambda v: 'Batch Size=' + str(v),
        'test': lambda x: '优化后'
    }
    for name, value in data.items():
        fig_id += 1
        fig_size = (5, 2.5) if name not in ['should_random_target', 'should_norm_advantage'] else (10, 2.5)
        fig = plt.figure(fig_id, fig_size)
        ax = fig.add_subplot(111)
        for v in value:
            ax.plot(v['ep_r'], label=labels[name](v['config'][name]))
        setlabel(ax)
        fig.savefig(picpath(name))

def params_table():
    files = os.listdir('./data')
    table = []
    for f in files:
        data = loadjson(f)
        if 'config' not in list(data):
            continue
        name = list(data['config'])[0]
        table.append([name, data['config'][name], data['time']])
    with open('table.csv', 'w') as f:
        f.write('\n'.join(map(lambda x: ','.join(map(str, x)), table)))

def plot_unity():
    data = loadcsvs()
    fig_size = (5, 2.5)
    fig = plt.figure(1, fig_size)
    name = ['unity-reward', 'unity-value-estimate', 'unity-loss']
    label = ['Reward', 'estimate', 'Value Loss']

    for i in range(3):
        plt.cla()
        ax = fig.add_subplot(111)

        current_data = np.array(data[i])[10:]
        ax.plot(current_data[:,0], current_data[:,1], label=label[i])
        setlabel(ax)
        plt.savefig(picpath(name[i]))

if __name__ == '__main__':
    # kl_vs_clip()
    # params_table()
    plot_unity()