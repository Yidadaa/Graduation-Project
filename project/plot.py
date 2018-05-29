from matplotlib import pyplot as plt
import json


with open('./clip_data.json', 'r') as f:
    clip_data = json.load(f)

with open('./kl_pen_data.json', 'r') as f:
    kl_data = json.load(f)

N = len(clip_data['ep_r'])

plt.figure(1)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

plt.sca(ax1)
# plt.plot(clip_data['ep_r'])
plt.plot(kl_data['ep_r'])
plt.sca(ax2)
plt.plot(kl_data['lambda'])
plt.show()