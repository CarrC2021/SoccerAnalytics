import numpy as np
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import os
from xGmodel import xGmodel
from WyscoutWrapper import WyscoutWrapper


HEADER = 403
CWD = os.path.dirname(os.path.realpath(__file__))
filename = f"{CWD}/models/notheaders.pickle"

model = xGmodel(filename)

wyscout = WyscoutWrapper()

shot_data = wyscout.load_all_events('Shot')
dictionary = {'id': HEADER}
shot_data = shot_data[shot_data['tags'].apply(lambda x: dictionary not in x)]
shot_data = wyscout.fixed_shot_data(shot_data)
goal_data = shot_data[shot_data['Goal'] == 1]

num = 68
#Create a 2D map of xG
pgoal_2d=np.zeros((num, num))
for x in range(num):
    for y in range(num):
        pgoal_2d[x,y] = model.calc_xG(x, y)

#plot pitch
pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.draw()
#plot probability
pos = ax.imshow(pgoal_2d, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=0.3, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Probability of goal', fontsize=30)
plt.xlim((0,68))
plt.ylim((0,60))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#plot pitch
pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
bin_statistic = pitch.bin_statistic(105 - shot_data['X'], shot_data['Y'], bins = 50)
bin_statistic_goals = pitch.bin_statistic(105 - goal_data['X'], goal_data['Y'], bins=50)
#normalize number of goals by number of shots
bin_statistic["statistic"] = bin_statistic_goals["statistic"]/bin_statistic["statistic"]
#plot heatmap
pcm = pitch.heatmap(bin_statistic, ax=ax["pitch"], cmap='Reds', edgecolor='white', vmin = 0, vmax = 0.6)
#make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Probability of scoring' , fontsize = 30)
plt.show()