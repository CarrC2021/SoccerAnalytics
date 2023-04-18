import numpy as np
from mplsoccer import VerticalPitch
import statsmodels.iolib
import matplotlib.pyplot as plt
import os
import xGmodel_class

CWD = os.path.dirname(os.path.realpath(__file__))
filename = f"{CWD}/models/headers.pickle"

model = xGmodel_class.xGmodel(filename)

#Create a 2D map of xG
pgoal_2d=np.zeros((68,68))
for x in range(68):
    for y in range(68):
        pgoal_2d[x,y] = model.calc_xG(x, y)

#plot pitch
pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.draw()
#plot probability
pos = ax.imshow(pgoal_2d, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=0.3, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Probability of goal')
plt.xlim((0,68))
plt.ylim((0,60))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()