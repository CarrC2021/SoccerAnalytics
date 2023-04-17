import numpy as np
from mplsoccer import VerticalPitch
import statsmodels.iolib
import matplotlib.pyplot as plt
import os

CWD = os.path.dirname(os.path.realpath(__file__))
with open(f"{CWD}/models/results.pickle", "rb") as f:
    model = statsmodels.iolib.smpickle.load_pickle(f)

print(model.summary())
print(model.params.index.values)
print(type(model.params.index.values))

b = model.params
model_variables = ['Angle','Distance','X','C', "X2", "C2", "AX"]

#return xG value for more general model
def calc_xG(sh):
   bsum=b[0]
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*sh[v]
   xG = 1/(1+np.exp(bsum))
   return xG


def shot_vars_from_x_and_y(x_coordinate, y_coordinate):
    shot_data = []
    angle = np.arctan(7.32*x_coordinate /(x_coordinate**2 + abs(y_coordinate-68/2)**2 - (7.32/2)**2))
    if angle < 0:
        angle = np.pi + angle
    shot_data.append(angle)
    shot_data.append(np.sqrt(x_coordinate**2 + abs(y_coordinate-68/2)**2))
    shot_data.append(x_coordinate**2 + abs(y_coordinate-68/2)**2)
    shot_data.append(x_coordinate)
    shot_data.append(x_coordinate*angle)
    shot_data.append(x_coordinate**2)
    shot_data.append(abs(y_coordinate-68/2))
    shot_data.append((y_coordinate-68/2)**2)
    return shot_data

#Create a 2D map of xG
pgoal_2d=np.zeros((68,68))
for x in range(68):
    for y in range(68):
        sh=dict()
        a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        sh['Angle'] = a
        sh['Distance'] = np.sqrt(x**2 + abs(y-68/2)**2)
        sh['D2'] = x**2 + abs(y-68/2)**2
        sh['X'] = x
        sh['AX'] = x*a
        sh['X2'] = x**2
        sh['C'] = abs(y-68/2)
        sh['C2'] = (y-68/2)**2
        pgoal_2d[x,y] = calc_xG(sh)

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