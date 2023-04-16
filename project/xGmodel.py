# Load in all match events
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib as plt
import mplsoccer
from sklearn.model_selection import train_test_split
import os

DUEL = 1
GOAL = 101
ASSIST = 301
KEYPASS = 302
LEFTFOOT = 402
RIGHTFOOT = 403


all_data = pd.DataFrame()
for file in os.listdir('/home/casey/Projects/SoccerAnalytics/project/data/Wyscout/events'):
    path = '/home/casey/Projects/SoccerAnalytics/project/data/Wyscout/events/' + file
    with open(path) as f:
        data = json.load(f)
    all_data = pd.concat([all_data, pd.DataFrame(data)])

train, test = train_test_split(all_data, test_size=.2)

shots = train.loc[train['subEventName'] == 'Shot']
#get shot coordinates as separate columns
shots["X"] = shots.positions.apply(lambda cell: (100 - cell[0]['x']) * 105/100)
shots["Y"] = shots.positions.apply(lambda cell: cell[0]['y'] * 68/100)
shots["C"] = shots.positions.apply(lambda cell: abs(cell[0]['y'] - 50) * 68/100)
#calculate distance and angle
shots["Distance"] = np.sqrt(shots["X"]**2 + shots["C"]**2)
shots["Angle"] = np.where(np.arctan(7.32 * shots["X"] / (shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * shots["X"] /(shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)), np.arctan(7.32 * shots["X"] /(shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) + np.pi)
#if you ever encounter problems (like you have seen that model treats 0 as 1 and 1 as 0) while modelling - change the dependant variable to object
shots["Goal"] = shots.tags.apply(lambda x: 1 if {'id':GOAL} in x else 0).astype(object)

    
pitch = mplsoccer.VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
#subtracting x from 105 but not y from 68 because of inverted Wyscout axis
#calculate number of shots in each bin
bin_statistic_shots = pitch.bin_statistic(105 - shots.X, shots.Y, bins=30)
#make heatmap
pcm = pitch.heatmap(bin_statistic_shots, ax=ax["pitch"], cmap='Reds', edgecolor='white', linewidth = 0.01)
#make legend
ax_cbar = fig.add_axes((0.95, 0.05, 0.04, 0.8))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Shot map - 2017/2018 Season in Top 5 Leagues' , fontsize = 30)
plt.show() 
