import sql_wrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
## import for one hot encoding
from sklearn.preprocessing import OneHotEncoder
#import for train test split
from sklearn.model_selection import train_test_split
import json
import mplsoccer

CM = [8, 11, 4, 8, 'Center Midfield']
TENSPACE = [11, 14, 4, 8, '10 Space']
EIGHTYARD = [14, 16, 4, 8, '8 Yard']
RIGHTHALFSPACE = [11, 14, 8, 10, 'Right Halfspace']
RIGHTWING = [11, 14, 10, 12, 'Right Wing']
RIGHTHALFSPACEDEEP = [7, 11, 8, 10, 'Right Halfspace Deep']
LEFTWING = [11, 14, 0, 2, 'Left Wing']
LEFTHALFSPACEDEEP = [7, 11, 2, 4, 'Left Halfspace Deep']
LEFTHALFSPACE = [11, 14, 2, 4, 'Left Halfspace']
DEEPCM = [6, 8, 4, 8, 'Deep Center Midfield']
DEEPLEFT = [6, 11, 0, 2, 'Deep Left Wing']
DEEPRIGHT = [6, 11, 10, 12, 'Deep Right Wing']

# load data using the sql wrapper 
db = sql_wrapper.SQLWrapper()
# from where does a player create the most xT? Let's consider passes that have xT > .05
WYID = 49876
passes = db.get_passes_by_player(WYID)
player_name = db.get_player_name(WYID)
passes = passes[passes['xT'] > .05]


# plot arrows on a pitch with custom dimensions x = 105, y = 68
pitch = mplsoccer.Pitch(line_color='black', pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
fig, ax = pitch.draw(figsize=(16, 12))
pitch.arrows(passes.X, passes.Y, passes.end_x, passes.end_y, width=1, ax=ax, color='black', zorder=1)
plt.title(f'{player_name} xT > .05', fontsize=20)
# save the plot
plt.savefig(f'./{player_name}_xT.png')