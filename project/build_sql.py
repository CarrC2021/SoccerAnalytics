"""
Calculating xT (position-based)
==============
Calculating Expected Threat
"""

#importing necessary libraries 
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
#opening data
import os
import pathlib
import warnings 
#used for plots
from mplsoccer import Pitch
from scipy.stats import binned_statistic_2d
import WyscoutWrapper
import sqlite3
import sql_wrapper
from xGmodel import xGmodel

NOTFOOT = 403 
RIGHTFOOT = 402
LEFTFOOT = 401
THROUGH = 901
ACCURATE = 1801
NOTACCURATE = 1802

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

# ##############################################################################
# # Opening data 
# # ------------
# # # In this section we implement the Expected Threat model in
# # # the same way described by `Karun Singh <https://karun.in/blog/expected-threat.html>`_.
# # # First, we open the data.
wyscout = WyscoutWrapper.WyscoutWrapper()
shot_df = wyscout.load_all_events(['Shot'])

#fix shot coordinates
shot_df = wyscout.fixed_shot_data(shot_df)
# # We will use the following columns:
shot_df = shot_df[['id', 'X', 'Y', 'Goal', 'matchId', 'teamId', 'playerId', 'subEventId', 'tags', 'eventSec', 'matchPeriod']]

# create an xG model using the models/notheaders.pickle file
foot_xGmodel = xGmodel('models/notheaders.pickle')

headers_xGmodel = xGmodel('models/headers.pickle')

sql = sql_wrapper.SQLWrapper()
foot_dict = sql.get_strong_foot()
print(foot_dict)

shot_df = wyscout.assign_body_part(shot_df, foot_dict)
shot_df['on_target'] = shot_df.apply(lambda x: 1 if {'id': ACCURATE} in x.tags else 0, axis=1)
#drop tags column since we don't need it anymore
shot_df.drop(columns=['tags'], inplace=True)

print(shot_df.head(10))

headers_df = shot_df.loc[shot_df['body_part'] == 'head/body']
headers_df = headers_xGmodel.assign_xG(headers_df)

notheaders_df = shot_df.loc[shot_df['body_part'] != 'head/body']
notheaders_df = foot_xGmodel.assign_xG(notheaders_df)

shot_df = pd.concat([headers_df, notheaders_df])

conn = sqlite3.connect('data/wyscout.db')

#save to database
shot_df.to_sql('shots', conn, if_exists='replace', index=False)

# load in the 2499719 match by matchId
# sql = sql_wrapper.SQLWrapper()

# passes = sql.get_passes_by_match(2499719)

# # get only passes where the matchPeriod is 1H
# passes = passes.loc[passes['matchPeriod'] == '1H']
# p

# #plot arrows of the passes using the X, Y, end_x, end_y columns with custom width of 68 and length of 105
# pitch = Pitch(pitch_type = 'custom', pitch_length = 105, pitch_width = 68)
# fig, ax = pitch.draw()
# pitch.arrows(passes['X'], passes['Y'], passes['end_x'], passes['end_y'], ax=ax, width=2, headwidth=10, color='black')
# plt.show()
# # save the figure
# fig.savefig('passing.png', dpi=300, bbox_inches='tight')

