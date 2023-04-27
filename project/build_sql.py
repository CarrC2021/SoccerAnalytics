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
import xGmodel

NOTFOOT = 403 
RIGHTFOOT = 402
LEFTFOOT = 401
THROUGH = 901
ACCURATE = 1801
NOTACCURATE = 1802

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

##############################################################################
# Opening data 
# ------------
# # In this section we implement the Expected Threat model in
# # the same way described by `Karun Singh <https://karun.in/blog/expected-threat.html>`_.
# # First, we open the data.
wyscout = WyscoutWrapper.WyscoutWrapper()
shot_df = wyscout.load_all_events(['Shot'])

#fix shot coordinates
shot_df = wyscout.fixed_shot_data(shot_df)
# # We will use the following columns:
shot_df = shot_df[['id', 'matchId', 'teamId', 'playerId', 'subEventId', 'tags', 'X', 'Y', 'Goal']]

xGmodel = xGmodel.xGmodel('models/notheaders.pickle')

headers_xGmodel = xGmodel.xGmodel('models/headers.pickle')

shot_df['body_part'] = shot_df.apply(lambda x: NOTFOOT if {'id': NOTFOOT} in x.tags else LEFTFOOT, axis=1)
shot_df['body_part'] = shot_df.apply(lambda x: LEFTFOOT if {'id': LEFTFOOT} in x.tags else RIGHTFOOT, axis=1)
shot_df['on_target'] = shot_df.apply(lambda x: ACCURATE if {'id': ACCURATE} in x.tags else NOTACCURATE, axis=1)
#drop tags column since we don't need it anymore
shot_df.drop(columns=['tags'], inplace=True)

headers_df = shot_df.loc[shot_df['body_part'] == NOTFOOT]
headers_df = headers_xGmodel.assign_xG(headers_df)

notheaders_df = shot_df.loc[shot_df['body_part'] != NOTFOOT]
notheaders_df = xGmodel.assign_xG(notheaders_df)

shot_df = pd.concat([headers_df, notheaders_df])

conn = sqlite3.connect('data/wyscout.db')

#save to database
shot_df.to_sql('shots', conn, if_exists='replace', index=False)