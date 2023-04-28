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
import scipy

NOTFOOT = 403 
RIGHTFOOT = 402
LEFTFOOT = 401
THROUGH = 901
ACCURATE = 1801
NOTACCURATE = 1802

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

def assign_start_and_end_sectors(df: pd.DataFrame) -> pd.DataFrame:
    #move start index - using the same function as mplsoccer, it should work
    df["start_sector"] = df.apply(lambda row: tuple([i[0] for i in scipy.stats.binned_statistic_2d(np.ravel(row.X), np.ravel(row.Y), 
                                                               values = "None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)

    #move end index
    df["end_sector"] = df.apply(lambda row: tuple([i[0] for i in scipy.stats.binned_statistic_2d(np.ravel(row.end_x), np.ravel(row.end_y), 
                                                               values = "None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)
    
    return df

wyscout = WyscoutWrapper.WyscoutWrapper()

df = wyscout.load_all_events(['Simple pass', 'High pass', 'Head pass', 'Smart pass', 'Cross'])
print(df.columns)
df = wyscout.fix_coordinates(df)
df = assign_start_and_end_sectors(df)
matrix = np.load('models/xT_5.npy')
df = df.loc[(((df["end_x"] != 0) & (df["end_y"] != 68)) & ((df["end_x"] != 105) & (df["end_y"] != 0)))]

df['xT'] = df.apply(lambda row: matrix[row.end_sector[1] - 1][row.end_sector[0] - 1] 
                    - matrix[row.start_sector[1] - 1][row.start_sector[0] - 1], axis = 1)

through = {'id': THROUGH}
df['through'] = df.apply(lambda row: 1 if through in row.tags else 0, axis = 1)
successful = {'id': ACCURATE}
df['successful'] = df.apply(lambda row: 1 if successful in row.tags else 0, axis = 1)
header = {'id': NOTFOOT}
df['body_part'] = df.apply(lambda row: 'head/body' if header in row.tags else 'foot', axis = 1)
left = {'id': LEFTFOOT}
df['body_part'] = df.apply(lambda row: 'left foot' if left in row.tags else 'right foot', axis = 1)
df = df[['id', 'playerId', 'X', 'Y', 'end_x', 'end_y', 'xT', 'body_part', 'successful', 'through',
            'subEventId', 'matchId', 'eventSec', 'matchPeriod']]

conn = sqlite3.connect(f'{os.path.dirname(os.path.realpath(__file__))}/data/wyscout.db')
df.to_sql('test_passes', conn, if_exists = 'replace', index=False)