"""
Calculating xT (position-based)
==============
Calculating Expected Threat
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import json

# plotting
import matplotlib.pyplot as plt

# opening data
import os
import pathlib
import warnings

# used for plots
from mplsoccer import Pitch
from scipy.stats import binned_statistic_2d
import WyscoutWrapper
import sqlite3
import xGmodel
import scipy
import time
import math

SHOT = 10
ASSIST = 301
KEYPASS = 302
NOTFOOT = 403
RIGHTFOOT = 402
LEFTFOOT = 401
THROUGH = 901
ACCURATE = 1801
NOTACCURATE = 1802

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


def assign_start_and_end_sectors(df: pd.DataFrame) -> pd.DataFrame:
    # move start index - using the same function as mplsoccer, it should work
    df["start_sector"] = df.apply(
        lambda row: tuple(
            [
                i[0]
                for i in scipy.stats.binned_statistic_2d(
                    np.ravel(row.X),
                    np.ravel(row.Y),
                    values="None",
                    statistic="count",
                    bins=(16, 12),
                    range=[[0, 105], [0, 68]],
                    expand_binnumbers=True,
                )[3]
            ]
        ),
        axis=1,
    )

    # move end index
    df["end_sector"] = df.apply(
        lambda row: tuple(
            [
                i[0]
                for i in scipy.stats.binned_statistic_2d(
                    np.ravel(row.end_x),
                    np.ravel(row.end_y),
                    values="None",
                    statistic="count",
                    bins=(16, 12),
                    range=[[0, 105], [0, 68]],
                    expand_binnumbers=True,
                )[3]
            ]
        ),
        axis=1,
    )

    return df


wyscout = WyscoutWrapper.WyscoutWrapper()

start_time = time.time()
df = wyscout.load_all_events(
    [
        "Simple pass",
        "High pass",
        "Smart pass",
        "Launch",
        "Cross",
        "Head pass",
        "Hand pass",
    ]
)
print("--- %s seconds --- to load all events" % (time.time() - start_time))
start_time = time.time()
df = wyscout.denormalize_coordinates(df)
print("--- %s seconds --- to denormalize the coordinates" % (time.time() - start_time))
start_time = time.time()

# df['isHeader'] = df.apply(lambda row: 1 if row.tags is not None and {'id': NOTFOOT} in row.tags else 0, axis = 1)

new_df = df.shift(-1)
df["nextPlayerId"] = new_df["playerId"].fillna(-1).astype(int)
# # figure out if the next subevent is a shot
# df["nextSubEventId"] = new_df["subEventId"].fillna(-1).astype(int)
# df["nextSubEventIsHeader"] = new_df['isHeader'].fillna(-1).astype(int)
# not_headers = xGmodel.xGmodel(f'models/notheaders.pickle')
# headers = xGmodel.xGmodel(f'models/headers.pickle')
# toAssignNotHeader = df[(df["nextSubEventId"] == SHOT) & (df["nextSubEventIsHeader"] == 0)]
# print(toAssignNotHeader)
# df.drop(df[(df["nextSubEventId"] == SHOT) & (df["nextSubEventIsHeader"] == 0)].index, inplace = True)
# toAssignNotHeader = wyscout.assign_body_part(toAssignNotHeader)
# toAssignNotHeader = not_headers.assign_xA(toAssignNotHeader)
# toAssignHeader = df[(df["nextSubEventId"] == SHOT) & (df["nextSubEventIsHeader"] == 1)]
# df.drop(df[(df["nextSubEventId"] == SHOT) & (df["nextSubEventIsHeader"] == 1)].index, inplace = True)
# toAssignHeader = headers.assign_xA(toAssignHeader)

# df.drop(df[df["subEventId"] == SHOT].index, inplace = True)
# df = pd.concat([df, toAssignHeader])
# df = pd.concat([df, toAssignNotHeader])

# print(df[df['xA'] > 0])

# print("--- %s seconds --- to perform xA assignment" % (time.time() - start_time))
# start_time = time.time()

# df['start_sector'] = df.apply(lambda row: (math.ceil(row.X * 16 / 105), math.ceil(row.Y * 12 / 68)), axis = 1)
# df['end_sector'] = df.apply(lambda row: (math.ceil(row.end_x * 16 / 105), math.ceil(row.end_y * 12 / 68)), axis = 1)
df = assign_start_and_end_sectors(df)
print(
    "--- %s seconds --- to perform the start and ending sectors assignment"
    % (time.time() - start_time)
)

df["successful"] = df.apply(
    func=lambda row: 1 if {"id": ACCURATE} in row.tags else 0, axis=1
)

# drop columns that are both successful and the end sector is out of bounds
to_drop = df.loc[
    (df["end_sector"].apply(lambda x: x[1]) <= 0)
    & (df["end_sector"].apply(lambda x: x[0]) <= 0)
    & (df["end_sector"].apply(lambda x: x[0]) >= 16)
    & (df["end_sector"].apply(lambda x: x[1]) >= 13)
    & (df["successful"] == 1)
]
# drop the rows that were found by to_drop
df = df.drop(to_drop.index)

start_time = time.time()
matrix = np.load("models/xT_7.npy")

through = {"id": THROUGH}
df["through"] = df.apply(lambda row: 1 if through in row.tags else 0, axis=1)
df = wyscout.assign_body_part(df)
df["assist"] = df.apply(func=lambda row: 1 if {"id": ASSIST} in row.tags else 0, axis=1)
df["keyPass"] = df.apply(
    func=lambda row: 1 if {"id": KEYPASS} in row.tags else 0, axis=1
)

print(
    "--- %s seconds --- to perform the remaining column additions"
    % (time.time() - start_time)
)
start_time = time.time()
## create a new column for the xT value of each pass where the xT is
# matrix[row.end_sector[1] - 1][row.end_sector[0] - 1] - matrix[row.start_sector[1] - 1][row.start_sector[0] - 1] when the pass is successful
# - matrix[row.start_sector[1] - 1][row.start_sector[0] - 1] when the pass is unsuccessful
df["xT"] = df.apply(
    lambda row: matrix[row.end_sector[1] - 1][row.end_sector[0] - 1]
    - matrix[row.start_sector[1] - 1][row.start_sector[0] - 1]
    if row.successful == 1
    else -matrix[row.start_sector[1] - 1][row.start_sector[0] - 1],
    axis=1,
)
print("--- %s seconds --- to perform the xT assignment" % (time.time() - start_time))
df = df[
    [
        "playerId",
        "X",
        "Y",
        "end_x",
        "end_y",
        "xT",
        "assist",
        "keyPass",
        "isLeftFoot",
        "isRightFoot",
        "isHeadOrBody",
        "successful",
        "through",
        "subEventId",
        "teamId",
        "matchId",
        "eventSec",
        "matchPeriod",
        "nextPlayerId",
    ]
]

conn = sqlite3.connect(f"{os.path.dirname(os.path.realpath(__file__))}/data/wyscout.db")
df.to_sql("passes", conn, if_exists="replace", index=False)
