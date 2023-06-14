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
import sql_wrapper
from xGmodel import xGmodel

NOTFOOT = 403
RIGHTFOOT = 402
LEFTFOOT = 401
THROUGH = 901
ACCURATE = 1801
NOTACCURATE = 1802

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# #load the data
wyscout = WyscoutWrapper.WyscoutWrapper()
shot_df = wyscout.load_all_events(["Shot"])

# #fix shot coordinates
shot_df = wyscout.fixed_shot_data(shot_df)
# # # We will use the following columns:
shot_df["isPenalty"] = shot_df.apply(
    lambda x: 1 if x.subEventName == "Penalty" else 0, axis=1
)
shot_df = shot_df[
    [
        "id",
        "X",
        "Y",
        "Goal",
        "matchId",
        "teamId",
        "playerId",
        "subEventId",
        "tags",
        "eventSec",
        "matchPeriod",
        "isPenalty",
    ]
]

# # create an xG model using the models/notheaders.pickle file
foot_xGmodel = xGmodel("models/notheaders.pickle")

headers_xGmodel = xGmodel("models/headers.pickle")

shot_df["on_target"] = shot_df.apply(
    lambda x: 1 if {"id": ACCURATE} in x.tags else 0, axis=1
)
# #drop tags column since we don't need it anymore
shot_df.drop(columns=["tags"], inplace=True)

print(shot_df.head(10))

headers_df = shot_df.loc[shot_df["isHeadOrBody"] == 1]
headers_df = headers_xGmodel.assign_xG(headers_df)

notheaders_df = shot_df.loc[shot_df["isHeadOrBody"] == 0]
notheaders_df = foot_xGmodel.assign_xG(notheaders_df)

shot_df = pd.concat([headers_df, notheaders_df])

# if the shot is a penalty assign the xG value to be .8
shot_df["xG"] = shot_df.apply(lambda x: 0.8 if x.isPenalty == 1 else x.xG, axis=1)

conn = sqlite3.connect("data/wyscout.db")

# #save to database
shot_df.to_sql("shots", conn, if_exists="replace", index=False)
