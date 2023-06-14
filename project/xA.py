# importing necessary libraries
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt

# used for plots
from mplsoccer import Pitch
from scipy.stats import binned_statistic_2d
import WyscoutWrapper
import sqlite3
import xGmodel
import time


SHOT = 10
ASSIST = 301
KEYPASS = 302
NOTFOOT = 403
RIGHTFOOT = 402
LEFTFOOT = 401
THROUGH = 901
ACCURATE = 1801
NOTACCURATE = 1802

wyscout = WyscoutWrapper.WyscoutWrapper()
start_time = time.time()

not_headers = xGmodel.xGmodel(f"models/notheaders.pickle")
headers = xGmodel.xGmodel(f"models/headers.pickle")

new_df = not_headers.assign_xG(
    new_df[(new_df["subEventId"] == SHOT) & (new_df["isHeader"] == 0)]
)
new_df = headers.assign_xG(new_df[(new_df["subEventId"] == SHOT) & (["isHeader"] == 1)])

df["xA"] = new_df["xG"].fillna(0).astype(float)
print(df[df["xA"] > 0])
