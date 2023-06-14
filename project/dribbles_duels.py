import WyscoutWrapper
import json
import pandas as pd
import numpy as np
import scipy
import sqlite3


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


# load all duels from the wyscout data
wyscout = WyscoutWrapper.WyscoutWrapper()
data = wyscout.load_all_events(
    [
        "Ground attacking duel",
        "Air duel",
        "Ground defending duel",
        "Ground loose ball duel",
    ]
)

# load the xT_7 numpy array
matrix = np.load("models/xT_7.npy")

data = wyscout.denormalize_coordinates(data)
data["won"] = data.apply(lambda x: 1 if {"id": 703} in x.tags else 0, axis=1)
data["neutral"] = data.apply(lambda x: 1 if {"id": 702} in x.tags else 0, axis=1)
data["lost"] = data.apply(lambda x: 1 if {"id": 701} in x.tags else 0, axis=1)
# find where the event is a dribble
data["is_take_on"] = data.apply(
    lambda x: 1 if {"id": 503} or {"id": 504} in x.tags else 0, axis=1
)
# where is_dribble is 0 see if {id: 504} is in the tags
# data['is_take_on'] = data.apply(lambda x: 1 if {'id': 504} in x.tags else 0 if x.is_take_on == 0 else 0, axis = 1)
data["interception"] = data.apply(lambda x: 1 if {"id": 1401} in x.tags else 0, axis=1)
data["ground_attacking_duel"] = data.apply(
    lambda x: 1 if x.subEventId == 11 else 0, axis=1
)
data["aerial_duel"] = data.apply(
    lambda x: 1 if x.subEventName == "Air duel" else 0, axis=1
)
data["ground_defending_duel"] = data.apply(
    lambda x: 1 if x.subEventName == "Ground defending duel" else 0, axis=1
)
data["ground_loose_ball_duel"] = data.apply(
    lambda x: 1 if x.subEventName == "Ground loose ball duel" else 0, axis=1
)


# assign sector so that we can calculate xT
data = assign_start_and_end_sectors(data)
# get the xT values for each event where is_take_on is 1
data["xT"] = 0
data["xT"] = data.apply(
    lambda row: matrix[row.end_sector[1] - 1][row.end_sector[0] - 1]
    - matrix[row.start_sector[1] - 1][row.start_sector[0] - 1]
    if (row.is_take_on == 1 & row.won == 1)
    else -matrix[row.start_sector[1] - 1][row.start_sector[0] - 1],
    axis=1,
)
data = data[
    [
        "matchId",
        "teamId",
        "playerId",
        "ground_attacking_duel",
        "aerial_duel",
        "ground_defending_duel",
        "ground_loose_ball_duel",
        "matchPeriod",
        "eventSec",
        "X",
        "Y",
        "end_x",
        "end_y",
        "won",
        "neutral",
        "lost",
        "interception",
        "is_take_on",
        "xT",
    ]
]

print(data["xT"])

# save the data to the duels table in the wyscout.db database
conn = sqlite3.connect("data/wyscout.db")
data.to_sql("duels", conn, if_exists="replace", index=False)
