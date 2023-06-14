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
import sql_wrapper
from xGmodel import xGmodel

## load the data from the players.json file
with open("data/Wyscout/players.json") as f:
    players = json.load(f)

df = pd.DataFrame(players)

# this code is largely taken from https://soccermatics.readthedocs.io/en/latest/gallery/lesson3/plot_RadarPlot.html

df["role"] = df.apply(lambda x: x["role"]["name"], axis=1)
df = df[
    [
        "wyId",
        "firstName",
        "middleName",
        "lastName",
        "shortName",
        "weight",
        "height",
        "role",
    ]
]

train = pd.DataFrame()
for file in os.listdir("data/Wyscout/events"):
    with open(f"data/Wyscout/events/{file}") as f:
        events = json.load(f)
    # use concatenate to add the data from each file to the train dataframe
    train = pd.concat([train, pd.DataFrame(events)])


minutes_per_game = pd.DataFrame()
for file in os.listdir("data/Wyscout/minutes"):
    with open(f"data/Wyscout/minutes/{file}") as f:
        minutes = json.load(f)
    # use concatenate to add the data from each file to the minutes_per_game dataframe
    minutes_per_game = pd.concat([minutes_per_game, pd.DataFrame(minutes)])

# possesion_dict = {}
# #for every row in the dataframe
# for i, row in minutes_per_game.iterrows():
#     #take player id, team id and match id, minute in and minute out
#     player_id, team_id, match_id = row["playerId"], row["teamId"], row["matchId"]
#     #create a key in dictionary if player encounterd first time
#     if not str(player_id) in possesion_dict.keys():
#         possesion_dict[str(player_id)] = {'team_passes': 0, 'all_passes' : 0}
#     min_in = row["player_in_min"]*60
#     min_out = row["player_out_min"]*60

#     #get the dataframe of events from the game
#     match_df = train.loc[train["matchId"] == match_id].copy()
#     #add to 2H the highest value of 1H
#     match_df.loc[match_df["matchPeriod"] == "2H", 'eventSec'] = match_df.loc[match_df["matchPeriod"] == "2H", 'eventSec'] + match_df.loc[match_df["matchPeriod"] == "1H"]["eventSec"].iloc[-1]
#     #take all events from this game and this period
#     player_in_match_df = match_df.loc[match_df["eventSec"] > min_in].loc[match_df["eventSec"] <= min_out]
#     #take all passes and won duels as described
#     all_passes = player_in_match_df.loc[player_in_match_df["eventName"].isin(["Pass", "Duel"])]
#     #adjusting for no passes in this period (Tuanzebe)
#     if len(all_passes) > 0:
#         #removing lost air duels
#         no_contact = all_passes.loc[all_passes["subEventName"].isin(["Air duel", "Ground defending duel","Ground loose ball duel"])].loc[all_passes.apply(lambda x:{'id':701} in x.tags, axis = 1)]
#         all_passes = all_passes.drop(no_contact.index)
#     #take team passes
#     team_passes = all_passes.loc[all_passes["teamId"] == team_id]
#     #append it {player id: {team passes: sum, all passes : sum}}
#     possesion_dict[str(player_id)]["team_passes"] += len(team_passes)
#     possesion_dict[str(player_id)]["all_passes"] += len(all_passes)

# #calculate possesion for each player
# percentage_dict = {key: value["team_passes"]/value["all_passes"] if value["all_passes"] > 0 else 0 for key, value in possesion_dict.items()}
# #create a dataframe
# percentage_df = pd.DataFrame(percentage_dict.items(), columns = ["playerId", "possesion"])
# percentage_df["playerId"] = percentage_df["playerId"].astype(int)
# # save the percentage_df dataframe to a file
# percentage_df.to_csv('data/possesion.csv', index=False)

## load the saved possession csv as a dataframe
possesion_df = pd.read_csv("data/possesion.csv")
# save this value in the database as a new column in the players table
sql = sql_wrapper.SQLWrapper()
# sql.add_column('players', 'possession', 'REAL')
for i, row in possesion_df.iterrows():
    sql.update_column("players", "possession", row.possesion, row.playerId)


minutes = (
    minutes_per_game.groupby(["playerId", "shortName"])
    .minutesPlayed.sum()
    .reset_index()
)
print(minutes.head(20))

# add minutes column to the players table in the database based on the minutes_per_game dataframe
# sql.add_column('players', 'minutes_played', 'INTEGER')

# update the players table with the minutes played for each player
for i, row in minutes.iterrows():
    sql.update_column("players", "minutes_played", row.minutesPlayed, row.playerId)
