import sql_wrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster

## import for one hot encoding
from sklearn.preprocessing import OneHotEncoder

# import for train test split
from sklearn.model_selection import train_test_split
import json

CM = [8, 11, 4, 8, "Center Midfield"]
TENSPACE = [11, 14, 4, 8, "10 Space"]
EIGHTYARD = [14, 16, 4, 8, "8 Yard"]
RIGHTHALFSPACE = [11, 14, 8, 10, "Right Halfspace"]
RIGHTWING = [11, 14, 10, 12, "Right Wing"]
RIGHTHALFSPACEDEEP = [7, 11, 8, 10, "Right Halfspace Deep"]
LEFTWING = [11, 14, 0, 2, "Left Wing"]
LEFTHALFSPACEDEEP = [7, 11, 2, 4, "Left Halfspace Deep"]
LEFTHALFSPACE = [11, 14, 2, 4, "Left Halfspace"]
DEEPCM = [6, 8, 4, 8, "Deep Center Midfield"]
DEEPLEFT = [6, 11, 0, 2, "Deep Left Wing"]
DEEPRIGHT = [6, 11, 10, 12, "Deep Right Wing"]

# load data using the sql wrapper
db = sql_wrapper.SQLWrapper()
# gets the passes from the central midfield area
passes = db.get_passes(TENSPACE, 8144)

# get a unique list of player ids in passes
player_ids = passes["playerId"].unique()

# loop through the player ids and get the passes for each player then calculate the likelihood of a pass being completed to the given
# sector of the field corresponding to the sector coordinates defined above
for player_id in player_ids:
    xT_from_sector = {}
    player_passes = passes[passes["playerId"] == player_id]
    total_passes = len(player_passes)
    print(f"Player {player_id} has {total_passes} passes")
    # loop through the sectors and sum up the xT of the passes that were completed to that sector
    for sector in [
        CM,
        TENSPACE,
        EIGHTYARD,
        RIGHTHALFSPACE,
        RIGHTWING,
        RIGHTHALFSPACEDEEP,
        LEFTWING,
        LEFTHALFSPACEDEEP,
        LEFTHALFSPACE,
        DEEPLEFT,
        DEEPRIGHT,
    ]:
        # get the passes that were completed to the sector
        sector_passes = player_passes[
            (player_passes["end_x"] >= sector[0] * 105 / 16)
            & (player_passes["end_x"] <= sector[1] * 105 / 16)
            & (player_passes["end_y"] >= sector[2] * 68 / 12)
            & (player_passes["end_y"] <= sector[3] * 68 / 12)
        ]
        # get only the passes completed to the sector
        sector_passes_completed = sector_passes[sector_passes["successful"] == 1]
        # sum up the xT of the passes completed to the sector
        sector_passes_completed_xT = sector_passes_completed["xT"].sum()
        # calculate the likelihood of a pass being completed to the sector
        if len(sector_passes) == 0:
            sector_passes_xT_adjusted = 0
        else:
            sector_passes_xT_adjusted = sector_passes_completed_xT / len(sector_passes)
        # add the adjusted to the dictionary
        xT_from_sector.update({sector[4]: sector_passes_xT_adjusted})

    # create a file for each player with the xT_from_sector dictionary and then
    # write the dictionary to the file using the json library
    with open(
        f"models/xT_from_sector/{player_id}_xT_from_{TENSPACE[4]}.json", "w"
    ) as f:
        json.dump(xT_from_sector, f)

# load the xT_to_sector dictionary from the file
with open(f"models/xT_from_sector/8144_xT_from_{TENSPACE[4]}.json", "r") as f:
    xT_from_sector = json.load(f)

# use seaborn to create a nice bar chart of the xT_to_sector dictionary and then save the plot
sns.barplot(x=list(xT_from_sector.keys()), y=list(xT_from_sector.values()))
# add a title to the plot
plt.title(f"xT from {TENSPACE[4]} for player 8144")
# make the plot bigger and have the labels be vertical so you can read them
plt.gcf().set_size_inches(10, 6)
plt.xticks(rotation=90)
# add labels to the x and y axis
plt.xlabel("Sector")
plt.ylabel("xT")
# add padding to the plot so the labels don't get cut off
plt.tight_layout()
# show the plot
plt.show()
plt.savefig(f"models/xT_from_sector/8144_xT_from_{TENSPACE[4]}.png")
