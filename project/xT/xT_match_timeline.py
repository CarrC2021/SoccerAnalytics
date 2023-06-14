import WyscoutWrapper
import xGmodel
import numpy as np
import sql_wrapper

# load the xT_7 matrix
matrix = np.load("models/xT_7.npy")
sql = sql_wrapper.SQLWrapper()


# load in a game by matchId Monaco vs PSG
matchId = 2500820
wyscout = WyscoutWrapper.WyscoutWrapper()
df = sql.get_shots_and_passes(matchId)
df = df[df["matchId"] == matchId]

shots = df[df["subEventId"] == 100]

# drop the shots from the dataframe
df = df[df["subEventId"] != 100]
# add xT values to the dataframe

# split the game into its two halves
first_half = df[df["matchPeriod"] == "1H"]
second_half = df[df["matchPeriod"] == "2H"]
first_half_shots = shots[shots["matchPeriod"] == "1H"]
second_half_shots = shots[shots["matchPeriod"] == "2H"]

# positive xT will represent xT for Monaco and negative xT will represent xT for PSG
xT_over_time_Monaco = []
xT_over_time_PSG = []
# sort the events by eventSec
first_half = first_half.sort_values(by=["eventSec"])
# look at the last row to see the ending eventSec
last_row = first_half.iloc[-1]
# get the ending eventSec
last_eventSec = last_row["eventSec"]
num_periods = 200
period = last_eventSec / 200
curr_xT_Monaco = 0
curr_xT_PSG = 0
curr_xG_Monaco = 0
curr_xG_PSG = 0


xG_over_time_Monaco = [0] * 400
xG_over_time_PSG = [0] * 400
# go through every row in shot and add the xG to the list
for index, row in first_half_shots.iterrows():
    i = int(row["eventSec"] / (last_eventSec / 200))
    # add the xG to the list
    if row["teamId"] == 19830:
        # locate which period the shot occured in
        xG_over_time_Monaco[i] = row["xG"]
    else:
        xG_over_time_PSG[i] = row["xG"]

for i in range(num_periods):
    # get events that occured between the current period and the next period
    events = first_half[
        (first_half["eventSec"] >= i * period)
        & (first_half["eventSec"] < (i + 1) * period)
    ]
    # calculate the net xT for the period for Monaco whose teamId is 19830
    Monaco_xT = events[(events["teamId"] == 19830) & (events["xT"] >= 0)]["xT"].sum()
    # calculate the net xT for the period for PSG whose teamId is 3767
    PSG_xT = events[(events["teamId"] == 3767) & (events["xT"] >= 0)]["xT"].sum()
    # add the net xT for the period to the list
    xT_over_time_Monaco.append(Monaco_xT + curr_xT_Monaco)
    xT_over_time_PSG.append(PSG_xT + curr_xT_PSG)
    curr_xT_Monaco += Monaco_xT
    curr_xT_PSG += PSG_xT

# do the same for the second half
# sort the events by eventSec
second_half = second_half.sort_values(by=["eventSec"])
# look at the last row to see the ending eventSec
last_row = second_half.iloc[-1]
# get the ending eventSec
last_eventSec = last_row["eventSec"]
num_periods = 200
period = last_eventSec / 200
for i in range(num_periods):
    # get events that occured between the current period and the next period
    events = second_half[
        (second_half["eventSec"] >= i * period)
        & (second_half["eventSec"] < (i + 1) * period)
    ]
    # calculate the net xT for the period for Monaco whose teamId is 19830
    Monaco_xT = events[(events["teamId"] == 19830) & (events["xT"] >= 0)]["xT"].sum()
    # calculate the net xT for the period for PSG whose teamId is 3767
    PSG_xT = events[(events["teamId"] == 3767) & (events["xT"] >= 0)]["xT"].sum()
    # add the net xT for the period to the list
    xT_over_time_Monaco.append(Monaco_xT + curr_xT_Monaco)
    xT_over_time_PSG.append(PSG_xT + curr_xT_PSG)
    curr_xT_Monaco += Monaco_xT
    curr_xT_PSG += PSG_xT

# go through every row in shot and add the xG to the list
for index, row in second_half_shots.iterrows():
    i = int(row["eventSec"] / (last_eventSec / 200)) + 200
    # add the xG to the list
    if row["teamId"] == 19830:
        # locate which period the shot occured in
        xG_over_time_Monaco[i] = row["xG"]
    else:
        xG_over_time_PSG[i] = row["xG"]

# Fill in all the zeros in the xG list with the first positive number to the left
for i in range(len(xG_over_time_Monaco)):
    if xG_over_time_Monaco[i] == 0:
        xG_over_time_Monaco[i] = xG_over_time_Monaco[i - 1]
    else:
        xG_over_time_Monaco[i] = xG_over_time_Monaco[i] + xG_over_time_Monaco[i - 1]
    if xG_over_time_PSG[i] == 0:
        xG_over_time_PSG[i] = xG_over_time_PSG[i - 1]
    else:
        xG_over_time_PSG[i] = xG_over_time_PSG[i] + xG_over_time_PSG[i - 1]

# plot the xT over time for Monaco and PSG
# also add the xG over time for Monaco and PSG
import matplotlib.pyplot as plt

plt.plot(xT_over_time_Monaco, label="Monaco xT", linestyle="--")
plt.plot(xT_over_time_PSG, label="PSG xT", linestyle="--")
plt.plot(xG_over_time_Monaco, label="Monaco xG")
plt.plot(xG_over_time_PSG, label="PSG xG")
plt.xlabel("Time")
plt.ylabel("xT/xG")
plt.title("xT over time for Monaco vs PSG")
plt.legend()
plt.show()
# save the figure
plt.savefig("plots/xT_over_time.png")
