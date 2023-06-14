import pandas as pd
import WyscoutWrapper
import matplotlib.pyplot as plt
import os

# load data from the json files in the data/Wyscout/events folder as a pandas dataframe
# and create a graph of the events
# the graph is saved as a png file in the data/Wyscout/graphs folder
wyscout = WyscoutWrapper.WyscoutWrapper()

events = wyscout.load_all()

# return the counts of the events sorted by event type
counts = events["eventName"].value_counts()
# divide each count by the number of unique matchids to get the average number of events per match
counts = counts / len(events["matchId"].unique())
# drop the 'Interruption', and 'Goalkeeper leaving line' events
counts = counts.drop(["Interruption", "Goalkeeper leaving line"])
print(counts)
print(
    counts["Pass"],
    counts["Duel"],
    counts["Foul"],
    counts["Offside"],
    counts["Shot"],
    counts["Free Kick"],
    counts["Others on the ball"],
    counts["Save attempt"],
)
# plot a bar chart of the counts
ax = counts.plot(
    kind="bar",
    title="Average Number of Events In a Match",
    figsize=(16, 10),
    fontsize=12,
)
ax.set_xlabel("Event Type", fontsize=12)
# make x axis label horizontal
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
ax.set_ylabel("Count", fontsize=12)
plt.savefig(os.path.join("plots", "event_type_counts.png"))
plt.show()
