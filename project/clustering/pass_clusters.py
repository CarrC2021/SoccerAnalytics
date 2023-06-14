import sql_wrapper
import WyscoutWrapper
import mplsoccer
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# import gridsearchcv from sklearn.model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import codecs

xT_val = 0.07
team_id = 1609

sql = sql_wrapper.SQLWrapper()
passes = sql.get_team_data_from_table("passes", team_id=team_id)
passes = passes[passes["successful"] == 1]
features = ["X", "Y", "end_x", "end_y", "xT", "angle", "distance"]

# for now we will choose an individual team use id 1646 Burnley
print(passes)
passes = passes[passes["xT"] > xT_val]
print(passes)
passes["angle"] = np.arctan2(
    passes["end_y"] - passes["Y"], passes["end_x"] - passes["X"]
)
passes["distance"] = np.sqrt(
    (passes["end_x"] - passes["X"]) ** 2 + (passes["end_y"] - passes["Y"]) ** 2
)
train, test = train_test_split(passes[features], test_size=0.2, random_state=42)


num_clusters = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

#  now we want to plot sse and then choose the best number of clusters
sse = []
for n in num_clusters:
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(train[features])
    sse.append(kmeans.inertia_)

# plot the elbow curve
fig, ax = plt.subplots()
ax.plot(num_clusters, sse)
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("SSE")
ax.set_title("Elbow Curve")
plt.savefig("plots/ElbowCurve.png")

# it seems like 10 clusters is roughly the best now lets plot these passes using mplsoccer
kmeans = KMeans(n_clusters=9, n_init=10, random_state=42)
kmeans.fit(train[features])
passes["cluster"] = kmeans.predict(passes[features])

# plot the passes using mplsoccer
pitch = mplsoccer.Pitch(
    line_color="black", pitch_type="custom", pitch_length=105, pitch_width=68
)
fig, axs = pitch.grid(
    ncols=3,
    nrows=3,
    grid_height=0.85,
    title_height=0.06,
    axis=False,
    endnote_height=0.04,
    title_space=0.04,
    endnote_space=0.01,
    space=0.1,
)

for clust, ax in zip(np.linspace(0, 11, 12), axs["pitch"].flat[:12]):
    to_draw = passes[passes["cluster"] == clust]
    pitch.scatter(to_draw["X"], to_draw["Y"], ax=ax, s=20, alpha=0.7)
    pitch.arrows(
        to_draw["X"],
        to_draw["Y"],
        to_draw["end_x"],
        to_draw["end_y"],
        ax=ax,
        lw=1,
        color="black",
    )
    # find the player receiving the most number of passes in the cluster
    # and plot their name
    player_counts = to_draw["playerId"].value_counts()
    next_player_counts = to_draw["nextPlayerId"].value_counts()
    most_frequent = player_counts.idxmax()
    most_frequent_received = next_player_counts.idxmax()
    percentage = player_counts[most_frequent] / len(to_draw)
    percentage_received = next_player_counts[most_frequent_received] / len(to_draw)

    ax.set_title(
        f"{sql.get_player_name(most_frequent)} {percentage:.2f} \n"
        + f"{sql.get_player_name(most_frequent_received)} {percentage_received:.2f}",
        ha="center",
        va="center",
    )

axs["title"].text(
    0.5,
    0.5,
    f"Arsenal Pass Clusters with Most Frequent Passers and Receievers when xT > {xT_val}",
    fontsize=20,
    ha="center",
    va="center",
)
plt.savefig(f"plots/Arsenal_pass_clusters{xT_val}.png")
