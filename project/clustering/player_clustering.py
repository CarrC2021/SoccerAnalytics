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
from sklearn.decomposition import PCA
import codecs
import RadarWrapper

cols = [
    "npxG",
    "npGoals",
    "final_third_passes_received",
    "passes_to_final_third",
    "threatening_passes",
    "progressive_passes",
    "pass_xT",
    "cross_xT",
    "long_pass_xT",
    "dribble_xT",
    "ground_attacking_duels_won",
    "aerial_duels_won",
    "ground_defending_duels_won",
    "loose_ball_duels_won",
    "aerial_win_rate",
    "ground_defensive_win_rate",
    "ground_attacking_win_rate",
    "loose_ball_win_rate",
    "long_pass_accuracy",
    "pass_accuracy",
    "cross_accuracy",
    "long_pass_attempts",
    "cross_attempts",
    "pass_attempts",
]


player_id = 38031
minutes_played = 400

sql = sql_wrapper.SQLWrapper()
p_radar = sql.get_radar_stats(player_id, cols, minutes_played)
# replace none with 0
p_radar = np.array(p_radar, dtype=np.float64)
p_radar = np.nan_to_num(p_radar, nan=0.0)
stats_to_cluster = [p_radar]

# get all forwards
forwards = sql.get_player_ids(minutes_played, sql.get_player_role(player_id))
forwards.remove(player_id)

ids = [player_id] + forwards
for player in forwards:
    radar = sql.get_radar_stats(player, cols, minutes_played)
    # replace none with 0
    radar = np.array(radar, dtype=np.float64)
    radar = np.nan_to_num(radar, nan=0.0)
    # # check that the stats are high enough
    # if radar[8] < 0.8 * p_radar[8] or radar[2] < 0.7 * p_radar[2] or radar[4] < .6 * p_radar[4]:
    #     continue
    stats_to_cluster.append(radar)

## now we want to scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stats_to_cluster)

# Now find the 10 players with closest euclidean distance to the player we are interested in
# using numpy norm
distances = []
player = stats_to_cluster[0]
for i in range(1, len(stats_to_cluster)):
    distances.append(np.linalg.norm(player - stats_to_cluster[i]))


# now we want to find the 10 players with the smallest distance by using the index of the smallest distances
indices = np.argsort(distances)[:15]
counter = 1
for index in indices:
    print(
        f"{sql.get_player_name(ids[index])}'s Euclidean distace is {distances[index] :.3f}"
    )
    counter += 1

# now we find the optimal number of clusters in order to cluster the forwards using the scaled data
# we will use the inertia attribute of the kmeans model to plot the elbow curve
inertia = []
for i in range(1, len(cols)):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# now we plot the inertia
plt.plot(range(1, len(cols)), inertia)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters for forwards")
plt.ylabel("Inertia")
plt.show()
plt.savefig("plots/elbow_forwards.png")

plt.clf()

# use PCA to determine the directions that maximize the variance in the data
# we will use the scaled data
pca = PCA()
pca.fit(scaled_data)
# now we want to plot the cumulative sum of the explained variance ratio
# we want to see how many components we need to explain 90% of the variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Sum of Explained Variance Ratio")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Sum of Explained Variance Ratio")
plt.show()
plt.savefig("plots/cumulative_sum_forwards.png")

print(pca.components_)

# clear the plot
plt.clf()

# save the pca to a file and pca components
np.savetxt("pca.txt", pca.components_)
np.savetxt("pca_explained_variance_forwards.txt", pca.explained_variance_ratio_)
# now we want to plot the pca components
# we will use the first two components
pca = PCA(n_components=2)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
# now we want to plot the pca data
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.title("PCA of forwards")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
# save the plot
plt.savefig("plots/pca_forwards.png")
plt.clf()

# it looks like 11 components explain 90% of the variance
# reduce the data to 11 components and then cluster the data
pca = PCA(n_components=11)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
# now we want to cluster the data
kmeans = KMeans(n_clusters=6, init="k-means++", random_state=42)
kmeans.fit(pca_data)
# now we want to plot the clusters
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_)
plt.title("PCA of forwards")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
# save the plot
plt.savefig("plots/pca_clusters_forwards.png")

print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
# now lets compare a few players within the same cluster using radar charts
# get the ids of some players in the same cluster
for i in range(len(stats_to_cluster)):
    stats_to_cluster[i] = np.append(stats_to_cluster[i], kmeans.labels_[i])
    stats_to_cluster[i] = np.append(stats_to_cluster[i], ids[i])

# sort on labels
stats_to_cluster.sort(key=lambda x: x[-2])
lows = []
# the minimum of each column
for i in range(len(cols)):
    lows.append(min(stats_to_cluster, key=lambda x: x[i])[i])
# now we want the highs
highs = []
for i in range(len(cols)):
    highs.append(max(stats_to_cluster, key=lambda x: x[i])[i])

print(len(lows))
print(len(highs))


# save to a file the first 5 players in each cluster as a dictionary
# the key will be the cluster number and the value will be a list of the first 5 players in that cluster
cluster_dict = {}
for i in range(7):
    cluster_dict[i] = []
for i in range(len(stats_to_cluster)):
    cluster_dict[kmeans.labels_[i]].append(int(stats_to_cluster[i][-1]))

# save the dictionary to a file
with open("cluster_dict_forwards.txt", "w") as f:
    for key, value in cluster_dict.items():
        f.write(f"{key} : {value}\n")
