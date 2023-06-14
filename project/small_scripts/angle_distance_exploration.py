import numpy as np
import pandas as pd
import sql_wrapper
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xGmodel

wyscout = sql_wrapper.SQLWrapper()

# get all shots
shots = wyscout.get_table("shots")

# angle of the shot using law of sines
shots["angle"] = np.arctan(
    7.32
    * shots["X"]
    / (shots["X"] ** 2 + abs(shots["Y"] - 68 / 2) ** 2 - (7.32 / 2) ** 2)
)
shots["distance"] = np.sqrt(shots["X"] ** 2 + abs(shots["Y"] - 68 / 2) ** 2)
# if shots['angle'] < 0 then angle = angle + np.pi
shots.loc[shots["angle"] < 0, "angle"] = shots["angle"] + np.pi


shotcount_hist = np.histogram(shots["angle"] * 180 / np.pi, bins=range(0, 150))

goals = shots[shots["Goal"] == 1]
goalcount_hist = np.histogram(goals["angle"] * 180 / np.pi, bins=range(0, 150))

np.seterr(divide="ignore", invalid="ignore")

goal_probability = np.divide(goalcount_hist[0], shotcount_hist[0])

angles = shotcount_hist[1]
avg_angle = (angles[1:] + angles[:-1]) / 2

# make single variable model of distance
test_model = smf.glm(
    formula="Goal ~ angle", data=shots, family=sm.families.Binomial()
).fit()

b = test_model.params

print(test_model.summary())
xG = 1 / (1 + np.exp(-(b[0] + b[1] * avg_angle * np.pi / 180)))

fig, ax = plt.subplots()
ax.plot(
    avg_angle,
    goal_probability,
    linestyle="none",
    marker=".",
    markersize=8,
    color="black",
)
ax.plot(avg_angle, xG, linestyle="solid", color="red")
ax.set_ylabel("Probability of goal")
ax.set_xlabel("Angle of shot (degrees)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()

fig.savefig("plots/angle_exploration.png", dpi=300, bbox_inches="tight")


# now explore distance
shotcount_hist = np.histogram(shots["distance"], bins=range(0, 50))
goalcount_hist = np.histogram(goals["distance"], bins=range(0, 50))

goal_probability = np.divide(goalcount_hist[0], shotcount_hist[0])
distance = shotcount_hist[1]
avg_distance = (distance[1:] + distance[:-1]) / 2

shots["distance_squared"] = shots["distance"] ** 2

# make single variable model of distance
test_model = smf.glm(
    formula="Goal ~ distance + distance_squared",
    data=shots,
    family=sm.families.Binomial(),
).fit()

b = test_model.params

print(test_model.summary())

print(test_model.pseudo_rsquared("mcf"))

xG = 1 / (1 + np.exp(-(b[0] + b[1] * avg_distance + b[2] * avg_distance**2)))

fig, ax = plt.subplots()
ax.plot(
    avg_distance,
    goal_probability,
    linestyle="none",
    marker=".",
    markersize=8,
    color="black",
)
ax.plot(avg_distance, xG, linestyle="solid", color="red")
ax.set_ylabel("Probability of goal")
ax.set_xlabel("Distance in meters")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()

fig.savefig("plots/distance_exploration.png", dpi=300, bbox_inches="tight")

notheaders = xGmodel.xGmodel("models/notheaders.pickle")

print(notheaders.model.pseudo_rsquared("mcf"))

headers = xGmodel.xGmodel("models/headers.pickle")

print(headers.model.pseudo_rsquared("mcf"))
