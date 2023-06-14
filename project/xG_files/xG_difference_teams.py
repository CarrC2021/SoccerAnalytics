import sql_wrapper
import sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression


# investigating if there is a correlation between a lower league position and a better xG difference

sql = sql_wrapper.SQLWrapper()

positions = {
    1: [1625, 676],
    2: [1611, 679],
    3: [1624, 675],
    4: [1612, 674],
    5: [1610, 682],
    6: [1609, 684],
    7: [1646, 680],
    8: [1623, 698],
    9: [1631, 701],
    10: [1613, 756],
    11: [1628, 691],
    12: [1659, 687],
    13: [1633, 692],
    14: [1644, 696],
    15: [1651, 695],
    16: [1673, 678],
    17: [1619, 712],
    18: [10531, 677],
    19: [1639, 714],
    20: [1627, 683],
}

shots = sql.get_table("shots")
X = []
# create y that starts from 20 and goes to 1 that follows the pattern
# 20, 20, 19, 19, 18, 18, ...
y = []
for position in range(20, 0, -1):
    teams = positions[position]
    for team in teams:
        opposing_xG = 0
        xG_diff = 0
        # find all unique matches that the team played in
        matches = shots[shots["teamId"] == team]["matchId"].unique()
        # for each match search and find the sum of the xG for the opposing team
        for match in matches:
            opposing_xG += shots[
                (shots["matchId"] == match) & (shots["teamId"] != team)
            ]["xG"].sum()
        xG_diff = shots[shots["teamId"] == team]["xG"].sum() - opposing_xG
        X.append(position)
        y.append(xG_diff)


# create a plot with x-axis being league position and y-axis being xG difference
plt.figure(figsize=(10, 10))
plt.xlabel("League Position")
plt.ylabel("xG Difference")
plt.title("xG Difference vs League Position")
plt.scatter(X, y, color="blue")

# use linear regression to see if there is a relationship between xG difference and league position
reg = LinearRegression()
X = np.array(X).reshape(-1, 1)
y = np.array(y)
reg.fit(X, y)

X_pred = np.arange(1, 21).reshape(-1, 1)
y_pred = reg.predict(X_pred)
# add the R^2 value to the plot in the top right corner
plt.text(
    0.65,
    0.95,
    "R^2 = " + str(round(reg.score(X, y), 3)),
    fontsize=12,
    transform=plt.gca().transAxes,
    color="black",
)

plt.plot(X_pred, y_pred, color="orange")

plt.savefig("plots/xG_diff_vs_league_position.png")

# now do the same thing but look at just xG for
X = []
y = []
for position in range(20, 0, -1):
    teams = positions[position]
    for team in teams:
        xG = shots[shots["teamId"] == team]["xG"].sum()
        X.append(position)
        y.append(xG)


# create a plot with x-axis being league position and y-axis being xG For
plt.figure(figsize=(10, 10))
plt.xlabel("League Position")
plt.ylabel("xG For")
plt.title("xG For vs League Position")
plt.scatter(X, y, color="blue")

# use linear regression to see if there is a relationship between xG difference and league position
reg = LinearRegression()
X = np.array(X).reshape(-1, 1)
y = np.array(y)
reg.fit(X, y)

X_pred = np.arange(1, 21).reshape(-1, 1)
y_pred = reg.predict(X_pred)
# add the R^2 value to the plot in the top right corner
plt.text(
    0.65,
    0.95,
    "R^2 = " + str(round(reg.score(X, y), 3)),
    fontsize=12,
    transform=plt.gca().transAxes,
    color="black",
)
plt.plot(X_pred, y_pred, color="orange")

plt.savefig("plots/xG_For_vs_league_position.png")


X = []
# create y that starts from 20 and goes to 1 that follows the pattern
# 20, 20, 19, 19, 18, 18, ...
y = []
for position in range(20, 0, -1):
    teams = positions[position]
    for team in teams:
        opposing_goals = 0
        goal_diff = 0
        # find all unique matches that the team played in
        matches = shots[shots["teamId"] == team]["matchId"].unique()
        # for each match search and find the sum of the xG for the opposing team
        for match in matches:
            opposing_goals += len(
                shots[
                    (shots["matchId"] == match)
                    & (shots["teamId"] != team)
                    & (shots["Goal"] == 1)
                ]
            )
        goal_diff = (
            len(shots[(shots["teamId"] == team) & (shots["Goal"] == 1)])
            - opposing_goals
        )
        X.append(position)
        y.append(goal_diff)


# create a plot with x-axis being league position and y-axis being xG difference
plt.figure(figsize=(10, 10))
plt.xlabel("League Position")
plt.ylabel("Goal Difference")
plt.title("Goal Difference vs League Position")
plt.scatter(X, y, color="blue")

# use linear regression to see if there is a relationship between xG difference and league position
reg = LinearRegression()
X = np.array(X).reshape(-1, 1)
y = np.array(y)
reg.fit(X, y)

X_pred = np.arange(1, 21).reshape(-1, 1)
y_pred = reg.predict(X_pred)
# add the R^2 value to the plot in the top right corner
plt.text(
    0.65,
    0.95,
    "R^2 = " + str(round(reg.score(X, y), 3)),
    fontsize=12,
    transform=plt.gca().transAxes,
    color="black",
)

plt.plot(X_pred, y_pred, color="orange")

plt.savefig("plots/goal_diff_vs_league_position.png")
