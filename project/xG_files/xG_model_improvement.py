import json
import numpy as np
from joblib import dump, load
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import sql_wrapper
import os
import WyscoutWrapper
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

DUEL = 1
GOAL = 101
ASSIST = 301
KEYPASS = 302
LEFTFOOT = 401
RIGHTFOOT = 402
HEADER = 403

CWD = os.path.dirname(os.path.realpath(__file__))

wyscout = WyscoutWrapper.WyscoutWrapper()

df = wyscout.load_all()

# Can we make an improvement to the model by considering the action leading to the shot?
# For example was the previous event a cross, a long pass, or a dribble?

df["previousSubEventId"] = df["subEventId"].shift(1)
df["previousEventId"] = df["eventId"].shift(1)
df["previousTags"] = df["tags"].shift(1)

df = df[df["eventId"] == 10]

df["isHeader"] = df.apply(lambda row: 1 if {"id": 403} in row.tags else 0, axis=1)
df = df[df["isHeader"] == 0]
df["Goal"] = df.apply(lambda row: 1 if {"id": 101} in row.tags else 0, axis=1)

df["previousActionIsCross"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 80 else 0, axis=1
)
df["previousActionIsHandPass"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 81 else 0, axis=1
)
df["previousActionIsHeadPass"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 82 else 0, axis=1
)
df["previousActionIsHighPass"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 83 else 0, axis=1
)
df["previousActionIsLaunch"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 84 else 0, axis=1
)
df["previousActionIsSimplePass"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 85 else 0, axis=1
)
df["previousActionIsSmartPass"] = df.apply(
    lambda row: 1 if row["previousSubEventId"] == 86 else 0, axis=1
)
df["previousActionIsShot"] = df.apply(
    lambda row: 1 if row["previousEventId"] == 10 else 0, axis=1
)
df["previousActionIsOtherOnTheBall"] = df.apply(
    lambda row: 1 if row["previousEventId"] == 7 else 0, axis=1
)
df["previousActionIsDuel"] = df.apply(
    lambda row: 1 if row["previousEventId"] == 1 else 0, axis=1
)

df = wyscout.fixed_shot_data(df)


df = df[
    [
        "Goal",
        "X",
        "Y",
        "C",
        "Distance",
        "Distance2",
        "Angle",
        "X2",
        "C2",
        "AX",
        "previousActionIsCross",
        "previousActionIsHandPass",
        "previousActionIsHighPass",
        "previousActionIsLaunch",
        "previousActionIsSimplePass",
        "previousActionIsSmartPass",
        "previousActionIsShot",
        "previousActionIsOtherOnTheBall",
        "previousActionIsDuel",
    ]
]

# list the model variables you want here
model_variables = [
    "Angle",
    "Distance",
    "Distance2",
    "X",
    "X2",
    "C",
    "C2",
    "AX",
    "previousActionIsCross",
    "previousActionIsHandPass",
    "previousActionIsHighPass",
    "previousActionIsLaunch",
    "previousActionIsSimplePass",
    "previousActionIsSmartPass",
    "previousActionIsShot",
    "previousActionIsOtherOnTheBall",
    "previousActionIsDuel",
]
model = " + ".join(model_variables)


# fit the model
# split the shot data into train and test sets
shots_train, shots_test = train_test_split(df, test_size=0.2, random_state=42)

test_model = smf.glm(
    formula="Goal ~ " + model, data=shots_train, family=sm.families.Binomial()
).fit()
# print summary
print(test_model.summary())
print(test_model.pseudo_rsquared("mcf"))
# save the summary to a txt file
with open(f"{CWD}/models/notheaders.txt", "w") as fh:
    fh.write(test_model.summary().as_text())

with open(f"{CWD}/models/notheaders.txt", "w") as fh:
    fh.write(test_model.summary().as_latex())

test_model.save(f"{CWD}/models/notheaders.pickle")
