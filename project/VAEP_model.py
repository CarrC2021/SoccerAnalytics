import os
import warnings
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# This file is built from the following tutorial:
# https://github.com/soccer-analytics-research/fot-valuing-actions/blob/master/notebooks/tutorial3-learn-models.ipynb

datafolder = os.path.join(os.getcwd(), "VAEP_data")
spadl_h5 = os.path.join(datafolder, "spadl.h5")
features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")
predictions_h5 = os.path.join(datafolder, "predictions.h5")

games = pd.read_hdf(spadl_h5, "games")
print(games)
# # select games from the first 5 matchweeks just to test the code
# games = games[games["game_day"] <= 5]
print("nb of games:", len(games))

# do train test split on games
traingames, testgames = train_test_split(games, test_size=0.2, random_state=42)

features = [
    fs.actiontype,
    fs.actiontype_onehot,
    # fs.bodypart,
    fs.bodypart_onehot,
    fs.result,
    fs.result_onehot,
    fs.goalscore,
    fs.startlocation,
    fs.endlocation,
    fs.movement,
    fs.space_delta,
    fs.startpolar,
    fs.endpolar,
    fs.team,
    # fs.time,
    fs.time_delta,
    # fs.actiontype_result_onehot
]

nb_prev_actions = 1

Xcols = fs.feature_column_names(features, nb_prev_actions)


def getXY(games, Xcols):
    # generate the columns of the selected feature
    X = []
    for game_id in tqdm.tqdm(games.game_id, desc="Loading features"):
        Xi = pd.read_hdf(features_h5, f"game_{game_id}")
        X.append(Xi[Xcols])
    X = pd.concat(X).reset_index(drop=True)

    # 2. Select label Y
    Ycols = ["scores", "concedes"]
    Y = []
    for game_id in tqdm.tqdm(games.game_id, desc="Loading labels"):
        Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
        Y.append(Yi[Ycols])
    Y = pd.concat(Y).reset_index(drop=True)
    return X, Y


X, Y = getXY(traingames, Xcols)
print("X:", list(X.columns))
print("Y:", list(Y.columns))

Y_hat = pd.DataFrame()
# train a model for each label column
# create a param grid with np.linspace for the number of estimators and a list of max_depth values
# create a list of every integer between 30 and 100 with a step size of 2
# n_estimators = np.linspace(30, 100, 36).astype(int)
n_estimators = [40, 50, 60, 70, 80, 90, 100]
param_grid = {"n_estimators": n_estimators, "max_depth": [3]}
optimal_params = {}
optimal_scores = {}
# track time to train for each label
start_time = time.time()
for col in tqdm.tqdm(list(Y.columns), desc="Training models"):
    # use grid search cross-validation to optimize the hyperparameters
    model = GridSearchCV(
        XGBClassifier(
            random_state=42,
        ),
        param_grid=param_grid,
        verbose=3,
        n_jobs=-1,
        cv=3
    )
    model.fit(X, Y[col])
    optimal_params[col] = model.best_params_
    optimal_scores[col] = model.best_score_
    elapsed_time = time.time() - start_time
    print(f"Elapsed time training the model for {col}: {elapsed_time:.2f} seconds")
    start_time = time.time()

testX, testY = getXY(testgames, Xcols)

def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

# retrain the model with the optimal hyperparameters
models = {}
for col in tqdm.tqdm(list(Y.columns), desc="Training models"):
    print(f"### {col} ###")
    print(f"Best params: {optimal_params[col]}")
    print(f"Best score: {optimal_scores[col]}")
    model = XGBClassifier(
        random_state=42,
        n_estimators=optimal_params[col]["n_estimators"],
        max_depth=optimal_params[col]["max_depth"],
    )
    model.fit(X, Y[col])
    models[col] = model
    Y_hat[col] = [p[1] for p in model.predict_proba(testX)]
    evaluate(testY[col], Y_hat[col])

# get rows with game id per action
A = []
for game_id in tqdm.tqdm(games.game_id, "Loading game ids"):
    Ai = pd.read_hdf(spadl_h5, f"actions/game_{game_id}")
    A.append(Ai[["game_id"]])
A = pd.concat(A)
A = A.reset_index(drop=True)

# concatenate action game id rows with predictions and save per game
grouped_predictions = pd.concat([A, Y_hat], axis=1).groupby("game_id")
for k, df in tqdm.tqdm(grouped_predictions, desc="Saving predictions per game"):
    df = df.reset_index(drop=True)
    df[Y_hat.columns].to_hdf(predictions_h5, f"game_{int(k)}")
