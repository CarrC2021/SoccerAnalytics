import os
import warnings
import tqdm
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import socceraction.spadl as spadl
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab

# code from https://github.com/ML-KULeuven/socceraction/blob/master/public-notebooks/2-compute-features-and-labels.ipynb


datafolder = os.path.join(os.getcwd(), "VAEP_data")
spadl_h5 = os.path.join(datafolder, "spadl.h5")
features_h5 = os.path.join(datafolder, "features.h5")
labels_h5 = os.path.join(datafolder, "labels.h5")

games = pd.read_hdf(spadl_h5, "games")
print("nb of games:", len(games))

xfns = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.bodypart,
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
    fs.time,
    fs.time_delta
]

# for game in tqdm.tqdm(list(games.itertuples()), desc=f"Generating and storing features in {features_h5}"):
#     actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")
#     gamestates = fs.gamestates(spadl.add_names(actions), 3)
#     gamestates = fs.play_left_to_right(gamestates, game.home_team_id)
    
#     X = pd.concat([fn(gamestates) for fn in xfns], axis=1)
#     X.to_hdf(features_h5, f"game_{game.game_id}")

yfns = [lab.scores, lab.concedes, lab.goal_from_shot]

for game in tqdm.tqdm(list(games.itertuples()), desc=f"Computing and storing labels in {labels_h5}"):
    actions = pd.read_hdf(spadl_h5, f"actions/game_{game.game_id}")   
    Y = pd.concat([fn(spadl.add_names(actions)) for fn in yfns], axis=1)
    Y.to_hdf(labels_h5, f"game_{game.game_id}")

