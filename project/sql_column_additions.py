import WyscoutWrapper
import sql_wrapper
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xGmodel
import os

CWD = os.path.dirname(os.path.realpath(__file__))
wyscout = WyscoutWrapper.WyscoutWrapper()
db = sql_wrapper.SQLWrapper()

free_kick_and_penalty = wyscout.load_all_events(['Free kick shot', 'Penalty'])
print(free_kick_and_penalty)
## fix the shot data coordinates
free_kick_and_penalty = wyscout.fixed_shot_data(free_kick_and_penalty)
## find the tag where the shot is on target
free_kick_and_penalty['on_target'] = free_kick_and_penalty.apply(lambda x: 1 if {'id': 1801} in x.tags else 0, axis=1)
free_kick_and_penalty['goal'] = free_kick_and_penalty.apply(lambda x: 1 if {'id': 101} in x.tags else 0, axis=1)
## find the body part of the shot
free_kick_and_penalty['body_part'] = free_kick_and_penalty.apply(lambda x: 'left foot' if {'id': 402} in x.tags else 'right foot', axis=1)
# penalty = free_kick_and_penalty.loc[free_kick_and_penalty['subEventName'] == 'Penalty']

free_kick_shots = wyscout.fixed_shot_data(free_kick_and_penalty)

penalty = free_kick_shots.loc[free_kick_shots['subEventName'] == 'Penalty']
penalty_xG = len(penalty[penalty['goal'] == 1]) / len(penalty)
print(penalty_xG)

free_kick = free_kick_shots.loc[free_kick_shots['subEventName'] == 'Free kick shot']
model = xGmodel.xGmodel(model_path='models/freekicks.pickle')
free_kick = model.assign_xG(df=free_kick)
free_kick = free_kick[free_kick['playerId', 'X', 'Y', 'xG', 'body_part',
            'subEventId', 'teamId', 'matchId', 'eventSec', 'matchPeriod']]
## save the dataframes to the sql database shots table
conn = sql_wrapper.SQLWrapper()
free_kick.to_sql('shots', conn, if_exists='append', index=False)