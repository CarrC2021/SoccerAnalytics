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
warnings.filterwarnings('ignore')

DUEL = 1
GOAL = 101
ASSIST = 301
KEYPASS = 302
LEFTFOOT = 401
RIGHTFOOT = 402
HEADER = 403

CWD = os.path.dirname(os.path.realpath(__file__))

sql = sql_wrapper.SQLWrapper()
strong_foot_dict = sql.get_strong_foot()

wyscout = WyscoutWrapper.WyscoutWrapper()
shots = wyscout.load_all_events(['Shot'])

# remove the rows where the shot was not a header 
shots = wyscout.fixed_shot_data(shots)
# locate where the shot was a header by seeing where header = {'id': 403} in tags
shots['header'] = shots.apply(lambda x: 1 if {'id': 403} in x.tags else 0, axis=1)
shots = shots.loc[shots['header'] == 0]

shots = shots[['Goal', 'X', 'Y', 'C', 'Distance', 'Angle', 'X2', 'C2', 'AX']]

# list the model variables you want here
model_variables = ['Angle','Distance','X','C', "X2", "C2", "AX"]
model = ' + '.join(model_variables)

#fit the model
test_model = smf.glm(formula="Goal ~ " + model, data=shots,
                           family=sm.families.Binomial()).fit()
#print summary
print(test_model.summary())
# save the summary to a txt file
with open(f'{CWD}/models/notheaders.txt', 'w') as fh:
    fh.write(test_model.summary().as_text())
b=test_model.params

test_model.save(f'{CWD}/models/notheaders.pickle')

# load the model using the xGmodel class
# model = xGmodel.xGmodel(model_path='models/notheaders.pickle')
