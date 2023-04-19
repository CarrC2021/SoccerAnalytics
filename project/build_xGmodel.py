import json
import numpy as np
from joblib import dump, load
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import os
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

all_data = pd.DataFrame()
for file in os.listdir(f'{CWD}/data/Wyscout/events'):
    path = f'{CWD}/data/Wyscout/events/{file}'
    with open(path) as f:
        data = json.load(f)
    data = pd.DataFrame(data)
    data = data.loc[data['subEventName'] == 'Shot']
    all_data = pd.concat([all_data, pd.DataFrame(data)])

shots = all_data.loc[all_data['subEventName'] == 'Shot']
dictionary = {'id': HEADER}
shots = shots[shots['tags'].apply(lambda x: dictionary in x)]
print(shots)
shots["X"] = shots.positions.apply(lambda cell: (100 - cell[0]['x']) * 105/100)
shots["Y"] = shots.positions.apply(lambda cell: cell[0]['y'] * 68/100)
shots["C"] = shots.positions.apply(lambda cell: abs(cell[0]['y'] - 50) * 68/100)
shots["Distance"] = np.sqrt(shots["X"]**2 + shots["C"]**2)
shots["Angle"] = np.where(np.arctan(7.32 * shots["X"] / (shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * shots["X"] /(shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)), np.arctan(7.32 * shots["X"] /(shots["X"]**2 + shots["C"]**2 - (7.32/2)**2)) + np.pi)
shots["Goal"] = shots.tags.apply(lambda x: 1 if {'id':GOAL} in x else 0).astype(object)
shots["X2"] = shots['X']**2
shots["C2"] = shots['C']**2
shots["AX"]  = shots['Angle']*shots['X']

shots = shots[['Goal', 'X', 'Y', 'C', 'Distance', 'Angle', 'X2', 'C2', 'AX']]
train, test = train_test_split(shots, test_size=.2)

# list the model variables you want here
model_variables = ['Angle','Distance','X','C', "X2", "C2", "AX"]
model = ' + '.join(model_variables)

#fit the model
test_model = smf.glm(formula="Goal ~ " + model, data=shots,
                           family=sm.families.Binomial()).fit()
#print summary
print(test_model.summary())
b=test_model.params

test_model.save(f'{CWD}/models/headers.pickle')