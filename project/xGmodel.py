import statsmodels.iolib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import pickle

class xGmodel:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.model_variables = ['Angle', 'Distance', 'X', 'C', 'X2', 'C2', 'AX']
        self.load_model()

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_summary(self):
        if self.model is not None:
            return self.model.summary()
        else:
            return None

    def get_model_params(self):
        if self.model is not None:
            return self.model.params
        else:
            return None
        
    def calc_xG(self, x: float, y: float) -> float:
        """
        Calculate the expected goal (xG) value for a given shot coordinates using the loaded stats model.

        Args:
            x (float): The x-coordinate of the shot.
            y (float): The y-coordinate of the shot.

        Returns:
            float: The calculated expected goal (xG) value.
        """
        if self.model is not None:
            sh = self.shot_vars_from_x_and_y(x, y)
            bsum = self.model.params[0]
            for i, v in enumerate(self.model_variables):
                bsum = bsum + self.model.params[i + 1] * sh[v]
            xG = 1 / (1 + np.exp(bsum))
            return xG
        else:
            return None
    
    def assign_xG(self, df: pd.DataFrame) -> pd.DataFrame:
        df['xG'] = df.apply(lambda x: self.calc_xG(x.X, x.Y), axis=1)
        return df

    def assign_xA(self, df: pd.DataFrame) -> pd.DataFrame:
        df['xA'] = df.apply(lambda row: self.calc_xG(105 - row.end_x, row.end_y), axis=1)
        return df

    @staticmethod
    def shot_vars_from_x_and_y(x_coordinate: float, y_coordinate: float) -> pd.DataFrame:
        shot_data=dict()
        a = np.arctan(7.32 *x_coordinate /(x_coordinate**2 + abs(y_coordinate-68/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        shot_data['Angle'] = a
        shot_data['Distance'] = np.sqrt(x_coordinate**2 + abs(y_coordinate-68/2)**2)
        shot_data['D2'] = x_coordinate**2 + abs(y_coordinate-68/2)**2
        shot_data['X'] = x_coordinate
        shot_data['AX'] = x_coordinate*a
        shot_data['X2'] = x_coordinate**2
        shot_data['C'] = abs(y_coordinate-68/2)
        shot_data['C2'] = (y_coordinate-68/2)**2
        return shot_data