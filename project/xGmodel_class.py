import statsmodels.iolib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

class xGmodel:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.model = None
        self.model_variables = ['Angle', 'Distance', 'X', 'C', 'X2', 'C2', 'AX']
        self.load_model()

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = statsmodels.iolib.smpickle.load_pickle(f)

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

    def calc_xG(self, shot_data: pd.DataFrame) -> float:
        """
        Calculate the expected goal (xG) value for a given shot data using the loaded stats model.

        Args:
            shot_data (pd.DataFrame): A DataFrame containing the shot data with columns matching the model variables.

        Returns:
            float: The calculated expected goal (xG) value.
        """
        if self.model is not None:
            bsum = self.model.params[0]
            for i, v in enumerate(self.model_variables):
                bsum = bsum + self.model.params[i + 1] * shot_data[v]
            xG = 1 / (1 + np.exp(bsum))
            return xG
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
        # Assumes that the df is properly formatted in order to calculate xG
        df.assign(xG = self.calc_xG(df))
        return df


    @staticmethod
    def shot_vars_from_x_and_y(x_coordinate: float, y_coordinate: float):
        shot_data = []
        angle = np.arctan(7.32 * x_coordinate / (x_coordinate**2 + abs(y_coordinate - 68/2)**2 - (7.32/2)**2))
        if angle < 0:
            angle = np.pi + angle
        shot_data.append(angle)
        shot_data.append(np.sqrt(x_coordinate**2 + abs(y_coordinate - 68/2)**2))
        shot_data.append(x_coordinate**2 + abs(y_coordinate - 68/2)**2)
        shot_data.append(x_coordinate)
        shot_data.append(x_coordinate * angle)
        shot_data.append(x_coordinate**2)
        shot_data.append(abs(y_coordinate - 68/2))
        shot_data.append((y_coordinate - 68/2)**2)
        return shot_data