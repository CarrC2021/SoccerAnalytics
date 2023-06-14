import json
import os
import numpy as np
import pandas as pd
import sql_wrapper


class WyscoutWrapper:
    def __init__(self) -> None:
        self.eventspath = (
            f"{os.path.dirname(os.path.realpath(__file__))}/data/Wyscout/events/"
        )
        self.id_dict = {
            "Duel": 1,
            "Air duel": 10,
            "Ground attacking duel": 11,
            "Ground defending duel": 12,
            "Ground loose ball duel": 13,
            "Foul": 2,
            "Hand foul": 21,
            "Late card foul": 22,
            "Out of game foul": 23,
            "Protest": 24,
            "Simulation": 25,
            "Time lost foul": 26,
            "Violent Foul": 27,
            "Free Kick": 3,
            "Corner": 30,
            "Free kick cross": 31,
            "Free kick shot": 32,
            "Goal kick": 33,
            "Penalty": 34,
            "Throw in": 35,
            "Goalkeeper leaving line": 4,
            "Ball out of the field": 50,
            "Whistle": 51,
            "Offside": 6,
            "Others on the ball": 7,
            "Acceleration": 70,
            "Clearance": 71,
            "Touch": 72,
            "Pass": 8,
            "Cross": 80,
            "Hand pass": 81,
            "Head pass": 82,
            "High pass": 83,
            "Launch": 84,
            "Simple pass": 85,
            "Smart pass": 86,
            "Reflexes": 9,
            "Save attempt": 90,
            "Shot": 10,
        }
        self.tag_dict = data = {
            401: "Left foot",
            402: "Right foot",
            403: "Head/body",
            1201: "Goal low center",
            1202: "Goal low right",
            1203: "Goal center",
            1204: "Goal center left",
            1205: "Goal low left",
            1206: "Goal center right",
            1207: "Goal high center",
            1208: "Goal high left",
            1209: "Goal high right",
            1210: "Out low right",
            1211: "Out center left",
            1212: "Out low left",
            1213: "Out center right",
            1214: "Out high center",
            1215: "Out high left",
            1216: "Out high right",
            1217: "Post low right",
            1218: "Post center left",
            1219: "Post low left",
            1220: "Post center right",
            1221: "Post high center",
            1222: "Post high left",
            1223: "Post high right",
        }

    def load_all(self) -> pd.DataFrame:
        all_data = pd.DataFrame()
        for file in os.listdir(self.eventspath):
            with open(f"{self.eventspath}/{file}") as f:
                data = json.load(f)
            data = pd.DataFrame(data)
            all_data = pd.concat([all_data, pd.DataFrame(data)])
        return all_data

    def load_all_events(self, event_name: str) -> pd.DataFrame:
        all_data = pd.DataFrame()
        for file in os.listdir(self.eventspath):
            with open(f"{self.eventspath}/{file}") as f:
                data = json.load(f)
            data = pd.DataFrame(data)
            data = data.loc[data["subEventName"] == f"{event_name}"]
            all_data = pd.concat([all_data, pd.DataFrame(data)])
        return all_data

    def load_all_events(self, event_names: list) -> pd.DataFrame:
        all_data = pd.DataFrame()
        for file in os.listdir(self.eventspath):
            with open(f"{self.eventspath}/{file}") as f:
                data = json.load(f)
            data = pd.DataFrame(data)
            data = data.loc[data["subEventName"].isin(event_names)]
            all_data = pd.concat([all_data, pd.DataFrame(data)])
        return all_data

    def load_all_events_from_competition(
        self, event_name: str, competition_name: str
    ) -> pd.DataFrame:
        with open(f"{self.eventspath}/{competition_name}") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df.loc[df["subEventName"] == f"{event_name}"]
        return df

    def load_player_from_competition(
        self, competition_name: str, player_name: str
    ) -> pd.DataFrame:
        with open(f"{self.eventspath}/{competition_name}") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Need to implement this function still
        df = df.loc[df["subEventName"] == f"{player_name}"]
        return df

    def denormalize_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        # exclude places where positions is not a list of two tuples
        data = data.loc[
            data.positions.apply(lambda cell: isinstance(cell, list) and len(cell) == 2)
        ]
        data["X"] = data.positions.apply(lambda cell: cell[0]["x"] * 105 / 100)
        data["Y"] = data.positions.apply(lambda cell: (100 - cell[0]["y"]) * 68 / 100)
        data["end_x"] = data.positions.apply(lambda cell: (cell[1]["x"]) * 105 / 100)
        data["end_y"] = data.positions.apply(
            lambda cell: (100 - cell[1]["y"]) * 68 / 100
        )
        return data

    def fix_shot_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        data["X"] = data.positions.apply(lambda cell: (100 - cell[0]["x"]) * 105 / 100)
        data["Y"] = data.positions.apply(lambda cell: cell[0]["y"] * 68 / 100)
        data["end_x"] = data.positions.apply(lambda cell: (cell[1]["x"]) * 105 / 100)
        data["end_y"] = data.positions.apply(
            lambda cell: (100 - cell[1]["y"]) * 68 / 100
        )
        return data

    def assign_body_part(self, data: pd.DataFrame) -> pd.DataFrame:
        data["isLeftFoot"] = data.tags.apply(lambda row: 1 if {"id": 401} in row else 0)
        data["isRightFoot"] = data.tags.apply(
            lambda row: 1 if {"id": 402} in row else 0
        )
        data["isHeadOrBody"] = data.tags.apply(
            lambda row: 1 if {"id": 403} in row else 0
        )
        # remove rows with playerId == 0 or NaN
        data = data.loc[data["playerId"] != 0]
        return data

    def fixed_shot_data(self, shots: pd.DataFrame) -> pd.DataFrame:
        shots = self.fix_shot_coordinates(shots)
        shots["C"] = shots.positions.apply(
            lambda cell: abs(cell[0]["y"] - 50) * 68 / 100
        )
        shots["Distance"] = np.sqrt(shots["X"] ** 2 + shots["C"] ** 2)
        shots["Distance2"] = shots["Distance"] ** 2
        shots["Angle"] = np.where(
            np.arctan(
                7.32
                * shots["X"]
                / (shots["X"] ** 2 + shots["C"] ** 2 - (7.32 / 2) ** 2)
            )
            > 0,
            np.arctan(
                7.32
                * shots["X"]
                / (shots["X"] ** 2 + shots["C"] ** 2 - (7.32 / 2) ** 2)
            ),
            np.arctan(
                7.32
                * shots["X"]
                / (shots["X"] ** 2 + shots["C"] ** 2 - (7.32 / 2) ** 2)
            )
            + np.pi,
        )
        shots["Goal"] = shots.tags.apply(lambda x: 1 if {"id": 101} in x else 0).astype(
            object
        )
        shots["X2"] = shots["X"] ** 2
        shots["C2"] = shots["C"] ** 2
        shots["AX"] = shots["Angle"] * shots["X"]
        return shots

    def only_shot_data(self, shots: pd.DataFrame) -> pd.DataFrame:
        shots = self.fixed_shot_data(shots)
        return shots
