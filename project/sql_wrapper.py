import sqlite3
import pandas as pd
import numpy as np
import os
import chardet


## SQL Wrapper class
class SQLWrapper:
    ## init function that connects to the database
    def __init__(self) -> None:
        self.db_path = f"{os.path.dirname(os.path.realpath(__file__))}/data/wyscout.db"
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

    # function that searches the passes table for all events that occur within the given x and y coordinates bound
    def get_passes(self, x1: float, x2: float, y1: float, y2: float) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE x BETWEEN {x1} AND {x2} AND y BETWEEN {y1} AND {y2}",
            self.conn,
        )
        return df

    # an overload of get_passes that uses integer coordinates instead of float coordinates
    # where the integer coordinates are the sector coordinates so they are multiplied by 105 for x and divided by
    # 16 and for y they are multiplied by 68 and divided by 12
    def get_passes(self, x1: int, x2: int, y1: int, y2: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE x BETWEEN {x1 * 105 / 16} AND {x2 * 105 / 16} AND y BETWEEN {y1 * 68 / 12} AND {y2 * 68 / 12}",
            self.conn,
        )
        return df

    # an overload of get_passes that uses an integer array of coordinates instead of float coordinates
    # where the integer coordinates are the sector coordinates so they are multiplied by 105 for x and divided by
    # 16 and for y they are multiplied by 68 and divided by 12
    def get_passes(self, sector: list) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE x BETWEEN {sector[0] * 105 / 16} AND {sector[1] * 105 / 16} AND y BETWEEN {sector[2] * 68 / 12} AND {sector[3] * 68 / 12}",
            self.conn,
        )
        return df

    # an overload of get_passes that uses an integer array of coordinates instead of float coordinates
    # where the integer coordinates are the sector coordinates so they are multiplied by 105 for x and divided by
    # 16 and for y they are multiplied by 68 and divided by 12 and also checks for the player_id
    def get_passes(self, sector: list, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE x BETWEEN {sector[0] * 105 / 16} AND {sector[1] * 105 / 16} AND y BETWEEN {sector[2] * 68 / 12} AND {sector[3] * 68 / 12} AND playerId = {player_id}",
            self.conn,
        )
        return df

    def get_passes_received(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE nextPlayerId = {player_id}", self.conn
        )
        return df

    def get_passes_by_player(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE playerId = {player_id}", self.conn
        )
        return df

    # function that searches the shots table for all events that occur by the given player using the playerId
    def get_shots_by_player(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM shots WHERE playerId = {player_id}", self.conn
        )
        return df

    # function that returns a player's name given their playerId by searching the players table
    def get_player_name(self, player_id: int) -> str:
        df = pd.read_sql_query(
            f"SELECT shortName FROM players WHERE wyId = {player_id}", self.conn
        )

        name = df["shortName"][0]

        decoded = bytes(name, "utf-8").decode("unicode_escape")

        return decoded

    # function that loads the teams table into a dataframe
    def get_teams(self) -> pd.DataFrame:
        df = pd.read_sql_query(f"SELECT * FROM teams", self.conn)
        return df

    # function that returns all shots that match the given matchId
    def get_shots_by_match(self, match_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM shots WHERE matchId = {match_id}", self.conn
        )
        return df

    # function that returns all shots that match the given matchId
    def get_passes_by_match(self, match_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE matchId = {match_id}", self.conn
        )
        return df

    # load the strong foot data from the players table as a dictionary with the
    # key being the player's id and the value being the player's strong foot
    def get_strong_foot(self) -> dict:
        df = pd.read_sql_query(f"SELECT wyId, foot FROM players ", self.conn)
        return dict(zip(df["wyId"], df["foot"]))

    # get the player's duels from the duels table
    def get_duels_by_player(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM duels WHERE playerId = {player_id}", self.conn
        )
        return df

    # get all player Ids from the players table
    def get_player_ids(self) -> list:
        df = pd.read_sql_query(f"SELECT wyId FROM players", self.conn)
        return df["wyId"].to_list()

    # get all player Ids where the minutes played is big enough from the players table
    def get_player_ids(self, minutes_played: int, position=None) -> list:
        if position:
            df = pd.read_sql_query(
                f"SELECT wyId FROM players WHERE minutes_played > {minutes_played} AND role = '{position}'",
                self.conn,
            )
            return df["wyId"].to_list()
        df = pd.read_sql_query(
            f"SELECT wyId FROM players WHERE minutes_played > {minutes_played}",
            self.conn,
        )
        return df["wyId"].to_list()

    # get minutes played from the players table by player id
    def get_minutes_played(self, player_id: int) -> int:
        df = pd.read_sql_query(
            f"SELECT minutes_played FROM players WHERE wyId = {player_id}", self.conn
        )
        return df["minutes_played"][0]

    # a function that returns a table in the database
    def get_table(self, table_name: str) -> list:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        print(df)
        return df

    # a function that returns a table in the database
    def get_team_data_from_table(self, table_name: str, team_id: str) -> list:
        df = pd.read_sql_query(
            f"SELECT * FROM {table_name} WHERE teamId = {team_id}", self.conn
        )
        print(df)
        return df

    # a function that returns a column from a given table in the database
    def get_column(
        self, table_name: str, column_name: str, role: str, minutes_played: int
    ) -> list:
        df = pd.read_sql_query(
            f"SELECT {column_name} FROM {table_name} WHERE role = '{role}' AND minutes_played > {minutes_played}",
            self.conn,
        )
        return df[column_name].to_list()

    # a function that returns a column from a given table in the database
    def get_player_column(self, player_id: int, column_name: str) -> list:
        df = pd.read_sql_query(
            f"SELECT {column_name} FROM players WHERE wyId = {player_id}", self.conn
        )
        return df[column_name].to_list()

    # write a function that adds a new column to a given table in the database
    def add_column(self, table_name: str, column_name: str, column_type: str) -> None:
        self.c.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )
        self.conn.commit()

    # write a function that updates a column entry in a given table in the database
    def update_column(
        self,
        table_name: str,
        column_name: str,
        column_value: str,
        player_id: int,
        commit=True,
    ) -> None:
        self.c.execute(
            f"UPDATE {table_name} SET {column_name} = {column_value} WHERE wyId = {player_id}"
        )
        if commit:
            self.conn.commit()

    # get minutes played from the players table by player id
    def get_possession(self, player_id: int) -> int:
        df = pd.read_sql_query(
            f"SELECT possession FROM players WHERE wyId = {player_id}", self.conn
        )
        return df["possession"][0]

    # return a list with all the player names and their respective npxG and npGoals
    def get_player_npxG(self) -> list:
        df = pd.read_sql_query(
            f"SELECT shortName, npxG, npGoals FROM players WHERE minutes_played > 400",
            self.conn,
        )
        return df.values.tolist()

    # return a list of the radar stats for a given player
    def get_radar_stats(self, player_id: int, cols: list, minutes_played: int) -> list:
        df = pd.read_sql_query(
            f"SELECT {', '.join(cols)} FROM players WHERE wyId = {player_id} AND minutes_played > {minutes_played}",
            self.conn,
        )
        return df.values.tolist()[0]

    def calc_efficiency_rates(self, player_id: int) -> list:
        df = pd.read_sql_query(
            f"SELECT * FROM duels WHERE playerId = {player_id}", self.conn
        )
        aerial_win_rate = (
            ground_defensive_win_rate
        ) = (
            ground_attacking_win_rate
        ) = loose_ball_win_rate = cross_accuracy = long_pass_accuracy = 0
        if len(df[df["aerial_duel"] == 1]) != 0:
            aerial_win_rate = len(
                df[(df["aerial_duel"] == 1) & (df["won"] == 1)]
            ) / len(df[df["aerial_duel"] == 1])
        if len(df[df["ground_defending_duel"] == 1]) != 0:
            ground_defensive_win_rate = len(
                df[(df["ground_defending_duel"] == 1) & (df["won"] == 1)]
            ) / len(df[df["ground_defending_duel"] == 1])
        if len(df[df["ground_attacking_duel"] == 1]) != 0:
            ground_attacking_win_rate = len(
                df[(df["ground_attacking_duel"] == 1) & (df["won"] == 1)]
            ) / len(df[df["ground_attacking_duel"] == 1])
        if len(df[df["ground_loose_ball_duel"] == 1]) != 0:
            loose_ball_win_rate = len(
                df[(df["ground_loose_ball_duel"] == 1) & (df["won"] == 1)]
            ) / len(df[df["ground_loose_ball_duel"] == 1])

        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE playerId = {player_id}", self.conn
        )
        # find passes that progressed the ball 25 meters in the attacking x direction
        long_passes = df[(df["end_x"] - df["X"] > 25)]
        if len(long_passes) != 0:
            long_pass_accuracy = len(long_passes[long_passes["successful"] == 1]) / len(
                long_passes
            )
        pass_accuracy = len(
            df[(df["successful"] == 1) & (df["subEventId"] != 80)]
        ) / len(df[df["subEventId"] != 80])
        if len(df[df["subEventId"] == 80]) != 0:
            cross_accuracy = len(
                df[(df["subEventId"] == 80) & (df["successful"] == 1)]
            ) / len(df[df["subEventId"] == 80])

        possession = self.get_player_column(player_id, "possession")[0]
        minutes_played = self.get_player_column(player_id, "minutes_played")[0]

        long_pass_attempts = len(long_passes) / possession * 90 / minutes_played
        pass_attempts = (
            len(df[df["subEventId"] != 80]) / possession * 90 / minutes_played
        )
        cross_attempts = (
            len(df[df["subEventId"] == 80]) / possession * 90 / minutes_played
        )

        # now save the values in the database for that player
        self.update_column("players", "aerial_win_rate", aerial_win_rate, player_id)
        self.update_column(
            "players", "ground_defensive_win_rate", ground_defensive_win_rate, player_id
        )
        self.update_column(
            "players", "ground_attacking_win_rate", ground_attacking_win_rate, player_id
        )
        self.update_column(
            "players", "loose_ball_win_rate", loose_ball_win_rate, player_id
        )
        self.update_column(
            "players", "long_pass_accuracy", long_pass_accuracy, player_id
        )
        self.update_column(
            "players", "long_pass_attempts", long_pass_attempts, player_id
        )
        self.update_column("players", "pass_accuracy", pass_accuracy, player_id)
        self.update_column("players", "pass_attempts", long_pass_attempts, player_id)
        self.update_column("players", "cross_accuracy", cross_accuracy, player_id)
        self.update_column("players", "cross_attempts", cross_attempts, player_id)
        # commit the changes
        self.conn.commit()

    # get a function that returns all shots and passes in a match
    def get_shots_and_passes(self, match_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE matchId = {match_id}", self.conn
        )
        df2 = pd.read_sql_query(
            f"SELECT * FROM shots WHERE matchId = {match_id}", self.conn
        )
        return pd.concat([df, df2])

    # get all duels for a player_id and calculate the duel win percentage
    def get_efficiency_rates(self, player_id: int) -> list:
        cols = [
            "aerial_win_rate",
            "ground_defensive_win_rate",
            "ground_offensive_win_rate",
            "loose_ball_win_rate",
            "long_pass_accuracy",
            "pass_accuracy",
            "cross_accuracy",
        ]
        return self.get_radar_stats(player_id, cols, 0)

    # get player role
    def get_player_role(self, player_id: int) -> str:
        df = pd.read_sql_query(
            f"SELECT role FROM players WHERE wyId = {player_id}", self.conn
        )
        return df["role"][0]
