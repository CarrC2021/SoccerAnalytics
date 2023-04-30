import sqlite3
import pandas as pd
import numpy as np
import os


## SQL Wrapper class
class SQLWrapper:
    ## init function that connects to the database
    def __init__(self) -> None:
        self.db_path = f'{os.path.dirname(os.path.realpath(__file__))}/data/wyscout.db'
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

    # function that searches the passes table for all events that occur within the given x and y coordinates bound
    def get_passes(self, x1: float, x2: float, y1: float, y2: float) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM test_passes WHERE x BETWEEN {x1} AND {x2} AND y BETWEEN {y1} AND {y2}", self.conn)
        return df

    # an overload of get_passes that uses integer coordinates instead of float coordinates
    # where the integer coordinates are the sector coordinates so they are multiplied by 105 for x and divided by 
    # 16 and for y they are multiplied by 68 and divided by 12
    def get_passes(self, x1: int, x2: int, y1: int, y2: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM test_passes WHERE x BETWEEN {x1 * 105 / 16} AND {x2 * 105 / 16} AND y BETWEEN {y1 * 68 / 12} AND {y2 * 68 / 12}", self.conn)
        return df

    # an overload of get_passes that uses an integer array of coordinates instead of float coordinates
    # where the integer coordinates are the sector coordinates so they are multiplied by 105 for x and divided by 
    # 16 and for y they are multiplied by 68 and divided by 12
    def get_passes(self, sector: list) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM test_passes WHERE x BETWEEN {sector[0] * 105 / 16} AND {sector[1] * 105 / 16} AND y BETWEEN {sector[2] * 68 / 12} AND {sector[3] * 68 / 12}", self.conn)
        return df
        
    # an overload of get_passes that uses an integer array of coordinates instead of float coordinates
    # where the integer coordinates are the sector coordinates so they are multiplied by 105 for x and divided by 
    # 16 and for y they are multiplied by 68 and divided by 12 and also checks for the player_id
    def get_passes(self, sector: list, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM test_passes WHERE x BETWEEN {sector[0] * 105 / 16} AND {sector[1] * 105 / 16} AND y BETWEEN {sector[2] * 68 / 12} AND {sector[3] * 68 / 12} AND playerId = {player_id}", self.conn)
        return df

    def get_passes_received(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM test_passes WHERE nextPlayerId = {player_id}", self.conn)
        return df
    
    def get_passes_by_player(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM test_passes WHERE playerId = {player_id}", self.conn)
        return df

    # function that searches the shots table for all events that occur by the given player using the playerId
    def get_shots_by_player(self, player_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM shots WHERE playerId = {player_id}", self.conn)
        return df

    # function that returns a player's name given their playerId by searching the players table
    def get_player_name(self, player_id: int) -> str:
        df = pd.read_sql_query(
            f"SELECT shortName FROM players WHERE wyId = {player_id}", self.conn)
        return df['shortName'][0]

    # function that loads the teams table into a dataframe
    def get_teams(self) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM teams", self.conn)
        return df

    # function that returns all shots that match the given matchId
    def get_shots_by_match(self, match_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM shots WHERE matchId = {match_id}", self.conn)
        return df

    # function that returns all shots that match the given matchId
    def get_passes_by_match(self, match_id: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM passes WHERE matchId = {match_id}", self.conn)
        return df

    # load the strong foot data from the players table as a dictionary with the 
    # key being the player's id and the value being the player's strong foot
    def get_strong_foot(self) -> dict:
        df = pd.read_sql_query(
            f"SELECT wyId, foot FROM players", self.conn)
        return dict(zip(df['wyId'], df['foot']))

    

   