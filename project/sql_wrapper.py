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

    

   