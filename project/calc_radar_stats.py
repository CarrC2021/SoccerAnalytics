import sql_wrapper
import radar
import json
import pandas as pd
import WyscoutWrapper

# Path: project/calc_radar_stats.py

# get all passes from the database that match a playerId
# get all shots from the database that match a playerId

sql = sql_wrapper.SQLWrapper()
player_id = 3359

player_name = sql.get_player_name(player_id=3359)
player_passes = sql.get_passes_by_player(player_id=3359)
player_passes_received = sql.get_passes_received(player_id=3359)
player_shots = sql.get_shots_by_player(player_id=3359)

xG = player_shots['xG'].sum()   # sum of all xG values for shots
goals = len(player_shots[player_shots['Goal'] == 1])  # sum of all non-penalty xG values for shots
threatening_passes_received = len(player_passes_received[player_passes_received['X'] > 2 / 3 * 105])  # all passes received that have an xG value greater than 0.05
passes_to_final_third = len(player_passes[player_passes['X'] > 2 / 3 * 105])   # all passes that go into the final third
xT = player_passes[player_passes['successful'] == 1]['xT'].sum()   # sum of all xT values for passes
print(xG, goals, threatening_passes_received, passes_to_final_third, xT)
cols = ['xG', 'Goals', 'Final 3rd Passes Received', 'Passes to Final Third', 'xT']
low = [0, 0, 0, 0, 0]
high = [xG+1, goals + 2, threatening_passes_received + 1, passes_to_final_third + 12, xT + 4]
radar = radar.RadarWrapper([xG, goals, threatening_passes_received, passes_to_final_third, xT], cols, low, high, None)

radar.plot_radar('Messi', 'FC Barcelona')

wyscout = WyscoutWrapper.WyscoutWrapper()
