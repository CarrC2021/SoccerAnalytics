import sql_wrapper
import RadarWrapper
import json
import pandas as pd
import WyscoutWrapper
import scipy
import math

# Path: project/calc_radar_stats.py

# get all passes from the database that match a playerId
# get all shots from the database that match a playerId

# convert the names of the cols data to more readable names for the radar chart
col_dict = {
    "npxG": "Non Penalty xG",
    "npGoals": "Non Penalty Goals",
    "assists": "Assists",
    "final_third_passes_received": "Final Third Passes Received",
    "passes_to_final_third": "Passes to Final Third",
    "threatening_passes": "Threatening Passes",
    "progressive_passes": "Progressive Passes",
    "pass_xT": "Pass xT",
    "long_pass_xT": "Long Pass xT",
    "cross_xT": "Cross xT",
    "dribble_xT": "Dribble xT",
    "ground_attacking_duels_won": "Ground Attacking Duels Won",
    "aerial_duels_won": "Aerial Duels Won",
    "ground_defending_duels_won": "Ground Defending Duels Won",
    "loose_ball_duels_won": "Loose Ball Duels Won",
    "long_pass_attempts": "Long Pass Attempts",
    "long_pass_accuracy": "Long Pass Accuracy",
    "cross_attempts": "Cross Attempts",
    "pass_attempts": "Pass Attempts",
    "aerial_win_rate": "Aerial Win Rate",
    "ground_defensive_win_rate": "Ground Defensive Win Rate",
    "ground_attacking_win_rate": "Ground Attacking Win Rate",
    "loose_ball_win_rate": "Loose Ball Win Rate",
    "pass_accuracy": "Pass Accuracy",
    "cross_accuracy": "Cross Accuracy",
    "dribble_attempts": "Dribble Attempts",
    "dribble_accuracy": "Dribble Accuracy",
}


sql = sql_wrapper.SQLWrapper()
# cols = ['npxG', 'npGoals', 'assists', 'final_third_passes_received', 'passes_to_final_third',
#  'threatening_passes', 'progressive_passes', 'pass_xT', 'long_pass_xT', 'cross_xT', 'dribble_xT', 'aerial_duels_won', 'ground_defending_duels_won', 'loose_ball_duels_won']


def calc_radar_stats(player_id: int, sql: sql_wrapper.SQLWrapper, vals_only=False):
    sql.calc_efficiency_rates(player_id)
    player_name = sql.get_player_name(player_id)
    player_passes = sql.get_passes_by_player(player_id)
    assists = len(player_passes[player_passes["assist"] == 1])
    passes_to_final_third = len(
        player_passes[player_passes["X"] > 2 / 3 * 105]
    )  # all passes that go into the final third
    threatening_passes = len(
        player_passes[player_passes["xT"] > 0.05]
    )  # all passes that have an xT value greater than 0.05
    progressive_passes = len(
        player_passes[player_passes["end_x"] - player_passes["X"] > 15]
    )  # all passes that progressed the ball 15 meters
    crosses = player_passes[player_passes["subEventId"] == 80]
    # drop crosses from player_passes
    player_passes = player_passes[player_passes["subEventId"] != 80]
    long_passes = player_passes[player_passes["subEventId"] == (83 or 84)]
    player_passes = player_passes[
        (player_passes["subEventId"] != 83) & (player_passes["subEventId"] != 84)
    ]
    player_passes_received = sql.get_passes_received(player_id)
    player_shots = sql.get_shots_by_player(player_id)
    player_duels = sql.get_duels_by_player(player_id)
    player_duels = player_duels[player_duels["won"] == 1]
    dribbles = player_duels[player_duels["is_take_on"] == 1]
    # only take the dribbles that move the ball a total of 10 meters or more and result in + .02 xT
    dribbles = dribbles[
        (dribbles["end_x"] - dribbles["X"] > 10) & dribbles["xT"] > 0.02
    ]
    minutes_played = sql.get_minutes_played(player_id)
    possession = sql.get_possession(player_id)
    xG = player_shots["xG"].sum()  # sum of all xG values for shots
    goals = len(
        player_shots[player_shots["Goal"] == 1]
    )  # sum of all non-penalty xG values for shots
    threatening_passes_received = len(
        player_passes_received[player_passes_received["X"] > 2 / 3 * 105]
    )  # all passes received that have an xG value greater than 0.05
    pass_xT = player_passes[
        (player_passes["successful"] == 1) & (player_passes["xT"] > 0)
    ][
        "xT"
    ].sum()  # sum of all xT values for
    cross_xT = crosses[(crosses["successful"] == 1) & (crosses["xT"] > 0)][
        "xT"
    ].sum()  # add the xT values for crosses
    long_pass_xT = long_passes[
        (long_passes["successful"] == 1) & (long_passes["xT"] > 0)
    ][
        "xT"
    ].sum()  # add the xT values for long passes
    dribble_xT = dribbles["xT"].sum()  # add the xT values for dribbles
    attacking_duels = len(
        player_duels[
            (player_duels["ground_attacking_duel"] == 1)
            & (player_duels["is_take_on"] == 0)
        ]
    )  # all ground duels won
    defending_duels = len(player_duels[player_duels["ground_defending_duel"] == 1])
    loose_ball_duels = len(player_duels[player_duels["ground_loose_ball_duel"] == 1])
    aerial_duels = len(
        player_duels[player_duels["aerial_duel"] == 1]
    )  # all aerial duels won
    vals = [
        xG,
        goals,
        assists,
        threatening_passes_received,
        passes_to_final_third,
        threatening_passes,
        progressive_passes,
        pass_xT,
        long_pass_xT,
        cross_xT,
        dribble_xT,
        aerial_duels,
        defending_duels,
        loose_ball_duels,
    ]
    # print(vals)
    # divide every value by the minutes played to get a per 90 value
    for i in range(len(cols)):
        vals[i] = (vals[i] / possession) * 90 / minutes_played
        sql.update_column("players", cols[i], vals[i], player_id, False)
    sql.conn.commit()
    if vals_only:
        return vals
    return dict(zip(cols, vals))


def calc_percentiles(player_id: int, cols: list):
    percentiles = []
    role = sql.get_player_column(player_id, "role")[0]
    for stat in cols:
        vals = sql.get_column("players", stat, role, 400)
        percentiles.append(
            scipy.stats.percentileofscore(
                vals, sql.get_player_column(player_id, stat), "weak"
            )[0]
        )
    return percentiles


def calc_lows_and_highs(player_id: int, cols: list):
    lows = []
    highs = []
    role = sql.get_player_column(player_id, "role")[0]
    for stat in cols:
        vals = sql.get_column("players", stat, role, 400)
        # replace NAN values with 0
        vals = [0 if math.isnan(x) else x for x in vals]
        lows.append(min(vals))
        highs.append(max(vals))
    return lows, highs


def compare_players(
    player_id1: int,
    player_id2: int,
    cols: list,
    lower_is_better=None,
    percentiles=False,
):
    player1 = sql.get_player_name(player_id1)
    player2 = sql.get_player_name(player_id2)
    if percentiles:
        player1_stats = calc_percentiles(player_id1, cols)
        player2_stats = calc_percentiles(player_id2, cols)
        lows = [0] * len(cols)
        highs = [100] * len(cols)
    else:
        player1_stats = sql.get_radar_stats(player_id1, cols, 400)
        player2_stats = sql.get_radar_stats(player_id2, cols, 400)
        lows, highs = calc_lows_and_highs(player_id1, cols)
    data = [player1_stats, player2_stats]
    col_names = [col_dict[i] for i in cols]
    radar = RadarWrapper.RadarWrapper(
        data=data,
        column_names=col_names,
        lows=lows,
        highs=highs,
        plot_percentiles=percentiles,
        lower_is_better=lower_is_better,
        is_comparison=True,
    )
    radar.plot_comparison_radar(player1, player2)


def plot_player_radar(
    player_id: int, team_name: str, cols: list, lower_is_better=None, percentiles=False
):
    player = sql.get_player_name(player_id)
    if percentiles:
        player_stats = calc_percentiles(player_id, cols)
        lows = [0] * len(cols)
        highs = [100] * len(cols)
    else:
        player_stats = sql.get_radar_stats(player_id, cols, 400)
        # replace NAN with 0
        lows, highs = calc_lows_and_highs(player_id, cols)
    col_names = [col_dict[i] for i in cols]
    radar = RadarWrapper.RadarWrapper(
        data=player_stats,
        column_names=col_names,
        lower_is_better=lower_is_better,
        lows=lows,
        highs=highs,
        plot_percentiles=percentiles,
    )
    radar.plot_radar(player, team_name)


# get all players from the database with minutes_played > 400 as a pandas dataframe
# players = sql.get_player_ids(400)
# all_stats = {}


# for player in players:
#     calc_radar_stats(int(player), sql, vals_only=True)

# removing xA for now
cols = [
    "npxG",
    "npGoals",
    "final_third_passes_received",
    "passes_to_final_third",
    "threatening_passes",
    "progressive_passes",
    "pass_xT",
    "cross_xT",
    "long_pass_xT",
    "dribble_xT",
    "aerial_duels_won",
    "ground_defending_duels_won",
    "loose_ball_duels_won",
    "long_pass_accuracy",
    "pass_accuracy",
    "long_pass_attempts",
    "cross_attempts",
    "pass_attempts",
]


# plot_player_radar(65596, 'Leipzig', cols, percentiles=True)
# compare_players(3359, 3322, cols)

plot_player_radar(21315, "Napoli", cols, percentiles=True)

compare_players(21315, 263802, cols, percentiles=True)

# open the cluster_dict.txt file and load it into a dictionary
# with open('cluster_dict_forwards.txt', 'r') as f:
#     cluster_dict = json.load(f)

# # for each cluster in the dictionary, plot the radar for each player in the cluster
# for cluster in cluster_dict:
#     players = cluster_dict[cluster]
#     counter = 0
#     prev_player = None
#     for player in players:
#         if counter > 4:
#             break
#         if prev_player is not None:
#             compare_players(prev_player, player, cols, percentiles=True)
#         print(f'Cluster {cluster} - Player {sql.get_player_name(player)}')
#         prev_player = player
#         counter += 1
