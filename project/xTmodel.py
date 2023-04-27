import WyscoutWrapper
import scipy.stats
import numpy as np
import pandas as pd
import mplsoccer
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

wyscout = WyscoutWrapper.WyscoutWrapper()

data = wyscout.load_all()

next_event = data.shift(-1, fill_value=0)
data["nextEvent"] = next_event["subEventName"]

data["kickedOut"] = data.apply(lambda x: 1 if x.nextEvent == "Ball out of the field" else 0, axis = 1)
#get move_df
move_df = data.loc[data['subEventName'].isin(['Simple pass', 'High pass', 'Head pass', 'Smart pass', 'Cross'])]
#filtering out of the field
delete_passes = move_df.loc[move_df["kickedOut"] == 1]
move_df = move_df.drop(delete_passes.index)

# fix coordinates for move_df
move_df = wyscout.fix_coordinates(move_df)

## exclude those rows for which the ball is out of bounds
move_df = move_df.loc[(((move_df["end_x"] != 0) & (move_df["end_y"] != 68)) & ((move_df["end_x"] != 105) & (move_df["end_y"] != 0)))]

pitch = mplsoccer.Pitch(line_color='black', pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)

move = pitch.bin_statistic(move_df.X, move_df.Y, statistic='count', bins=(16, 12), normalize=False)
move_count = move['statistic']

#get shot df
shot_df = data.loc[data['subEventName'] == 'Shot']

# fix coordinates for shot_df
shot_df = wyscout.fix_coordinates(shot_df)

#create 2D histogram of these
pitch = mplsoccer.Pitch(line_color='black', pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
shot = pitch.bin_statistic(shot_df.X, shot_df.Y, statistic='count', bins=(16, 12), normalize=False)
shot_count = shot['statistic']

#get goal df
pitch = mplsoccer.Pitch(line_color='black', pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
goal_df  = shot_df.loc[shot_df.apply(lambda x: {'id':101} in x.tags, axis = 1)]
goal = pitch.bin_statistic(goal_df.X, goal_df.Y, statistic='count', bins=(16, 12), normalize=False)

#calculate probabilities
goal_count = goal["statistic"]
move_probability = move_count/(move_count+shot_count)
shot_probability = shot_count/(move_count+shot_count)
goal_probability = goal_count/(shot_count)
goal_probability[np.isnan(goal_probability)] = 0

# Locating which bin the event starts in
move_df["start_sector"] = move_df.apply(lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.X), np.ravel(row.X),
                                                               values = "None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)
print(move_df['start_sector'])
# Locating which bin the event ends in
move_df["end_sector"] = move_df.apply(lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.end_x), np.ravel(row.end_y),
                                                               values = "None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)
print(move_df['end_sector'])

#df with summed events from each index
df_count_starts = move_df.groupby(["start_sector"],)["eventId"].count().reset_index()
df_count_starts.rename(columns = {'eventId':'count_starts'}, inplace=True)
print(len(df_count_starts))
print(df_count_starts)

transition_matrices = []
for i, row in df_count_starts.iterrows():
    start_sector = row['start_sector']
    count_starts = row['count_starts']
    #get all events that started in this sector
    this_sector = move_df.loc[move_df["start_sector"] == start_sector]
    df_count_ends = this_sector.groupby(["end_sector"])["eventId"].count().reset_index()
    df_count_ends.rename(columns = {'eventId':'count_ends'}, inplace=True)
    T_matrix = np.zeros((12, 16))
    for j, row2 in df_count_ends.iterrows():
        end_sector = row2["end_sector"]
        value = row2["count_ends"]
        T_matrix[end_sector[1] - 1][end_sector[0] - 1] = value
    T_matrix = T_matrix / count_starts
    transition_matrices.append(T_matrix)

transition_matrices_array = np.array(transition_matrices)
np.save('transition_matrix.npy', transition_matrices_array)
print(transition_matrices_array.shape)
xT = np.zeros((12, 16))
for i in range(5):
    shoot_expected_payoff = goal_probability*shot_probability
    print(shoot_expected_payoff.shape)
    print(move_probability.shape)
    print(np.sum(transition_matrices_array*xT, axis = 2).shape)
    print(np.sum(np.sum(transition_matrices_array*xT, axis = 2), axis = 1).shape)
    move_expected_payoff = move_probability*(np.sum(np.sum(transition_matrices_array*xT, axis = 2), axis = 1).reshape(16,12).T)
    xT = shoot_expected_payoff + move_expected_payoff

    #let's plot it!
    fig, ax = mplsoccer.pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.01, title_space=0, endnote_space=0)
    goal["statistic"] = xT
    pcm  = mplsoccer.pitch.heatmap(goal, cmap='Oranges', edgecolor='grey', ax=ax['pitch'])
    labels = mplsoccer.pitch.label_heatmap(goal, color='blue', fontsize=9,
                             ax=ax['pitch'], ha='center', va='center', str_format="{0:,.2f}", zorder = 3)
    #legend to our plot
    ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
    cbar = plt.colorbar(pcm, cax=ax_cbar)
    fig.suptitle(f'Expected Threat matrix after {str(i+1)} moves', fontsize = 30)
    plt.savefig(f'xT_{str(i+1)}.png', dpi = 300)
    np.save(f'xT_{str(i+1)}.npy', xT)
    plt.show()

#only successful
successful_moves = move_df.loc[move_df.apply(lambda x:{'id':1801} in x.tags, axis = 1)]
#calculatexT
successful_moves["xT_added"] = successful_moves.apply(lambda row: xT[row.end_sector[1] - 1][row.end_sector[0] - 1]
                                                      - xT[row.start_sector[1] - 1][row.start_sector[0] - 1], axis = 1)
#only progressive
value_adding_actions = successful_moves.loc[successful_moves["xT_added"] > 0]
