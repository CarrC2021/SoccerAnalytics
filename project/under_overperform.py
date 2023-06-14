import WyscoutWrapper
import xGmodel
import json
import sql_wrapper
import codecs

# for each player in the sql database with minutes_played > 400 calculate npxG - npGoals

sql = sql_wrapper.SQLWrapper()
ids = sql.get_player_ids(minutes_played=400)
final_list = []
for player in ids:
    player_shots = sql.get_shots_by_player(player)
    xG = player_shots["xG"].sum()  # sum of all xG values for shots
    goals = len(
        player_shots[player_shots["Goal"] == 1]
    )  # sum of all non-penalty xG values for shots
    final_list.append([player, xG - goals, xG, goals])
# print top 15 players with the highest npxG - npGoals
final_list.sort(key=lambda x: x[1], reverse=True)
## Biggest Underperformers
print("Biggest Underperformers")
for entry in final_list[:15]:
    player_name = sql.get_player_name(entry[0])
    decoded_name = codecs.decode(player_name, "unicode_escape")
    print(f"{player_name}: {entry[1]}")

print("\n\n\n")
print("Biggest Overperformers")
## Biggest Overperformers
for entry in final_list[-15:]:
    player_name = sql.get_player_name(entry[0])
    decoded_name = codecs.decode(player_name, "unicode_escape")
    print(f"{player_name}: {-entry[1]}")
