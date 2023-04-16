from mplsoccer import statsbomb

## Some key statistics we are interested in measuring
BALL_RECOVERY = 2
DUEL = 4
BLOCK = 6
CLEARANCE = 9
INTERCEPTION = 10
DRIBBLE = 14
SHOT = 16
PRESSURE = 17
FOUL_WON = 21
FOUL_COMMITTED = 22
SHIELD = 28
PASS = 30
FIFTYFIFTY = 33
MISCONTROL = 38
DRIBBLED_PAST = 39
CARRY = 43


directory_path = '/home/casey/Projects/SoccerAnalytics/project/data/Statsbomb/data'
competitions_file= directory_path + '/competitions.json'
data = statsbomb.Sblocal(dataframe=True)

data.competition(competitions_file)

events, related, freeze, tactics = data.event(directory_path + '/events/3788741.json')

print(events.loc[events['type_id'] == SHOT])