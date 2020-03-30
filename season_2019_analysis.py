import pandas as pd
import numpy as np

#TODO: just load needed columns: game_date, player_name, game_year, bases, xbpp 
numeric_cols = ['game_year', 'bases', 'xBPP', "inning", "at_bat_number", "pitch_number", "release_speed", 'balls', 'strikes']
filter_cols = ['game_date', 'player_name'] + numeric_cols
game_year = 2019.0

#TODO: load csv
data = pd.read_csv("combined_xBPP.csv", header=0, usecols=filter_cols)
print("number of negative predicitons")
print(data[data['xBPP'] < 0].count())
#TODO: pitcher evaluation 2019: expected bases, base difference, xBPP difference, 

#TODO: filter all pitches not from game_year
data = data.dropna()
print(data["game_year"].unique())
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric)
s_data = data[data['game_year'] == game_year].copy()
#TODO: top pitch 2019
top = s_data.loc[s_data['xBPP'].idxmin()]
print("top", top)

#TODO: bottom pitch 2019
bot = s_data.loc[s_data['xBPP'].idxmax()]
print("bot", bot)

#TODO: calculate expected bases
s_data = s_data.drop(['release_speed', 'balls', 'strikes', 'inning', 'at_bat_number', 'pitch_number'], axis=1)
p_data = s_data.groupby(['player_name', 'game_year'], sort=False).sum()
p_data.loc[:, 'pitches'] = s_data.groupby(['player_name', 'game_year'], sort=False).count().loc[:, 'bases']
p_data = p_data.rename({'xBPP': 'xB'}, axis=1)
p_data.loc[:, 'xBPP'] = p_data.loc[:, 'xB'] / p_data.loc[:, 'pitches']

#TODO: calculate base difference
p_data.loc[:, 'BPP'] = p_data.loc[:, 'bases'] / p_data.loc[:, 'pitches']
p_data.loc[:, 'xBPP-diff'] = p_data.loc[:, 'xBPP'] - p_data.loc[:, 'BPP']
p_data.loc[:, 'xB-diff'] = p_data.loc[:, 'xB'] - p_data.loc[:, 'bases']
print(p_data)

#p_data.to_csv("2019_combined_analysis.csv")
