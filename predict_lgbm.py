import lightgbm as lgb
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_hdf('pitches.h5', 'data')
lgbm = load('lgbm.joblib')
pitchers = data.player_name.unique()
#print(data.columns)
data_columns = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_spin_rate']
X = data.loc[:, data_columns].values
sc = StandardScaler()
x = sc.fit_transform(X)
prediction = lgbm.predict(x)
res = data.loc[:,['player_name', 'game_year', 'bases']]
res.loc[:, 'xBPP'] = prediction
res.to_csv("xBPP.csv", index=False)
