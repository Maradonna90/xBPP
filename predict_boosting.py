import lightgbm as lgb
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import numpy as np

lgbm = load('models/lgbm.joblib')
cat = load('models/cb.joblib')
xgb = load('models/xgb.joblib')
models = [lgbm, cat, xgb]
names = ['lgbm', 'cb', 'xgb']
data_columns = ['release_speed', 'release_pos_x', 'release_pos_z', 'release_pos_y', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_spin_rate']
data = pd.read_hdf('data/pitches.h5', key)
print(data.shape)
X = data.loc[:, data_columns].values
sc = StandardScaler()
x = sc.fit_transform(X)
for model, name in zip(models, names):
    prediction = model.predict(x)
    data.loc[:, name] = prediction
data.loc[:,"xBPP"] = data.loc[:, "lgbm"] * 0.5 + data.loc[:, "xgb"] * 0.3 + data.loc[:, 'cb'] * 0.2
data.to_csv("results/combined_xBPP.csv", index=False, mode='w')
