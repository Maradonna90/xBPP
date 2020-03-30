import lightgbm as lgb
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import numpy as np

store = pd.HDFStore('pitches.h5')
store_keys = store.keys()
store.close()
lgbm = load('lgbm.joblib')
cat = load('cb.joblib')
models = [lgbm, cat]
names = ['lgbm', 'cat']
data_columns = ['release_speed', 'release_pos_x', 'release_pos_z', 'release_pos_y', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_spin_rate']
for key in store_keys:
    print(key)
    data = pd.read_hdf('pitches.h5', key)
    #else:
    #    data = pd.read_hdf('pitches.h5', key, start=2)
    X = data.loc[:, data_columns].values
    sc = StandardScaler()
    x = sc.fit_transform(X)
    for model, name in zip(models, names):
        prediction = model.predict(x)
        data.loc[:, name] = prediction
    data.loc[:,"xBPP"] = data.loc[:, "lgbm"] * 0.5 + data.loc[:, "cat"] * 0.5
    if key == '/data0':
        data.to_csv("combined_xBPP.csv", index=False, mode='a')
    else:
         data.to_csv("combined_xBPP.csv", index=False, mode='a', header=False)
