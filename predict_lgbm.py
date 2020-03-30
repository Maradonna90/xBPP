import lightgbm as lgb
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import numpy as np

store = pd.HDFStore('pitches.h5')
store_keys = store.keys()
store.close()
lgbm = load('lgbm.joblib')
data_columns = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_spin_rate']
for key in store_keys:
    print(key)
    #if key == '/data0':
    data = pd.read_hdf('pitches.h5', key)
    #else:
    #    data = pd.read_hdf('pitches.h5', key, start=2)
    X = data.loc[:, data_columns].values
    sc = StandardScaler()
    x = sc.fit_transform(X)
    prediction = lgbm.predict(x)
    data.loc[:, 'xBPP'] = prediction
    if key == '/data0':
        data.to_csv("full_xBPP.csv", index=False, mode='a')
    else:
         data.to_csv("full_xBPP.csv", index=False, mode='a', header=False)
