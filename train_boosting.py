import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from math import sqrt
import xgboost as xgb
from catboost import CatBoostRegressor

def rmse(y_actual, y_predicted):
    #print(y_actual, y_predicted)
    return sqrt(mean_squared_error(y_actual, y_predicted))

def train(models, x_train, y_train, x_test, y_test, names):
    preds = pd.DataFrame()
    for model, name in zip(models, names):
        clf = make_pipeline(StandardScaler(), model)
        scorer = make_scorer(rmse, greater_is_better=False)
        scores = cross_val_score(clf, x_train, y_train, cv=5, scoring=scorer)
        print(name+" train_avg_score:", np.mean(scores))
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        scores = rmse(y_test, predict)
        preds[name] = predict
        print(name+" test_avg_score:", np.mean(scores))
        dump(model, "models/"+name+'.joblib')

    print("combines results...")
    #TODO: evaluate weights for boosting predictions
    preds.loc[:,"com_pred"] = preds.loc[:, "lgbm"] * 0.5 + preds.loc[:, "xgb"] * 0.3 + preds.loc[:, "cb"] * 0.2
    preds.loc[preds.com_pred < 0, 'com_pred'] = 0
    scores = (y_test, preds.loc[:, 'com_pred'])
    print("combined test_avg_score: {}".format(np.mean(scores)))

    #FEATURE IMPORTANCE
    #feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, data.iloc[:, 0: -1].columns)), columns=['Value','Feature'])
    #plt.figure(figsize=(20, 10))
    #sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    #plt.title('LightGBM Features (avg over folds)')
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('lgbm_importances-02.png')


def main():
    data_columns = ['release_speed', 'release_pos_x', 'release_pos_z', 'release_pos_y', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_spin_rate', 'bases']
    #TODO: add pitch  type
    #data_columns = ['release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'bases']

    data = pd.read_hdf('data/pitches.h5', columns=data_columns)
    print(data.shape)
    print(data.isna().sum())
    data = data.dropna()
    print(data.shape)
    #SPLIT DATAFRAME IN X AND Y
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    # SPLIT INTO TRAIN AND VAL SET
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    #SCALE FEATURES AND TARGET
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    #BUILD MODEL
    cb_model = CatBoostRegressor(eval_metric='RMSE', verbose=False)
    xgb_model = xgb.XGBRegressor(eval_metric='rmse')
    lgbm = lgb.LGBMRegressor(metric='rmse')
    train([lgbm, xgb_model, cb_model], x_train, y_train, x_test, y_test, ['lgbm', 'xgb', 'cb'])

if __name__ == "__main__":
    main()
