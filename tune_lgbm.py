import pandas as pd
#import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from math import sqrt
import optuna.integration.lightgbm as lgb

def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))

def train(x_train, y_train, x_test, y_test, name):
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_test, label=y_test)
    #TODO: for better accuracy
        #TODO: large max_bin, small lr with large num_iterations, boosting: dart
    #TODO: avoid overfitting
        #TODO: small max_bin, use min_data_in_leaf and min_sum_hessian_in_leaf, min_gain_to_split, max_depth, extra_trees
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "max_bin" : 512
    }

    best_params, tuning_history = dict(), list()


    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        best_params=best_params,
        tuning_history=tuning_history,
        verbose_eval=100,
        early_stopping_rounds=100,
    )

    #clf = make_pipeline(StandardScaler(), model)
    scorer = make_scorer(rmse, greater_is_better=False)
    #scores = cross_val_score(clf, x_train, y_train, cv=5, scoring=scorer)
    #print("train_avg_score:", np.mean(scores))
    #clf.fit(x_train, y_train)
    prediction = np.rint(model.predict(x_test, num_iteration=model.best_iteration))
    #predict = clf.predict(x_test)
    scores = rmse(y_test, prediction)
    print("Number of finished trials: {}".format(len(tuning_history)))
    print("Best params:", best_params)
    print("  RMSE = {}".format(scores))
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    #FEATURE IMPORTANCE
    #feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, data.iloc[:, 0: -1].columns)), columns=['Value','Feature'])
    #plt.figure(figsize=(20, 10))
    #sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    #plt.title('LightGBM Features (avg over folds)')
    #plt.tight_layout()
    #plt.show()
    #plt.savefig('lgbm_importances-02.png')

    #dump(model, 'lgbm-'+name+'.joblib')

def main():
    #JUST GET FEATURE COLS (incl. pitch_type)
    data_columns = ['release_speed', 'release_pos_x', 'release_pos_z', 'release_pos_y', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'release_spin_rate', 'bases']
    data = pd.read_csv('full_xBPP.csv', usecols=data_columns, header=0)
    data = data.dropna()
    #SPLIT DATAFRAME IN X AND Y
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    # SPLIT INTO TRAIN AND VAL SET
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    #SCALE FEATURES AND TARGET
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    #TRAIN MODEL
    train(x_train, y_train, x_test, y_test, "lgbm-tune")

if __name__ == "__main__":
    main()
