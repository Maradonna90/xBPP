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
#LOAD DATA
data = pd.read_hdf('pitches.h5', 'data')

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
lgbm = lgb.LGBMRegressor()
clf = make_pipeline(StandardScaler(), lgbm)
scorer = make_scorer(mean_squared_error, greater_is_better=False)
scores = cross_val_score(clf, x_train, y_train, cv=5, scoring=scorer)
print("train_avg_score:", np.mean(scores))
clf.fit(x_train, y_train)
predict = clf.predict(x_test)
scores = mean_squared_error(y_test, predict)
#print("x_test:", x_test[20:30])
#print("y_test:", y_test[20:30])
#print("predict:", predict[20:30])
print("test_avg_score:", np.mean(scores))

#FEATURE IMPORTANCE
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, data.iloc[:, 0: -1].columns)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
#plt.show()
plt.savefig('lgbm_importances-01.png')

dump(lgbm, 'lgbm.joblib')
