 #Importing Libraries or Packages that are needed throughout the Program
import numpy as np
import pandas as pd
import random
import gc
import copy
import sklearn


from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# model
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def load_data():
    train = pd.read_csv('../input/train_2016_v2.csv')
    properties = pd.read_csv('../input/properties_2016.csv')
    sample = pd.read_csv('../input/sample_submission.csv')
    
    print("Preprocessing...")
    for c, dtype in zip(properties.columns, properties.dtypes):
        if dtype == np.float64:
            properties[c] = properties[c].astype(np.float32)
            
    print("Set train/test data...")
    id_feature = ['heatingorsystemtypeid','propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
        'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'typeconstructiontypeid']
    for c in properties.columns:
        properties[c]=properties[c].fillna(-1)
        if properties[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
        if c in id_feature:
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
            dum_df = pd.get_dummies(properties[c])
            dum_df = dum_df.rename(columns=lambda x:c+str(x))
            properties = pd.concat([properties,dum_df],axis=1)
        #
    # Make train and test dataframe
    #
    train = train.merge(properties, on='parcelid', how='left')
    sample['parcelid'] = sample['ParcelId']
    test = sample.merge(properties, on='parcelid', how='left')

    # drop out ouliers
    train = train[train.logerror > -0.4]
    train = train[train.logerror < 0.42]

    train["transactiondate"] = pd.to_datetime(train["transactiondate"])
    train["Month"] = train["transactiondate"].dt.month

    x_train = train.drop(['parcelid', 'logerror','transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
    y_train = train["logerror"].values
    
    test_10 = copy.deepcopy(test)
    test_11 = copy.deepcopy(test)
    test_12 = copy.deepcopy(test)
    test_10["Month"] = 10
    test_11["Month"] = 11
    test_12["Month"] = 12
    test_10 = test_10[x_train.columns]
    test_11 = test_11[x_train.columns]
    test_12 = test_12[x_train.columns]
    del train, test   
    return x_train, y_train, test_10, test_11, test_12



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T1,T2,T3):
        X = np.array(X)
        y = np.array(y)
        T1 = np.array(T1)
        T2 = np.array(T2)
        T3 = np.array(T3)


        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test_10 = np.zeros((T1.shape[0], len(self.base_models)))
        S_test_11 = np.zeros((T2.shape[0], len(self.base_models)))
        S_test_12 = np.zeros((T3.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i_10 = np.zeros((T1.shape[0], self.n_splits))
            S_test_i_11 = np.zeros((T2.shape[0], self.n_splits))
            S_test_i_12 = np.zeros((T3.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]                

                S_train[test_idx, i] = y_pred
                S_test_i_10[:, j] = clf.predict(T1)[:]
                S_test_i_11[:, j] = clf.predict(T2)[:]
                S_test_i_12[:, j] = clf.predict(T3)[:]
            S_test_10[:, i] = S_test_i_10.mean(axis=1)
            S_test_11[:, i] = S_test_i_11.mean(axis=1)
            S_test_12[:, i] = S_test_i_12.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res1 = self.stacker.predict(S_test_10)[:]
        res2 = self.stacker.predict(S_test_11)[:]
        res3 = self.stacker.predict(S_test_12)[:]
        return res1,res2,res3

x_train, y_train, x_test_10, x_test_11, x_test_12 = load_data()

# xgb params
xgb_params = {}
#xgb_params['objective'] = 'reg:linear'
#xgb_params['eval_metric'] = 'mae'
xgb_params['n_estimators'] = 250
xgb_params['min_child_weight'] = 12
xgb_params['learning_rate'] = 0.37
xgb_params['max_depth'] = 6
xgb_params['subsample'] = 0.77
xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
xgb_params['base_score'] = 0
#xgb_params['seed'] = 400
xgb_params['silent'] = 1

# rf params
rf_params = {}


# XGB model
xgb_model = XGBRegressor(**xgb_params)

# lgb model
lgb_model = LGBMRegressor()

# RF model
rf_model = RandomForestRegressor(**rf_params)

# ET model
et_model = ExtraTreesRegressor()

# SVR model
# SVM is too slow in more then 10000 set
#svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.05)

# DecsionTree model
dt_model = DecisionTreeRegressor()

# AdaBoost model
ada_model = AdaBoostRegressor()

stack = Ensemble(n_splits=5,
        stacker=LinearRegression(),
        base_models=(rf_model, xgb_model, lgb_model, et_model, ada_model, dt_model))

y_test_10,y_test_11,y_test_12 = stack.fit_predict(x_train, y_train, x_test_10, x_test_11, x_test_12)
# y_test_11 = stack.fit_predict(x_train, y_train, x_test_10)
# y_test_12 = stack.fit_predict(x_train, y_train, x_test_10)

from datetime import datetime
print("submit...")
# pre = y_test
sub = pd.read_csv('../input/sample_submission.csv')
# for c in sub.columns[sub.columns != 'ParcelId']:
sub['201610'] = y_test_10
sub['201611'] = y_test_11
sub['201612'] = y_test_12
sub['201710'] = y_test_10
sub['201711'] = y_test_11
sub['201712'] = y_test_12
submit_file = '{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
sub.to_csv(submit_file, index=False,  float_format='%.4f')
