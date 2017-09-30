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
    train = pd.read_csv('../train_2016_v2.csv')
    properties = pd.read_csv('../properties_2016.csv')
    sample = pd.read_csv('../sample_submission.csv')

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

    print("Feature Engg")
    properties['N-LivingAreaProp'] = properties['calculatedfinishedsquarefeet'] / properties[
        'lotsizesquarefeet']
    properties['N-ValueRatio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
    properties['N-ValueProp'] = properties['structuretaxvaluedollarcnt'] / properties[
        'landtaxvaluedollarcnt']
    properties['N-TaxScore'] = properties['taxvaluedollarcnt'] * properties['taxamount']
    zip_count = properties['regionidzip'].value_counts().to_dict()
    properties['N-zip_count'] = properties['regionidzip'].map(zip_count)
    city_count = properties['regionidcity'].value_counts().to_dict()
    properties['N-city_count'] = properties['regionidcity'].map(city_count)
    group = properties.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    properties['N-Avg-structuretaxvaluedollarcnt'] = properties['regionidcity'].map(group)
    properties['N-Dev-structuretaxvaluedollarcnt'] = abs(
        (properties['structuretaxvaluedollarcnt'] - properties['N-Avg-structuretaxvaluedollarcnt'])) / properties[
                                                       'N-Avg-structuretaxvaluedollarcnt']
    
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

    print("Dropping")
    toDrop = []
    for c in x_train.columns:
        if (x_train[c]==-1).sum()>70000:
            toDrop.append(c)

    print(toDrop)
    x_train = x_train.drop(toDrop,axis=1)
    test_10 = copy.deepcopy(test)

    test_10["Month"] = 11

    x_test_10 = test_10[x_train.columns]

    del train, test

    return x_train, y_train, x_test_10



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        print("yaha")
        T = np.array(T)
        # T[1] = np.array(T[1])
        # T[2] = np.array(T[2])

        print("yaha to aa gya")
        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test_10 = np.zeros((T.shape[0], len(self.base_models)))
        # S_test_11 = np.zeros((T2.shape[0], len(self.base_models)))
        # S_test_12 = np.zeros((T3.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i_10 = np.zeros((T.shape[0], self.n_splits))
            # S_test_i_11 = np.zeros((T2.shape[0], self.n_splits))
            # S_test_i_12 = np.zeros((T3.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                print ("Fit Model %d fold %d" % (i, j))
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i_10[:, j] = clf.predict(T)[:]
                # S_test_i_11[:, j] = clf.predict(T2)[:]
                # S_test_i_12[:, j] = clf.predict(T3)[:]
            S_test_10[:, i] = S_test_i_10.mean(axis=1)
            # S_test_11[:, i] = S_test_i_11.mean(axis=1)
            # S_test_12[:, i] = S_test_i_12.mean(axis=1)
            # del S_test_i_10,S_test_i_11,S_test_i_12
            # gc.collect()

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()
        print("fitting final")
        self.stacker.fit(S_train, y)
        print("predicting for ",i)
        res = self.stacker.predict(S_test_10)[:]
        # print("predicting for 11")
        # res2 = self.stacker.predict(S_test_11)[:]
        # print("predicting for 12")
        # res3 = self.stacker.predict(S_test_12)[:]
        # del S_test_10,S_test_11,S_test_12

        return res

x_train, y_train, x_test_10 = load_data()



### NN ###
# Neural Network
len_x = int(x_train.shape[1])

print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

print("\nFitting neural network model...")
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)

print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
y_pred_ann = nn.predict(x_test)

print( "\nPreparing results for write..." )
nn_pred = y_pred_ann.flatten()
print( "Type of nn_pred is ", type(nn_pred) )
print( "Shape of nn_pred is ", nn_pred.shape )

print( "\nNeural Network predictions:" )
print( pd.DataFrame(nn_pred).head() )







# rf params
rf_params = {}
rf_params['n_estimators'] = 50
rf_params['max_depth'] = 8
rf_params['min_samples_split'] = 100
rf_params['min_samples_leaf'] = 30

# xgb params
xgb_params = {}
xgb_params['n_estimators'] = 50
xgb_params['min_child_weight'] = 12
xgb_params['learning_rate'] = 0.27
xgb_params['max_depth'] = 6
xgb_params['subsample'] = 0.77
xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
xgb_params['base_score'] = 0
#xgb_params['seed'] = 400
xgb_params['silent'] = 1


# lgb params
lgb_params = {}
lgb_params['n_estimators'] = 50
lgb_params['max_bin'] = 10
lgb_params['learning_rate'] = 0.321 # shrinkage_rate
lgb_params['metric'] = 'l1'          # or 'mae'
lgb_params['sub_feature'] = 0.34
lgb_params['bagging_fraction'] = 0.85 # sub_row
lgb_params['bagging_freq'] = 40
lgb_params['num_leaves'] = 512        # num_leaf
lgb_params['min_data'] = 500         # min_data_in_leaf
lgb_params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params['verbose'] = 0
lgb_params['feature_fraction_seed'] = 2
lgb_params['bagging_seed'] = 3


# XGB model
xgb_model = XGBRegressor(**xgb_params)

# lgb model
lgb_model = LGBMRegressor(**lgb_params)

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

print("aag ya")
stack = Ensemble(n_splits=5,
        stacker=LinearRegression(),
        base_models=(rf_model, xgb_model, lgb_model, et_model, ada_model, dt_model))
print("haga")
y_test_10 = stack.fit_predict(x_train, y_train, x_test_10)
# y_test_11 = stack.fit_predict(x_train, y_train, x_test_10)
# y_test_12 = stack.fit_predict(x_train, y_train, x_test_10)

from datetime import datetime
print("submit...")
# pre = y_test
sub = pd.read_csv('../sample_submission.csv')
# for c in sub.columns[sub.columns != 'ParcelId']:
sub['201610'] = y_test_10
sub['201611'] = y_test_10
sub['201612'] = y_test_10
sub['201710'] = y_test_10
sub['201711'] = y_test_10
sub['201712'] = y_test_10
submit_file = '{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
sub.to_csv(submit_file, index=False,  float_format='%.4f')
