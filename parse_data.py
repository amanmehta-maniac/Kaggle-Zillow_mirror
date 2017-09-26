#Importing Libraries or Packages that are needed throughout the Program
import numpy as np
import pandas as pd
import xgboost as  xgb
import random
import datetime as dt
import gc
import copy
 #Load the Datasets #

     # We need to load the datasets that will be needed to train our machine learning algorithms, handle our data and make predictions.
 #Note that these datasets are the ones that are already provided once
 # you enter the competition by accepting terms and conditions

train = pd.read_csv('../train_2016_v2.csv' , parse_dates=["transactiondate"])
properties = pd.read_csv('../properties_2016.csv')
test = pd.read_csv('../sample_submission.csv')
test= test.rename(columns={'ParcelId': 'parcelid'}) #To make it easier for merging datasets on same column_id later

# Analyse the Dimensions of our Datasets.

print("Training Size:" + str(train.shape))
print("Property Size:" + str(properties.shape))
print("Sample Size:" + str(test.shape))


     # Type Converting the DataSet

     # The processing of some of the algorithms can be made quick if data representation is made in int/float32 instead of int/float64.
 # Therefore, in order to make sure that all of our columns types are in
 # float32, we are implementing the following lines of code #


for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)
    if dtype == np.int64:
        properties[c] = properties[c].astype(np.int32)

for column in test.columns:
    if test[column].dtype == int:
        test[column] = test[column].astype(np.int32)
    if test[column].dtype == float:
        test[column] = test[column].astype(np.float32)

#living area proportions
properties['living_area_prop'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']
#tax value ratio
properties['value_ratio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
#tax value proportions
properties['value_prop'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']


###Merging the Datasets ###

# We are merging the properties dataset with training and testing dataset for model building and testing prediction #

df_train = train.merge(properties, how='left', on='parcelid')
df_test = test.merge(properties, how='left', on='parcelid')

df_train['month'] = df_train['transactiondate'].dt.month

### Remove previos variables to keep some memory
del properties, train
gc.collect();


print('Memory usage reduction...')
df_train[['latitude', 'longitude']] /= 1e6
df_test[['latitude', 'longitude']] /= 1e6

df_train['censustractandblock'] /= 1e12
df_test['censustractandblock'] /= 1e12


# Label Encoding For Machine Learning &amp; Filling Missing Values

# We are now label encoding our datasets. All of the machine learning algorithms employed in scikit learn assume that the data being fed to them is in numerical form. LabelEncoding ensures that all of our categorical variables are in numerical representation. Also note that we are filling the missing values in our dataset with a zero before label encoding them. This is to ensure that label encoder function does not experience any problems while carrying out its operation #

from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()
for c in df_train.columns:
    df_train[c]=df_train[c].fillna(-1)
    if df_train[c].dtype == 'object':
        lbl.fit(list(df_train[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))

for c in df_test.columns:
    df_test[c]=df_test[c].fillna(0)
    if df_test[c].dtype == 'object':
        lbl.fit(list(df_test[c].values))
        df_test[c] = lbl.transform(list(df_test[c].values))


### Rearranging the DataSets ###

# We will now drop the features that serve no useful purpose. We will also split our data and divide it into the representation to make it clear which features are to be treated as determinants in predicting the outcome for our target feature. Make sure to include the same features in the test set as were included in the training set #

toDrop = set()

countOfEmpty=0
for c in df_train.columns:
    Empty = df_train[c]
    if df_train[c].dtype!='datetime64[ns]':
    	countOfEmpty = (Empty==-1).sum()
    else:
	countOfEmpty = 0
    if(countOfEmpty >= 20000):
        toDrop.add(c)

df_train = df_train.replace(-1,0)
df_test = df_test.replace(-1,0)
print "Printing the elements to be dropped"
for each in toDrop:
    print each

toDrop.add('parcelid')
toDrop.add('logerror')
toDrop.add('transactiondate')
toDrop.add('propertyzoningdesc')
toDrop.add('propertycountylandusecode')

toDropList = []

for each in toDrop:
    toDropList.append(each)

x_train = df_train.drop(toDropList, axis=1)

toDropList.append('201610')
toDropList.append('201611')
toDropList.append('201612')
toDropList.append('201710')
toDropList.append('201711')
toDropList.append('201712')
toDropList.remove('logerror')
toDropList.remove('transactiondate')

x_test = df_test.drop(toDropList, axis = 1)

count = 0
for c in x_train.columns:
    count=count+1

print "Total columns in training:",count

count=0
for c in x_test.columns:
    count=count+1

print "Total columns in testing:",count

print "Number of Items in test data"
print len(x_test)

x_train = x_train.values
y_train = df_train['logerror'].values

### Cross Validation ###

# We are dividing our datasets into the training and validation sets so that we could monitor and the test the progress of our machine learning algorithm. This would let us know when our model might be over or under fitting on the dataset that we have employed. #

from sklearn.model_selection import train_test_split

X = x_train
y = y_train

Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, random_state=42)

#Preparing testing data for month 10, 11, 12 seperately
x_test10 = copy.deepcopy(x_test)
x_test10['month'] = 10

x_test11 = copy.deepcopy(x_test)
x_test11['month'] = 11

x_test12 = copy.deepcopy(x_test)
x_test12['month'] = 12
#Implement the Xgboost#


# We can now select the parameters for Xgboost and monitor the progress of results on our validation set. The explanation of the xgboost parameters and what they do can be found on the following link http://xgboost.readthedocs.io/en/latest/parameter.html #

dtrain = xgb.DMatrix(Xtrain, label=ytrain)
#dtrain = xgb.DMatrix(X, label=y)
dvalid = xgb.DMatrix(Xvalid, label=yvalid)

dtest10 = xgb.DMatrix(x_test10.values)
dtest11 = xgb.DMatrix(x_test11.values)
dtest12 = xgb.DMatrix(x_test12.values)
print "Length of test data of month 10 again", dtest10.num_row()
print "Length of test data of month 11 again", dtest11.num_row()
print "Length of test data of month 12 again", dtest12.num_row()

# Try different parameters!
xgb_params = {'min_child_weight': 5, 'eta': 0.035, 'colsample_bytree': 0.5, 'max_depth': 4,
            'subsample': 0.85, 'lambda': 0.8, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,
            'eval_metric': 'mae', 'objective': 'reg:linear' }

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
#watchlist = [(dtrain,'train')]


model_xgb = xgb.train(xgb_params, dtrain, 1000, watchlist, early_stopping_rounds=100,
                  maximize=False, verbose_eval=10)

#Predicting the results#

# Let us now predict the target variable for our test dataset. All we have to do now is just fit the already trained model on the test set that we had made merging the sample file with properties dataset #

Predicted_test_xgb10 = model_xgb.predict(dtest10)
Predicted_test_xgb11 = model_xgb.predict(dtest11)
Predicted_test_xgb12 = model_xgb.predict(dtest12)

#Submitting the Results

#Once again load the file and start submitting the results in each column

sample_file = pd.read_csv('../sample_submission.csv')
sample_file['201610'] = Predicted_test_xgb10
sample_file['201611'] = Predicted_test_xgb11
sample_file['201612'] = Predicted_test_xgb12
sample_file['201710'] = Predicted_test_xgb10
sample_file['201711'] = Predicted_test_xgb11
sample_file['201712'] = Predicted_test_xgb12

print('Preparing the csv file ...')
sample_file.to_csv('xgb_predicted_results.csv', index=False, float_format='%.4f')
print("Finished writing the file")
