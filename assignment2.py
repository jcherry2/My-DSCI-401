# Josiah Cherry
# assignment 2

# BASELINE MODEL 
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import *
import pprint
from sklearn import preprocessing
import matplotlib.pyplot as plt

# import data
train = pd.read_csv('C:\GitHub\DSCI401\data\AmesHousingSetA.csv')
test = pd.read_csv('C:\GitHub\DSCI401\data\AmesHousingSetB.csv')


categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
train_num = train[numerical_features]
train_cat = train[categorical_features]

# replace num var na with mean
train_num = train_num.fillna(train_num.mean())
# replace cat var na with dummies
train_cat = pd.get_dummies(train_cat)


train = pd.concat([train_cat,train_num],axis=1)

features = list(train)
features.remove('SalePrice')
data_x = train[features]
data_y = train['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()

linear_mod.fit(x_train,y_train)
preds = linear_mod.predict(x_test)

print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)]))



							   
							   
# BEST MODEL						   
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import *
import pprint
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import skew

train = pd.read_csv('C:\GitHub\DSCI401\data\AmesHousingSetA.csv')
test = pd.read_csv('C:\GitHub\DSCI401\data\AmesHousingSetB.csv')

train["SalePrice"] = np.log1p(train["SalePrice"])
num_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[num_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
train[skewed_feats] = np.log1p(train[skewed_feats])

corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


train = train.drop('PID', 1)


categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
train_num = train[numerical_features]
train_cat = train[categorical_features]

# replace num var na with mean
train_num = train_num.fillna(train_num.mean())
# replace cat var na with dummies
train_cat = pd.get_dummies(train_cat)

train = pd.concat([train_cat,train_num],axis=1)
features = list(train)
features.remove('SalePrice')
data_x = train[features]
data_y = train['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()

linear_mod.fit(x_train,y_train)
preds = linear_mod.predict(x_test)

print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)]))

