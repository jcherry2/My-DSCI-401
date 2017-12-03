import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import tree
from sklearn import naive_bayes
from datautil import *

# Basic Models
data = pd.read_csv('C:\GitHub\DSCI401\data\churn_data.csv')
print(data.head())


features = list(data)
features.remove('Churn')
data_x = data[features]
data_y = data['Churn']
data_x = pd.get_dummies(data_x)

le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# decision tree with gini impurity criterion
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)

# decision tree with entropy criterion
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

# Naive bayesian classifier
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)


# Best Model
data = pd.read_csv('C:\GitHub\DSCI401\data\churn_data.csv')

del data['FamilySize']
del data['CustID']

# label encoding for categorical variables
for i in data.columns:
    if data[i].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(list(data[i].values)) 
        data[i] = le.transform(list(data[i].values))
		
features = list(data)
features.remove('Churn')
data_x = data[features]
data_y = data['Churn']
#data_x = pd.get_dummies(data_x)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)


# validation set
datatest = pd.read_csv('C:\GitHub\DSCI401\data\churn_validation.csv')

del datatest['FamilySize']
del datatest['CustID']

# label encoding for categorical variables
for i in datatest.columns:
    if datatest[i].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(list(datatest[i].values)) 
        datatest[i] = le.transform(list(datatest[i].values))

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
		
features = list(datatest)
features.remove('Churn')
data_x = datatest[features]
data_y = datatest['Churn']

Dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
Dtree_gini_mod.fit(x_train, y_train)
preds_gini = Dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)


