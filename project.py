import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import *
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# import data
data = pd.read_csv('C:\GitHub\DSCI401\data\\nba_data2.csv')

# delete variables not wanted in model
del data['TEAM']
del data['MIN']
del data['BLKA']
del data['GP']

# determine correlations and print chart
corr = data.corr()
fig = plt.figure(figsize=(16,15))
ax = fig.add_subplot(111)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, 
           xticklabels=corr.columns.values,
           yticklabels=corr.index.values,
           cmap=cmap)
ax.xaxis.tick_top()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

# delete some of the correlated variables
del data['OREB']
del data['DREB']
del data['PFD']
del data['3PA']
del data['FTA']
del data['FGM']
del data['FT%']

features = list(data)
features.remove('W')
data_x = data[features]
data_y = data['W']

# create a linear regression model
model = linear_model.LinearRegression()

# split training and test sets from main set
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)


# fit the model
model.fit(x_train,y_train)

# make predictions on test data 
preds = model.predict(x_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 

# k-best feature selection to build the model with the best 25% features
selector_f = SelectKBest(f_regression, k=3)
selector_f.fit(x_train, y_train)
xt_train, xt_test = selector_f.transform(x_train), selector_f.transform(x_test)
model = linear_model.LinearRegression()
model.fit(xt_train, y_train)
preds = model.predict(xt_test)
pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))

print('MSE, MAE, R^2, EVS (Top 3 Model): ' + \
							   str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 	


