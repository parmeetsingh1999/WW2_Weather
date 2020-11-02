# Importing libraries

import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import os
from sklearn.model_selection import train_test_split
from statsmodels import api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd 

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
warnings.filterwarnings('ignore')

# Loading the dataset

data = pd.read_csv('../input/weatherww2/Summary of Weather.csv')
data = data[['MinTemp', 'MaxTemp']]
data.head()

plt.figure(figsize = (10, 5))
plt.scatter(data['MinTemp'], data['MaxTemp'], s = 10)
plt.xlabel('Min Temperature Celsius', fontsize = 15)
plt.ylabel('Max Temperature Celsius', fontsize = 15)
plt.show()

data.drop(data[(data['MinTemp'] < -15) & (data['MaxTemp'] > 15)].index, inplace = True)
data.drop(data[(data['MinTemp'] > 8) & (data['MaxTemp'] < -15)].index, inplace = True)

plt.figure(figsize = (10, 5))
plt.scatter(data['MinTemp'], data['MaxTemp'], s = 10)
plt.xlabel('Min Temperature Celsius', fontsize = 15)
plt.ylabel('Max Temperature Celsius', fontsize = 15)
plt.show()

# Performing Linear Regression

#Scatter plot with few possible regression line
plt.figure(figsize = (10, 5))
plt.scatter(data['MinTemp'], data['MaxTemp'], s = 10)
plt.xlabel('Min temperature celsius', fontsize = 15)
plt.ylabel('Max temperature celsius', fontsize = 15)

x1 = [-38, 35]
y1 = [-32, 53]
plt.plot(x1, y1, color = 'orange')

x2 = [-38, 35]
y2 = [-27, 44]
plt.plot(x2, y2, color = 'red')
plt.show()

# Generic steps in modeling

x = data['MinTemp']
y = data['MaxTemp']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 1)

x_train.head()

y_train.head()

# Building a linear model

x_train_sm = sm.add_constant(x_train)
lr = sm.OLS(y_train, x_train_sm).fit()

lr.params

lr.summary()

#Best fit line
plt.figure(figsize = (10, 5))
plt.scatter(x_train, y_train, s = 10)
plt.plot(x_train, 10.686345 + 0.920631 * x_train, 'r')
plt.xlabel('Min temperature celsius', fontsize = 15)
plt.ylabel('Max temperature celsius', fontsize = 15)
plt.show()

# Residual Analysis

#Distribution of error terms
y_train_pred = lr.predict(x_train_sm)
res = y_train - y_train_pred

fig = plt.figure(figsize = (8, 4))
sns.distplot(res, bins = 15)
fig.suptitle('Error terms', fontsize = 15)
plt.xlabel('y_train - y_train-pred', fontsize = 15)
plt.show()

#Looking for patterns in the residuals
plt.figure(figsize = (8, 4))
plt.scatter(x_train, res)
plt.show()

# Prediction on test set

x_test_sm = sm.add_constant(x_test)
y_pred = lr.predict(x_test_sm)

y_pred.head()

mean_squared_error(y_test, y_pred)

#RMSE
np.sqrt(mean_squared_error(y_test, y_pred))

#Checking the r-squared on the test set
r_squared = r2_score(y_test, y_pred)
r_squared

# Visualizing the fit on test set

plt.figure(figsize = (10, 5))
plt.scatter(x_test, y_test, s = 10)
plt.plot(x_test, 10.686345 + 0.920631 * x_test, 'r')
plt.show()

# Linear Regression using linear_model in sklearn

x_train_lm, x_test_lm, y_train_lm, y_test_lm = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 100)

x_train_lm.shape

x_train_lm = x_train_lm.values.reshape(-1, 1)
x_test_lm = x_test_lm.values.reshape(-1, 1)

print(x_train_lm.shape)
print(x_test_lm.shape)
print(y_train_lm.shape)
print(y_test_lm.shape)

lm = LinearRegression()
lm.fit(x_train_lm, y_train_lm)

print(lm.intercept_)
print(lm.coef_)
