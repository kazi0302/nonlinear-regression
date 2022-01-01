'''
Final Project: Nonlinear Regression Model with GPU Acceleration
Author: Hirokatsu Suzuki
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
import warnings
warnings.filterwarnings("ignore")

#Read files and extract needed part
df = pd.read_csv('CA_new.csv')
df = df.truncate(before=0,after=360)
temp_df = df[['date','positive','death']]
temp_df['date'] = pd.to_datetime(temp_df['date'])
new_df = temp_df.sort_values(by=['date'])

#Reformat datetime to integer from day 1
start_week = 1
covid_data = pd.DataFrame()
for index,row in new_df.iterrows():
  death = row[2]
  positive = row[1]
  new_data = [start_week,int(positive),int(death)]
  covid_data = covid_data.append([new_data], ignore_index=True)
  start_week += 1

#Rename the column names
covid_data = covid_data.rename({0: 'Day', 1: 'Cases', 2:'Deaths'}, axis=1)

#Reformat the column values for RF regression
day = []
cases = []
for w in covid_data['Day']:
  day.append(w)
for c in covid_data['Cases']:
  cases.append(c)

day = np.array(day).reshape(-1,1)
cases = np.array(cases).reshape(-1,1)

#Split dataset and create model
X_train, X_test, y_train, y_test = train_test_split(day, cases, test_size=0.8, random_state=42)
regression = RandomForestRegressor(n_estimators=10000, random_state=42, n_jobs=4)

#Record the start time of regression process
start = time()

#Train the model with splitted data
regression.fit(X_train,y_train)

#Record the end time and print the runtime
end = time()
result = end - start
print('%.3f seconds' % result)

prediction = regression.predict(X_test)

# (Parallelizing) Repeat the processes with difference number of core(s)
jobs = [1,2,3,4]
results = []
for j in jobs:
  start = time()
  model = RandomForestRegressor(n_estimators=10000, random_state=42, n_jobs=j)
  model.fit(X_test, y_test)
  end = time()
  result = end - start
  print('>cores=%d: %.3f seconds' % (j, result))
  results.append(result)

#Perform model analysis
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


