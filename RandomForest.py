# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:23:48 2020

@author: Ishan Nilotpal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('Data_Train.xlsx')
pd.set_option('display.max_columns',None)

df.dropna(inplace=True)

df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.day
df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.month

df = df.drop('Date_of_Journey',axis=1)

df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute

df = df.drop('Dep_Time',axis=1)

df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute

df = df.drop('Arrival_Time',axis=1)

duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h "+ duration[i]
            
Duration_Hour = []
Duration_min = []

for i in range(len(duration)):
    Duration_Hour.append(int(duration[i].split(sep='h')[0]))
    Duration_min.append(int(duration[i].split(sep='m')[0].split()[-1]))

df['Duration_Hour'] = Duration_Hour
df['Duration_min'] = Duration_min

df.drop('Duration',axis=1,inplace=True)

sns.catplot(y='Price',x='Airline',data=df.sort_values("Price",ascending=False),kind="boxen",height=6,aspect=3)

Airline = df['Airline']
Airline = pd.get_dummies(Airline,drop_first=True)

Source = df['Source']
Source = pd.get_dummies(Source,drop_first= True)

destination = df['Destination']
destination = pd.get_dummies(destination,drop_first=True)

##Route and No. of stops columns actually represent the same thing so  Route Column is dropped
##80% of the Additional Column is 'no info' so it is not significant  

df = df.drop(['Route','Additional_Info'],axis=1) 

df.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)

df = pd.concat([df,Airline,Source,destination],axis=1)

df = df.drop(['Airline','Source','Destination'],axis=1)


##Loading and Preprocessing of test data
dft = pd.read_excel('Test_set.xlsx')
pd.set_option('display.max_columns',None)

dft.dropna(inplace=True)

dft['Journey_day'] = pd.to_datetime(dft['Date_of_Journey'],format="%d/%m/%Y").dt.day
dft['Journey_month'] = pd.to_datetime(dft['Date_of_Journey'],format="%d/%m/%Y").dt.month

dft = dft.drop('Date_of_Journey',axis=1)

dft['Dep_Hour'] = pd.to_datetime(dft['Dep_Time']).dt.hour
dft['Dep_min'] = pd.to_datetime(dft['Dep_Time']).dt.minute

dft = dft.drop('Dep_Time',axis=1)

dft['Arrival_Hour'] = pd.to_datetime(dft['Arrival_Time']).dt.hour
dft['Arrival_min'] = pd.to_datetime(dft['Arrival_Time']).dt.minute

dft = dft.drop('Arrival_Time',axis=1)

duration1 = list(dft["Duration"])

for i in range(len(duration1)):
    if len(duration1[i].split()) != 2:
        if 'h' in duration1[i]:
            duration1[i] = duration1[i].strip() + " 0m"
        else:
            duration1[i] = "0h "+ duration1[i]
            
Duration_Hour = []
Duration_min = []

for i in range(len(duration1)):
    Duration_Hour.append(int(duration1[i].split(sep='h')[0]))
    Duration_min.append(int(duration1[i].split(sep='m')[0].split()[-1]))

dft['Duration_Hour'] = Duration_Hour
dft['Duration_min'] = Duration_min

dft.drop('Duration',axis=1,inplace=True)


Airline = dft['Airline']
Airline = pd.get_dummies(Airline,drop_first=True)

Source = dft['Source']
Source = pd.get_dummies(Source,drop_first= True)

destination = dft['Destination']
destination = pd.get_dummies(destination,drop_first=True)


dft = dft.drop(['Route','Additional_Info'],axis=1) 

dft.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)

dft = pd.concat([dft,Airline,Source,destination],axis=1)

dft = dft.drop(['Airline','Source','Destination'],axis=1)

x = df.drop(['Price','Trujet'],axis=1)
y = df['Price']

from sklearn.ensemble import ExtraTreesRegressor
Selection = ExtraTreesRegressor()

Selection.fit(x, y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.250046807714,random_state=0)


plt.figure(figsize=(12,18))
feat_importances = pd.Series(Selection.feature_importances_,index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
 
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor()
Regressor.fit(x_train,y_train)

predict = Regressor.predict(x_test)


from sklearn.metrics import r2_score
print(r2_score(predict,y_test))

##To increase prediction Hyper parameter tuning

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = Regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

import pickle
file = open('flight_rf.pkl','wb')
pickle.dump(Regressor,file)

rf_random.fit(x_train,y_train)
predict = rf_random.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(predict,y_test))





