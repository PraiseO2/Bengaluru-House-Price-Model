# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:06:01 2023

@author: CEO
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


df = pd.read_csv('Bengaluru_House_Data.csv')
#null = df.isna().sum()
'''DATA CLEANING'''
#drop features we're not interested in
df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')
#drop null features
df2 = df2.dropna()
#check for unique values
unique = df2['size'].unique()
null = df2.isna().sum()
#change the unique values to float instead of string
df2['bedrooms'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))

#create a function to output ranged data
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
#show the values that are ranged
new = df2[-df2['total_sqft'].apply(is_float)]
#create a function to find the mean of ranged data
def convert1(x):
    sep = x.split('-')
    if len(sep)==2:
        return(float(sep[0])+float(sep[1]))/2
    try:
        return float(x)
    except:
        return None
    
#print(convert1('100-200')

df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert1)

df3['price_per_sqft'] = (df3['price']*100000)/df3['total_sqft']

df4 = df3.copy()
df4.isna().sum()
df4 = df4.dropna()
#print(df4.isna().sum())
#print(df4.shape)

#remove spaces in location and group same locations together
df4['location'] = df4.location.apply(lambda x: x.strip())
loc_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
#check the number of locations which sum are =< 10
len(loc_stats[loc_stats<=10])
#create a new ddf and check the new unique locations in it
loc_stats_10 = loc_stats[loc_stats<=10]
len(df4['location'].unique())
#put those filtered data into same set "others'
df4['location'] = df4['location'].apply(lambda x: 'Other' if x in loc_stats_10 else x) 
len(df4['location'].unique())

#detect outliers in the price 
df4[df4['total_sqft']/df4['bedrooms']<300]
#print(nem.median())
#df4[df4['total_sqft']/df4['bedrooms']<300].head()
df5 = df4[-(df4['total_sqft']/df4['bedrooms']<300)]
#df5 = df4[-(df4.total_sqft/df4.bedrooms<300)]
#remove outliers for price 
def remove_outliers(x):
    x_out = pd.DataFrame()
    for ker, subdf in x.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        red_x = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        x_out = pd.concat([x_out, red_x], ignore_index=True)
    return x_out

df6 = remove_outliers(df5)

#remove outliers for same location but different price ranges
def plot_scatter(x, location):
    bed2 = x[(x.location==location)&(x.bedrooms==2)]
    bed3 = x[(x.location==location)&(x.bedrooms==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bed2.total_sqft, bed2.price_per_sqft, color = 'blue', label='2bed', s=50)
    plt.scatter(bed3.total_sqft, bed3.price_per_sqft, marker = '+', color = 'green', label='3bed', s=50)
    plt.xlabel('Total Sqft');plt.ylabel('Price per sqft')
    plt.title(location)
    plt.legend()
    
#plot_scatter(df6, 'Rajaji Nagar')
#remove outliers for same location but different price ranges
def remove_bed_out(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bedrooms_stats = {}
        for bedrooms, bedrooms_df in location_df.groupby('bedrooms'):
            bedrooms_stats[bedrooms] = {
                'mean': np.mean(bedrooms_df.price_per_sqft),
                'std': np.std(bedrooms_df.price_per_sqft),
                'count': bedrooms_df.shape[0]
            }
        for bedrooms, bedrooms_df in location_df.groupby('bedrooms'):
            stats = bedrooms_stats.get(bedrooms-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bedrooms_df[bedrooms_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df7 = remove_bed_out(df6)

df7 = remove_bed_out(df6)
#plot_scatter(df7, 'Hebbal')
'''
plt.hist(df7.price_per_sqft,rwidth=0.8)
plt.xlabel('price per sqft');plt.ylabel('count')'''
#clean data for unnecessarily much bathrooms
df7.bath.unique()
df7[df7.bath>10]
'''
plt.hist(df7.bath,rwidth=0.8)
plt.xlabel('bath');plt.ylabel('count')'''
df7[df7.bath>df7.bedrooms+2]
df8 = df7[df7.bath<df7.bedrooms+2]
#drop our non-needed features
df9 = df8.drop(['size', 'price_per_sqft'], axis = 'columns')

'''DATA MODELLING'''
#convert location features to dummies and drop text column
dummies = pd.get_dummies(df9.location)
df10 = pd.concat([df9, dummies.drop('Other', axis='columns')], axis='columns')
df11 = df10.drop('location', axis='columns')
#feature and output
X = df11.drop('price', axis=1)
y = df11['price']
#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#try linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
#try different moddels and params to determine the best
def best_model(X,y):
    algos = {
        'linear_regression':{
            'model': LinearRegression(),
            'params':{
                'positive':[True, False]
            }
        },
        'lasso':{
            'model': Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params':{
                'criterion':['mse', 'friedman_mse'],
                'splitter':['best', 'random']
            }  
        }
    }
              
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
           'model': algo_name,
           'best_score': gs.best_score_,
           'best_params': gs.best_params_
       })
    return pd.DataFrame(scores, columns=['model','best_score','best_params'])
             
best = best_model(X,y)          
#predict price for each dummy location
def predict_price(location,sqft,bath,bedrooms):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath 
    x[2] = bedrooms
    if loc_index >=0:
        x[loc_index] = 1 
    return model.predict([x])[0]
#predict house prices 
pred = predict_price('1st Phase JP Nagar', 1000, 2, 2)
pred2 = predict_price('1st Phase JP Nagar', 1000, 3, 3)
pred3 = predict_price('1st Phase JP Nagar', 1000, 4, 4)
pred4 = predict_price('Indira Nagar', 1000, 5, 5)
pred5 = predict_price('Indira Nagar', 1000, 6, 6)         
#dump as pickle              
import pickle
with open('Bengalaru_homw_prices_model.pickle','wb') as f:
    pickle.dump(model, f)  
#save as json
import json
columns = {
    'data_columns':[col.lower() for col in X.columns]
    }
with open('columns.json', 'w') as f:
     f.write(json.dumps(columns))             
                
                
                
                
                
                
                
                
                
                
                


























