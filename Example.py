"""
@author: Daniel Moreira de Sousa
@my github: https://github.com/DanielMSousa
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class model_evaluator:
    def __init__(self):
        
        #you can add models here
        self.models = [
            ('Linear regression', LinearRegression()),
            ('Polynomial regression', LinearRegression()),
            ('SVR', SVR()),
            ('Decision Tree', DecisionTreeRegressor()),
            ('Random Forest', RandomForestRegressor(random_state=42)),
            ('KNN regressor', KNeighborsRegressor()),
        ]
    
        self.cv_models = []
    
        self.predictions = []
        
        #These are your regression metrics
        self.metrics = []
        
        #Feel free to change the grid search params
        self.grid_searchs = {
            'SVR': {
                'C':[0.1, 1, 5, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel':['rbf'],
                'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'Random Forest': {
                'n_estimators': [100, 150],
                'random_state': [42]
            },
            'KNN regressor': {
                'n_neighbors': [3, 4, 5, 8, 10],
            }
        }
    
    def generate_metrics(self, name, model, X_test, y_test):
        self.cv_models.append((name, model))
        p = model.predict(X_test)
                
        d = {
        'Name': name,
        'r2 score': r2_score(y_test, p),
        'MRE': mean_squared_error(y_test, p),
        'MRSE': mean_squared_error(y_test, p) ** (1/2),
        'MAE': mean_absolute_error(y_test, p),
        }
        self.metrics.append(d)
    
    def evaluate_models(self, X_train, X_test, y_train, y_test, cv=None, verb=0):
        for name, model in self.models:
            if name == 'Linear regression':
                y_train.columns = ['target']
                fd = pd.concat([X_train, y_train], axis=1)
                corr = fd.corr()['target'].drop('target')             
                a = np.absolute(corr.values).max()
                ind = corr[(corr == a) | (corr == -a)].index[0]
                lr_train = X_train[ind]

                model = LinearRegression()

                model.fit(lr_train.values.reshape(-1, 1), y_train)
                
                self.generate_metrics(name, model, X_test[ind].values.reshape(-1, 1), y_test)
                
                continue
                
            if(cv==True and name in self.grid_searchs.keys()):
                print(f'Cross validating {name}, it may take some time!')
                m = GridSearchCV(model, param_grid = self.grid_searchs[name], refit=True, verbose=verb)
                m.fit(X_train, y_train.values.ravel())
                self.generate_metrics(name+'_cv', m, X_test, y_test)
                
            model.fit(X_train, y_train.values.ravel())
            
            self.generate_metrics(name, model, X_test, y_test)

        self.models.extend(self.cv_models)    
            
        self.metrics = pd.DataFrame(self.metrics)
        
        self.metrics = self.metrics.set_index('Name').sort_values(by=['r2 score'], ascending=False)
    
    def select_best_r2(self):
        best = self.metrics[self.metrics['r2 score'] == self.metrics['r2 score'].max()]
        print(best)
        for model in self.models:
            if model[0] == best.index:
                return model[1]

# In this example we'll be using the boston housing dataset
from sklearn.datasets import load_boston
data = load_boston()

X = pd.DataFrame(data['data'], columns=data.feature_names)
y = pd.DataFrame(data['target'], columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

scaler.fit_transform(X_train)
scaler.fit_transform(X_test)

#Without grid search
tester = model_evaluator()
tester.evaluate_models(X_train, X_test, y_train, y_test)

print(tester.metrics)

print()

model = tester.select_best_r2()
print(model)

print()

#With grid search
tester2 = model_evaluator()

#only use this if you have a lot of processing and RAM!
tester2.evaluate_models(X_train, X_test, y_train, y_test, cv=True)

print(tester2.metrics)

print()

model2 = tester2.select_best_r2()
print(model2)