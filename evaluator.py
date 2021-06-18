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

class model_evaluator:
    def __init__(self):
        #You can add more models here
        self.models = [
            #use this structure: (name, model())
            ('Linear regression', LinearRegression()),
            ('Multiple linear regression', LinearRegression()),
            ('SVR', SVR()),
            ('Decision Tree', DecisionTreeRegressor(random_state=42)),
            ('Random Forest', RandomForestRegressor(random_state=42)),
            ('KNN regressor', KNeighborsRegressor()),
        ]
    
        self.cv_models = []
    
        self.predictions = []

        self.metrics = []
        
        #You can change the grid search params here
        #make sure the keys are the same name inside of self.models
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
        
        #you can add more evaluation metrics in this dict here
        #y test are the correct values and p the predicted values
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
            #This does only the simple linear regression
            #Using only the most correlated column
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
            
            #this is where we do the grid search cross-validation
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
    
    #This method returns the model that had the best R2 score in the evaluation
    def select_best_r2(self):
        best = self.metrics[self.metrics['r2 score'] == self.metrics['r2 score'].max()]
        print(best)
        for model in self.models:
            if model[0] == best.index:
                return model[1]