# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:53:08 2018

@author: t_alli
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from scipy.stats import randint, uniform

from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

#Load data
filename = 'wrangled_cleveland.csv'
cleveland = pd.read_csv(filename)

# Split data into training and test sets
X = cleveland.iloc[:,0:13]
Y = cleveland.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify = Y)

# Map columns to transformations 
mapper = DataFrameMapper([
    # Standardize continuous features 
    (['Age'], preprocessing.StandardScaler()), 
    (['resting_bp'], preprocessing.StandardScaler()),
    (['serum_chol'], preprocessing.StandardScaler()),
    (['max_hr'], preprocessing.StandardScaler()),
    (['ST_dep'], preprocessing.StandardScaler()),
    # Convert categorical features to one-hot encoded labels
    (['Sex'], preprocessing.OneHotEncoder()), 
    (['Chest_pain_type'], preprocessing.OneHotEncoder()), 
    (['blood_sugar'], preprocessing.OneHotEncoder()),
    (['ecg'], preprocessing.OneHotEncoder()),
    (['e_i_angina'], preprocessing.OneHotEncoder()),
    (['slope'], preprocessing.OneHotEncoder()),
    (['ca'], preprocessing.OneHotEncoder()),
    (['thal'], preprocessing.OneHotEncoder())
])

clf_names = ['Support Vector Machine', 
             'Logistic Regression',
             'K Nearest Neighbours',
            ]

algorithms = [SVC(),
              LogisticRegression(),
              KNeighborsClassifier(),
             ]

pipelines = []

# create list containing a pipeline construct for each classifier
for clf in algorithms:
    pipelines.append(Pipeline([
        ('mapper_and_scaler', mapper),
        ('classifier', clf)
    ]))

# Set random search parameters
hyperparameters = {0: {'classifier__C' : uniform(0.05, 10),
                       'classifier__gamma' : uniform(0.0001,0.2)},
                   1: {'classifier__C' : uniform(0.05, 10)},
                   2: {'classifier__n_neighbors' : np.arange(1, 21, 2),
                       'classifier__leaf_size' : randint(1,100)},
                  }  

# Set number of random search iterations
n_iter_search = 5

clfs = []
f1_scores = []

#For each classifier:
for idx, pipeline in enumerate(pipelines):
    print('\nEstimator: {}'.format(clf_names[idx]))
    
    # Implement a random search over the parameters. Uses K fold cross validation with K = 5  
    clfs.append(RandomizedSearchCV(pipeline, param_distributions = hyperparameters[idx], n_iter = n_iter_search, cv = 5))
    
    # Fit model
    clfs[idx].fit(X_train, Y_train)
    
    print('Best parameters: {}'.format(clfs[idx].best_params_))
    print('Best Training Score {}'.format(clfs[idx].best_score_))
    
    # Use model to predict Y values based on test set
    y_pred = clfs[idx].predict(X_test)
    
    # Evaluate model via precision, recall and F1 scores
    print('Test set recall {}'.format(recall_score(Y_test, y_pred)))
    print('Test set precision {}'.format(precision_score(Y_test, y_pred)))
    fc = f1_score(Y_test, y_pred)
    print('Test set f1 score {}'.format(fc))
    
    # Save F1 score
    f1_scores.append(fc)
    
best_classifier = f1_scores.index(max(f1_scores))

print('\nClassifier with the highest test set f1 score: {}'.format(clf_names[best_classifier]))