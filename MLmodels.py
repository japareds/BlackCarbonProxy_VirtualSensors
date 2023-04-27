#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:20:19 2023

@author: jparedes
"""
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate,learning_curve, validation_curve,GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV,RFE
from sklearn import svm, tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

#%%
def get_algorithm_params(algorithm,X_train,feature_selection):
    n_samples,n_features = X_train.shape
    
    if feature_selection and algorithm=='SVR':
        grid_params = {
            'selector__estimator__C':np.logspace(-1,1,3),
            'selector__estimator__epsilon':np.logspace(-3,-1,3),
            'selector__n_features_to_select':np.arange(1,X_train.shape[1]),
            'model__C':np.logspace(-3,3,7),
            'model__gamma':np.logspace(-3,3,7),
            'model__epsilon':np.linspace(0.2,0.8,4)
            }
        #model = svm.SVR(kernel='rbf')
        model = svm.LinearSVR(
            loss = 'epsilon_insensitive',
            random_state=92,
            max_iter=100000,
            tol=1e-4
            )
        
    
    elif algorithm=='RF':
        
        model = RandomForestRegressor(criterion='squared_error', min_samples_leaf=2,max_leaf_nodes=None,
                                      bootstrap=True, oob_score=False, random_state=92,
                                      n_jobs=2, verbose=1, warm_start=False)

        grid_params = {
            'model__n_estimators':[100,500,1000,3000],
            'model__max_depth':[20,10,5,3],
            'model__min_samples_split':[5,2],
            'model__max_samples':[1.0,0.66],
            'model__max_features':[1.0,0.5,0.33]
            }   
    
    elif algorithm=='MLP':
        
        
        model = MLPRegressor(solver='adam',learning_rate='adaptive',batch_size=32,
                                 max_iter=1000,shuffle=False,random_state=92,warm_start=False,tol=1e-4,early_stopping=True)
        grid_params = {
            'model__learning_rate_init':np.logspace(-4,-2,3),
            'model__hidden_layer_sizes':[(int(np.ceil(1.33*n_features)),),
                                             (n_features,),
                                             (int(0.5*n_features),),
                                             (n_features,n_features),
                                             (int(0.5*n_features),int(0.5*n_features))],
            'model__alpha':np.logspace(-6,-2,3),
            'model__activation':['tanh','relu']
            }

    return grid_params,model

def BFS(X_train,Y_train,cv,algorithm):
    grid_params, model_rfe =  get_algorithm_params(algorithm,X_train,feature_selection=True)
    scaler = StandardScaler()
    selector = RFE(
        estimator=model_rfe,
        step=1,
        verbose=1
        )
    model = svm.SVR(kernel='rbf')
    pipe = Pipeline([('scaler',scaler),('selector',selector),('model',model)])
    
    # gridsearch
    gs = GridSearchCV(pipe, grid_params, 
                      scoring='neg_root_mean_squared_error',
                      n_jobs=2, 
                      refit=True, 
                      cv=cv,
                      verbose=1,
                      pre_dispatch=4,
                      return_train_score=True)
    
    start_time = time.time()
    gs.fit(X_train,Y_train.values.ravel())
    end_time = time.time()
    
    gs_results = pd.DataFrame(gs.cv_results_)
    gs_results = gs_results.sort_values(by='rank_test_score')
    print('Grid search finished in %.2f'%(end_time-start_time))
    print(f'Best hyperparameters {gs.best_params_}')
    print('Features importance in decreasing order (the first feature is the most important)')
    print([x for _, x in sorted(zip(gs.best_estimator_['selector'].ranking_, X_train.columns))])
    
    return gs,gs_results


#%%
def model_fit(X_train,Y_train,cv,algorithm):
    """
    
    Grid Search CV hyperparameters optimization

    Parameters
    ----------
    X_train : pandas DataFrame
            predictors training set
            
    Y_train : pandas DataFrame
            BC concentration training set
            
    cv : sklearn cv
    cross-validation scheme
    
    model : str
            ML algorithm for regression
            
    perform_pca : bool
        include or not dimensionality reduction via PCA 

    Returns
    -------
    gs : sklearn gridsearch object
        fitted grid for different hyperparameters combinations   
    
    results : pandas DataFrame
            gridsearch results for different hyperparamters combinations

    """
    
    n_samples,n_features = X_train.shape
    # Select a model
    if algorithm not in ['SVR','RF','MLP']:
        raise Exception(f'{algorithm} not implemented. Choose between [SVR,RF,MLP]')
    
    if algorithm=='SVR':
        print('---------------\nFitting SVR\n----------------')
        grid_params = {
            'model__C':np.logspace(-3,3,7),
            'model__gamma':np.logspace(-3,3,7),
            'model__epsilon':np.linspace(0.1,0.8,5)
            }
        scaler = StandardScaler()
        model = svm.SVR(kernel='rbf')
        pipe = Pipeline([('scaler',scaler),('model', model)])
            
    
    elif algorithm=='RF':
        print('----------\nFitting RF\n----------')
        model = RandomForestRegressor(criterion='squared_error', min_samples_leaf=2,max_leaf_nodes=None,
                                      bootstrap=True, oob_score=False, random_state=92,
                                      n_jobs=2, verbose=1, warm_start=False)

        grid_params = {
            'model__n_estimators':[100,500,1000,3000],
            'model__max_depth':[20,10,5,3],
            'model__min_samples_split':[5,2],
            'model__max_samples':[1.0,0.66],
            'model__max_features':[1.0,0.5,0.33]
            }   
    
        pipe = Pipeline(steps=[('model', model)])


    elif algorithm=='MLP':
        print('-----------\nFitting MLP\n----------')
        
        model = MLPRegressor(solver='adam',learning_rate='adaptive', activation='tanh',batch_size=32,
                                 max_iter=1000,shuffle=False,random_state=92,warm_start=False,tol=1e-4,early_stopping=False)            
        grid_params = {
            'model__learning_rate_init':np.logspace(-4,-2,3),
            'model__hidden_layer_sizes':[(int(np.ceil(1.33*n_features)),),
                                             (n_features,),
                                             (int(0.5*n_features),),
                                             (n_features,n_features),
                                             (int(0.5*n_features),int(0.5*n_features))],
            'model__alpha':np.logspace(-6,-2,3)
            }
        
        
        scaler = StandardScaler()
        pipe = Pipeline([('scaler',scaler),('model', model)])

    # gridsearch
    print('Fitting model ',pipe)
    start_time = time.time()
    
    if model =='RF':
        n_jobs = 2
        pre_dispatch = 2
    else:
        n_jobs = 2
        pre_dispatch = 4
        
    gs = GridSearchCV(pipe, grid_params, scoring='neg_root_mean_squared_error',
                      n_jobs=n_jobs, refit=True, cv=cv, verbose=1,
                      pre_dispatch=pre_dispatch, return_train_score=True)
    gs.fit(X_train,Y_train.values.ravel())
    end_time = time.time()
    
    gs_results = pd.DataFrame(gs.cv_results_)
    gs_results = gs_results.sort_values(by='rank_test_score')
    print('Grid search cv finished in %.2f'%(end_time-start_time))
    
    return gs,gs_results

        
#%% model selection and scoring

def SelectModel(algorithm='SVR'):
    if algorithm not in ['SVR','RF','MLP']:
        raise Exception(f'{algorithm} not implemented. Choose between [SVR,RF,MLP]')
    
    if algorithm=='SVR':
        print('---------------\nSelecting optimal SVR model\n----------------')
        C = 10#1000
        e = 0.1#0.275
        g = 0.01#0.01
        scaler = StandardScaler()
        model = svm.SVR(kernel='rbf',C=C,epsilon=e,gamma=g)
        pipe = Pipeline([('scaler',scaler),('model', model)])
        print(f'Loading model {pipe}')


    elif model=='RF':
        print('----------\nSelecting optimal RF model\n----------')
        n_estimators = 0.0
        max_depth = 0.0
        min_samples_split = 0.0
        max_samples = 0.0
        max_features = 0.0
        
        model = RandomForestRegressor(criterion='squared_error', min_samples_leaf=2,max_leaf_nodes=None,
                                      bootstrap=True, oob_score=False, random_state=92,
                                      n_jobs=2, verbose=1, warm_start=False,
                                      n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      min_samples_split=min_samples_split,
                                      max_samples = max_samples,
                                      max_features=max_features)
        pipe = Pipeline(steps=[('model', model)])
        
    elif model=='MLP':
        print('-----------\nSelecting optimal MLP model\n----------')
        lr=0.0
        hs = (1,)
        alpha = 0.0
        bs = 0.
        scaler = StandardScaler()
        model = MLPRegressor(solver='adam',learning_rate='adaptive', activation='tanh',
                              max_iter=5000,shuffle=True,random_state=92,warm_start=False,tol=1e-4,early_stopping=False,
                              learning_rate_init=lr,
                              hidden_layer_sizes = hs,
                              alpha=alpha,
                              batch_size=bs)
        pipe = Pipeline([('scaler',scaler),('model', model)])
         
    return pipe

def scoring(Y_true,y_pred):
    """
    Computes scoring metrics for predictions
    """
    RMSE = np.sqrt(mean_squared_error(Y_true,y_pred))
    R2 = r2_score(Y_true,y_pred)
    return RMSE,R2


    