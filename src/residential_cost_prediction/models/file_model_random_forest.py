#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:07:24 2024

@author: cristiantobar
"""

# from sklearn.ensemble import RandomForestRegressor

# def rf_regresion(X_train, y_train):
#     modelo = RandomForestRegressor(n_estimators=100, random_state=42)  # Ajusta los hiperparámetros si es necesario
#     modelo.fit(X_train, y_train)
#     return modelo    

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def grid_search_rf(X, y, cv=5, n_jobs=-1, random_state=42):
    model = RandomForestRegressor(random_state=random_state)

    param_grid = {
        "n_estimators"      : [100, 300, 500],
        "max_depth"         : [None, 5, 10, 20],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf"  : [1, 2, 4],
        "max_features"      : ["sqrt", "log2", 1.0]
    }

    search = GridSearchCV(
        estimator  = model,
        param_grid = param_grid,
        cv         = cv,
        scoring    = "neg_mean_absolute_error",
        n_jobs     = n_jobs
    )

    search.fit(X, y)
    return search.best_params_

def rf_regresion(X_train, y_train, rf_params=None, random_state=42):
    if rf_params is None:
        
        print("****[WARNING]**** USING DEFAULT RF PARAMETERS")
        
        rf_params = {
            "n_estimators"     : 100,
            "max_depth"        : None,
            "min_samples_split": 2,
            "min_samples_leaf" : 1,
            "max_features"     : 1.0
        }

    modelo = RandomForestRegressor(
        n_estimators      = rf_params["n_estimators"],
        max_depth         = rf_params["max_depth"],
        min_samples_split = rf_params["min_samples_split"],
        min_samples_leaf  = rf_params["min_samples_leaf"],
        max_features      = rf_params["max_features"],
        random_state      = random_state
    )

    modelo.fit(X_train, y_train)
    return modelo