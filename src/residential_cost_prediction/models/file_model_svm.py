#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:41:07 2024

@author: cristiantobar
"""

# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline

# def svm_regresion(X_train, y_train):
#     modelo = make_pipeline(StandardScaler(), SVR(kernel='linear', C=100, epsilon=0.2))  # Ajusta los hiperparámetros si es necesario
#     modelo.fit(X_train, y_train)
#     return modelo

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def grid_search_svr(X, y, cv=5, n_jobs=-1):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])

    param_grid = {
        "svr__kernel":  ["linear", "rbf"],
        "svr__C":       [1, 10, 100, 500],
        "svr__epsilon": [0.01, 0.1, 0.2, 0.5],
        "svr__gamma":   ["scale", "auto"]
    }

    search = GridSearchCV(
        estimator  = pipe,
        param_grid = param_grid,
        cv         = cv,
        scoring    = "neg_mean_absolute_error",
        n_jobs     = n_jobs
    )

    search.fit(X, y)
    return search.best_params_

def svm_regresion(X_train, y_train, svr_params=None):
    if svr_params is None:
        print("****[WARNING]**** USING DEFAULT SVR PARAMETERS")
        
        svr_params = {
            "kernel"  : "linear",
            "C"       : 100,
            "epsilon" : 0.2,
            "gamma"   : "scale"
        }

    modelo = SVR(
        kernel  = svr_params["kernel"],
        C       = svr_params["C"],
        epsilon = svr_params["epsilon"],
        gamma   = svr_params["gamma"]
    )

    modelo.fit(X_train, y_train)
    return modelo