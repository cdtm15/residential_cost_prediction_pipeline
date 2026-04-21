#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:47:37 2026

@author: cristian
"""

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from residential_cost_prediction.models.file_model_ann_v2 import ann_regresion
from residential_cost_prediction.models.file_model_random_forest import rf_regresion
from residential_cost_prediction.models.file_model_svm import svm_regresion

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def evaluate_model_with_cv(X, y, ml_tech, n_splits=5, n_repeats=5, random_state=42, ann_params=None):
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    r2_scores           = []
    mae_scores          = []
    accuracy_scores     = []
    relative_errors_all = []

    y_test_all = []
    y_pred_all = []

    #for train_idx, test_idx in rkf.split(X):
    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler          = StandardScaler()
        X_train         = scaler.fit_transform(X_train)
        X_test          = scaler.transform(X_test)

        if ml_tech == 'ANN':
            #model = ann_regresion(X_train, y_train)
            model = ann_regresion(X_train, y_train, ann_params=ann_params)
            
        elif ml_tech == 'SVM':
            model = svm_regresion(X_train, y_train)

        elif ml_tech == 'RF':
            model = rf_regresion(X_train, y_train)

        else:
            raise ValueError(f"ml_tech no reconocido: {ml_tech}")

        y_pred = model.predict(X_test).flatten()

        r2     = r2_score(y_test, y_pred)
        mae    = mean_absolute_error(y_test, y_pred)

        # Ojo: esta métrica no es estándar en regresión
        accuracy = 1 - np.mean(np.abs((y_test - y_pred) / y_test))

        relative_errors = (y_test - y_pred) / y_test * 100

        r2_scores.append(r2)
        mae_scores.append(mae)
        accuracy_scores.append(accuracy)
        relative_errors_all.extend(relative_errors.tolist())

        y_test_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        
        if fold % 10 == 0:
            logging.info(f"Modelo: {ml_tech} | Fold {fold+1}/{n_splits*n_repeats}")
    return {
        'r2_mean':             np.mean(r2_scores),
        'r2_std':              np.std(r2_scores, ddof=1),
        'mae_mean':            np.mean(mae_scores),
        'mae_std':             np.std(mae_scores, ddof=1),
        'acc_mean':            np.mean(accuracy_scores),
        'acc_std':             np.std(accuracy_scores, ddof=1),
        'relative_errors_all': relative_errors_all,
        'y_test_all':          np.array(y_test_all),
        'y_pred_all':          np.array(y_pred_all)
    }