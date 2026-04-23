#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:41:07 2024

@author: cristiantobar
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

def svm_regresion(X_train, y_train):
    modelo = make_pipeline(StandardScaler(), SVR(kernel='linear', C=100, epsilon=0.2))  # Ajusta los hiperparámetros si es necesario
    modelo.fit(X_train, y_train)
    return modelo