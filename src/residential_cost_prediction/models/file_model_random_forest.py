#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:07:24 2024

@author: cristiantobar
"""

from sklearn.ensemble import RandomForestRegressor

def rf_regresion(X_train, y_train):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)  # Ajusta los hiperparámetros si es necesario
    modelo.fit(X_train, y_train)
    return modelo    