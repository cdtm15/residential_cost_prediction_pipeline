#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:02:36 2026

@author: cristian
"""
import tensorflow as tf
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

def build_ann(n_features, n_units=64, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(n_units, activation='relu'),
        tf.keras.layers.Dense(n_units // 2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model

def random_search_ann(X, y, n_iter=10):

    model = KerasRegressor(
        model=build_ann,
        epochs=200,
        batch_size=32,
        verbose=0
    )
    
    param_dist = {
        "model__n_units": [32, 64, 128],
        "model__lr": [1e-2, 1e-3, 1e-4],
        "batch_size": [16, 32],
        "epochs": [100, 200]
    }
    
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    search.fit(X, y)
    
    return search.best_params_


#modelo = build_ann(
#    n_units=best_params["model__n_units"],
#    lr=best_params["model__lr"]
#)