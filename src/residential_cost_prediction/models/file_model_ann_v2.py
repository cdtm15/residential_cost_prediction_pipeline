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
from sklearn.model_selection import GridSearchCV

def build_ann(meta, n_units, learning_rate):
    n_features = meta["n_features_in_"]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(n_units, activation='relu'),
        tf.keras.layers.Dense(max(n_units // 2, 1), activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model

def random_search_ann(X, y, n_iter=25, random_state=42):
    model = KerasRegressor(
        model=build_ann,
        epochs=200,
        batch_size=32,
        verbose=0
    )

    # param_dist = {
    #     "model__n_units": [32, 64, 128],
    #     "model__learning_rate": [1e-2, 1e-3, 1e-4],
    #     "batch_size": [16, 32],
    #     "epochs": [100, 200]
    # }
    
    # param_dist = {
    #     "model__n_units": [32, 64, 128, 256],
    #     "model__learning_rate": [1e-2, 1e-3, 1e-4, 5e-4, 1e-4],
    #     "batch_size": [8, 16, 32, 64],
    #     "epochs": [200, 300, 500]
    # }
    
    param_dist = {
        "model__n_units": [32, 64],
        "model__learning_rate": [1e-2, 1e-3],
        "batch_size": [8, 16],
        "epochs": [300, 500]
    }

    # search = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions=param_dist,
    #     n_iter=n_iter,
    #     cv=5,
    #     scoring='neg_mean_absolute_error',
    #     n_jobs=-1,
    #     random_state=random_state
    # )
    
    search = GridSearchCV(
            estimator=model,
            param_grid=param_dist,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
            )

    search.fit(X, y)

    return search.best_params_


def ann_regresion(X_train, y_train, ann_params=None):
        
    if ann_params is None:
        ann_params = {
            "model__n_units": 64,
            "model__learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 200
        }

    n_units = ann_params.get("model__n_units")
    learning_rate = ann_params.get("model__learning_rate")
    batch_size = ann_params.get("batch_size")
    epochs = ann_params.get("epochs")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(n_units, activation='relu'),
        tf.keras.layers.Dense(max(n_units // 2, 1), activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae']
    )

    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=20,
    #     restore_best_weights=True
    # )

    model.fit(
        X_train,
        y_train,
        #validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        #callbacks=[early_stop]
    )

    return model