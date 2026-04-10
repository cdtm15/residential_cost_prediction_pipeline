#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:41:30 2024

@author: cristiantobar
"""

import tensorflow as tf


def ann_regresion(X_train, y_train):
    # Crear el modelo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train, epochs=500, verbose=0)
    
    return modelo