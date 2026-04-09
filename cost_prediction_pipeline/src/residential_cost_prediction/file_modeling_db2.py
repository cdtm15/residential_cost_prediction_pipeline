#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:03:48 2024

@author: cristiantobar
"""
import numpy as np
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from file_model_ct import model_ct
from file_model_random_forest import model_random_forest
from file_model_logistic_reg import model_logistic_reg
from file_model_knn import model_knn
from file_model_svm import model_svm
from file_model_ann import model_ann

models         = ['ct', 'forest', 'logistic', 'knn', 'svm', 'ann']

def modeling_db2(df, sorted_features):
        
    feat_names = list(sorted_features.index.values)
    sorted_df  = df[feat_names]
    table_met = np.zeros((len(feat_names),10,6))

    
    for id_model, model in enumerate(models):
        for index in range(len(feat_names)):
                        
            elastic_df   = sorted_df.iloc[:,0:index+1]
            
            X_encoded    = elastic_df.copy()
            num_features = X_encoded.shape[1]
            Y            = df['output'].copy()
            
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3, random_state=42)
    
            "Modeling"
            
            if model == 'ct':
                results = model_ct(X_train, X_test, y_train, y_test, X_encoded, False)
                
            if model == 'forest':
                results = model_random_forest(X_train, X_test, y_train, y_test, X_encoded)

            if model == 'logistic':
                results = model_logistic_reg(X_train, X_test, y_train, y_test, X_encoded)

            if model == 'knn':
                results = model_knn(X_train, X_test, y_train, y_test, X_encoded)

            if model == 'svm':
                results = model_svm(X_train, X_test, y_train, y_test, X_encoded)

            if model == 'ann':
                results = model_ann(X_train, X_test, y_train, y_test, X_encoded)
                
            "Evaluation"
            table_met[index,:,id_model]  = [num_features] + results
    
    return table_met 
            
        
        
    #Generar sobre esto la clusterización jerarquica aglomerada para
    #generar el Y, estoy generando distintos outputs de la clusterización jerarquica
    
    #Poner el Y normal para la tarea de regresión
    
    #Poner un Y con umbral por costoso y barato para la tarea de clasificación
    
    #Enviar a los algoritmos el df con el y
    
    #Obtener R2, y las métricas que manejamos en una tabla
    
    
