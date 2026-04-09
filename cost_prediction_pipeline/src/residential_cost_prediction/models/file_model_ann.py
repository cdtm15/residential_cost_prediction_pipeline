#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:41:30 2024

@author: cristiantobar
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def model_ann(X_train, X_test, y_train, y_test, X_encoded):
    n_train = len(X_train)
    n_test  = len(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    
    
    clf = MLPClassifier(solver='lbfgs', alpha = 1e-5, 
                        hidden_layer_sizes=(10,4), random_state=1)
    
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    
    n_feat  = X_train.shape[1]

    
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    accu           = (tp+tn)/(tp+fp+fn+tn)
    sensi          = (tp)/(tp+fn)
    speci          = (tn)/(tn+fp)
    f1             = (2*tp)/((2*tp)+fp+fn)
    mse            = mean_squared_error(y_test, y_predict)
    r2             = r2_score(y_test, y_predict)
        
    return [n_feat, n_train, n_test, r2, accu, sensi, speci, f1, mse]
    