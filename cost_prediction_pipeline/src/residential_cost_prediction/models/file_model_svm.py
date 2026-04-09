#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:41:07 2024

@author: cristiantobar
"""
import pandas as pd # to load and manipulate data and for One-Hot Encoding
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

def model_svm(X_train, X_test, y_train, y_test, X_encoded):
    n_train = len(X_train)
    n_test  = len(X_test)

    # X_train_scaled = scale(X_train)
    # X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # X_test_scaled  = scale(X_test)
    # X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    #breakpoint()
    
    clf = SVC(kernel='linear', random_state = 42)
    clf.fit(X_train, y_train)
    
    n_feat  = X_train.shape[1]

    
    y_predict      = clf.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    accu           = (tp+tn)/(tp+fp+fn+tn)
    sensi          = (tp)/(tp+fn)
    speci          = (tn)/(tn+fp)
    f1             = (2*tp)/((2*tp)+fp+fn)
    mse            = mean_squared_error(y_test, y_predict)
    r2             = r2_score(y_test, y_predict)

    return [n_feat, n_train, n_test, r2, accu, sensi, speci, f1, mse]