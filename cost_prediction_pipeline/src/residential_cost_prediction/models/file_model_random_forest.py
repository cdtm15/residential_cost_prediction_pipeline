#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:07:24 2024

@author: cristiantobar
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, f1_score
import pandas as pd # to load and manipulate data and for One-Hot Encoding
import matplotlib.pyplot as plt


def model_random_forest(X_train, X_test, y_train, y_test, X_encoded):    
    n_train = len(X_train)
    n_test  = len(X_test)
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    rf.score(X_test, y_test)
    
    n_feat  = X_train.shape[1]
    
    # accu   = accuracy_score(y_test, y_predict)
    # r2     = r2_score(y_test, y_predict)
    # mse    = mean_squared_error(y_test, y_predict)
    # recall = recall_score(y_test, y_predict)
    # f1     = f1_score(y_test, y_predict)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    r2             = r2_score(y_test, y_predict)
    mse    = mean_squared_error(y_test, y_predict)

    accu  = (tp+tn)/(tp+fp+fn+tn)
    sensi = (tp)/(tp+fn)
    speci = (tn)/(tn+fp)
    f1    = (2*tp)/((2*tp)+fp+fn)
    
    features = pd.DataFrame(rf.feature_importances_, index=X_train.columns)
    
    importances = rf.feature_importances_
    
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    
    # fig, ax = plt.subplots()
    # features.plot.bar(yerr)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
        
    return [n_feat, n_train, n_test, r2, accu, sensi, speci, f1, mse]