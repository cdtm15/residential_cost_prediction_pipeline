#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:05:33 2024

@author: cristiantobar
"""

import pandas as pd # to load and manipulate data and for One-Hot Encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import cross_val_score # for cross validation
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score



def model_ct(X_train, X_test, y_train, y_test, X_encoded, print_figures):
    ## create a decisiont tree and fit it to the training data
    clf_dt = DecisionTreeClassifier(random_state=42)
    clf_dt = clf_dt.fit(X_train, y_train)

    true_label  = "Delayed"
    false_label = "On-time"
    
    print("Printing Preliminary Classification tree")
    
    if print_figures == True:
        ## NOTE: We can plot the tree and it is huge!
        plt.figure(figsize=(37, 18))
        plot_tree(clf_dt, 
                  filled=True, 
                  rounded=True, 
                  class_names=[false_label, true_label], 
                  feature_names=X_encoded.columns); 
    
        plt.show()
    
        ## plot_confusion_matrix() will run the test data down the tree and draw
        ## a confusion matrix.
        ConfusionMatrixDisplay.from_estimator(clf_dt, 
                                              X_test, 
                                              y_test, 
                                              display_labels=[false_label, true_label])
     #%% Cross complexity Pruning, se identifican los alphas que garantizan la mayor precisión entre
     #los distintos tipos de arbol que son generados por cada variación de Alpha
    print("Evaluating Alpha value vs. accuracy")

    path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
    ccp_alphas = path.ccp_alphas # extract different values for alpha
    ccp_alphas = ccp_alphas[:-2] # exclude the maximum value for alpha

    clf_dts = [] # create an array that we will put decision trees into

    ## now create one decision tree per value for alpha and store it in the array
    for ccp_alpha in ccp_alphas:
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf_dt.fit(X_train, y_train)
        clf_dts.append(clf_dt)


    train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
    test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

    if print_figures == True:
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
        ax.legend()
    #%% Validación cruzada para hacer las pruebas de si este nuevo arbol con este alpha le va bien
    # con otras variaciones en las divisiones para training y testing datasets.

    
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.020) # create the tree with ccp_alpha=0.016
    ## now use 5-fold cross validation create 5 different training and testing datasets that
    ## are then used to train and test the tree.
    ## NOTE: We use 5-fold because we don't have tons of data...
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5) 
    df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
    # df.plot(x='tree', y='accuracy', marker='o', linestyle='--')


    ## create an array to store the results of each fold during cross validiation
    alpha_loop_values = []

    ## For each candidate value for alpha, we will run 5-fold cross validation.
    ## Then we will store the mean and standard deviation of the scores (the accuracy) for each call
    ## to cross_val_score in alpha_loop_values...
    for ccp_alpha in ccp_alphas:
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
        alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
    
    ## Now we can draw a graph of the means and standard deviations of the scores
    ## for each candidate value for alpha
    alpha_results = pd.DataFrame(alpha_loop_values, 
                                  columns=['alpha', 'mean_accuracy', 'std'])
    
    
    if print_figures == True:
        alpha_results.plot(x='alpha', 
                        y='mean_accuracy', 
                        yerr='std', 
                        marker='o', 
                        linestyle='--')

    #%%
    # alpha_results[(alpha_results['alpha'] > 0.014)
    #               &
    #               (alpha_results['alpha'] < 0.015)]


    # ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014) 
    #                                 & 
    #                                 (alpha_results['alpha'] < 0.015)]['alpha']
    try:

        ideal_ccp_alpha = alpha_results.loc[alpha_results['mean_accuracy'].idxmax(),'alpha']
        #ideal_ccp_alpha = 0.071
        alpha_results_array = alpha_results.to_numpy()
        alpha_variety       = alpha_results_array[:,0]

        # ## Build and train a new decision tree, only this time use the optimal value for alpha
        clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
        clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train) 
        accu_val = clf_dt_pruned.score(X_test, y_test)
        
        y_predict = clf_dt_pruned.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
        mse          = mean_squared_error(y_test, y_predict)
        r2           = r2_score(y_test, y_predict)

        
        accu  = (tp+tn)/(tp+fp+fn+tn)
        sensi = (tp)/(tp+fn)
        speci = (tn)/(tn+fp)
        f1    = (2*tp)/((2*tp)+fp+fn)
        
        print("Printing Pruned Classification tree")

        #ConfusionMatrixDisplay.from_estimator(clf_dt_pruned, X_test, y_test,display_labels=[false_label, true_label])

        if print_figures == True:
            plt.figure(figsize=(15, 10))
            plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=[false_label, true_label], feature_names=X_encoded.columns); 
            plt.savefig('ct.pdf')
            plt.show()
            # #plt.figure(figsize=(15, 7.5))
            # #plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=[false_label, true_label],); 
            # #plt.show()
            
            ConfusionMatrixDisplay.from_estimator(clf_dt_pruned, 
                                                  X_test, 
                                                  y_test, 
                                                  display_labels=[false_label, true_label])
            plt.show()
            
        #print("***TREE RULES FOR alpha: %s" %alpha)
        #verify_tree_rules(clf_dt_pruned, X_test)
        
        n_feat  = X_train.shape[1]
        n_train = len(X_train)
        n_test  = len(X_test)
        
        return [n_feat, n_train, n_test, r2, accu, sensi, speci, f1, mse]    
    except Exception:
            breakpoint()