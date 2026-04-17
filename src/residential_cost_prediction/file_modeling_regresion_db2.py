#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:14:50 2024

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import os

from residential_cost_prediction.models.file_model_ann import ann_regresion
from residential_cost_prediction.models.file_model_random_forest import rf_regresion
from residential_cost_prediction.models.file_model_svm import svm_regresion
from file_cross_validation import evaluate_model_with_cv


def modeling_regresion_db2(df_clustered, nature, ml_tech, folder_path, sorted_feat_subproj, outlier_scenario):
    
    df_subproj_ext = df_clustered.drop(['built_area',
    'lot_area',
    'total_prelim_cost_est',
    'prelim_cost_est_est',
    'equi_prelim_cost',
    'duration',
    'unit_price',
    'output', 'actual_sale_price'], axis=1).copy()
    
    #sorted_feat_subproj = feature_importance(df_subproj_ext, nature, folder_path) 
        
    # 2. Definir las variables de entrada y salida
    features_iniciales = ['built_area', 'lot_area', 'total_prelim_cost_est', 'prelim_cost_est_est', 
                          'equi_prelim_cost', 'duration', 'unit_price']
    features_adicionales = list(sorted_feat_subproj.index)
        
    target = 'actual_construction_cost'
    
    resultados = []
    errores_relativos = []
    num_features_list = []  # Lista para almacenar el número de features para cada error

    # 5. Generar los plots en una matriz
    num_cols = 4  # Número de columnas en la matriz de subplots
    num_rows = int(np.ceil((len(features_adicionales)+1)/ num_cols))  # Número de filas
        
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols , 3.5 * num_rows))  # Ajusta figsize según sea necesario
    

    for i in range(0, len(features_adicionales)+1):        # Calcular la posición del subplot en la matriz
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]  # Manejar caso de una sola fila
                
        # 3. Crear y evaluar el modelo con diferentes combinaciones de variables
        if i==0:
            features = features_iniciales
        else: 
            features = features_iniciales + features_adicionales[:i]
        
        X = df_clustered[features].values
        y = df_clustered[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
                
        if ml_tech == 'ann':
            modelo = ann_regresion(X_train, y_train)
        
        if ml_tech == 'svm':
            modelo = svm_regresion(X_train, y_train)
            
        if ml_tech == 'rf':
            modelo = rf_regresion(X_train, y_train)
    
        y_pred = modelo.predict(X_test).flatten() 
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
                
        # Accuracy no se usa normalmente en regresión, pero se puede calcular como 1 - error relativo
        accuracy = 1 - np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) 
        errores = (y_test - y_pred.flatten()) / y_test * 100
        errores = errores.tolist()        
        
        if i==0:
            resultados.append([len(features), 'fixed_baseline', r2, accuracy, mae])
        else:
            resultados.append([len(features), features_adicionales[i-1], r2, accuracy, mae])
        #errores_relativos.append(errores)
        errores_relativos.extend(errores)
        num_features_list.extend([len(features)] * len(errores))  # Agregar el número de features correspondiente a cada error
        
        # Calcular la regresión lineal para la ecuación en el plot
        reg = LinearRegression().fit(y_test.reshape(-1, 1), y_pred)
        pendiente = reg.coef_[0]
        intercepto = reg.intercept_
        
        # Crear el plot
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Ideal')
        
        ax.set_title(f'Number of features: {len(features)}')
        ax.set_xlabel('Actual Construction Cost ('+r'$ \$/m² $)')
        ax.set_ylabel('Predicted Construction Cost ('+r'$ \$/m² $)')
        
        # Agregar la ecuación y R² al plot
        ecuacion = f'y = {pendiente:.4f}x + {intercepto:.0f}\nR² = {r2:.4f}'
        ax.text(0.05, 0.95, ecuacion, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    
    # Ocultar subplots vacíos si hay menos plots que espacios en la matriz
    if len(resultados) < num_rows * num_cols:
        for i in range(len(resultados), num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.axis('off')  # Ocultar el subplot
    
    plt.tight_layout()
    #plt.suptitle(nature+'_'+ml_tech)
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, 'predicted_cost_'+nature+'_'+ml_tech+'_'+outlier_scenario+'.pdf') 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
        
    df_results = pd.DataFrame(resultados)
    df_results.columns = ['Num Features', 'Feature', 'R²', 'Accuracy', 'MAE']
        
    data = pd.DataFrame({'Number of Features': num_features_list, 'Relative error (%)': errores_relativos})
        
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Number of Features', y='Relative error (%)', data=data, width=0.95, palette='Set2')
    plt.title('Relative Errors by Number of Features')
    plt.xticks(rotation=45, ha='right') 
    plt.ylim([-100, 100])  # Ajusta los valores según tus necesidades
    
    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, 'error_relative'+nature+'_'+ml_tech+'_'+outlier_scenario+'.pdf') 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    
    return df_results, sorted_feat_subproj
