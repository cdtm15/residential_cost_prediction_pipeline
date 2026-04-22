#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:56:41 2026

@author: cristian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import logging
from sklearn.preprocessing import StandardScaler

from residential_cost_prediction.file_cross_validation import evaluate_model_with_cv
from residential_cost_prediction.models.file_model_ann_v2 import random_search_ann

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def modeling_regresion_db2_cv(
    df_clustered,
    nature,
    ml_tech,
    folder_path,
    sorted_feat_subproj,
    outlier_scenario,
    n_splits=5,
    n_repeats=5,
    random_state=42
):
        
    df_subproj_ext = df_clustered.drop([
        'built_area',
        'lot_area',
        'total_prelim_cost_est',
        'prelim_cost_est_est',
        'equi_prelim_cost',
        'duration',
        'unit_price',
        'output',
        'actual_sale_price'
    ], axis=1).copy()

    features_iniciales = [
        'built_area',
        'lot_area',
        'total_prelim_cost_est',
        'prelim_cost_est_est',
        'equi_prelim_cost',
        'duration',
        'unit_price'
    ]
        
    features_adicionales = list(sorted_feat_subproj.feature)
    target = 'actual_construction_cost'
    ann_params = None

    if ml_tech == 'ann':
        baseline_features = features_iniciales
        X_base = df_clustered[baseline_features].values
        y_base = df_clustered[target].values
    
        scaler_base = StandardScaler()
        X_base = scaler_base.fit_transform(X_base)
    
        logging.info("Iniciando búsqueda de hiperparámetros ANN con fixed baseline...")
        ann_params = random_search_ann(X_base, y_base, n_iter=10, random_state=random_state)
        logging.info(f"Mejores hiperparámetros ANN: {ann_params}")
    
    resultados = []
    errores_relativos = []
    num_features_list = []

    num_cols = 4
    num_rows = int(np.ceil((len(features_adicionales) + 1) / num_cols))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.5 * num_cols, 3.5 * num_rows)
    )

    axes = np.array(axes).reshape(num_rows, num_cols)
    
    for i in range(0, len(features_adicionales) + 1):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        if i == 0:
            features = features_iniciales
            feature_name = 'fixed_baseline'
        else:
            features = features_iniciales + features_adicionales[:i]
            feature_name = features_adicionales[i - 1]
        
        print(f"Adding the market feature: {i}: {feature_name}")
        
        X = df_clustered[features].values
        y = df_clustered[target].values

        cv_results = evaluate_model_with_cv(
            X            = X,
            y            = y,
            ml_tech      = ml_tech,
            n_splits     = n_splits,
            n_repeats    = n_repeats,
            random_state = random_state,
            ann_params   = ann_params
        )
                
        r2_mean = cv_results['r2_mean']
        r2_std = cv_results['r2_std']
        mae_mean = cv_results['mae_mean']
        mae_std = cv_results['mae_std']
        acc_mean = cv_results['acc_mean']
        acc_std = cv_results['acc_std']

        y_test_all = cv_results['y_test_all']
        y_pred_all = cv_results['y_pred_all']

        resultados.append([
            len(features),
            feature_name,
            r2_mean,
            r2_std,
            acc_mean,
            acc_std,
            mae_mean,
            mae_std
        ])
        
        logging.info(f"Modelo: {ml_tech} | Iteración {i} | Num features: {len(features)} | Feature añadida: {feature_name}")
        
        errores_relativos.extend(cv_results['relative_errors_all'])
        num_features_list.extend([len(features)] * len(cv_results['relative_errors_all']))

        reg = LinearRegression().fit(y_test_all.reshape(-1, 1), y_pred_all)
        pendiente = reg.coef_[0]
        intercepto = reg.intercept_

        ax.scatter(y_test_all, y_pred_all, alpha=0.4)
        ax.plot(
            [min(y_test_all), max(y_test_all)],
            [min(y_test_all), max(y_test_all)],
            '--',
            color='red',
            label='Ideal'
        )

        ax.set_title(f'Number of features: {len(features)}')
        ax.set_xlabel('Actual Construction Cost (' + r'$ \$/m² $)')
        ax.set_ylabel('Predicted Construction Cost (' + r'$ \$/m² $)')

        ecuacion = (
            f'y = {pendiente:.4f}x + {intercepto:.0f}\n'
            f'R² = {r2_mean:.4f} ± {r2_std:.4f}\n'
            f'MAE = {mae_mean:.4f} ± {mae_std:.4f}'
        )
        ax.text(
            0.05,
            0.95,
            ecuacion,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top'
        )

    total_plots = num_rows * num_cols
    for j in range(len(features_adicionales) + 1, total_plots):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)

    pdf_path = os.path.join(
        folder_path,
        f'predicted_cost_{nature}_{ml_tech}_{outlier_scenario}_cv.pdf'
    )
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    df_results = pd.DataFrame(resultados, columns=[
        'Num Features',
        'Feature',
        'R² Mean',
        'R² Std',
        'Accuracy Mean',
        'Accuracy Std',
        'MAE Mean',
        'MAE Std'
    ])

    data = pd.DataFrame({
        'Number of Features': num_features_list,
        'Relative error (%)': errores_relativos
    })

    plt.figure(figsize=(6, 6))
    sns.boxplot(
        x='Number of Features',
        y='Relative error (%)',
        data=data,
        width=0.95,
        palette='Set2'
    )
    plt.title('Relative Errors by Number of Features (Repeated CV)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([-100, 100])

    pdf_path = os.path.join(
        folder_path,
        f'error_relative_{nature}_{ml_tech}_{outlier_scenario}_cv.pdf'
    )
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return df_results, sorted_feat_subproj, ann_params