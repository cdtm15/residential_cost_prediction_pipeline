#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:57:24 2026

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import shap
import os


def plot_shap_and_perf(model, X_test, df_perf, folder_path, outlier_scenario, nature, models, colors, top_n=None, save_tiff=False):
    """
    model: modelo entrenado (árbol, RF, XGB, etc.)
    X_test: DataFrame con las features de test
    df_perf: DataFrame de performance con columnas tipo:
             Num_features, Market_feature,
             ANN_R2_mean, ANN_R2_std, ANN_MAE_mean, ANN_MAE_std,
             SVM_R2_mean, SVM_R2_std, SVM_MAE_mean, SVM_MAE_std,
             RF_R2_mean,  RF_R2_std,  RF_MAE_mean,  RF_MAE_std
    nature: string para el nombre del archivo (p.ej. 'small_projects')
    models: lista de modelos, por ejemplo ["ANN", "SVM", "RF"]
    colors: diccionario con colores por modelo
    top_n: si quieres limitar a las top-n features en SHAP
    """
    
    
    # =========================================================
    # 1) SHAP: explainer y valores
    # =========================================================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # regresión: (n_samples, n_features)

    # Ajuste de fuente global (ajusta a tu gusto)
    plt.rcParams.update({'font.size': 8})

    # Selección de columnas (top_n) para SHAP
    if top_n is not None:
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)[:top_n]
        X_show = X_test.iloc[:, order]
        shap_show = shap_values[:, order]
    else:
        X_show = X_test
        shap_show = shap_values

    # =========================================================
    # 2) Figure con GridSpec: izquierda SHAP, derecha R2/MAE
    # =========================================================
    fig = plt.figure(figsize=(8, 5))

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 2],  # izquierda más ancha para SHAP
        wspace=0.3,
        hspace=0.1
    )

    # Eje SHAP ocupa las dos filas de la primera columna
    ax_shap = fig.add_subplot(gs[:, 0])

    # Ejes de performance en la segunda columna (arriba/abajo)
    # --- LO IMPORTANTE: comparten eje X ---
    ax_r2  = fig.add_subplot(gs[0, 1], sharex=None)
    ax_mae = fig.add_subplot(gs[1, 1], sharex=ax_r2)

    # =========================================================
    # 2.1) SHAP summary (dot) en ax_shap
    # =========================================================
    plt.sca(ax_shap)  # VERY IMPORTANT: SHAP usa eje actual

    shap.summary_plot(
        shap_show,
        X_show,
        show=False,
        plot_size=None  # <- CLAVE: no cambies el tamaño de la figura
    )
    ax_shap.set_title("a)", fontsize=14)

    # Si el título de SHAP queda raro (porque SHAP manipula el eje),
    # se puede volver a forzar:
    # ax_shap.set_ylabel("Features")
    # ax_shap.set_xlabel("SHAP value")

    # =========================================================
    # 2.2) Gráfico R² y MAE en los ejes de la derecha
    # =========================================================
    #models = ["ANN", "SVM", "RF"]
    
    x = df_perf["Num_features"]

    # ---- Subplot de R² ----
    for m in models:
        r2_mean_col = f"{m}_R2_Mean"
        r2_std_col = f"{m}_R2_Std"

        if r2_mean_col not in df_perf.columns or r2_std_col not in df_perf.columns:
            raise KeyError(f"Faltan columnas para {m}: {r2_mean_col} o {r2_std_col}")

        y_mean = df_perf[r2_mean_col]
        y_std = df_perf[r2_std_col]
        color = colors.get(m, None)

        ax_r2.plot(
            x,
            y_mean,
            marker="o",
            label=m,
            linewidth=2,
            color=color
        )

        ax_r2.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            color=color,
            alpha=0.18
        )

    ax_r2.set_ylim(0.5, 1.0)
    ax_r2.set_ylabel(r"$R^2$")
    ax_r2.tick_params(labelbottom=False)
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_title("b)", fontsize=14)
    ax_r2.legend(title="Model", loc="lower right", fontsize=10)

    # ax_r2.axvline(
    #     x=20,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.5,
    #     alpha=0.8,
    # )

    # ---- Subplot de MAE ----
    mae_upper_values = []

    for m in models:
        mae_mean_col = f"{m}_MAE_Mean"
        mae_std_col = f"{m}_MAE_Std"

        if mae_mean_col not in df_perf.columns or mae_std_col not in df_perf.columns:
            raise KeyError(f"Faltan columnas para {m}: {mae_mean_col} o {mae_std_col}")

        y_mean = df_perf[mae_mean_col]
        y_std = df_perf[mae_std_col]
        color = colors.get(m, None)

        ax_mae.plot(
            x,
            y_mean,
            marker="o",
            label=m,
            linewidth=2,
            color=color
        )

        ax_mae.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            color=color,
            alpha=0.18
        )

        mae_upper_values.append((y_mean + y_std).max())

    ax_mae.set_ylim(0, max(mae_upper_values) * 1.1)
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, alpha=0.3)

    # ax_mae.axvline(
    #     x=20,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.5,
    #     alpha=0.9,
    # )

    ax_mae.set_xticks(df_perf["Num_features"])
    ax_mae.set_xticklabels(df_perf["Market_feature"], rotation=60, ha="right")
    ax_mae.set_xlabel("Market variable added (Incrementally)")

    # =========================================================
    # 3) Guardar y mostrar
    # =========================================================
    if nature == "small_projects":
        title_label = "Small projects"
    if nature == "large_projects":
        title_label = "Large projects"
    
    fig.suptitle(
        title_label,
        fontsize=16,
        x=0.38, 
        y=1.02                # desplaza hacia arriba para que no se superponga
    )
            
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f"shap_and_perf_{nature}_{outlier_scenario}.pdf") 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
