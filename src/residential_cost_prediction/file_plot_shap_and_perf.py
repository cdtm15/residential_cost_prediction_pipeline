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

# ---------------------------
# 1. Crear el DataFrame
# ---------------------------
data = {
    "Num_features": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "Market_feature": [
        "Fixed baseline features",
        "cumulative_liquidity",
        "city_population",
        "private_sector_investment",
        "land_price_index",
        "construc_cost_priv_time_fin",
        "gold_price",
        "num_build_permit",
        "total_floor_area_permit",
        "BSI_num_contracts",
        "bank_loans_num",
        "street_exchange_rate_to_dollar",
        "WPI_mater_price",
        "construc_cost_priv_time_start",
        "bank_loans_amou",
        "stock_market",
        "loan_interest_rate",
        "exchange_rate_to_dollar",
        "consumer_price_index",
        "cpi_services",
    ],
    # ANN
    "ANN_R2":  [0.855, 0.752, 0.736, 0.797, 0.792, 0.812, 0.850, 0.791, 0.824, 0.885, 0.884,
                0.874, 0.860, 0.876, 0.890, 0.884, 0.868, 0.894, 0.882, 0.888],
    "ANN_MAE": [59.028, 84.524, 82.479, 71.863, 73.871, 68.085, 61.023, 79.662, 71.474, 56.445,
                55.595, 58.874, 62.142, 55.703, 56.992, 57.283, 62.509, 55.545, 59.310, 55.900],
    # SVM
    "SVM_R2":  [0.802, 0.849, 0.835, 0.837, 0.838, 0.821, 0.820, 0.812, 0.812, 0.815, 0.814,
                0.818, 0.817, 0.805, 0.811, 0.813, 0.813, 0.809, 0.809, 0.809],
    "SVM_MAE": [63.449, 48.032, 52.260, 53.241, 52.067, 55.317, 56.954, 58.101, 57.947, 57.803,
                58.007, 57.905, 57.910, 59.508, 59.095, 59.095, 59.549, 60.022, 60.032, 60.033],
    # RF
    "RF_R2":   [0.703, 0.691, 0.679, 0.686, 0.677, 0.677, 0.680, 0.671, 0.703, 0.684, 0.661,
                0.670, 0.665, 0.669, 0.673, 0.665, 0.681, 0.675, 0.659, 0.667],
    "RF_MAE":  [65.824, 63.085, 65.558, 63.204, 63.866, 65.753, 66.118, 66.512, 64.696, 65.846,
                69.929, 68.605, 68.899, 68.939, 67.408, 69.400, 68.337, 68.593, 68.071, 69.461],
}

# ---------------------------
# 1. Crear el DataFrame
# ---------------------------
data2 = {
    "Num_features": [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
    "Market_feature": [
        "Fixed baseline features",
        "construc_cost_priv_time_fin",
        "land_price_index",
        "gold_price",
        "bank_loans_amou",
        "city_population",
        "num_build_permit",
        "private_sector_investment",
        "total_floor_area_permit",
        "bank_loans_num",
        "BSI_num_contracts",
        "stock_market",
        "construc_cost_priv_time_start",
        "street_exchange_rate_to_dollar",
        "exchange_rate_to_dollar",
        "cumulative_liquidity",
        "WPI_mater_price",
        "consumer_price_index",
        "loan_interest_rate",
        "cpi_services",
    ],
    # ANN
    "ANN_R2":  [0.925, 0.928,0.928,0.937,0.927,0.918,0.936,0.934,0.932,0.932,0.939,
                0.947,0.952,0.962,0.959,0.956,0.960,0.953,0.962,0.956],
    "ANN_MAE": [32.550, 33.432,34.235,29.822,32.629,34.824,30.150,32.262,32.226,31.341,30.691,
                27.683,27.062,24.812,26.437,26.369,26.411,28.316,24.661,27.052],
    # SVM
    "SVM_R2":  [0.820, 0.860,0.862,0.856,0.856,0.857,0.861,0.860,0.860,0.860,0.858,
                0.857,0.857,0.862,0.859,0.862,0.860,0.859,0.860,0.859],
    "SVM_MAE": [52.512, 46.030,44.644,44.552,43.502,43.383,43.426,44.514,45.078,44.853,44.869,
                45.289,45.283,44.070,44.538,43.926,44.254,44.428,44.438,44.513],
    # RF
    "RF_R2":   [0.885, 0.919,0.932,0.929,0.928,0.926,0.931,0.934,0.930,0.934,0.931,
                0.931,0.929,0.933,0.931,0.930,0.931,0.928,0.934,0.928],
    "RF_MAE":  [37.015, 33.190,31.331,32.162,32.285,32.707,31.732,30.961,31.816,31.000,31.602,
                39.065,31.313,31.097,31.269,31.460,31.410,31.928,30.786,31.765],
}




def plot_shap_and_perf(model, X_test, df_perf, nature, top_n=None, save_tiff=False):
    """
    model: modelo entrenado (árbol, RF, XGB, etc.)
    X_test: DataFrame con las features de test
    df_perf: DataFrame de performance (tipo data o data2 que mostraste)
    nature: string para el nombre del archivo (p.ej. 'small_project')
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
    ax_shap.set_title(f"a)", fontsize=14)

    # Si el título de SHAP queda raro (porque SHAP manipula el eje),
    # se puede volver a forzar:
    # ax_shap.set_ylabel("Features")
    # ax_shap.set_xlabel("SHAP value")

    # =========================================================
    # 2.2) Gráfico R² y MAE en los ejes de la derecha
    # =========================================================
    models = ["ANN", "SVM", "RF"]
    colors = {
        "ANN": "tab:blue",
        "SVM": "tab:orange",
        "RF": "tab:green",
    }

    # ---- Subplot de R² ----
    for m in models:
        ax_r2.plot(
            df_perf["Num_features"],
            df_perf[f"{m}_R2"],
            marker="o",
            label=m,
            linewidth=2,
            color=colors.get(m, None),
        )
    ax_r2.set_ylim(0.5, 1.0)
    ax_r2.set_ylabel("$R^2$")
    ax_r2.tick_params(labelbottom=False)
    ax_r2.grid(True, alpha=0.3)
    ax_r2.set_title("b)", fontsize=14)
    
    ax_r2.axvline(
        x=20,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    # ---- Subplot de MAE ----
    for m in models:
        ax_mae.plot(
            df_perf["Num_features"],
            df_perf[f"{m}_MAE"],
            marker="o",
            label=m,
            linewidth=2,
            color=colors.get(m, None),
        )
    ax_mae.set_ylim(0, max(df_perf[[f"{m}_MAE" for m in models]].max()) * 1.1)
    ax_mae.set_ylabel("MAE")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(title="Model", loc="best", fontsize=10)
    
    # Línea vertical en MAE
    ax_mae.axvline(
        x=20,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
    )
            
    # Etiquetas del eje X con el nombre de la variable de mercado añadida
    ax_mae.set_xticks(df_perf["Num_features"])
    ax_mae.set_xticklabels(df_perf["Market_feature"], rotation=60, ha="right")
    ax_mae.set_xlabel("Market variable added (Incrementally)")
    #ax_mae.tick_params(labelsize=20)

    # =========================================================
    # 3) Guardar y mostrar
    # =========================================================
    fig.suptitle(
        "Large projects",
        fontsize=16,
        x=0.38, 
        y=1.02                # desplaza hacia arriba para que no se superponga
    )
            
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f"shap_and_perf_{nature}.pdf") 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
