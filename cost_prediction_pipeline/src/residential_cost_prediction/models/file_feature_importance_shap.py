#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:08:54 2026

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score
import os


def feature_importance(df, folder_path, nature = "test", top_n=None, save_tiff=False):
    
    """
   Entrena XGBoost, grafica importancia por gain y gráficos SHAP:
   - SHAP summary (dot)
   - SHAP summary (bar)
   - SHAP dependence plot para la feature más importante
   Args
   ----
   df : pd.DataFrame (incluye 'actual_construction_cost')
   nature : str        (sufijo de archivo)
   top_n : int|None    (si se indica, limita a top_n features en las figuras)
   save_tiff : bool    (si True, además guarda TIFF a 300 dpi)
   """
    # -------------------- datos --------------------
    X = df.drop('actual_construction_cost', axis=1).copy()
    y = df['actual_construction_cost'].copy()
 
    # Asegurar solo numéricas para evitar sorpresas con SHAP/corr
    X = X.select_dtypes(include=[np.number])
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
 
    # -------------------- modelo --------------------
    # Nota: 'reg:linear' está deprecado; usar 'reg:squarederror'
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
 
    # =========================================================
    # 1) Importancia por gain (XGBoost)
    # =========================================================
    gain = model.get_booster().get_score(importance_type='gain')
    imp_df = pd.DataFrame({'feature': list(gain.keys()), 'gain': list(gain.values())})
    if imp_df.empty:
        print("No se pudieron extraer importancias del booster (gain).")
    else:
        imp_df = imp_df.sort_values('gain', ascending=False)
        if top_n is not None:
            imp_df = imp_df.head(top_n)
 
        vals = imp_df['gain'].values
        labels_val = [f"{v:.2f}" for v in vals]
 
    
    # =========================================================
    # 2) SHAP: explainer y valores
    # =========================================================
    # Para árboles, TreeExplainer es muy eficiente
    
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
 
    # En algunos entornos, SHAP usa pyplot global; forzamos estilo limpio
    plt.rcParams.update({'font.size': 20})
 
    # -------------------- SHAP summary (dot) --------------------
    plt.figure(figsize=(20, 6 + 0.15 * X_test.shape[1]))
    # Si se desea limitar a top_n, reducimos X_test y shap_values con esas columnas
    if top_n is not None:
        # orden global por |SHAP| medio
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)[:top_n]
        X_show = X_test.iloc[:, order]
        shap_show = shap_values[:, order]
    else:
        X_show = X_test
        shap_show = shap_values
 
    shap.summary_plot(
        shap_show,
        X_show,
        show=False  # para poder guardar
    )
    #plt.title(f"SHAP Summary (dot) — {nature}", fontsize=13, weight='bold')
    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f"shap_summary_dot_{nature}.pdf")    
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    

    feature_important = model.get_booster().get_score(importance_type='gain')
    keys              = list(feature_important.keys())
    values            = list(feature_important.values())
    sorted_features   = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

    
    return sorted_features
