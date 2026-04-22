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


def feature_importance(df, folder_path, nature, outlier_scenario, top_n=None, save_tiff=False):
    
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
    # 2) SHAP: explainer y valores
    # =========================================================
    # Para árboles, TreeExplainer es muy eficiente
    
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
 
    # En algunos entornos, SHAP usa pyplot global; forzamos estilo limpio
    plt.rcParams.update({'font.size': 20})
 
    # Importancia SHAP consistente con la figura
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if top_n is not None:
        top_features = shap_importance["feature"].iloc[:top_n].tolist()
        X_show = X_test[top_features]
        shap_show = shap_values[:, [X_test.columns.get_loc(f) for f in top_features]]
    else:
        X_show = X_test
        shap_show = shap_values

    plt.figure(figsize=(20, 6 + 0.15 * X_show.shape[1]))
    shap.summary_plot(
        shap_show,
        X_show,
        show=False,
        #sort=False   # respeta el orden que ya definiste
    )
    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f"shap_summary_dot_{nature}_{outlier_scenario}.pdf")    
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    

    # feature_important = model.get_booster().get_score(importance_type='gain')
    # keys              = list(feature_important.keys())
    # values            = list(feature_important.values())
    # sorted_features   = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

    
    return shap_importance, model, X_test
