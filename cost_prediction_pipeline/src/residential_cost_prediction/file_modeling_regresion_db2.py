#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:14:50 2024

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


#def feature_importance(df, nature):
def feature_importance(df, nature, folder_path, top_n=None, save_tiff=False):

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
 
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35*len(imp_df))))
        y_pos = range(len(imp_df))
 
        ax.barh(y_pos, vals, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_df['feature'], fontsize=11)
        ax.invert_yaxis()
 
        ax.set_xlabel("Feature importance (gain)", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.set_title(f"XGBoost Feature Importance (gain) — R²={r2:.3f}", fontsize=13, weight='bold')
 
        max_val = vals.max() if len(vals) else 1.0
        pad = 0.02 * max_val
        ax.set_xlim(0, max_val * 1.12)
 
        for i, v in enumerate(vals):
            inside = v > 0.90 * max_val
            if inside:
                ax.text(v - pad, i, labels_val[i], va='center', ha='right', fontsize=10, color='white')
            else:
                ax.text(v + pad, i, labels_val[i], va='center', ha='left', fontsize=10, color='black', clip_on=False)
 
        ax.xaxis.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
        plt.tight_layout()        
        os.makedirs(folder_path, exist_ok=True)
        pdf_path = os.path.join(folder_path, f'feature_importance_external_{nature}.pdf') 
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.show()
    
    # =========================================================
    # 2) SHAP: explainer y valores
    # =========================================================
    # Para árboles, TreeExplainer es muy eficiente
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
 
    # En algunos entornos, SHAP usa pyplot global; forzamos estilo limpio
    plt.rcParams.update({'font.size': 40})
 
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
    pdf_path = os.path.join(folder_path, f'shap_summary_dot_{nature}.pdf') 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
 
    # -------------------- SHAP summary (bar) --------------------
    #plt.figure(figsize=(8, max(4, 0.3 * X_show.shape[1])))
    plt.figure(figsize=(8,5))
    shap.summary_plot(
        shap_show,
        X_show,
        plot_type='bar',
        show=False
    )
    plt.title(f"SHAP Global Importance (|mean SHAP|) — {nature}", fontsize=13, weight='bold')
    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f'shap_summary_bar_{nature}.pdf') 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    
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
    
        
    
    # Para el caso de proyectos pequeños
    plot_shap_and_perf(model, X_test, pd.DataFrame(data), nature="small_projects", top_n=20)
    
    # Para el otro conjunto
    plot_shap_and_perf(model, X_test, pd.DataFrame(data2), nature="large_projects", top_n=20)
    
 
    # -------------------- SHAP dependence (top feature) --------------------
    # Tomamos la feature con mayor |SHAP| medio
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    top_feat = X_test.columns[top_idx]
 
    plt.figure(figsize=(7,5))
    shap.dependence_plot(
        top_feat,
        shap_values,
        X_test,
        interaction_index='auto',  # deja que SHAP elija interacción informativa
        show=False
    )
    plt.title(f"SHAP Dependence — {top_feat}", fontsize=13, weight='bold')
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f'shap_dependence_{nature}_{top_feat}.pdf') 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
        
    # shap_values es un arreglo [n_obs, n_features]
    # X_test tiene las columnas (features)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Construir dataframe ordenado
    shap_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    feature_important = model.get_booster().get_score(importance_type='gain')
    keys              = list(feature_important.keys())
    values            = list(feature_important.values())
    sorted_features   = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
   
    return shap_importance


def svm_regresion(X_train, y_train):
    modelo = make_pipeline(StandardScaler(), SVR(kernel='linear', C=1.0, epsilon=0.2))  # Ajusta los hiperparámetros si es necesario
    modelo.fit(X_train, y_train)
    return modelo

def ann_regresion(X_train, y_train):
    # Crear el modelo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train, epochs=500, verbose=0)
    
    return modelo

def ann_regresion_2(X_train, y_train):
    breakpoint()
    # Split validación
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Modelo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7)

    # Entrenar
    history = modelo.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    return history
    
def rf_regresion(X_train, y_train):
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)  # Ajusta los hiperparámetros si es necesario
    modelo.fit(X_train, y_train)
    return modelo    

def modeling_regresion_db2(df_clustered, nature, ml_tech, folder_path):
    
    df_subproj_ext = df_clustered.drop(['built_area',
    'lot_area',
    'total_prelim_cost_est',
    'prelim_cost_est_est',
    'equi_prelim_cost',
    'duration',
    'unit_price',
    'output', 'actual_sale_price'], axis=1).copy()
    
    sorted_feat_subproj = feature_importance(df_subproj_ext, nature, folder_path) 

    # 2. Definir las variables de entrada y salida
    features_iniciales = ['built_area', 'lot_area', 'total_prelim_cost_est', 'prelim_cost_est_est', 
                          'equi_prelim_cost', 'duration', 'unit_price']
    features_adicionales = list(sorted_feat_subproj.feature)
        
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
    plt.savefig('predicted_cost_'+nature+'_'+ml_tech+'.pdf',bbox_inches='tight')
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
    pdf_path = os.path.join(folder_path, 'error_relative'+nature+'_'+ml_tech+'.pdf') 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    
    return df_results, sorted_feat_subproj
