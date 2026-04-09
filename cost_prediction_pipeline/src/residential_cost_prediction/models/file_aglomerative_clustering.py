#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:10:43 2026

@author: cristiantobar
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.gridspec as gridspec
import os


def aglomerative_clustering_db2(df, currency, full_df, folder_path):
    data_scaled = normalize(df)
    data_scaled = pd.DataFrame(data_scaled, columns=df.columns)
    data_scaled.head()
    
    plt.figure(figsize=(12, 6))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
    plt.savefig('dendogram.pdf',bbox_inches='tight')
    plt.show()
    
    cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')  
    y_predict = cluster.fit_predict(data_scaled)
      
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns = ['output'] #1 para descarado, 0 para no descarado
    
    df_temp = df.reset_index()
    df_temp_res = df_temp.drop(['index'], axis = 1).copy()
    
    new_df = pd.concat([df_temp_res, y_predict], axis = 1)
    
    full_df_temp = full_df.reset_index()
    full_df_temp_res = full_df_temp.drop(['index'], axis = 1).copy()
    new_full_df = pd.concat([full_df_temp_res, y_predict], axis = 1)
    
    X_encoded    = new_df.drop(['output'], axis = 1).copy()
    num_features = X_encoded.shape[1]
    Y            = new_df['output'].copy() 
    #Y_or         = df['weeks_delay'].copy()

    sns.set(style="whitegrid")
    
    # Aseguramos que output sea categórico
    hue_series = pd.Categorical(new_df["output"])
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(
        nrows=4, ncols=4,
        height_ratios=[1, 1, 1, 1],
        wspace=0.1, hspace=2
    )
    
    # -------- COLUMN 1: Size --------
    ax_built = fig.add_subplot(gs[0, 0])
    sns.boxplot(x=new_df["built_area"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_built)
    ax_built.set_title("Built_area")
    ax_built.set_xlabel(r"$m^{2}$")
    
    ax_lot = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=new_df["lot_area"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_lot)
    ax_lot.set_title("lot_area")
    ax_lot.set_xlabel(r"$m^{2}$")
    
    # Quitamos leyendas duplicadas
    ax_lot.get_legend().remove()
    ax_built.legend_.set_title("output")
    
    # Las celdas (2,0) y (3,0) quedan vacías
    ax_empty_20 = fig.add_subplot(gs[2, 0])
    ax_empty_20.axis("off")
    ax_empty_30 = fig.add_subplot(gs[3, 0])
    ax_empty_30.axis("off")
    
    # -------- COLUMN 2: Cost --------
    ax_prelim = fig.add_subplot(gs[0, 1])
    sns.boxplot(x=new_df["prelim_cost_est_est"], hue=hue_series, width=0.8,
                palette="Set2", ax=ax_prelim)
    ax_prelim.set_title("prelim_cost_est_est")
    ax_prelim.set_xlabel(r"$\$/m^{2}$")
    
    ax_equi = fig.add_subplot(gs[1, 1])
    sns.boxplot(x=new_df["equi_prelim_cost"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_equi)
    ax_equi.set_title("equi_prelim_cost")
    ax_equi.set_xlabel(r"$\$/m^{2}$")
    
    ax_total = fig.add_subplot(gs[2, 1])
    sns.boxplot(x=new_df["total_prelim_cost_est"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_total)
    ax_total.set_title("total_prelim_cost_est")
    ax_total.set_xlabel(r"$")
    
    ax_actual = fig.add_subplot(gs[3, 1])
    sns.boxplot(x=new_df["actual_construction_cost"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_actual)
    ax_actual.set_title("actual_construction_cost")
    ax_actual.set_xlabel(r"$\$/m^{2}$")
    
    # Quitamos todas las leyendas menos la de la primera columna
    for ax in [ax_prelim, ax_equi, ax_total, ax_actual]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    
    # -------- COLUMN 3: Duration --------
    ax_dur = fig.add_subplot(gs[0, 2])
    sns.boxplot(x=new_df["duration"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_dur)
    ax_dur.set_title("duration")
    ax_dur.set_xlabel("Weeks")
    leg = ax_dur.get_legend()
    if leg is not None:
        leg.remove()
    
    # Celdas vacías debajo
    for r in [1, 2, 3]:
        ax = fig.add_subplot(gs[r, 2])
        ax.axis("off")
    
    # -------- COLUMN 4: Sale price --------
    ax_unit = fig.add_subplot(gs[0, 3])
    sns.boxplot(x=new_df["unit_price"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_unit)
    ax_unit.set_title("unit_price")
    ax_unit.set_xlabel(r"$\$/m^{2}$")
    
    leg = ax_unit.get_legend()
    if leg is not None:
        leg.remove()
    
    for r in [1, 2, 3]:
        ax = fig.add_subplot(gs[r, 3])
        ax.axis("off")
    
    # -------- Dendrograma ocupando el bloque inferior derecho --------
    ax_dend = fig.add_subplot(gs[1:3, 3:4])
    
    Z = shc.linkage(data_scaled, method='ward')
    shc.dendrogram(Z, ax=ax_dend)

    
    ax_dend.set_title("Dendrogram", fontsize=12)
    ax_dend.set_xlabel("Projects")      # cambia etiqueta si quieres
    ax_dend.set_ylabel("Distance")
     
            
    # -------- Títulos de columnas --------
    fig.text(0.20, 0.96, "Size", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    fig.text(0.42, 0.96, "Cost", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    fig.text(0.61, 0.96, "Duration", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    fig.text(0.81, 0.96, "Sale price", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    
    # -------- Etiquetas de filas --------
    fig.text(0.1, 0.63, "Project physical and financial features",
             rotation="vertical", ha="center", va="center",
             fontsize=12, fontweight="bold")
    fig.text(0.1, 0.15, "Output",
             rotation="vertical", ha="center", va="center",
             fontsize=12, fontweight="bold")
    
    # 1. Obtener handles y labels de cualquier subplot (por ejemplo ax_built)
    handles, labels = ax_built.get_legend_handles_labels()
    
    # 2. Crear legend global en la esquina inferior derecha
    fig.legend(
        handles, labels,
        title="Project type",
        loc="lower right",
        bbox_to_anchor=(0.90, 0.02),   # Ajusta ligeramente para moverlo
        frameon=True
    )
    
    # 3. Remover la leyenda del subplot original
    ax_built.get_legend().remove()
    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, "boxplots_layout_all.pdf") 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    
    return new_df, new_full_df
