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
from sklearn.metrics import silhouette_score
import numpy as np

def aglomerative_clustering_db2(df, currency, full_df, folder_path, outlier_scenario):
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
    
    sil_scores = []
    k_values = range(2, 7)
    
    for k in k_values:
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = model.fit_predict(data_scaled)
        sil = silhouette_score(data_scaled, labels)
        sil_scores.append(sil)
    
    # fig = plt.figure(figsize=(10, 5))
    # gs = gridspec.GridSpec(
    #     nrows=4, ncols=4,
    #     height_ratios=[1, 1, 1, 1],
    #     wspace=0.1, hspace=2
    # )
    
    best_idx = int(np.argmax(sil_scores))
    best_k = k_values[best_idx]
    best_score = sil_scores[best_idx]
    
    
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(
        nrows=4, ncols=4,
        width_ratios=[1.1, 1.1, 1.0, 1.2],
        height_ratios=[1, 1, 1, 1],
        wspace=0.35, hspace=1.5
    )
    
    
    
    # -------- COLUMN 1: Size --------
    ax_built = fig.add_subplot(gs[0, 0])
    sns.boxplot(x=new_df["built_area"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_built)
    ax_built.set_title("Built area")
    ax_built.set_xlabel(r"$m^{2}$")
    
    ax_lot = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=new_df["lot_area"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_lot)
    ax_lot.set_title("Lot area")
    ax_lot.set_xlabel(r"$m^{2}$")
    
    # # Quitamos leyendas duplicadas
    # ax_lot.get_legend().remove()
    # ax_built.legend_.set_title("output")
    
    # Las celdas (2,0) y (3,0) quedan vacías
    ax_empty_20 = fig.add_subplot(gs[2, 0])
    ax_empty_20.axis("off")
    ax_empty_30 = fig.add_subplot(gs[3, 0])
    ax_empty_30.axis("off")
    
    # -------- COLUMN 2: Cost --------
    ax_prelim = fig.add_subplot(gs[0, 1])
    sns.boxplot(x=new_df["prelim_cost_est_est"], hue=hue_series, width=0.8,
                palette="Set2", ax=ax_prelim)
    ax_prelim.set_title("Prelim. cost")
    ax_prelim.set_xlabel(r"$\$/m^{2}$")
    
    ax_equi = fig.add_subplot(gs[1, 1])
    sns.boxplot(x=new_df["equi_prelim_cost"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_equi)
    ax_equi.set_title("Equi. prelim. cost")
    ax_equi.set_xlabel(r"$\$/m^{2}$")
    
    ax_total = fig.add_subplot(gs[2, 1])
    sns.boxplot(x=new_df["total_prelim_cost_est"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_total)
    ax_total.set_title("Total prelim. cost")
    ax_total.set_xlabel(r"$")
    
    ax_actual = fig.add_subplot(gs[3, 1])
    sns.boxplot(x=new_df["actual_construction_cost"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_actual)
    ax_actual.set_title("Actual construction cost")
    ax_actual.set_xlabel(r"$\$/m^{2}$")
    
    # # Quitamos todas las leyendas menos la de la primera columna
    # for ax in [ax_prelim, ax_equi, ax_total, ax_actual]:
    #     leg = ax.get_legend()
    #     if leg is not None:
    #         leg.remove()
    
    # -------- COLUMN 3: Duration --------
    ax_dur = fig.add_subplot(gs[0, 2])
    sns.boxplot(x=new_df["duration"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_dur)
    ax_dur.set_title("Duration")
    ax_dur.set_xlabel("Weeks")
    
    # leg = ax_dur.get_legend()
    # if leg is not None:
    #     leg.remove()
    
    ax_unit = fig.add_subplot(gs[1, 2])
    sns.boxplot(x=new_df["unit_price"], hue=hue_series, width=.8,
                palette="Set2", ax=ax_unit)
    ax_unit.set_title("Sale price")
    ax_unit.set_xlabel(r"$\$/m^{2}$")
    
    # # Celdas vacías debajo
    # for r in [1, 2, 3]:
    #     ax = fig.add_subplot(gs[r, 2])
    #     ax.axis("off")
    
    ax_empty_22 = fig.add_subplot(gs[2, 2])
    ax_empty_22.axis("off")

    ax_empty_32 = fig.add_subplot(gs[3, 2])
    ax_empty_32.axis("off")
    
    # -------- COLUMN 4: Sale price --------
    
    
    
    
    # ax_sil = fig.add_subplot(gs[1, 3])

    # ax_sil.plot(k_values, sil_scores, marker='o')
    # ax_sil.set_title("Silhouette score", fontsize=11)
    # ax_sil.set_xlabel("k")
    # ax_sil.set_ylabel("Score")
    
    # # marcar k=2
    # best_k = k_values[np.argmax(sil_scores)]
    # best_score = max(sil_scores)
    
    # ax_sil.axvline(x=best_k, linestyle='--', alpha=0.6)
    # ax_sil.scatter(best_k, best_score)
    
    # ax_sil.text(
    #     best_k, best_score,
    #     f"  k={best_k}",
    #     verticalalignment='bottom'
    # )
    
    # # -------- Dendrograma ocupando el bloque inferior derecho --------
    # #ax_dend = fig.add_subplot(gs[1:3, 3:4])
    # ax_dend = fig.add_subplot(gs[2:4, 3])
    
    # Z = shc.linkage(data_scaled, method='ward')
    # shc.dendrogram(Z, ax=ax_dend)

    
    # ax_dend.set_title("Dendrogram", fontsize=12)
    # ax_dend.set_xlabel("Projects")      # cambia etiqueta si quieres
    # ax_dend.set_ylabel("Distance")
     
        
    # -------------------------
    # Silhouette score occupies top half
    ax_sil = fig.add_subplot(gs[0:2, 3])
    ax_sil.plot(k_values, sil_scores, marker='o', linewidth=2)
    ax_sil.axvline(best_k, linestyle='--', alpha=0.7)
    ax_sil.axhline(best_score, linestyle=':', alpha=0.5)
    ax_sil.scatter(best_k, best_score, s=90, zorder=5)

    ax_sil.annotate(
        f"Highest at k={best_k}",
        xy=(best_k, best_score),
        xytext=(best_k + 0.25, best_score + 0.02),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=11
    )

    ax_sil.set_title("Silhouette score", fontsize=13)
    ax_sil.set_xlabel("Number of clusters (k)")
    ax_sil.set_ylabel("Score")
    ax_sil.set_xticks(k_values)
    ax_sil.set_ylim(min(sil_scores) - 0.03, max(sil_scores) + 0.05)

    # Dendrogram occupies bottom half
    ax_dend = fig.add_subplot(gs[2:4, 3])
    Z = shc.linkage(data_scaled, method='ward')
    shc.dendrogram(
        Z,
        ax=ax_dend,
        no_labels=True,
        color_threshold=None
    )

    ax_dend.set_title("Hierarchical clustering dendrogram", fontsize=13)
    ax_dend.set_xlabel("Projects")
    ax_dend.set_ylabel("Distance")
    
    # =========================
    # 5) Remove duplicated legends
    # =========================
    all_axes = [
        ax_built, ax_lot, ax_prelim, ax_equi, ax_total, ax_actual,
        ax_dur, ax_unit
    ]

    for ax in all_axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # # Global legend
    # handles, labels = ax_built.get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     title="Project type",
    #     loc="lower left",
    #     bbox_to_anchor=(0.95, 0.12),
    #     frameon=True
    # )
    
    # -------- Títulos de columnas --------
    fig.text(0.20, 0.96, "Size", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    fig.text(0.42, 0.96, "Cost", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    fig.text(0.61, 0.96, "Duration and sale price", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="navy")
    fig.text(0.81, 0.96, "Cluster validation", ha="center", va="bottom",
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
        loc="lower left",
        bbox_to_anchor=(0.55, 0.1),   # Ajusta ligeramente para moverlo
        frameon=True
    )
    
    # 3. Remover la leyenda del subplot original
    # ax_built.get_legend().remove()
    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f"boxplots_layout_all_{outlier_scenario}.pdf") 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    
    return new_df, new_full_df
