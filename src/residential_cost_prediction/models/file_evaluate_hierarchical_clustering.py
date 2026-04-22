#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:39:38 2026

@author: cristiantobar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.utils import resample


def evaluate_hierarchical_clustering(df, folder_path, max_k=6, n_boot=100, random_state=42):
    os.makedirs(folder_path, exist_ok=True)

    # =========================
    # 1) Escalado
    # =========================
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    X = pd.DataFrame(X, columns=df.columns)

    # =========================
    # 2) Dendrograma
    # =========================
    Z = shc.linkage(X, method='ward')

    plt.figure(figsize=(12, 6))
    shc.dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.xlabel("Projects")
    plt.ylabel("Fusion distance")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "dendrogram_ward.pdf"), bbox_inches="tight")
    plt.show()

    # Distancias de fusión
    merge_distances = Z[:, 2]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(merge_distances) + 1), merge_distances, marker='o')
    plt.title("Fusion distances across hierarchical merges")
    plt.xlabel("Merge step")
    plt.ylabel("Ward linkage distance")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "fusion_distances.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # 3) Validez interna para distintos k
    # =========================
    results = []

    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        results.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        })

    results_df = pd.DataFrame(results)

    # =========================
    # 4) Estabilidad bootstrap
    # =========================
    stability_results = []

    # Soluciones base
    base_labels_dict = {}
    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        base_labels_dict[k] = model.fit_predict(X)

    for k in range(2, max_k + 1):
        ari_scores = []

        for b in range(n_boot):
            # bootstrap sample indices
            idx = resample(np.arange(len(X)), replace=True, random_state=random_state + b)
            X_boot = X.iloc[idx]

            model_boot = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
            boot_labels = model_boot.fit_predict(X_boot)

            # comparar con clustering original en las mismas observaciones
            original_labels_subset = base_labels_dict[k][idx]

            ari = adjusted_rand_score(original_labels_subset, boot_labels)
            ari_scores.append(ari)

        stability_results.append({
            "k": k,
            "ARI_mean": np.mean(ari_scores),
            "ARI_std": np.std(ari_scores)
        })

    stability_df = pd.DataFrame(stability_results)

    # Unir resultados
    summary_df = results_df.merge(stability_df, on="k")

    # =========================
    # 5) Gráficos de validación
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(summary_df["k"], summary_df["silhouette"], marker='o')
    axes[0, 0].set_title("Silhouette score")
    axes[0, 0].set_xlabel("Number of clusters (k)")
    axes[0, 0].set_ylabel("Score")

    axes[0, 1].plot(summary_df["k"], summary_df["calinski_harabasz"], marker='o')
    axes[0, 1].set_title("Calinski-Harabasz index")
    axes[0, 1].set_xlabel("Number of clusters (k)")
    axes[0, 1].set_ylabel("Score")

    axes[1, 0].plot(summary_df["k"], summary_df["davies_bouldin"], marker='o')
    axes[1, 0].set_title("Davies-Bouldin index")
    axes[1, 0].set_xlabel("Number of clusters (k)")
    axes[1, 0].set_ylabel("Score")

    axes[1, 1].errorbar(
        summary_df["k"],
        summary_df["ARI_mean"],
        yerr=summary_df["ARI_std"],
        marker='o',
        capsize=4
    )
    axes[1, 1].set_title("Bootstrap stability (ARI)")
    axes[1, 1].set_xlabel("Number of clusters (k)")
    axes[1, 1].set_ylabel("Adjusted Rand Index")

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "cluster_validation_summary.pdf"), bbox_inches="tight")
    plt.show()

    # =========================
    # 6) Elegir k=2 y perfilar clusters
    # =========================
    final_model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    final_labels = final_model.fit_predict(X)

    profile_df = df.copy()
    profile_df["cluster"] = final_labels

    cluster_profile = profile_df.groupby("cluster").agg(["median", "mean", "std", "min", "max"])

    summary_df.to_csv(os.path.join(folder_path, "cluster_validation_metrics.csv"), index=False)
    cluster_profile.to_csv(os.path.join(folder_path, "cluster_profile.csv"))

    return summary_df, cluster_profile, final_labels
