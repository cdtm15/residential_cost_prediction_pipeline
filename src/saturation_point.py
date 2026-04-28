#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:49:38 2026

@author: cristiantobar
"""

import pickle

# Open the file in read-binary mode ('rb')
with open('perf_no_outliers_v1_27_april.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)

projects = data['no_outliers']
small_projects_ANN = projects['proj_0_perf_pred']['ANN'][0]
small_projects_SVR = projects['proj_0_perf_pred']['SVM'][0]
small_projects_RF  = projects['proj_0_perf_pred']['RF'][0]

large_projects_ANN = projects['proj_1_perf_pred']['ANN'][0]
large_projects_SVR = projects['proj_1_perf_pred']['SVM'][0]
large_projects_RF  = projects['proj_1_perf_pred']['RF'][0]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_moving_variation(
    df,
    model_name,
    k_col="Num Features",
    r2_col="R² Mean",
    mae_col="MAE Mean",
    window=4
):
    """
    Calcula la variación local en ventana móvil para R² y MAE.
    """

    df = df.sort_values(k_col).reset_index(drop=True).copy()

    records = []

    for i in range(0, len(df) - window + 1):
        sub = df.iloc[i:i + window]

        k = df.loc[i, k_col]

        r2_range = sub[r2_col].max() - sub[r2_col].min()
        mae_range = sub[mae_col].max() - sub[mae_col].min()

        records.append({
            "Model": model_name,
            "K": k,
            "R2_variation": r2_range,
            "MAE_variation": mae_range
        })

    return pd.DataFrame(records)

def classify_saturation(
    df,
    k_star,
    k_col="Num Features",
    r2_col="R² Mean",
    mae_col="MAE Mean",
    eps_r2=0.01,
    eps_mae=2.0
):
    """
    Clasifica el tipo de saturación comparando el desempeño en K*
    contra el mejor desempeño global del modelo.
    """

    if k_star is None:
        return "No saturation"

    df = df.sort_values(k_col).reset_index(drop=True).copy()

    row_sat = df[df[k_col] == k_star]

    if row_sat.empty:
        return "K* not found"

    r2_sat = row_sat[r2_col].iloc[0]
    mae_sat = row_sat[mae_col].iloc[0]

    r2_best = df[r2_col].max()
    mae_best = df[mae_col].min()

    if (abs(r2_sat - r2_best) <= eps_r2) and (abs(mae_sat - mae_best) <= eps_mae):
        return "Efficient saturation"

    elif r2_sat < (r2_best - 3 * eps_r2):
        return "Degradation saturation"

    else:
        return "Inefficient saturation"

def detect_saturation_from_variation(
    df_var,
    eps_r2=0.01,
    eps_mae=2.0
):
    """
    Detecta el primer K donde se cumple simultáneamente:
    R2_variation < eps_r2 AND MAE_variation < eps_mae
    """

    df_var = df_var.sort_values("K").reset_index(drop=True)

    condition = (
        (df_var["R2_variation"] < eps_r2) &
        (df_var["MAE_variation"] < eps_mae)
    )

    if condition.any():
        first_idx = condition.idxmax()

        return {
            "K_star": df_var.loc[first_idx, "K"],
            "Model": df_var.loc[first_idx, "Model"],
            "R2_variation_at_K": df_var.loc[first_idx, "R2_variation"],
            "MAE_variation_at_K": df_var.loc[first_idx, "MAE_variation"]
        }

    return {
        "K_star": None,
        "Model": df_var["Model"].iloc[0],
        "message": "No saturation point found."
    }

def detect_saturation_forward(
    df,
    k_col="Num Features",
    r2_col="R² Mean",
    mae_col="MAE Mean",
    eps_r2=0.01,
    eps_mae=2.0,
    min_points=3
):
    """
    Detecta el primer K donde el desempeño entra en régimen permanente
    usando FORWARD STABILITY.

    Criterio:
    Desde K hasta el final:
        max - min < epsilon

    Parámetros
    ----------
    min_points : int
        Número mínimo de puntos restantes para validar estabilidad
        (evita detectar saturación al final por falta de datos)
    """

    df = df.sort_values(k_col).reset_index(drop=True).copy()
    n = len(df)

    for i in range(n):

        # evitar detección trivial al final
        if (n - i) < min_points:
            continue

        sub = df.iloc[i:]

        r2_range = sub[r2_col].max() - sub[r2_col].min()
        mae_range = sub[mae_col].max() - sub[mae_col].min()

        if (r2_range < eps_r2) and (mae_range < eps_mae):
            return {
                "K_star": df.loc[i, k_col],
                "Feature_at_K": df.loc[i, "Feature"] if "Feature" in df.columns else None,
                "R2_range_forward": r2_range,
                "MAE_range_forward": mae_range,
                "remaining_points": n - i
            }

    return {
        "K_star": None,
        "message": "No forward saturation point found"
    }

def plot_saturation_variations(
    df_ann,
    df_svr,
    df_rf,
    window=4,
    eps_r2=0.01,
    eps_mae=2.0,
    title="Moving-window variation and saturation point"
):
    """
    Genera un plot con:
    - Arriba: variación local de R²
    - Abajo: variación local de MAE
    - Tres modelos en cada subplot
    - Línea horizontal del umbral
    - Línea vertical del K* de saturación por modelo
    """

    model_data = {
        "ANN": df_ann,
        "SVR": df_svr,
        "RF": df_rf
    }

    all_var = []
    saturation_results = {}

    for model_name, df in model_data.items():
        df_var = compute_moving_variation(
            df=df,
            model_name=model_name,
            window=window
        )

        sat = detect_saturation_forward(
            df,
            eps_r2=eps_r2,
            eps_mae=eps_mae,
            min_points=3
        )
        
        sat_type = classify_saturation(
            df=df,
            k_star=sat.get("K_star"),
            eps_r2=eps_r2,
            eps_mae=eps_mae
        )
        
        sat["Saturation_type"] = sat_type
        
        all_var.append(df_var)
        saturation_results[model_name] = sat

    df_all_var = pd.concat(all_var, ignore_index=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6, 6),
        sharex=True
    )

    ax_r2 = axes[0]
    ax_mae = axes[1]

    # -------------------------
    # R² variation
    # -------------------------
    for model_name in model_data.keys():
        sub = df_all_var[df_all_var["Model"] == model_name]

        ax_r2.plot(
            sub["K"],
            sub["R2_variation"],
            marker="o",
            label=model_name
        )

    ax_r2.axhline(
        eps_r2,
        linestyle="--",
        linewidth=1.5,
        label=fr"$\epsilon_{{R^2}}$ = {eps_r2}"
    )

    ax_r2.set_ylabel(r"Moving-window variation in $R^2$")
    ax_r2.set_title(title)
    ax_r2.grid(alpha=0.3)
    ax_r2.legend()
    ax_r2.set_ylim(0, 0.06)

    # -------------------------
    # MAE variation
    # -------------------------
    for model_name in model_data.keys():
        sub = df_all_var[df_all_var["Model"] == model_name]

        ax_mae.plot(
            sub["K"],
            sub["MAE_variation"],
            marker="o",
            label=model_name
        )

    ax_mae.axhline(
        eps_mae,
        linestyle="--",
        linewidth=1.5,
        label=fr"$\epsilon_{{MAE}}$ = {eps_mae}"
    )

    ax_mae.set_xlabel("Number of features")
    ax_mae.set_ylabel("Moving-window variation in MAE")
    ax_mae.grid(alpha=0.3)
    ax_mae.legend()
    ax_mae.set_ylim(0, 10)

    # -------------------------
    # Vertical saturation lines
    # -------------------------
    
    for model_name, sat in saturation_results.items():
        k_star = sat["K_star"]
        feature_name = sat.get("Feature_at_K", "")
        label_text = f"{model_name}\nK*={k_star}\n{feature_name}"
        
        if k_star is not None:
            ax_r2.axvline(
                k_star,
                linestyle=":",
                linewidth=1.8
            )

            ax_mae.axvline(
                k_star,
                linestyle=":",
                linewidth=1.8
            )

            ax_r2.text(
                k_star,
                ax_r2.get_ylim()[1],
                label_text,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )
            
            ax_mae.text(
                k_star,
                ax_mae.get_ylim()[1],
                label_text,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )
    plt.tight_layout()
    plt.show()

    return df_all_var, saturation_results

df_var_small, sat_small = plot_saturation_variations(
    df_ann=small_projects_ANN,
    df_svr=small_projects_SVR,
    df_rf=small_projects_RF,
    window=4,
    eps_r2=0.02,
    eps_mae=2.0,
    title="Small projects: saturation based on moving-window performance variation"
)

df_var_large, sat_large = plot_saturation_variations(
    df_ann = large_projects_ANN,
    df_svr = large_projects_SVR,
    df_rf  = large_projects_RF,
    window=4,
    eps_r2=0.02,
    eps_mae=2.0,
    title="Large projects: saturation based on moving-window performance variation"
)

sat_small

for model, result in sat_small.items():
    print(model, result)