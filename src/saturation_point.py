#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:49:38 2026

@author: cristiantobar
"""

import pickle
import os

data_location    = "/Users/cristiantobar/data_projects/residential_cost_prediction_pipeline/data"
data_filename    = ["database_2_csv.csv"]
output_folder    = "/Users/cristiantobar/data_projects/residential_cost_prediction_pipeline/assets/output_figures"


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

def compute_forward_variation(
    df,
    model_name,
    k_col="Num Features",
    r2_col="R² Mean",
    mae_col="MAE Mean"
):
    """
    Calcula la variación forward para R² y MAE.

    Para cada K:
        R2_forward_variation = max(R2 desde K hasta n) - min(R2 desde K hasta n)
        MAE_forward_variation = max(MAE desde K hasta n) - min(MAE desde K hasta n)
    """

    df = df.sort_values(k_col).reset_index(drop=True).copy()

    records = []

    for i in range(len(df)):
        sub = df.iloc[i:]

        records.append({
            "Model": model_name,
            "K": df.loc[i, k_col],
            "Feature": df.loc[i, "Feature"] if "Feature" in df.columns else None,
            "R2_forward_variation": sub[r2_col].max() - sub[r2_col].min(),
            "MAE_forward_variation": sub[mae_col].max() - sub[mae_col].min()
        })

    return pd.DataFrame(records)

def plot_forward_saturation_variations(
    df_ann,
    df_svr,
    df_rf,
    eps_r2,
    eps_mae,
    min_points,
    title,
    folder_path,
    nature
):
    """
    Genera un plot alineado con el criterio forward stability:

    - Arriba: variación forward de R²
    - Abajo: variación forward de MAE
    - Tres modelos por subplot
    - Línea horizontal del umbral
    - Línea vertical del K* de saturación por modelo

    El K* se detecta cuando:
        max(R²_k:n) - min(R²_k:n) < eps_r2
        AND
        max(MAE_k:n) - min(MAE_k:n) < eps_mae
    """

    model_data = {
        "ANN": df_ann,
        "SVR": df_svr,
        "RF": df_rf
    }

    all_var = []
    saturation_results = {}

    for model_name, df in model_data.items():

        df_var = compute_forward_variation(
            df=df,
            model_name=model_name
        )

        sat = detect_saturation_forward(
            df=df,
            eps_r2=eps_r2,
            eps_mae=eps_mae,
            min_points=min_points
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
        figsize=(4, 6),
        sharex=True
    )

    ax_r2 = axes[0]
    ax_mae = axes[1]

    # -------------------------
    # R² forward variation
    # -------------------------
    for model_name in model_data.keys():
        sub = df_all_var[df_all_var["Model"] == model_name]

        ax_r2.plot(
            sub["K"],
            sub["R2_forward_variation"],
            marker="o",
            linewidth=1.8,
            label=model_name
        )

    ax_r2.axhline(
        eps_r2,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=fr"$\epsilon_{{R^2}}$ = {eps_r2}"
    )

    ax_r2.set_ylabel(r"Forward variation in $\overline{R^2}$")
    ax_r2.set_title(title)
    ax_r2.grid(alpha=0.3)
    ax_r2.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )
    
    
    ax_r2.set_ylim(0, 0.15)

    # -------------------------
    # MAE forward variation
    # -------------------------
    for model_name in model_data.keys():
        sub = df_all_var[df_all_var["Model"] == model_name]

        ax_mae.plot(
            sub["K"],
            sub["MAE_forward_variation"],
            marker="o",
            linewidth=1.8,
            label=model_name
        )

    ax_mae.axhline(
        eps_mae,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label=fr"$\epsilon_{{MAE}}$ = {eps_mae}"
    )

    ax_mae.set_xlabel("Number of features")
    ax_mae.set_ylabel(r"Forward variation in $\overline{MAE}$")
    ax_mae.grid(alpha=0.3)
    #ax_mae.legend()
    ax_mae.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )    
    ax_mae.set_ylim(0, 20)

    # -------------------------
    # Vertical saturation lines
    # -------------------------
    for model_name, sat in saturation_results.items():

        k_star = sat.get("K_star")

        if k_star is not None:

            feature_name = sat.get("Feature_at_K", "")
            sat_type = sat.get("Saturation_type", "")

            label_text = (
                f"{model_name}\n"
                f"K*={k_star}\n"
                f"{feature_name}"
            )

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
                ha="left",
                fontsize=8
            )

            ax_mae.text(
                k_star,
                ax_mae.get_ylim()[1],
                label_text,
                rotation=90,
                va="top",
                ha="left",
                fontsize=8
            )

    
    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, f"saturation_{nature}.pdf") 
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return df_all_var, saturation_results



df_forward_var_small, sat_small = plot_forward_saturation_variations(
    df_ann=small_projects_ANN,
    df_svr=small_projects_SVR,
    df_rf=small_projects_RF,
    eps_r2=0.02,
    eps_mae=2.0,
    min_points=3,
    title="Small projects: stationary regime-based saturation",
    folder_path=output_folder,
    nature = "Small_projects"
)

df_forward_var_large, sat_large = plot_forward_saturation_variations(
    df_ann = large_projects_ANN,
    df_svr = large_projects_SVR,
    df_rf  = large_projects_RF,
    eps_r2=0.02,
    eps_mae=2.0,
    min_points=3,
    title="Large projects: stationary regime-based saturation",
    folder_path=output_folder,
    nature = "Large_projects"
)

def plot_forward_saturation_2x2(
    small_data,
    large_data,
    eps_r2=0.02,
    eps_mae=2.0,
    min_points=3,
    folder_path=None
):

    datasets = {
        "Small projects": small_data,
        "Large projects": large_data
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 6),
        sharex='col'
    )
    fig.subplots_adjust(wspace=0.25)
    saturation_results_all = {}

    for col_idx, (title, model_data) in enumerate(datasets.items()):

        all_var = []
        saturation_results = {}

        for model_name, df in model_data.items():

            df_var = compute_forward_variation(
                df=df,
                model_name=model_name
            )

            sat = detect_saturation_forward(
                df=df,
                eps_r2=eps_r2,
                eps_mae=eps_mae,
                min_points=min_points
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

        # -------------------------
        # Subplots
        # -------------------------
        ax_r2 = axes[0, col_idx]
        ax_mae = axes[1, col_idx]

        # R2
        for model_name in model_data.keys():
            sub = df_all_var[df_all_var["Model"] == model_name]

            ax_r2.plot(
                sub["K"],
                sub["R2_forward_variation"],
                marker="o",
                linewidth=1.8,
                label=model_name
            )

        ax_r2.axhline(eps_r2, linestyle="--", color="black")
        ax_r2.text(
                0.98,
                eps_r2,
                fr"$\epsilon_{{R^2}}={eps_r2}$",
                transform=ax_r2.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.15)
            )
        ax_r2.set_title(title)
        ax_r2.set_ylabel(r"$\Delta^{forward} \overline{R^2}$")
        ax_r2.grid(alpha=0.3)
        ax_r2.set_ylim(0, 0.15)

        # MAE
        for model_name in model_data.keys():
            sub = df_all_var[df_all_var["Model"] == model_name]

            ax_mae.plot(
                sub["K"],
                sub["MAE_forward_variation"],
                marker="o",
                linewidth=1.8
            )

        ax_mae.axhline(eps_mae, linestyle="--", color="black")
        ax_mae.text(
                0.98,
                eps_mae,
                fr"$\epsilon_{{MAE}}={eps_mae}$",
                transform=ax_mae.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.15)
            )
        ax_mae.set_ylabel(r"$\Delta^{forward} \overline{MAE}$")
        ax_mae.set_xlabel("Number of features")
        ax_mae.grid(alpha=0.3)
        ax_mae.set_ylim(0, 20)

        # Saturation lines
        for model_name, sat in saturation_results.items():

            k_star = sat.get("K_star")

            if k_star is not None:

                feature_name = sat.get("Feature_at_K", "")

                label_text = f"{model_name}\nK*={k_star}\n{feature_name}"

                ax_r2.axvline(k_star, linestyle=":", linewidth=1.5)
                ax_mae.axvline(k_star, linestyle=":", linewidth=1.5)

                # R2 subplot
                ax_r2.text(
                    k_star,
                    ax_r2.get_ylim()[1]*0.95,
                    label_text,
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.2)
                )
                
                # MAE subplot
                # ax_mae.text(
                #     k_star,
                #     ax_mae.get_ylim()[1]*0.95,
                #     label_text,
                #     rotation=90,
                #     va="top",
                #     ha="left",
                #     fontsize=7,
                #     bbox=dict(boxstyle="round,pad=0.2", alpha=0.2)
                # )

        saturation_results_all[title] = saturation_results

    # -------------------------
    # Global legend
    # -------------------------
    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center right",
       # bbox_to_anchor=(1.08, 0.5)
    )

    #plt.tight_layout(rect=[0, 0, 0.9, 1])

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(f"{folder_path}/saturation_2x2.pdf", bbox_inches="tight")

    plt.show()

    return saturation_results_all

small_data = {
    "ANN": small_projects_ANN,
    "SVM": small_projects_SVR,
    "RF": small_projects_RF
}

large_data = {
    "ANN": large_projects_ANN,
    "SVM": large_projects_SVR,
    "RF": large_projects_RF
}

results = plot_forward_saturation_2x2(
    small_data=small_data,
    large_data=large_data,
    eps_r2=0.02,
    eps_mae=2.0,
    min_points=3,
    folder_path=output_folder
)

for model, result in sat_small.items():
    print(
        model,
        "| K* =", result.get("K_star"),
        "| Feature =", result.get("Feature_at_K"),
        "| Type =", result.get("Saturation_type"),
        "| R2 forward range =", result.get("R2_range_forward"),
        "| MAE forward range =", result.get("MAE_range_forward")
    )
    
for model, result in sat_large.items():
    print(
        model,
        "| K* =", result.get("K_star"),
        "| Feature =", result.get("Feature_at_K"),
        "| Type =", result.get("Saturation_type"),
        "| R2 forward range =", result.get("R2_range_forward"),
        "| MAE forward range =", result.get("MAE_range_forward")
    )