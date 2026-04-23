#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:31:17 2026

@author: cristiantobar
"""

#Subpipelines
from residential_cost_prediction.file_data_preparation_db2 import data_preparation_db2
from residential_cost_prediction.file_data_reception_db2 import data_reception_db2
#from residential_cost_prediction.file_modeling_regresion_db2 import modeling_regresion_db2
from residential_cost_prediction.file_modeling_regresion_v3 import modeling_regresion_db2_cv

#Models
from residential_cost_prediction.models.file_aglomerative_clustering import aglomerative_clustering_db2
from residential_cost_prediction.models.file_feature_importance_shap import feature_importance
from residential_cost_prediction.file_plot_shap_and_perf import plot_shap_and_perf
from residential_cost_prediction.models.file_evaluate_hierarchical_clustering import evaluate_hierarchical_clustering

#Utils
from residential_cost_prediction.utils.file_get_perf_summary import build_perf_summary

data_location    = "/Users/cristian/data_projects/residential_cost_prediction_pipeline/data"
data_filename    = ["database_2_csv.csv"]
output_folder    = "/Users/cristian/data_projects/residential_cost_prediction_pipeline/assets/output_figures"
merged_path      = data_location + "/" + data_filename[0]
currency         = ['Million COPm', 'DOLLARm']
dollar2cop       = 4333.11
outlier_scenario = ["with_outliers", "no_outliers"]
models           = ["ANN", "SVM", "RF"]
#models           = ["SVM"]
# model_map        = {
#                     'ann': 'ANN',
#                     'svm': 'SVM',
#                     'rf': 'RF'
#                     }

colors = {
        "ANN": "tab:blue",
        "SVM": "tab:orange",
        "RF": "tab:green",
         }

def cost_pipeline_run(data_path, output_path, outlier_flag, outlier_scenario):
    df             = data_reception_db2(merged_path)
    filt_df        = data_preparation_db2(df, currency[1], dollar2cop, outlier_flag)
    #corr, desc    = data_understanding_db2(filt_df, currency[1])

    df_int = filt_df[['built_area',
                    'lot_area',
                    'total_prelim_cost_est',
                    'prelim_cost_est_est',
                    'equi_prelim_cost',
                    'duration',
                    'unit_price',
                    'actual_construction_cost']]
    # df_ext = df_wo_outliers.iloc[:,13:32]

    #Project Specific Clustering and Feature Importance
    sorted_features, _, _           = feature_importance(df_int, output_folder, 'internal', outlier_scenario)
    
    #summary_df, cluster_profile, final_labels = evaluate_hierarchical_clustering(df_int, output_folder)
    
    
    df_clustered, full_df_clustered = aglomerative_clustering_db2(df_int, currency, filt_df, output_folder, outlier_scenario)
    
    
    df_proj_0           = full_df_clustered[full_df_clustered['output']==0]
    df_proj_1           = full_df_clustered[full_df_clustered['output']==1]

    # #np.save('project_1.npy', df_proj_1)
    # #df_proj_1.to_csv('out.csv', index=False)  
    # #corr_0, desc_0      = data_understanding_db2(df_proj_1, currency[0])
    
    
    df_proj_0_ext = df_proj_0.drop(['built_area',
                    'lot_area',
                    'total_prelim_cost_est',
                    'prelim_cost_est_est',
                    'equi_prelim_cost',
                    'duration',
                    'unit_price', 
                    'actual_sale_price', 'output'], axis=1).copy()
    
    df_proj_1_ext = df_proj_1.drop(['built_area',
                    'lot_area',
                    'total_prelim_cost_est',
                    'prelim_cost_est_est',
                    'equi_prelim_cost',
                    'duration',
                    'unit_price',
                    'actual_sale_price', 'output'], axis=1).copy()
    
    
    external_features_proj_0, model_shap_ext_0, x_test_0 = feature_importance(df_proj_0_ext, output_folder, 'ext_project_0', outlier_scenario)
    external_features_proj_1, model_shap_ext_1, x_test_1 = feature_importance(df_proj_1_ext, output_folder, 'ext_project_1', outlier_scenario)
            
    #reg_proj_0_ann, proj_0_sorted_feat = modeling_regresion_db2(df_proj_0, 'proj_0', 'ann', output_folder, outlier_scenario)
    # reg_proj_0_svm, _ = modeling_regresion_db2(df_proj_0, 'proj_0', 'svm', output_folder, external_features_proj_0, outlier_scenario)
    # reg_proj_0_rf, _  = modeling_regresion_db2(df_proj_0, 'proj_0', 'rf', output_folder, external_features_proj_0, outlier_scenario)
    
    
    all_perf_proj_0 = {}
    all_perf_proj_1 = {}
    # for model in models:
    #     all_perf_proj_0[model] = modeling_regresion_db2_cv(df_proj_0, 'proj_0', model, output_folder, external_features_proj_0, outlier_scenario)
    #     all_perf_proj_1[model] = modeling_regresion_db2_cv(df_proj_1, 'proj_1', model, output_folder, external_features_proj_1, outlier_scenario)
    
    for model in models:
        all_perf_proj_0[model] = modeling_regresion_db2_cv(df_proj_0, 'proj_0', model, output_folder, external_features_proj_0, outlier_scenario)
        all_perf_proj_1[model] = modeling_regresion_db2_cv(df_proj_1, 'proj_1', model, output_folder, external_features_proj_1, outlier_scenario)
        
    proj_0_summary = build_perf_summary(all_perf_proj_0, models)
    proj_1_summary = build_perf_summary(all_perf_proj_1, models)
    
    
    #reg_proj_1_ann, proj_1_sorted_feat = modeling_regresion_db2(df_proj_1, 'proj_1', 'ann', output_folder, outlier_scenario)
    # reg_proj_1_svm, _ = modeling_regresion_db2(df_proj_1, 'proj_1', 'svm', output_folder, external_features_proj_1, outlier_scenario)
    # reg_proj_1_rf, _  = modeling_regresion_db2(df_proj_1, 'proj_1', 'rf', output_folder, external_features_proj_1, outlier_scenario)
        
    # # Para el caso de proyectos pequeños
    plot_shap_and_perf(model_shap_ext_0, x_test_0, proj_0_summary, output_folder, outlier_scenario, "small_projects", models, colors, top_n=20)
    plot_shap_and_perf(model_shap_ext_1, x_test_1, proj_1_summary, output_folder, outlier_scenario, "large_projects", models, colors, top_n=20)
    
    
    # # Para el otro conjunto
    # plot_shap_and_perf(model, X_test, pd.DataFrame(data2), nature="large_projects", top_n=20)
    
    performance_project= {}
    performance_project[outlier_scenario] =  {
        #"feature_importace_proj_0" : proj_0_sorted_feat,
        #"feature_importace_proj_1" : proj_1_sorted_feat,
        "proj_0_perf_pred": all_perf_proj_0,
        "proj_1_perf_pred": all_perf_proj_1,
        #"reg_proj_0_ann":reg_proj_0_ann,
        #"reg_proj_1_ann":reg_proj_1_ann,
        #"reg_proj_0_svm":reg_proj_0_svm,
        #"reg_proj_1_svm":reg_proj_1_svm,
        #"reg_proj_0_rf":reg_proj_0_rf,
        #"reg_proj_1_rf":reg_proj_1_rf,
        }
    
    return performance_project


#perf_with_outliers = cost_pipeline_run(merged_path, output_folder, False, "with_outliers")
perf_no_outliers   = cost_pipeline_run(merged_path, output_folder, True, "no_outliers")

import pickle
# 2. Save (Serialize) to a file
with open('perf_no_outliers_v1_23_april.pkl', 'wb') as f:
    pickle.dump(perf_no_outliers, f)