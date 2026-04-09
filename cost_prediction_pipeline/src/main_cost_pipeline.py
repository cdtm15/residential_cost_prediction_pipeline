#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:52:17 2026

@author: cristiantobar
"""

## New version of the cost prediction pipeline
import numpy as np
# from file_data_reception_db2 import data_reception_db2
# from file_data_understanding_db2 import data_understanding_db2
#from file_data_preparation_db2 import data_preparation_db2
#from file_modeling_db2 import modeling_db2
#from file_modeling_regresion_db2 import modeling_regresion_db2

#Subpipelines
from residential_cost_prediction.file_data_preparation_db2 import data_preparation_db2
from residential_cost_prediction.file_data_reception_db2 import data_reception_db2
from residential_cost_prediction.file_modeling_regresion_db2 import modeling_regresion_db2

#Models
from residential_cost_prediction.models.file_aglomerative_clustering import aglomerative_clustering_db2
from residential_cost_prediction.models.file_feature_importance_shap import feature_importance


#data_location       = "/Users/cristiantobar/Library/CloudStorage/OneDrive-CDTCreaTIC/Fortalecimiento/Investigacion/doctorado_cristian/procesamiento_datos/base_datos_3/datos"
data_location  = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/base_datos_3/datos"
data_filename  = ["database_2_csv.csv"]
output_folder  = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/cost_prediction_pipeline/assets/output_figures"
merged_path    = data_location + "/" + data_filename[0]
currency       = ['Million COPm', 'DOLLARm']
dollar2cop     = 4333.11



df             = data_reception_db2(merged_path)
filt_df        = data_preparation_db2(df, currency[1], dollar2cop, True)
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
sorted_features = feature_importance(df_int, output_folder)
df_clustered, full_df_clustered    = aglomerative_clustering_db2(df_int, currency, filt_df, output_folder)

df_proj_0           = full_df_clustered[full_df_clustered['output']==0]
df_proj_1           = full_df_clustered[full_df_clustered['output']==1]

# #np.save('project_1.npy', df_proj_1)
# #df_proj_1.to_csv('out.csv', index=False)  
# #corr_0, desc_0      = data_understanding_db2(df_proj_1, currency[0])

reg_proj_0_ann, proj_0_sorted_feat = modeling_regresion_db2(df_proj_0, 'proj_0', 'ann')
reg_proj_0_svm, _ = modeling_regresion_db2(df_proj_0, 'proj_0', 'svm')
reg_proj_0_rf, _  = modeling_regresion_db2(df_proj_0, 'proj_0', 'rf')

reg_proj_1_ann, proj_1_sorted_feat = modeling_regresion_db2(df_proj_1, 'proj_1', 'ann')
reg_proj_1_svm, _ = modeling_regresion_db2(df_proj_1, 'proj_1', 'svm')
reg_proj_1_rf, _  = modeling_regresion_db2(df_proj_1, 'proj_1', 'rf')

