# -*- coding: utf-8 -*-

import pandas as pd

def build_perf_summary(project_dict):
    """
    Construye un DataFrame resumen a partir de un diccionario tipo:
    {
        ...: {
            'ann': [df0, df1, ...],
            'svm': [df0, df1, ...],
            'rf' : [df0, df1, ...]
        }
    }
    
    Toma el DataFrame en el índice 0 para cada modelo y devuelve un único
    DataFrame con columnas:
    Num_features, Market_feature, ANN_R2, ANN_MAE, SVM_R2, SVM_MAE, RF_R2, RF_MAE
    """
    # Si el diccionario principal tiene un solo subdiccionario adentro
    # tomamos ese contenido. Ajusta esto si tu estructura es distinta.
    #inner_dict = next(iter(project_dict.values()))
    
    model_map = {
        'ann': 'ANN',
        'svm': 'SVM',
        'rf': 'RF'
    }
    
    merged_df = None
    
    for model_key, model_name in model_map.items():
        df_model = project_dict[model_key][0].copy()
        
        # Nos quedamos solo con las columnas que interesan
        df_model = df_model[['Num Features', 'Feature', 'R²', 'MAE']].copy()
        
        # Renombramos para el formato final
        df_model = df_model.rename(columns={
            'Num Features': 'Num_features',
            'Feature': 'Market_feature',
            'R²': f'{model_name}_R2',
            'MAE': f'{model_name}_MAE'
        })
        
        # Merge incremental por llaves comunes
        if merged_df is None:
            merged_df = df_model
        else:
            merged_df = pd.merge(
                merged_df,
                df_model,
                on=['Num_features', 'Market_feature'],
                how='outer'
            )
    
    return merged_df