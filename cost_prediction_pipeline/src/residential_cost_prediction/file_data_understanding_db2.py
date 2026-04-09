#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:06:11 2024

@author: cristiantobar
"""
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np



def data_understanding_db2(df, currency):
    "¿Cuántos registros hay?"
    print('Cantidad de Filas y columnas:',df.shape)
    print('Nombre columnas:',df.columns)
    #Tabla descriptiva de los datos
    descriptives = df.describe()

    "¿Están todas las filas completas ó tenemos campos con valores nulos?"
    df.info()

    "¿Que datos son discretos y cuales continuos?"
    df.head(5)
    # df['start_quarter'].unique()
    # df['completion_quarter'].unique()
    #df['location'].unique()
    #df['loan_interest_rate'].unique()
    
    
    "¿Hay correlación entre features (características)?"
    
    
    df_int = df[['built_area',
    'lot_area',
    'total_prelim_cost_est',
    'prelim_cost_est_est',
    'equi_prelim_cost',
    'duration',
    'unit_price',
    'actual_construction_cost']]
            
    df_ext = df.drop(['built_area',
    'lot_area',
    'total_prelim_cost_est',
    'prelim_cost_est_est',
    'equi_prelim_cost',
    'duration',
    'unit_price',
    'actual_construction_cost',
    ],axis = 1).copy()
        
    corre = df_int.corr()
    mask = np.triu(np.ones_like(corre, dtype=bool))       # Sample figsize in inches
    fig, ax = plt.subplots(figsize=(12,8))  
    sb.heatmap(corre, mask=mask, cmap='coolwarm', annot=True, annot_kws={"size": 12})
    sb.set(font_scale=1.5)
    ax.grid(False)
    plt.savefig('corr_interna_db2.pdf',bbox_inches='tight')
    plt.show()
    
    corre = df_ext.corr()
    mask = np.triu(np.ones_like(corre, dtype=bool))       # Sample figsize in inches
    fig, ax = plt.subplots(figsize=(20,15))  
    sb.heatmap(corre, mask=mask, cmap='coolwarm', annot=True, annot_kws={"size": 14})
    sb.set(font_scale=1.5)
    ax.grid(False)
    plt.savefig('corr_externa_db2.pdf',bbox_inches='tight')
    plt.show()
    
    corre = df.corr()
    mask = np.triu(np.ones_like(corre, dtype=bool))       # Sample figsize in inches
    fig, ax = plt.subplots(figsize=(30,20))  
    sb.heatmap(corre, mask=mask, cmap='coolwarm', annot=True, annot_kws={"size": 12})
    sb.set(font_scale=1.5)
    ax.grid(False)
    plt.savefig('corr_db2.pdf',bbox_inches='tight')
    plt.show()
    
    # Matriz de correlación
    corre = df.corr(numeric_only=True)  # por si hay columnas no numéricas
    mask = np.triu(np.ones_like(corre, dtype=bool))
    
    # Figura
    fig, ax = plt.subplots(figsize=(18, 14))  # ajusta según #features
    sb.set_theme(style="white")
    
    # Heatmap
    sb.heatmap(
        corre, 
        mask=mask,
        cmap="RdBu_r",    # paleta divergente
        vmin=-1, vmax=1,  # escala fija
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation coefficient"}
    )
    
    # Mejorar etiquetas
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    ax.set_title("Correlation Heatmap of Variables", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    
    # Guardar en formato de alta calidad
    plt.savefig("corr_db2_new.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    # "¿Siguen alguna distribución?"
    # for group in df['location'].unique():
    #     filtered_df = df[df['location']==group] 
    #     filtered_df['built_area'].hist(figsize=(8, 4), bins=25, label = group, alpha= 0.7)
    # plt.xlabel('\n Squared meters \n')
    # plt.ylabel('\n Construction projects \n')
    # plt.legend()
    # plt.show()
    
    # for group in ['location']:
    #     df[group].hist(bins=25, label = group, alpha= 0.7)
    # plt.xlabel('\n N/A \n')
    # plt.ylabel('\n Construction projects \n')
    # plt.legend()
    # plt.show() 
    
    for group in ['built_area', 'lot_area']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n Squared meters \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['prelim_cost_est_est', 'equi_prelim_cost']:
        df[group].hist(figsize=(8, 4), bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency)
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['prelim_cost_est_est', 'equi_prelim_cost','actual_construction_cost']:
        df[group].hist(figsize=(8, 4), bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency)
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['duration']:
        df[group].hist(figsize=(8, 4), bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n Weeks \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
       
    for group in ['unit_price', 'actual_sale_price']:
        df[group].hist(figsize=(8, 4), bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency)
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['loan_interest_rate']:
        df[group].hist(figsize=(8, 4), bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n % \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    # for group in ['start_year', 'completion_year']:
    #     df[group].hist(bins=25, label = group, alpha= 0.7)
    # plt.xlabel('\n Dates? \n')
    # plt.ylabel('\n Construction projects \n')
    # plt.legend()
    # plt.show()
    
    # for group in ['start_quarter', 'completion_quarter']:
    #     df[group].hist(bins=25, label = group, alpha= 0.7)
    # plt.xlabel('\n Dates? \n')
    # plt.ylabel('\n Construction projects \n')
    # plt.legend()
    # plt.show()
    
    for group in ['num_build_permit']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['BSI_num_contracts']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A  \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['WPI_mater_price']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['total_floor_area_permit']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['bank_loans_num']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['consumer_price_index']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['cpi_services']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['stock_market']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['city_population']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n N/A \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['cumulative_liquidity']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency)
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['private_sector_investment']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n 10000000 IRRm \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['land_price_index']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency)
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['construc_cost_priv_time_start']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency +'/m2')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['construc_cost_priv_time_fin']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel(currency +'/m2')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['gold_price']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n IRRm \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['exchange_rate_to_dollar']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n IRRm \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    for group in ['street_exchange_rate_to_dollar']:
        df[group].hist(bins=25, label = group, alpha= 0.7)
    plt.xlabel('\n IRRm \n')
    plt.ylabel('\n Construction projects \n')
    plt.legend()
    plt.show()
    
    "¿Cuales son los Outliers? (unos pocos datos aislados que difieren drásticamente del resto y “contaminan” ó desvían las distribuciones)"
    df.boxplot(column=['built_area', 'lot_area'])
    plt.show()
    df.boxplot(column=['total_prelim_cost_est', 'prelim_cost_est_est', 'equi_prelim_cost'])
    plt.show()
    df.boxplot(column=['total_prelim_cost_est', 'prelim_cost_est_est', 'equi_prelim_cost','actual_construction_cost'])
    plt.show()
    df.boxplot(column=['duration'])
    plt.show()
    df.boxplot(column=['unit_price', 'actual_sale_price'])
    plt.show()
    df.boxplot(column=['loan_interest_rate'])
    plt.show()
    # df.boxplot(column=['start_year', 'completion_year'])
    # plt.show()
    # df.boxplot(column=['start_quarter', 'completion_quarter'])
    # plt.show()
    df.boxplot(column=['num_build_permit'])
    plt.show()
    df.boxplot(column=['BSI_num_contracts'])
    plt.show()
    df.boxplot(column=['WPI_mater_price'])
    plt.show()
    df.boxplot(column=['total_floor_area_permit'])
    plt.show()
    df.boxplot(column=['bank_loans_num'])
    plt.show()
    df.boxplot(column=['consumer_price_index'])
    plt.show()
    df.boxplot(column=['cpi_services'])
    plt.show()
    df.boxplot(column=['stock_market'])
    plt.show()
    df.boxplot(column=['city_population'])
    plt.show()
    df.boxplot(column=['cumulative_liquidity'])
    plt.show()
    df.boxplot(column=['private_sector_investment'])
    plt.show()
    df.boxplot(column=['land_price_index'])
    plt.show()
    df.boxplot(column=['construc_cost_priv_time_start'])
    plt.show()
    df.boxplot(column=['construc_cost_priv_time_fin'])
    plt.show()
    df.boxplot(column=['gold_price'])
    plt.show()
    df.boxplot(column=['exchange_rate_to_dollar'])
    plt.show()
    df.boxplot(column=['street_exchange_rate_to_dollar'])
    plt.show()

    # df.hist(column=['built_area', 'lot_area'],  bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
    # plt.title('Built area', fontsize = 20)
    # plt.xlabel('\n Squared meters \n', fontsize = 20)
    # plt.ylabel('\n Construction projects \n', fontsize = 20)
    # plt.tick_params( labelsize = 20)
    # plt.show()

    
    # df.hist(column='lot_area',  bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
    # plt.title('Lot area', fontsize = 20)
    # plt.xlabel('\n Squared meters \n', fontsize = 20)
    # plt.ylabel('\n Construction projects \n', fontsize = 20)
    # plt.tick_params( labelsize = 20)
    # plt.show()

    # df.boxplot(column='built_area', by = 'location')
    # plt.suptitle('')
    
    return corre, descriptives
    
