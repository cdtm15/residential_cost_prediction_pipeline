#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:55:57 2024

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


def remove_outliers(df, df_encoded_ff):
    # OPTION 3: iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    df_filtered          = df.copy()
    df_orig              = df_encoded_ff.copy()
    df_new               = df.drop('loan_interest_rate',axis= 1).copy()

    iqr                  = df_new.quantile(0.75, numeric_only=False) - df_new.quantile(0.25, numeric_only=False)
    lim                  = np.abs((df_new- df_new.median()) / iqr) < 2.22
    cols                 = df_new.select_dtypes('number').columns  # limits to a (float), b (int) and e (timedelta)
    df_orig.loc[:, cols] = df_new.where(lim, np.nan)
    df_orig.dropna(subset=cols, inplace=True) # drop rows with NaN in numerical columns
    df_filtered.loc[:, cols] = df_new.where(lim, np.nan)
    df_filtered.dropna(subset=cols, inplace=True)
    return df_orig, df_filtered

#def feature_importance(df):



def data_preparation_db2(df, currency, dollar2cop, remove_outliers_flag):
    
    df_encoded = pd.get_dummies(df, columns=[
                                            'loan_interest_rate'
                                            ])
        
    df['total_prelim_cost_est']        = df['total_prelim_cost_est']*10000000
    df['prelim_cost_est_est']          = df['prelim_cost_est_est']*10000
    df['equi_prelim_cost']             = df['equi_prelim_cost']*10000
    df['unit_price']                   = df['unit_price']*10000
    df['actual_sale_price']            = df['actual_sale_price']*10000
    df['actual_construction_cost']     = df['actual_construction_cost']*10000
    
        
    df['cumulative_liquidity']         = df['cumulative_liquidity']*10000000
    df['private_sector_investment']    = df['private_sector_investment']*10000000
    df['land_price_index']             = df['land_price_index']*10000000
    df['bank_loans_amou']              = df['bank_loans_amou']*10000000
    df['construc_cost_priv_time_fin']  = df['construc_cost_priv_time_fin']*10000
    df['construc_cost_priv_time_start']= df['construc_cost_priv_time_fin']*10000
    df['duration']                     = df['duration']*13

    if currency == 'Million COPm':
        #VALOR DEL DOLAR DE 10 DE NOVIEMBRE DE 2024
        df['total_prelim_cost_est']         = ((df['total_prelim_cost_est']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['prelim_cost_est_est']           = ((df['prelim_cost_est_est']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['equi_prelim_cost']              = ((df['equi_prelim_cost']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['unit_price']                    = ((df['unit_price']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['actual_sale_price']             = ((df['actual_sale_price']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['actual_construction_cost']      = ((df['actual_construction_cost']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        
        df['cumulative_liquidity']          = ((df['cumulative_liquidity']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['private_sector_investment']     = ((df['private_sector_investment']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['land_price_index']              = ((df['land_price_index']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['bank_loans_amou']               = ((df['bank_loans_amou']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['construc_cost_priv_time_fin']   = ((df['construc_cost_priv_time_fin']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['construc_cost_priv_time_start'] = ((df['construc_cost_priv_time_start']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
        df['gold_price']                    = ((df['gold_price']/df['exchange_rate_to_dollar'])*dollar2cop)/1000000
    
    if currency == 'DOLLARm':
        df['total_prelim_cost_est']         = df['total_prelim_cost_est']/df['exchange_rate_to_dollar']
        df['prelim_cost_est_est']           = df['prelim_cost_est_est']/df['exchange_rate_to_dollar']
        df['equi_prelim_cost']              = df['equi_prelim_cost']/df['exchange_rate_to_dollar']
        df['unit_price']                    = df['unit_price']/df['exchange_rate_to_dollar']
        df['actual_sale_price']             = df['actual_sale_price']/df['exchange_rate_to_dollar']
        df['actual_construction_cost']      = df['actual_construction_cost']/df['exchange_rate_to_dollar']
        
        df['cumulative_liquidity']          = df['cumulative_liquidity']/df['exchange_rate_to_dollar']
        df['private_sector_investment']     = df['private_sector_investment']/df['exchange_rate_to_dollar']
        df['land_price_index']              = df['land_price_index']/df['exchange_rate_to_dollar']
        df['bank_loans_amou']               = df['bank_loans_amou']/df['exchange_rate_to_dollar']
        df['construc_cost_priv_time_fin']   = df['construc_cost_priv_time_fin']/df['exchange_rate_to_dollar']
        df['construc_cost_priv_time_start'] = df['construc_cost_priv_time_start']/df['exchange_rate_to_dollar']
        df['gold_price']                    = df['gold_price']/df['exchange_rate_to_dollar']
    
    
    descriptives = df.describe().round(2)

    if remove_outliers_flag == True:
        df_wo_out_encoded, df_wo_out  = remove_outliers(df, df_encoded)
        #df_int = df_wo_outliers.iloc[:, [1,2,3,4,5,6,7,32]]
       
    else:
        df_wo_out = df.drop('loan_interest_rate',axis= 1)
        
    return df_wo_out
    
    

