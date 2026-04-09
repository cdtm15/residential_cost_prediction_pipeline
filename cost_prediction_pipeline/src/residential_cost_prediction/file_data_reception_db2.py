#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:01:09 2024

@author: cristiantobar
"""
import pandas as pd

def data_reception_db2(datapath):
    old_df         = pd.read_csv(datapath, header=None, sep=';')
    df             = old_df.drop([0,1])

    df.columns     = ['start_year',
                      'start_quarter',
                      'completion_year',
                      'completion_quarter',
                      
                      'location',
                      'built_area',
                      'lot_area',
                      'total_prelim_cost_est',
                      'prelim_cost_est_est',
                      'equi_prelim_cost',
                      'duration',
                      'unit_price',
                      
                      'num_build_permit',
                      'BSI_num_contracts',
                      'WPI_mater_price',
                      'total_floor_area_permit',
                      'cumulative_liquidity',
                      'private_sector_investment',
                      'land_price_index',
                      'bank_loans_num',
                      'bank_loans_amou',
                      'loan_interest_rate',
                      'construc_cost_priv_time_fin',
                      'construc_cost_priv_time_start',
                      'exchange_rate_to_dollar',
                      'street_exchange_rate_to_dollar',
                      'consumer_price_index',
                      'cpi_services',
                      'stock_market',
                      'city_population',
                      'gold_price',
                      
                      'actual_sale_price',
                      'actual_construction_cost',
                      ]
    
    
    df["start_year"] = df.start_year.astype(float)
    df["start_quarter"] = df.start_quarter.astype(float)
    df["completion_year"] = df.completion_year.astype(float)
    df["completion_quarter"] = df.completion_quarter.astype(float)
                      
    df["location"] = df.location.astype(float)
    df["built_area"] = df.built_area.astype(float)
    df["lot_area"] = df.lot_area.astype(float)
    df["total_prelim_cost_est"] = df.total_prelim_cost_est.astype(float)
    df["prelim_cost_est_est"] = df.prelim_cost_est_est.astype(float)
    df["equi_prelim_cost"] = df.equi_prelim_cost.astype(float)
    df["duration"] = df.duration.astype(float)
    df["unit_price"] = df.unit_price.astype(float)
                      
    df["num_build_permit"] = df.num_build_permit.astype(float)
    df["BSI_num_contracts"] = df.BSI_num_contracts.astype(float)
    df["WPI_mater_price"] = df.WPI_mater_price.astype(float)
    df["total_floor_area_permit"] = df.total_floor_area_permit.astype(float)
    df["cumulative_liquidity"] = df.cumulative_liquidity.astype(float)
    df["private_sector_investment"] = df.private_sector_investment.astype(float)
    df["land_price_index"] = df.land_price_index.astype(float)
    df["bank_loans_num"] = df.bank_loans_num.astype(float)
    df["bank_loans_amou"] = df.bank_loans_amou.astype(float)
    df["loan_interest_rate"] = df.loan_interest_rate.astype(float)
    df["construc_cost_priv_time_fin"] = df.construc_cost_priv_time_fin.astype(float)
    df["construc_cost_priv_time_start"] = df.construc_cost_priv_time_start.astype(float)
    df["exchange_rate_to_dollar"] = df.exchange_rate_to_dollar.astype(float)
    df["street_exchange_rate_to_dollar"] = df.street_exchange_rate_to_dollar.astype(float)
    df["consumer_price_index"] = df.consumer_price_index.astype(float)
    df["cpi_services"] = df.cpi_services.astype(float)
    df["stock_market"] = df.stock_market.astype(float)
    df["city_population"] = df.city_population.astype(float)
    df["gold_price"] = df.gold_price.astype(float)
    
    df["actual_sale_price"] = df.actual_sale_price.astype(float)
    df["actual_construction_cost"] = df.actual_construction_cost.astype(float)
    
    df = df.drop(['start_year', 'start_quarter', 'completion_year', 'completion_quarter', 'location'], axis=1).copy()   
    
    return df