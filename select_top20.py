#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 22:31:47 2021

@author: qinfang
"""

import pandas as pd
import numpy as np
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import scipy.optimize as sco
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
import os
from top20_by_sector import *

def select_top20_stocks_by_portfolio_type(predicted_returns_dir, final_df_filtered,sectors_by_portfolio_types):
    final_df_filtered.index = list(range(final_df_filtered.index.shape[0]))
    risk_levels = ['high','medium','low']
    top20_by_portfolio_types = []
    for k, v in sectors_by_portfolio_types.items():
        predicted_df_by_portfolio_types = []
        for i, gsector in enumerate(v['gsector']):
            predicted_df = pd.read_csv(os.path.join(predicted_returns_dir, f"sector{gsector}/df_predict_best.csv"))
            nearest_trade_date = sorted(predicted_df.iloc[:, 0])[-1]
            my_predicted_returns = list(predicted_df.iloc[-1, 1:].values)
            all_tics_in_sector = predicted_df.columns[1:]
            cur_df = pd.DataFrame({'predicted_returns': my_predicted_returns, 'tic': all_tics_in_sector,
                                   'sector':[v['GICS'][i]]*len(my_predicted_returns), 'tradedate':[nearest_trade_date]*len(my_predicted_returns)})
            predicted_df_by_portfolio_types.append(cur_df)
        predicted_df_by_portfolio_types = pd.concat(predicted_df_by_portfolio_types)
        predicted_df_by_portfolio_types['portfolio_type'] = k
        predicted_df_by_portfolio_types.index = list(range(predicted_df_by_portfolio_types.index.shape[0]))

        for level in risk_levels:
            cur_result = pd.merge(final_df_filtered, predicted_df_by_portfolio_types, how='inner', on=['tic', 'tradedate'])
            cur_result = cur_result.loc[:, ['predicted_returns', 'tradedate', 'tic', 'risk_level', 'portfolio_type']]
            cur_result = cur_result.loc[cur_result.loc[:,'risk_level']==level,:].drop_duplicates()
            cur_result = cur_result.sort_values(['tradedate','predicted_returns'], ascending=[False, False]).iloc[:20]
            top20_by_portfolio_types.append(cur_result)

    return top20_by_portfolio_types


def get_return_and_info_table(selected_stock,df_price):
    trade_date=list(selected_stock.tradedate.unique())
    all_date=df_price.datadate.unique()
    all_return_table={}
    #all_predicted_return={}
    all_stocks_info = {}
    #for i in range(0,1):
    for i in range(len(trade_date)):
        #match trading date
        index = selected_stock.tradedate==trade_date[i]
        #get the corresponding trade period's selected stocks' name
        stocks_name=selected_stock.tic[selected_stock.tradedate==trade_date[i]].values
        temp_info = selected_stock[selected_stock.tradedate==trade_date[i]]
        temp_info = temp_info.reset_index()
        del temp_info['index']
        all_stocks_info[trade_date[i]] = temp_info
        #get the corresponding trade period's selected stocks' predicted return
        #asset_expected_return=selected_stock[index].predicted_returns.values
        #get current trade date and calculate trade date last year, it has to be a business date
        last_year_tradedate=int((trade_date[i]-round(trade_date[i]/10000)*10000)+round(trade_date[i]/10000-1)*10000)
        convert_to_yyyymmdd=dt.datetime.strptime(str(last_year_tradedate), '%Y%m%d').strftime('%Y-%m-%d')
        #determine the business date
        #print(convert_to_yyyymmdd)
        ts = pd.Timestamp(convert_to_yyyymmdd) 
        bd = pd.tseries.offsets.BusinessDay(n =1) 
        new_timestamp = ts - bd 
        lastY_tradedate = int(new_timestamp.date().strftime('%Y%m%d'))
        get_date_index=(all_date<trade_date[i]) & (all_date>lastY_tradedate)
        get_date=all_date[get_date_index]
        #get adjusted price table
        return_table=pd.DataFrame()
        for m in range(len(stocks_name)):
            #get stocks's name
            index_tic=(df_price.tic==stocks_name[m])
            #get this stock's all historicall price from sp500_price
            sp500_temp=df_price[index_tic]
            merge_left_data_table = pd.DataFrame(get_date)
            merge_left_data_table.columns = ['datadate']
            temp_price=merge_left_data_table.merge(sp500_temp, on=['datadate'], how='left')
            temp_price = temp_price.dropna()
            temp_price['daily_return']=temp_price.adj_price.pct_change()
            return_table=return_table.append(temp_price,ignore_index=True)
        all_return_table[trade_date[i]] = return_table

    return all_stocks_info,  all_return_table

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(mean_returns.shape[0])
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def calculate_weight(all_stocks_info, all_return_table, trade_dates, save_path):
    stocks_weight_table = pd.DataFrame([])
    for i in range(len(trade_dates)):
        # get selected stocks information
        p1_alldata = all_stocks_info[trade_dates[i]]
        # sort it by tic
        p1_alldata = p1_alldata.sort_values('tic')
        p1_alldata = p1_alldata.reset_index()
        del p1_alldata['index']
        # get selected stocks tic
        pivot_returns = all_return_table[trade_dates[i]].pivot_table(index='datadate', columns='tic', values='daily_return')
        # use the predicted returns as the Expected returns to feed into the portfolio object
        mu, S = pivot_returns.mean(), pivot_returns.cov()
        num_portfolios = 25000
        risk_free_rate = 0.0178
        results, weights = random_portfolios(num_portfolios, mu, S, risk_free_rate)
        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
        print(weights[max_sharpe_idx], weights[max_sharpe_idx].shape)
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=pivot_returns.columns, columns=['allocation'])
        max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
        print(max_sharpe_allocation.allocation)
        min_vol_idx = np.argmin(results[0])
        sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=pivot_returns.columns, columns=['allocation'])
        min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
        p1_alldata['mean_weight'] = list(max_sharpe_allocation.allocation.values)
        p1_alldata['min_weight'] = list(min_vol_allocation.allocation.values)
        print(p1_alldata.head())
        print(f"sdp, rp, | {sdp}, {rp} :: sdp_min, rp_min | {sdp_min}, {rp_min}")
        stocks_weight_table = stocks_weight_table.append(pd.DataFrame(p1_alldata), ignore_index=True)
        print(trade_dates[i], ": Done")
    stocks_weight_table.to_csv(save_path,index=False)
    return stocks_weight_table
    
    
def process_price_table(input_file, price_start_date):
       
    df_prices = pd.read_csv(input_file)
    df_price = df_prices[['Date','tic','Adj Close']]
    df_price['datadate'] =df_price['Date'].apply(lambda x: x.replace('-','')).astype(int)
    df_price = df_price.rename(columns={'Adj Close':'adj_price'})
    df_price = df_price[df_price['datadate']>=price_start_date]
    df_price = df_price[['datadate', 'tic', 'adj_price']]
    return df_price         



if __name__ == '__main__':
    ## generate portfolio type and sectors mapping
    today = dt.datetime.today().strftime('%Y%m%d')
    predicted_returns_dir, outp_fp = 'results', "top20-ressull3000-based-on-sharpe-ratio.csv"
    final_df = pd.read_csv('--ress3k_fundamental_final.csv')
    final_df_top20 = pd.read_csv(outp_fp)
    final_df_filtered = split_sector(final_df, final_df_top20)
    sectors_by_portfolio_types = {'tech': {'GICS':['Technology', 'Communication Services'],'gsector':[30, 60]},
                                  'cyclical':{'GICS':['Basic Materials', 'Real Estate',
                                                      'Financial Services','Industrials','Consumer Cyclical'],
                                              'gsector':[15, 35, 40, 20,25]},
                                  'defensive':{'GICS':['Utilities', 'Consumer Defensive'],'gsector':[55, 45]},
                                  'all':{'GICS':['Healthcare', 'Basic Materials', 'Industrials',
                                                'Consumer Cyclical', 'Technology', 'Real Estate',
                                                'Financial Services', 'Consumer Defensive', 'Energy',
                                                'Utilities','Communication Services', 'unknown'],
                                         'gsector':[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]}}

    top20_by_portfolio_types = select_top20_stocks_by_portfolio_type(predicted_returns_dir, final_df_filtered, sectors_by_portfolio_types)
    trade_dates = list(np.unique(list(map(lambda x: x.tradedate.unique()[0], top20_by_portfolio_types))))
    price_start_date = dt.datetime.strptime(str(min(trade_dates)),'%Y%m%d') + dt.timedelta(-365)
    price_start_date = int(dt.datetime.strftime(price_start_date, '%Y%m%d'))
    # input_file = 'raw_data/russeTop600_stock_df.csv'
    df_price = process_price_table('raw_data/russel3000_stock.csv', price_start_date)
    for i, top20_by_portfolio_type in enumerate(top20_by_portfolio_types):
        weights_res_dir = 'results/weights'
        risk, portfolio_type = top20_by_portfolio_type.risk_level.unique()[0], top20_by_portfolio_type.portfolio_type.unique()[0]
        fp_output = f"stocks_weight_table{today}_{risk}_{portfolio_type}.csv"
        all_stocks_info,  all_return_table  = get_return_and_info_table(top20_by_portfolio_type,df_price)
        stocks_weight_table = calculate_weight(all_stocks_info,all_return_table,trade_dates, os.path.join(weights_res_dir,fp_output))
