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


today = dt.datetime.today().strftime('%Y%m%d')

def select_top20_risk_sector_stocks(final_file_prefix, final_file_folder, output_path):
    final_folder_file_list = os.listdir(final_file_folder)
    final_file_list = []
    for file in final_folder_file_list:
        if final_file_prefix in file and '.csv' in file:
            final_file_list.append(file)
    max_trade_date = max(final_file_list).split('_')[-1][:-4]
    final_fa_file = os.path.join(final_file_folder,final_file_prefix+'_'+max_trade_date+'.csv')
    final_fa_df = pd.read_csv(final_fa_file)
    final_fa_df = final_fa_df.sort_values(['tic','datadate']).drop_duplicates(subset = ['tradedate','tic'],keep='last')
    mapping_sectors = {'Technology': 'Tech Stocks',
                     'Basic Materials': 'Cyclical Stocks',
                     'Consumer Cyclical': 'Cyclical Stocks',
                     'Financial Services': 'Cyclical Stocks',
                     'Real Estate': 'Cyclical Stocks',
                     'Consumer Defensive': 'Defensive Stocks',
                     'Utilities': 'Defensive Stocks',
                     'Industrials': 'Other stocks',
                     'Healthcare': 'Other stocks',
                     'Communication Services': 'Other stocks',
                     'Energy': 'Other stocks'}

    sector_list = []
    for i in range(11):
        sector = f'sector{i*5+10}'
        sector_list.append(sector)
    all_pred_list = []
    for sector in sector_list:
        if os.path.exists(f'results/{sector}/df_predict_best.csv'):
            sect_df = pd.read_csv(f'results/{sector}/df_predict_best.csv')
            res = pd.DataFrame(sect_df.iloc[:, 1:].values.T)
            res.columns = sect_df.iloc[:,0].tolist()
            res['tic'] = sect_df.columns[1:]
            res['sector'] = sector
            all_pred_list.append(res)    
    all_pred_df = pd.concat(all_pred_list)
    trading_dates = list(all_pred_df.columns[:-2])
    
    res_list = []
    for trade_date in trading_dates:
        pred_df = all_pred_df[[trade_date,'tic','sector']]
        pred_df = pred_df.rename(columns={trade_date:'predicted_return'})
        pred_df['trade_date'] = trade_date
        fa_df = final_fa_df[final_fa_df['tradedate'] == trade_date][['X6_PS', 'GICS Sector','tic']]
        pred_final = pred_df.join(fa_df.set_index('tic'),on='tic',how='left').dropna()
        pred_final['koin_sector'] = pred_final['GICS Sector'].apply(lambda x: mapping_sectors[x])
        koin_sectors = pred_final['koin_sector'].unique().tolist()
        for koin_sector in koin_sectors:
            koin_sector_df = pred_final[pred_final['koin_sector'] == koin_sector]
            if len(koin_sector_df) < 3:
                koin_sector_df['risk_level'] = 'medium'
            else:
                
                koin_sector_df['risk_level'] = pd.qcut(koin_sector_df['X6_PS'], q=3,labels=['low','medium','high'])
            risk_levels = koin_sector_df['risk_level'].unique().tolist()

            for risk_level in risk_levels:
                one_risk_df = koin_sector_df[koin_sector_df['risk_level']==risk_level]
                one_risk_df = one_risk_df.sort_values('predicted_return',ascending=False)
                if len(one_risk_df) <= 20:
                    res_list.append(one_risk_df)
                else:               
                    res_list.append(one_risk_df.iloc[:20])    
    res_final= pd.concat(res_list)
    res_final.to_csv(output_path)
    return res_final

    
    
    
def select_top20_stocks():
    sector_list = []
    for i in range(11):
        sector = f'sector{i*5+10}'
        sector_list.append(sector)
    res_list = []
    for sector in sector_list:
        if os.path.exists(f'results/{sector}/df_predict_best.csv'):
            sect_df = pd.read_csv(f'results/{sector}/df_predict_best.csv')
            res = pd.DataFrame(sect_df.iloc[:, 1:].values.T)
            res.columns = sect_df.iloc[:,0].astype(str).tolist()
            res['tic'] = sect_df.columns[1:]
            res['sector'] = sector
            trading_dates = sect_df.iloc[:,0].astype(str)
        
            for i in range(len(trading_dates)):
                one_trade_rank = res[[trading_dates[i],'tic','sector']].dropna().sort_values(trading_dates[i],ascending=False)
                if len(one_trade_rank) >= 5:
                    #top20 = one_trade_rank.iloc[:int(0.2 * len(one_trade_rank))]
                    top20 = one_trade_rank.iloc[:20]
                    top20['trade_date'] = int(trading_dates[i])
                    top20 = top20.rename(columns={trading_dates[i]:"predicted_return"})
                    res_list.append(top20)
    if len(res_list) > 0 :
        top20_df = pd.concat(res_list).sort_values(['trade_date','predicted_return'],ascending=[False, False])
        top20_df.to_csv(f'top20_stocks_{today}.csv',index=False)
    
        return top20_df
    return None



def get_return_and_info_table(selected_stock,df_price):
    trade_date=selected_stock.trade_date.unique()
    all_date=df_price.datadate.unique()
    all_return_table={}
    #all_predicted_return={}
    all_stocks_info = {}
    #for i in range(0,1):
    for i in range(len(trade_date)):
        #match trading date
        index = selected_stock.trade_date==trade_date[i]
        print(trade_date[i])
        #get the corresponding trade period's selected stocks' name
        stocks_name=selected_stock.tic[selected_stock.trade_date==trade_date[i]].values
        temp_info = selected_stock[selected_stock.trade_date==trade_date[i]]
        temp_info = temp_info.reset_index()
        del temp_info['index']
        all_stocks_info[trade_date[i]] = temp_info
        #get the corresponding trade period's selected stocks' predicted return
        asset_expected_return=selected_stock[index].predicted_return.values
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


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    table = df.pivot(columns='ticker')
    # By specifying col[1] in below list comprehension
    # You can select the stock names under multi-level column
    table.columns = [col[1] for col in table.columns]
    table.head()
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)


    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)

def calculate_weight(all_stocks_info,all_return_table,save_path):
    stocks_weight_table = pd.DataFrame([])
    for i in range(len(trade_date)):
        # get selected stocks information
        p1_alldata = all_stocks_info[trade_date[i]]
        # sort it by tic
        p1_alldata = p1_alldata.sort_values('tic')
        p1_alldata = p1_alldata.reset_index()
        del p1_alldata['index']
        # get selected stocks tic
        p1_stock = p1_alldata.tic

        pivot_returns = all_return_table[trade_date[i]].pivot_table(index='datadate', columns='tic', values='daily_return')
        # use the predicted returns as the Expected returns to feed into the portfolio object
        mu, S = pivot_returns.mean(), pivot_returns.cov()
        num_portfolios = 25000
        risk_free_rate = 0.0178
        results, weights = random_portfolios(num_portfolios, mu, S, risk_free_rate)
        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=pivot_returns.columns, columns=['allocation'])
        max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
        print(max_sharpe_allocation.allocation)
        # max_sharpe_allocation = list(max_sharpe_allocation.loc['allocation'].values)

        min_vol_idx = np.argmin(results[0])
        sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=pivot_returns.columns, columns=['allocation'])
        min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
        # min_vol_allocation = list(min_vol_allocation.loc['allocation'].values)

        p1_alldata['mean_weight'] = list(max_sharpe_allocation.allocation.values)
        p1_alldata['min_weight'] = list(min_vol_allocation.allocation.values)

        print(p1_alldata.head())

        print(f"sdp, rp, | {sdp}, {rp} :: sdp_min, rp_min | {sdp_min}, {rp_min}")

        # # get predicted return from selected stocks
        # p1_predicted_return=p1_alldata.pivot_table(index = 'trade_date',columns = 'tic', values = 'predicted_return')
        # # use the predicted returns as the Expected returns to feed into the portfolio object
        # mu = p1_predicted_return.T.values
        #
        # # get the 1-year historical return
        # all_return_table = all_return_table.dropna()
        # p1_return_table=all_return_table[trade_date[i]]
        # p1_return_table_pivot=p1_return_table.pivot_table(index = 'datadate',columns = 'tic', values = 'daily_return')
        # # use the 1-year historical return table to calculate covariance matrix between selected stocks
        # S = risk_models.sample_cov(p1_return_table_pivot)
        # #del S.index.name
        # print(mu.shape, S.shape)
        # # mean variance
        # ef_mean = EfficientFrontier(mu, S,weight_bounds=(0, 1))
        # raw_weights_mean = ef_mean.max_sharpe()
        # cleaned_weights_mean = ef_mean.clean_weights()
        #print(raw_weights_mean)
        #ef.portfolio_performance(verbose=True)
    
        # # minimum variance
        # ef_min = EfficientFrontier([0]*len(p1_stock), S,weight_bounds=(0, 1))
        # raw_weights_min = ef_min.max_sharpe()
        # cleaned_weights_min = ef_min.clean_weights()
        # #print(cleaned_weights_min)
        #
        # p1_alldata['mean_weight'] = cleaned_weights_mean.values()
        # p1_alldata['min_weight'] = cleaned_weights_min.values()
        #
        #ef.portfolio_performance(verbose=True)
        stocks_weight_table = stocks_weight_table.append(pd.DataFrame(p1_alldata), ignore_index=True)
        print(trade_date[i], ": Done")
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
    selected_stock = select_top20_stocks()
    #selected_stock = pd.read_csv('top20_stocks_20210926.csv')
    trade_date=selected_stock.trade_date.unique()
    price_start_date = dt.datetime.strptime(str(min(trade_date)),'%Y%m%d') + dt.timedelta(-122) 
    price_start_date = int(dt.datetime.strftime(price_start_date, '%Y%m%d'))
    input_file = 'raw_data/russeTop600_stock_df.csv'
    df_price = process_price_table('raw_data/russel3000_stock.csv', price_start_date)
    print(selected_stock.tic.unique().shape)
    print(df_price.tic.unique().shape)
    all_stocks_info,  all_return_table  = get_return_and_info_table(selected_stock,df_price)
    print(len(all_stocks_info))
    print(len(all_return_table))

    stocks_weight_table = calculate_weight(all_stocks_info,all_return_table,f"stocks_weight_table{today}.csv")
    print(stocks_weight_table.shape)
    print(stocks_weight_table.head())