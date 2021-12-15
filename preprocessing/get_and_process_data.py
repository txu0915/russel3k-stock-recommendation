#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:47 2021

@author: qinfang
"""

#import yahoo_fin.stock_info as si
import pandas as pd
import datetime as dt
import requests
import os
from config import config
import yfinance as yf
import numpy as np
import pandas_market_calendars as mcal



    

def get_key_metrics(tic,limit=100):

    api_key = config.api_key
    v3_string = 'https://financialmodelingprep.com/api/v3'
   
    try:
        target_url = f'{v3_string}/key-metrics/{tic}?period=quarter&limit={limit}&apikey={api_key}'
        #print(target_url)
        res = requests.get(target_url).json()
#             #res = res.get('historical')
#             return pd.DataFrame(res).drop(columns = "label")
        return res
    except Exception as e:
        print(e)
        print(f'get_price failed for {tic}')
        pass

def get_tics_key_metrics(km_file, tic_list):        
    exist_file = os.path.exists(km_file)
    if exist_file:
        old_km = pd.read_csv(km_file)
        new_data = []
        old_tics = old_km['symbol'].unique()
        for tic in tic_list:
            if tic in old_tics:
                last_date = old_km[old_km['symbol'] == tic].date.max()
                max_dt = dt.datetime.strptime(last_date, '%Y-%m-%d')
                today = dt.datetime.now()
                num_quarters = (today-max_dt).days//90+2
                new_tic_km = get_key_metrics(tic,limit=num_quarters)
                new_data.append(pd.DataFrame(new_tic_km))
            else:
                one_km_raw = get_key_metrics(tic,limit=100)
                one_tic_km_raw = pd.DataFrame(one_km_raw)
                new_data.append(one_tic_km_raw)
        if len(new_data)>0:
            new_km_df = pd.concat(new_data)
            km_df = old_km.append(new_km_df)
        else:
            km_df = old_km
    else:
        kms = []
        idx = 0
        for tic in tic_list:
            if not idx%100:
                print(f"....{tic} | progress: {idx/len(tic_list)*100}%....")
            one_km_raw = get_key_metrics(tic,limit=100)
            one_tic_km_raw = pd.DataFrame(one_km_raw)
            kms.append(one_tic_km_raw)
            idx += 1
        if len(kms)>0:
            km_df = pd.concat(kms)
        else:
            return None
        

    km_df = km_df.drop_duplicates(subset=['date','symbol']).sort_values(['symbol','date']).reset_index(drop=True)
    km_df.to_csv(km_file, index=False)

    
    return km_df


#get income-statement data
def get_income_statements(tic,limit=100):

    api_key = config.api_key
    v3_string = 'https://financialmodelingprep.com/api/v3'
   
    try:
        target_url = f'{v3_string}/income-statement/{tic}?period=quarter&limit={limit}&apikey={api_key}'
        res = requests.get(target_url).json()
        return res
    except Exception as e:
        print(e)
        print(f'get_price failed for {tic}')
        pass 

def get_tics_incom_states(fa_file,tic_list):
    exist_file = os.path.exists(fa_file)
    if exist_file:
        old_fa = pd.read_csv(fa_file)
        new_data = []
        old_tics = old_fa['symbol'].unique()
        for tic in tic_list:
            if tic in old_tics:
                last_date = old_fa[old_fa['symbol'] == tic].date.max()
                max_dt = dt.datetime.strptime(last_date, '%Y-%m-%d')
                today = dt.datetime.now()
                num_quarters = (today-max_dt).days//90+2
                new_tic_fa = get_income_statements(tic,limit=num_quarters)
                new_data.append(pd.DataFrame(new_tic_fa))
            else:
                one_fa_raw = get_income_statements(tic,limit=100)
                one_tic_fa_raw = pd.DataFrame(one_fa_raw)
                new_data.append(one_tic_fa_raw)
        if len(new_data)>0:
            new_fa_df = pd.concat(new_data)
            fa_df = old_fa.append(new_fa_df)
        else:
            fa_df = old_fa
    else:
        fas = []
        idx = 0
        for tic in tic_list:
            if not idx%100:
                print(f"....{tic} | progress: {idx/len(tic_list)*100}%....")
            one_fa_raw = get_income_statements(tic,limit=100)
            one_tic_fa_raw = pd.DataFrame(one_fa_raw)
            fas.append(one_tic_fa_raw)
            idx += 1
        if len(fas)>0:
            fa_df = pd.concat(fas)
        else:
            return None
        

    fa_df = fa_df.drop_duplicates(subset=['date','symbol']).sort_values(['symbol','date']).reset_index(drop=True)
    fa_df.to_csv(fa_file, index=False)
    
    return fa_df
    
 
    
    
#get stocks information
def get_stocks_data(stock_file,tic_list,lookback_days):
    
    end = dt.datetime.now()
    exist_file = os.path.exists(stock_file)

    if exist_file:
        old_data = pd.read_csv(stock_file, parse_dates=['Date'])
        old_data = old_data.drop_duplicates(subset=['tic','Date'])
        new_data_list = []
        old_tics = old_data['tic'].unique().tolist()
        for tick in tic_list:
            print("....",tick, '....')
            if tick not in old_tics:
                df_tick = pull_data_by_tic(tick,end + dt.timedelta(days=-lookback_days),end)
                if len(df_tick) > 0:
                        new_data_list.append(df_tick)
            else:
                start = old_data[old_data['tic'] == tick].Date.max()+dt.timedelta(days=1)
                if start <= end:
                    df_tick = pull_data_by_tic(tick,start ,end)
                    if len(df_tick) > 0:
                        new_data_list.append(df_tick)

        if len(new_data_list) > 0 :
            new_data = pd.concat(new_data_list)
            df = old_data.append(new_data)
            df = df.drop_duplicates(subset=['tic','Date'])
            df.to_csv(stock_file, index=False)
        else:
            df = old_data
        return df
    else:
        stocks = []
        idx = 0
        for tic in tic_list:
            if not idx%100:
                print(f"....{tic} | progress: {idx/len(tic_list)*100}%....")
            df_tic = yf.download(tic,  end + dt.timedelta(days=-lookback_days), end, progress=False)
            df_tic = df_tic.reset_index()
            df_tic['tic'] = tic
            stocks.append(df_tic)
            idx += 1
        if len(stocks) > 0:
            df = pd.concat(stocks)
            df = df.drop_duplicates(subset=['tic','Date'])
            df.to_csv(stock_file, index=False)
            return df

    return None

def pull_data_by_tic(ticker, start, end):
    df_ticker = yf.download(ticker,start,end, progress=False)
    df_ticker = df_ticker.reset_index()
    df_ticker['tic'] = ticker
    return df_ticker
    
    
#processing data    
quarters = config.quarters
quarters2 = config.quarters2


trade_month = ['03', '03', '06', '06', '06', '09', '09', '09', '12', '12', '12', '03']
def get_year_quarter(x):
    return int(str(x.year) + quarters[x.quarter-1])
def get_quarters(x):
    return str(x.year) + quarters2[x.quarter-1]
def get_quarter_num(x):
    return x.quarter
# =============================================================================
# def get_trade_date(x):
#     return (str(x.year) if x.quarter<4 else str(x.year+1))+trade_month[(x.quarter+1)%4-1]+'01' 
# =============================================================================
def get_trade_date(x):
    return (str(x.year) if x.month<12 else str(x.year+1))+trade_month[x.month-1]+'01' 
def get_trade_month(x):
    return int((str(x.year) if x.month<12 else str(x.year+1))+trade_month[x.month-1])


        
def processing_raw_data(km_df,fa_df,stocks_df,sector_tic, gvkey_conm_unique, final_file_prefix):

    df1 = km_df[['symbol','date','period','roe','peRatio','priceToSalesRatio','evToSales','pbRatio',
                        'pocfratio','currentRatio','enterpriseValueOverEBITDA','evToOperatingCashFlow',
                        'debtToAssets','netDebtToEBITDA','debtToEquity','payablesTurnover','daysSalesOutstanding',
                        'daysPayablesOutstanding']]
    df1 = df1.rename(columns = {'symbol':'tic','period':'quarter','roe':'X4_ROE','peRatio':'X5_PE',
                                'priceToSalesRatio':'X6_PS','evToSales':'X9_OM','pbRatio':'X10_PB','pocfratio':'X11_PCFO',
                                'currentRatio':'X12_CR','enterpriseValueOverEBITDA':'X13_EM','evToOperatingCashFlow':'X14_EVCFO',
                                'debtToAssets':'X15_LTDTA','netDebtToEBITDA':'X16_WCR','debtToEquity':'X17_DE','payablesTurnover':'X18_QR',
                                'daysSalesOutstanding':'X19_DSI','daysPayablesOutstanding':'X20_DPO'})
    df1['datadate'] = pd.to_datetime(df1['date'])
    df1['datadate'] = df1['datadate'].apply(lambda x: dt.datetime.strftime(x,'%Y%m%d'))
    df1 = df1.set_index(['datadate','tic'])
    
    
    df2 = fa_df[['date','symbol','period','revenue','eps','operatingIncomeRatio','netIncomeRatio','grossProfitRatio']]
    df2 = df2.rename(columns={'symbol':'tic','revenue':'X1_REVGH','eps':'X2_EPS','operatingIncomeRatio':'X3_ROA',
                              'netIncomeRatio':'X7_NPM','grossProfitRatio':'X8_GPM'})
    df2 = df2.loc[df2.loc[:, 'date'] >= '1900-01-01',:] ## removing unrealistic data
    df2['datadate2'] = pd.to_datetime(df2['date'])
    df2['datadate'] = df2['datadate2'].apply(lambda x: dt.datetime.strftime(x,'%Y%m%d'))
    df2 = df2.set_index(['datadate','tic'])                                                

    df = pd.merge(df1,df2.drop('date',axis=1), on=(['datadate','tic']), how='inner')
    
    df = df.reset_index()
    df['fyearq'] = df['datadate2'].apply(lambda x: dt.datetime.strftime(x,'%Y'))
    df['datacqtr'] = df['datadate2'].apply(get_quarters)
    df['datafqtr'] = df['fyearq']+ df['quarter']
    df['tradedate'] = df['datadate2'].apply(get_trade_date)
    df['fqtr'] = df['datadate2'].apply(get_quarter_num)                                                
    
    
    df_gvkey = df.join(gvkey_conm_unique, on='tic',how='left')
                                                     
    fakm_final = df_gvkey[['tic','datadate','tradedate','fyearq','fqtr','conm','datacqtr','datafqtr',
                         'X1_REVGH','X2_EPS','X3_ROA','X4_ROE','X5_PE','X6_PS','X7_NPM','X8_GPM','X9_OM','X10_PB',
                         'X11_PCFO','X12_CR','X13_EM','X14_EVCFO','X15_LTDTA','X16_WCR','X17_DE','X18_QR','X19_DSI',
                         'X20_DPO']] 
    fc = ['X1_REVGH','X2_EPS','X3_ROA','X4_ROE','X5_PE','X6_PS','X7_NPM','X8_GPM','X9_OM','X10_PB',
                         'X11_PCFO','X12_CR','X13_EM','X14_EVCFO','X15_LTDTA','X16_WCR','X17_DE','X18_QR','X19_DSI',
                         'X20_DPO']
    fakm_final.to_csv('fakm_final.csv')
    fakm_final['month'] = fakm_final['tradedate'].astype(int)//100
 
    # combine with y_return from stock_df   

    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df['datadate'] = pd.to_datetime(stocks_df['Date']).apply(lambda x: dt.datetime.strftime(x,'%Y%m%d'))
    stocks_df['datadate'] = stocks_df['datadate'].astype(int)

    nyse = mcal.get_calendar('NYSE')
    trade_dates = nyse.schedule(start_date=stocks_df['Date'].min()+dt.timedelta(days=-60), end_date=stocks_df['Date'].max()+dt.timedelta(days=210))
    trade_dates_list = list(trade_dates.index)
    
    trading_dates = []
    for i in range(len(trade_dates_list)):
        date = trade_dates_list[i].strftime('%Y%m%d')
        trading_dates.append(int(date))
    
    #trading_dates_set = set(trading_dates)
    tdf = pd.DataFrame({'trading_date':trading_dates})
    tdf['month'] = tdf['trading_date'].astype(int)//100
    
    tdfm = tdf.groupby('month')['trading_date'].min().reset_index()

    funda_df = fakm_final.join(tdfm.set_index('month'),on='month',how='left')
    funda_df = funda_df.rename(columns={'tradedate':'tradedate_fake','trading_date':'tradedate'})
    # print(funda_df['tradedate'].shape, funda_df[['tradedate']].dropna().shape)
    # print(funda_df[funda_df['tradedate'].isna()])
    #funda_df['tradedate'] = funda_df['tradedate'].astype(int)
    stocks_df['tradedate'] = stocks_df['datadate'].astype(int)
    stocks_df['tradedate_price'] = stocks_df['Adj Close']
    
    fa_quarter_prices = funda_df.join(stocks_df[['tradedate','tradedate_price','tic']].set_index(['tradedate','tic']),
                                      on=['tradedate','tic'],how='left').dropna()

    fa_quarter_prices['next_trading_month'] = fa_quarter_prices['tradedate'].apply(lambda x: dt.datetime.strptime(str(int(x)), '%Y%m%d')).apply(get_trade_month)
    fa_prices_df = fa_quarter_prices.join(tdfm.rename(columns={'month':'next_trading_month',
                                                                'trading_date':'next_trading_date'}).set_index('next_trading_month'),
                                                                  on='next_trading_month',how='left')

    fa_prices_df['next_trading_date'] = fa_prices_df['next_trading_date'].apply(lambda x: int(x) if x and ~np.isnan(x) else None)
    
    stocks_df['next_trading_date'] = stocks_df['datadate'].astype(int)
    stocks_df['next_trading_date_price'] = stocks_df['Adj Close']
    all_quartly_prices = fa_prices_df.join(stocks_df[['next_trading_date','next_trading_date_price','tic']].set_index(['next_trading_date','tic']),
                                      on=['next_trading_date','tic'],how='left')
    all_quartly_prices['y_return'] = np.log(all_quartly_prices['next_trading_date_price']/all_quartly_prices['tradedate_price'])

    
    fundamental_all = pd.merge(all_quartly_prices,sector_tic[['GICS Sector','tic']], on='tic',how='left')
    sector_list = fundamental_all['GICS Sector'].unique().tolist()
    
    sector_map = {}
    for i in range(len(sector_list)):
        sector_map[sector_list[i]] = i*5+10
        
    fundamental_all['GICS Sector'] = fundamental_all['GICS Sector'].astype(str)
    fundamental_all['gsector'] = fundamental_all['GICS Sector'].apply(lambda x: -1 if x not in sector_map else sector_map[x])
    #del sp500_fundamental_final['GICS Sector']
    print(fundamental_all.tail())
    
    fundamental_clean_table = fundamental_all
    max_trade_date = fundamental_clean_table.tradedate.max()
    keep = fundamental_clean_table[(fundamental_clean_table.tradedate == max_trade_date)]
    keep = keep.append(fundamental_clean_table[(fundamental_clean_table.tradedate < max_trade_date)].dropna(subset=['y_return']+fc))
    keep['tradedate'] = keep['tradedate'].astype(int)
    max_trade_date = keep.tradedate.max()
    final_file = final_file_prefix + f'_{max_trade_date}.csv'
    keep.to_csv(final_file,index=False)
    keep.to_csv(final_file_prefix+'.csv',index=False)
    return keep

def create_dir_if_needed(filepath):
    dir, _ = os.path.split(filepath)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    
def split_sector(final_df,final_df_top20, folder_path='Data/pre-focasting_data/'):
    create_dir_if_needed(folder_path)
    gsectors = final_df.gsector.unique()
    filtered_russell_top_20 = final_df_top20.tic.unique()
    final_df.loc[:,'filter_idx'] = final_df.loc[:,'tic'].apply(lambda x: x in filtered_russell_top_20)
    final_df = final_df.loc[final_df.loc[:,'filter_idx']==True]
    dfs_filtered = []
    for sector in gsectors:
        sector_df = final_df[final_df.gsector == sector]
        sector_df.loc[:,'risk_level'] = pd.qcut(sector_df.loc[:,'X6_PS'], 3, labels=["low", "medium", "high"])
        file, file_low, file_medium, file_high = f'sector{sector}-v2.csv', f'sector{sector}-v2-low-risk.csv', f'sector{sector}-v2-medium-risk.csv', f'sector{sector}-v2-high-risk.csv'
        sector_df.to_csv(os.path.join(folder_path, file),index=False)
        for file_to_save, risk_level in zip([file_low, file_medium, file_high], ['low', 'medium', 'high']):
            sector_df_curr_risk_level = sector_df.loc[sector_df.loc[:,'risk_level'] == risk_level, ]
            sector_df_curr_risk_level.to_csv(os.path.join(folder_path, file_to_save),index=False)
            dfs_filtered.append(sector_df_curr_risk_level)
    final_df_filtered = pd.concat(dfs_filtered).sort_values(['risk_level', 'tradedate'], ascending=[True, False])
    return final_df_filtered

