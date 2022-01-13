#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:10:41 2021

@author: qinfang
"""

import pandas as pd
import numpy as np
import datetime as dt
from config import config
import pandas_market_calendars as mcal

quarters = ['01','04','07','10']
quarters2 = ['Q1','Q2','Q3','Q4']
trade_month = ['01',  '04',  '07','10']
tic_list = config.tic_list
#sector_tic = config.sector_tic
sector_tic = pd.read_csv('sector_tic.csv')
   
km_df = aapl_wmt_km
fa_df = aapl_wmt_fa
stocks_df = aapl_wmt_stock[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'tic']]
gvkey_conm = pd.read_csv('sp500_gvkey_conm.csv')
gvkey_conm_unique = gvkey_conm.drop_duplicates(subset=['tic']).set_index('tic')


def get_year_quarter(x):
    return int(str(x.year) + quarters[x.quarter-1])
def get_quarters(x):
    return str(x.year) + quarters2[x.quarter-1]
def get_quarter_num(x):
    return x.quarter
def get_trade_date(x):
    return (str(x.year) if x.quarter<4 else str(x.year+1))+trade_month[(x.quarter+1)%4-1]+'01' 
def get_trade_month(x):
    return int((str(x.year) if x.quarter<4 else str(x.year+1))+trade_month[(x.quarter+1)%4-1])


        
def processing_raw_data(km_df,fa_df,stocks_df):

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
    
    
    df2 = fa_df[['date','symbol','period','revenue','eps','operatingIncomeRatio',
                            'netIncomeRatio','grossProfitRatio']]
    df2 = df2.rename(columns={'symbol':'tic','revenue':'X1_REVGH','eps':'X2_EPS','operatingIncomeRatio':'X3_ROA',
                              'netIncomeRatio':'X7_NPM','grossProfitRatio':'X8_GPM'})
    df2['datadate2'] = pd.to_datetime(df2['date'])
    df2['datadate'] = df2['datadate2'].apply(lambda x: dt.datetime.strftime(x,'%Y%m%d'))
    df2 = df2.set_index(['datadate','tic'])                                                

    df = pd.merge(df1,df2, on=(['datadate','tic']), how='left')
    
    df = df.reset_index()
    df['fyearq'] = df['datadate2'].apply(lambda x: dt.datetime.strftime(x,'%Y'))
    df['datacqtr'] = df['datadate2'].apply(get_quarters)
    df['datafqtr'] = df['fyearq']+ df['quarter']
    df['tradedate'] = df['datadate2'].apply(get_trade_date)
    df['fqtr'] = df['datadate2'].apply(get_quarter_num)                                                
    
    
    df_gvkey = df.join(gvkey_conm_unique, on='tic',how='left')
                                                     
    fakm_final = df_gvkey[['gvkey','tic','datadate','tradedate','fyearq','fqtr','conm','datacqtr','datafqtr',
                         'X1_REVGH','X2_EPS','X3_ROA','X4_ROE','X5_PE','X6_PS','X7_NPM','X8_GPM','X9_OM','X10_PB',
                         'X11_PCFO','X12_CR','X13_EM','X14_EVCFO','X15_LTDTA','X16_WCR','X17_DE','X18_QR','X19_DSI',
                         'X20_DPO']] 
    
    fakm_final.to_csv('fakm_final.csv')
    fakm_final['month'] = fakm_final['tradedate'].astype(int)//100
 
    # combine with y_return from stock_df   

    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df['datadate'] = pd.to_datetime(stocks_df['Date']).apply(lambda x: dt.datetime.strftime(x,'%Y%m%d'))
    stocks_df['datadate'] = stocks_df['datadate'].astype(int)

    nyse = mcal.get_calendar('NYSE')
    trade_dates = nyse.schedule(start_date=stocks_df['Date'].min()+dt.timedelta(days=-60), end_date=stocks_df['Date'].max()+dt.timedelta(days=120))
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

    all_quartly_prices.to_csv('fundamental_final_{today}.csv',index=False)
    
    fundamental_all = pd.merge(all_quartly_prices,sector_tic[['GICS Sector','tic']], on='tic',how='left')
    sector_list = fundamental_all['GICS Sector'].unique().tolist()
    
    sector_map = {}
    for i in range(len(sector_list)):
        sector_map[sector_list[i]] = i*5+10
        
    fundamental_all['GICS Sector'] = fundamental_all['GICS Sector'].astype(str)
    fundamental_all['gsector'] = fundamental_all['GICS Sector'].apply(lambda x: -1 if x not in sector_map else sector_map[x])
    #del sp500_fundamental_final['GICS Sector']
    fundamental_clean_table = fundamental_all.dropna()
    fundamental_clean_table.to_csv('fundamental_clean_table.csv',index=False)
    print(fundamental_clean_table.head())
    return fundamental_clean_table    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
