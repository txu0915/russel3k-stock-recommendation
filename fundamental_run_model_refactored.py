import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import traceback
import sys
import os
sys.path.append('models')
from py_scripts.ml_model import *
import time

inputfile_fundamental = "--ress3k_fundamental_final.csv"
input_sector_dir = 'pre-focasting_data'
fundamental_total = pd.read_csv(inputfile_fundamental)
unique_sectors_files = list(map(lambda x: f'sector{x}-v2.csv', fundamental_total.gsector.unique()))

for sector_file in unique_sectors_files:
    # if sector_file != 'sector60-v2.csv':
    #     continue
    print(f'now processing sector {sector_file}')
    inputfile_sector = sector_file
    sector_data = pd.read_csv(input_sector_dir+'/'+inputfile_sector)
    unique_ticker = sorted(sector_data.tic.unique())
    unique_datetime = sorted(sector_data.tradedate.unique())
    if len(unique_datetime) < 21:
        first_trade_date_index = len(unique_datetime)-1
    else:
        first_trade_date_index = 20
    testing_windows = 4
    # get all backtesting period trade dates
    trade_date = unique_datetime[first_trade_date_index:]
    # variable column name
    label_column = 'y_return'
    date_column = 'tradedate'
    tic_column = 'tic'
    no_feature_column_names = ['gvkey', 'tic', 'datadate', 'rdq', 'tradedate', 'fyearq', 'fqtr',\
       'conm', 'datacqtr', 'datafqtr', 'gsector','y_return']
    features_column = ['X1_REVGH','X2_EPS','X3_ROA','X4_ROE','X5_PE','X6_PS','X7_NPM','X8_GPM','X9_OM','X10_PB','X11_PCFO',\
                       'X12_CR','X13_EM','X14_EVCFO','X15_LTDTA','X16_WCR','X17_DE','X18_QR','X19_DSI','X20_DPO']
    start = time.time()
    model_result = run_4model(sector_data,
                                       features_column,
                                       label_column,
                                       date_column,
                                       tic_column,
                                       unique_ticker,
                                       unique_datetime,
                                       trade_date,
                                       first_trade_date_index,
                                       testing_windows)
    end = time.time()
    print('Time Spent: ', (end - start) / 60, ' minutes')
    save_model_result(model_result, sector_file[:8])



