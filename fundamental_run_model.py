import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import traceback
import sys
sys.path.append('models')
import ml_model




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    #sector name
    parser.add_argument('-sector_name','--sector_name_input', type=str,  required=True,help='sector name: i.e. sector10')

    # file name
    parser.add_argument('-fundamental','--fundamental_input', type=str,  required=True,help='inputfile name for fundamental table')
    parser.add_argument('-sector','--sector_input', type=str,  required=True,help='inputfile name for individual sector')
    
    # rolling window variables
    parser.add_argument("-first_trade_index", default=20, type=int)
    parser.add_argument("-testing_window", default=4, type=int)
    
    # column name
    parser.add_argument("-label_column", default='y_return', type=str)
    parser.add_argument("-date_column", default='tradedate', type=str)
    parser.add_argument("-tic_column", default='tic', type=str)
    parser.add_argument("-no_feature_column_names", default = ['gvkey', 'tic', 'datadate', 'rdq', 'tradedate', 'fyearq', 'fqtr',
       'conm', 'datacqtr', 'datafqtr', 'gsector','y_return'], type=list,help='column names that are not fundamental features')
    parser.add_argument("-features_column", default = ['X1_REVGH','X2_EPS','X3_ROA','X4_ROE','X5_PE','X6_PS','X7_NPM',
                                                     'X8_GPM','X9_OM','X10_PB','X11_PCFO','X12_CR','X13_EM','X14_EVCFO',
                                                     'X15_LTDTA','X16_WCR','X17_DE','X18_QR','X19_DSI','X20_DPO'], 
                                                    type=list,help='column names that are fundamental features')
    
    parser.add_argument("-run_last", default=0, type=int) 
    args = parser.parse_args()
    #load fundamental table
    inputfile_fundamental = args.fundamental_input
    
    fundamental_total=pd.read_csv(inputfile_fundamental)
    #fundamental_total=fundamental_total[fundamental_total['tradedate']]
    #get all unique quarterly date
    unique_datetime = sorted(fundamental_total.tradedate.unique())
    print(len(unique_datetime))
    # load sector data
    inputfile_sector = args.sector_input
    sector_data=pd.read_csv(inputfile_sector)

    #get sector unique ticker
    unique_ticker=sorted(sector_data.tic.unique())
    run_last = args.run_last
    if run_last == 1:
        first_trade_date_index = len(unique_datetime)-1
    else:
        first_trade_date_index=args.first_trade_index

    #testing window
    testing_windows = args.testing_window

    #get all backtesting period trade dates
    trade_date=unique_datetime[first_trade_date_index:]
    
    #variable column name
    label_column = args.label_column
    date_column = args.date_column
    tic_column = args.tic_column
    
    no_feature_column_names = args.no_feature_column_names
    features_column = args.features_column
    #features_column = [x for x in sector_data.columns.values if x not in no_feature_column_names]
    print(features_column)
 
    sector_name = args.sector_name_input
    
    try:
        start = time.time()
        model_result=ml_model.run_4model(sector_data,
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
        print('Time Spent: ',(end-start)/60,' minutes')
        ml_model.save_model_result(model_result,sector_name)

    except Exception as e:
        print(e)

    

# python3 fundamental_run_model.py -sector_name sector10 -fundamental Data/fundamental_final_table.xlsx -sector Data/1-focasting_data/sector10_clean.xlsx 
 # python fundamental_run_model.py   -sector_name sector10 -fundamental __ress3k_fundamental_final.csv -sector Data/1-focasting_data/sector15.csv -first_trade_index 83