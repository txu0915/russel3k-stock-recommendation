
import yahoo_fin.stock_info as si
import datetime as dt
import pandas as pd

api_key = '26f02b30c6137017402dcd75e2499440'


today_month = dt.datetime.now().strftime('%Y%m')


#sector_tic = pd.read_csv('sector_tic.csv')
quarters = ['01','04','07','10']
quarters2 = ['Q1','Q2','Q3','Q4']
