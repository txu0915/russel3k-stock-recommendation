#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from py_scripts.get_and_process_data import *
import os
from py_scripts.top20_by_sector import *


# if __name__ == '__main__':
final_file_prefix='--ress3k_fundamental_final'
final_file_folder = '.'
final_folder_file_list = os.listdir(final_file_folder)
final_file_list = []
for file in final_folder_file_list:
    if final_file_prefix in file and '.csv' in file:
        final_file_list.append(file)
pull_data = True
# if len(final_file_list) > 0:
#     max_trade_date = max(final_file_list).split('_')[-1][:-4]
#     assert(len(max_trade_date) == 8)
#     next_trade_date = get_trade_date(pd.to_datetime(max_trade_date))
#
#     today = dt.datetime.today().strftime('%Y%m%d')
#     if next_trade_date >today:
#         pull_data = False
if pull_data:
    sector_tic = pd.read_csv('sector_tic.csv')
    russell3000 = pd.read_csv('russel3k_tic.csv')
    russell3000_tics = russell3000.ticker.unique().tolist()

    gvkey_conm = pd.read_csv('conm_tic.csv')
    gvkey_conm_unique = gvkey_conm.drop_duplicates(subset=['tic']).set_index('tic')
    km_df = get_tics_key_metrics('raw_data/russel3000_km.csv', russell3000_tics)
    fa_df = get_tics_incom_states('raw_data/russel3000_fa.csv',russell3000_tics)
    stock_df = get_stocks_data('raw_data/russel3000_stock.csv',russell3000_tics,3600)

# =============================================================================
    km_df = pd.read_csv('raw_data/russel3000_km.csv')
    fa_df = pd.read_csv('raw_data/russel3000_fa.csv')
    stock_df = pd.read_csv('raw_data/russel3000_stock.csv')
# =============================================================================

    final_file_prefix = os.path.join(final_file_folder, final_file_prefix)
    final_df = processing_raw_data(km_df,fa_df,stock_df,sector_tic, gvkey_conm_unique, final_file_prefix=final_file_prefix)
    outp_fp = "top20-ressull3000-based-on-sharpe-ratio.csv"
    final_df_top20 = get_top20(get_res_df(final_df), outp_fp)

    final_df_filtered = split_sector(final_df,final_df_top20)
    print(final_df_top20.shape,final_df_top20.head(),final_df_top20.columns)

    ##
    final_df_filtered.groupby('tic', as_index=False)['y_return'].agg({'historical_std':np.std})




