import pandas as pd

def get_normalized_return_with_sector(df,t,init_idx=0, cur_idx=-1):
    '''
        pass index of sorted df to determine the return interval
        by default, it calculates the first and last day normalized gain
    '''
    fir_price = df[df['tic'] == t]['tradedate_price'].iloc[init_idx]
    cur_price = df[df['tic'] == t]['tradedate_price'].iloc[cur_idx]
    sector = df[df['tic'] == t]['GICS Sector'].iloc[0]
    return cur_price / fir_price - 1, sector

def get_sharpe_ratio(df,t):
    '''
        pass index of sorted df to determine the return interval
        by default, it calculates the first and last day normalized gain
    '''
    returns = df[df['tic'] == t]['y_return']
    len_of_duration = returns.shape[0]
    sector = df[df['tic'] == t]['GICS Sector'].iloc[0]
    sharpe_ratio = (len_of_duration)**0.5*returns.mean()/returns.std()
    #print(sharpe_ratio, sector)
    return sharpe_ratio, sector

def get_res_df(df):
    tic, norm_gain, sector = [], [], []
    for t in df['tic'].unique(): #bottleneck, check pandas docs if possible
        tic.append(t)
        g, s = get_sharpe_ratio(df,t)
        norm_gain.append(g)
        sector.append(s)
    df_result = pd.DataFrame(list(zip(tic,norm_gain,sector)),columns=['tic','norm_gain','sector'])
    return df_result.sort_values(['sector','norm_gain'],ascending=[True, False])

def get_top20(df_result,outp_fp=None):
    df_top20_sector = []
    for s in df_result['sector'].unique():
        temp = df_result[df_result['sector'] == s]
        df_top20_sector.append(temp[0:int(len(temp)*0.2)])
    df_res = pd.concat(df_top20_sector,ignore_index=True)
    if outp_fp:
        df_res.to_csv(outp_fp)
    return df_res


def run(fp,outp_fp=None):
	'''
		fp: your input csv file path
		outp_fp: your final output csv path
	'''

	df = pd.read_csv(fp)
	return get_top20(get_res_df(df), outp_fp)

## updates: change to risk-adjusted return