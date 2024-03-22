import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def clean_data(df_name):
    index_count = df_name.shape[0]
    drop_columns = set()
    for column in df_name.columns:
        missing_count = df_name[column].isna().sum()
        if missing_count/index_count > 0.5:
            drop_columns.add(column)
        else:
            df_name[column].interpolate(inplace=True)
            df_name[column].fillna(df_name[column].loc[df_name[column].first_valid_index()],inplace=True)
    return drop_columns


class DataConstruct:
    drop_columns = set()

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_space = self.cal_return()

    def read_data(self):
        file_names = os.listdir(self.data_path)
        data_space = []
        for file_name in file_names:
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_path, file_name)
                df_name = file_name[:-4]
                globals()[df_name] = pd.read_csv(file_path)
                data_space.append(df_name)
                globals()[df_name]['TRADE_DT'] = pd.to_datetime(globals()[df_name]['TRADE_DT'],format='%Y%m%d')
                globals()[df_name].set_index('TRADE_DT',inplace=True)
        return data_space

    def process_data_get_factors(self):
        data_space = self.read_data()
        for df_name in data_space:
            DataConstruct.drop_columns = DataConstruct.drop_columns | clean_data(globals()[df_name])
        for df_name in data_space:
            to_drop = DataConstruct.drop_columns & set(globals()[df_name].columns)# & set(globals()['industry'].index)
            globals()[df_name].drop(columns=to_drop,inplace=True)
        return data_space

    def cal_return(self):
        data_space = self.process_data_get_factors()
        global stock_return, stock_return_next
        stock_return = globals()['close'].pct_change()
        stock_return_next = stock_return.shift(periods=-1)
        #print(stock_return)
        data_space.extend(['stock_return','stock_return_next'])
        return data_space


def backtest_group(factor, bins_n):
    if factor not in globals().keys():
        print("因子"+factor+"不存在可用数据，无法回测")
        return 0
    globals()[factor+'_t'] = globals()[factor].T
    #print(globals()[factor+'_t'])
    globals()[factor+'_t'].drop(globals()[factor+'_t'].columns[0],axis=1,inplace=True)
    labels = ['第' + str(i + 1) + '分位' for i in range(bins_n)]
    globals()[factor + '_t_groups'] = pd.DataFrame(index=globals()[factor+'_t'].index)
    globals()[factor + '_groups_return'] = pd.DataFrame(index=globals()[factor + '_t'].columns, columns=labels)
    globals()[factor + '_groups_value'] = pd.DataFrame(index=globals()[factor + '_t'].columns, columns=labels)
    globals()[factor + '_groups_value'].iloc[0,:] = 1.0000
    pre_value = globals()[factor + '_groups_value'].iloc[0,:]
    pre_return = {key: 0 for key in labels}
    for trade_date in globals()[factor+'_t'].columns:
        globals()[factor + '_t_groups'][trade_date] = pd.qcut(globals()[factor+'_t'][trade_date],bins_n,labels=labels)
        for bin in labels:
            stock_for_bin = list(globals()[factor + '_t_groups'][globals()[factor + '_t_groups'][trade_date]==bin].index)
            globals()[factor + '_groups_return'].loc[trade_date,bin] = globals()['stock_return'].loc[trade_date,stock_for_bin].mean()
            globals()[factor + '_groups_value'].loc[trade_date,bin] = pre_value[bin] * (1 + pre_return[bin])
        pre_value = globals()[factor + '_groups_value'].loc[trade_date,:]
        pre_return = globals()[factor + '_groups_return'].loc[trade_date,:]
    #print(globals()[factor + '_groups_return'])
    #print(globals()[factor + '_groups_value'])
    globals()[factor + '_groups_return'].plot(kind='line')
    plt.title(factor + '因子分组回测（因子组合收益率）')
    plt.show()
    globals()[factor + '_groups_value'].plot(kind='line')
    plt.title(factor + '因子分组回测（因子组合净值）')
    plt.show()
    return 0


def factor_mv_ind_neutral(factor):
    factor_data = globals()[factor].drop(globals()[factor].index[0],axis=0)
    #print(factor_data)
    return_data = globals()['stock_return'].drop(globals()['stock_return'].index[0],axis=0)
    mv_data = globals()['mv'].drop(globals()['mv'].index[0],axis=0)
    #regression = pd.DataFrame(index=factor_data.columns,columns=['stock_code','return','factor','mv','industry'])
    factor_neutral = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
    global industry
    industry = pd.read_csv('industry.csv')
    industry.set_index('stock_code',inplace=True)
    industry.drop(columns=['stock_name','sw_industry_code'],inplace=True)
    industry_dummy = pd.get_dummies(globals()['industry']['sw_industry_name'],drop_first=True)
    for trade_date in factor_data.index:
        regression = pd.DataFrame(index=factor_data.columns)
        regression['return'] = return_data.loc[trade_date].tolist()
        regression['factor'] = factor_data.loc[trade_date].tolist()
        regression['mv'] = mv_data.loc[trade_date].tolist()
        #set1 = set(regression.index)
        regression = pd.concat([regression,industry_dummy],axis=1,join='inner')
        #set2 = set(regression.index)
        #print(set1.symmetric_difference(set2))
        X = regression[['factor','mv']+list(industry_dummy.columns)]
        X = sm.add_constant(X)
        y = regression['return']
        #print(regression)
        model = sm.OLS(y, X.astype(float)).fit()
        print(model.summary())
        factor_neutral.loc[trade_date] = model.resid.tolist()
    #print(factor_neutral)
    return factor_neutral


def backtest_icir(factor):
    #factor_data = factor_mv_ind_neutral(factor)
    factor_data = globals()[factor]
    factor_data.drop(factor_data.index[-1],axis=0,inplace=True)
    next_return = globals()['stock_return_next'].drop(globals()['stock_return_next'].index[-1],axis=0)
    #print(factor_data.shape,next_return.shape)
    globals()[factor+'_ic'] = pd.DataFrame(index=factor_data.index,columns=['normal_ic','rank_ic'])
    for trade_date in factor_data.index:
        globals()[factor + '_ic'].loc[trade_date,'normal_ic'], _ = pearsonr(factor_data.loc[trade_date],next_return.loc[trade_date])
        globals()[factor + '_ic'].loc[trade_date, 'rank_ic'], _ = spearmanr(factor_data.loc[trade_date],next_return.loc[trade_date])
    print(f'【{factor}因子IC分析】')
    for ttype in ['normal', 'rank']:
        print(f'{ttype}_ic均值：'+str(round(globals()[factor+'_ic'][f'{ttype}_ic'].mean(),4))+f'，{ttype}_ic标准差：'+str(round(globals()[factor+'_ic'][f'{ttype}_ic'].std(),4))+f'，{ttype}_icir：'+str(round(globals()[factor+'_ic'][f'{ttype}_ic'].mean()/globals()[factor+'_ic'][f'{ttype}_ic'].std(),4))+'，IC>0占比：'+str(round(len(globals()[factor+'_ic'][f'{ttype}_ic'][globals()[factor+'_ic'][f'{ttype}_ic']>0])/len(globals()[factor+'_ic'][f'{ttype}_ic']),4)))
    #print('normal_ic均值：'+str(globals()[factor+'_ic']['normal_ic'].mean())+'，normal_icir：'+str(globals()[factor+'_ic']['normal_ic'].std()))
    #print('rank_ic均值：'+str(globals()[factor+'_ic']['rank_ic'].mean())+'，rank_icir：'+str(globals()[factor+'_ic']['rank_ic'].std()))
    fig, ax1 = plt.subplots()
    ax1.plot(globals()[factor + '_ic'].index,globals()[factor + '_ic']['normal_ic'],color='b')
    ax1.set_xlabel('TRADE_DT')
    ax1.set_ylabel('normal_ic',color='b')
    ax1.axhline(y=0, color='b', linestyle='--')
    ax2 = ax1.twinx()
    ax2.plot(globals()[factor + '_ic'].index,globals()[factor + '_ic']['rank_ic'],color='r')
    ax2.set_ylabel('rank_ic',color='r')
    ax2.axhline(y=0, color='r', linestyle='--')
    plt.title(factor+'因子回测（ic时间序列变化图）')
    plt.show()

DataConstruct('div_datas/')
#backtest_group('div_12m',5)
#factor_mv_ind_neutral('div_12m')
backtest_icir('div_12m')

