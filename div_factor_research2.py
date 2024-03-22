import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
from scipy.stats import pearsonr, spearmanr, zscore
import statsmodels.api as sm
from itertools import combinations, permutations
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
    drop_columns = {'000046.SZ', '002002.SZ', '000666.SZ', '002013.SZ'}

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_space = self.cal_return()
        #self.write_code_list()

    def read_data(self):
        file_names = os.listdir(self.data_path)
        data_space = []
        for file_name in file_names:
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_path, file_name)
                df_name = file_name[:-4]
                globals()[df_name] = pd.read_csv(file_path)
                if df_name=='RESSET_FINRATIO':
                    globals()[df_name]['TRADE_DT'] = pd.to_datetime(globals()[df_name]['TRADE_DT'], format='%Y/%m/%d')
                else:
                    globals()[df_name]['TRADE_DT'] = pd.to_datetime(globals()[df_name]['TRADE_DT'],format='%Y%m%d')
                globals()[df_name]['TRADE_DT'] = globals()[df_name]['TRADE_DT'].dt.to_period('M')
                if df_name in ['RevenueTechnicalFactor1', 'RESSET_FINRATIO']:
                    for column in globals()[df_name].columns[2:]:
                        data_space.append(column)
                        globals()[column] = globals()[df_name].pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values=column)
                else:
                    data_space.append(df_name)
                    globals()[df_name].set_index('TRADE_DT',inplace=True)
        return data_space

    def process_data_get_factors(self):
        data_space = self.read_data()
        #stocks = globals()['CurRt'].columns.tolist()
        for df_name in data_space:
            DataConstruct.drop_columns = DataConstruct.drop_columns | clean_data(globals()[df_name])
        for df_name in data_space:
            to_drop = DataConstruct.drop_columns & set(globals()[df_name].columns)# & set(globals()['industry'].index)
            globals()[df_name].drop(columns=to_drop,inplace=True)
            #globals()[df_name] = globals()[df_name][stocks]
        globals()['pe_ttm'] = globals()['pe_ttm'].apply(lambda x: 1.0000/x)
        globals()['mv'] = globals()['mv'].apply(np.log)
        return data_space

    def cal_return(self):
        data_space = self.process_data_get_factors()
        global stock_return, stock_return_next
        stock_return = globals()['close'].pct_change()
        stock_return_next = stock_return.shift(periods=-1)
        data_space.extend(['stock_return','stock_return_next'])
        return data_space
    
    '''def cal_volatility(self):
        data_space = self.cal_return()
        global stock_volatility
        stock_volatility = stock_return.rolling(window=12,min_periods=1).std()
        stock_volatility.interpolate(inplace=True)
        stock_volatility.fillna(0, inplace=True)
        print(stock_volatility)
        data_space.extend(['stock_volatility'])
        return data_space'''

    def write_code_list(self):
        df = pd.DataFrame(globals()['pe_ttm'].columns.tolist())
        df.to_csv('code_list1.csv', index=False, header=False)
        return 0

    def print_all(self):
        for df_name in self.data_space:
            print('{%s} (%d * %d):' % (df_name,globals()[df_name].shape[0],globals()[df_name].shape[1]))
            print(globals()[df_name].iloc[:5,:5])
            print('\n')
        #print(list(set(globals()['pe_ttm'].columns.tolist())-set(globals()['CurRt'].columns.tolist())))


def backtest_group(factor, bins_n):
    if factor not in globals().keys():
        print("因子"+factor+"不存在可用数据，无法回测")
        return 0
    globals()[factor+'_t'] = globals()[factor].T
    #print(globals()[factor+'_t'])
    #globals()[factor+'_t'].drop(globals()[factor+'_t'].columns[0],axis=1,inplace=True)
    labels = ['第' + str(i + 1) + '分位' for i in range(bins_n)]
    globals()[factor + '_t_groups'] = pd.DataFrame(index=globals()[factor+'_t'].index)
    globals()[factor + '_groups_return'] = pd.DataFrame(index=globals()[factor + '_t'].columns, columns=labels+['基准组合'])
    globals()[factor + '_groups_value'] = pd.DataFrame(index=globals()[factor + '_t'].columns, columns=labels+['基准组合'])
    globals()[factor + '_groups_value'].iloc[0] = 1.0000
    pre_value = globals()[factor + '_groups_value'].iloc[0]
    #pre_return = {key: 0 for key in labels+['基准组合']}
    i=0
    for trade_date in globals()[factor+'_t'].columns[1:]:
        globals()[factor + '_t_groups'][trade_date] = pd.qcut(globals()[factor+'_t'][globals()[factor+'_t'].columns[i]],bins_n,labels=labels)
        for bin in labels:
            stock_for_bin = list(globals()[factor + '_t_groups'][globals()[factor + '_t_groups'][trade_date]==bin].index)
            globals()[factor + '_groups_return'].loc[trade_date,bin] = globals()['stock_return'].loc[trade_date,stock_for_bin].mean()
            globals()[factor + '_groups_value'].loc[trade_date,bin] = pre_value[bin] * (1 + globals()[factor + '_groups_return'].loc[trade_date,bin])
        globals()[factor + '_groups_return'].loc[trade_date,'基准组合'] = globals()['stock_return'].loc[trade_date].mean()
        globals()[factor + '_groups_value'].loc[trade_date,'基准组合'] = pre_value['基准组合'] * (1 + globals()[factor + '_groups_return'].loc[trade_date,'基准组合'])
        pre_value = globals()[factor + '_groups_value'].loc[trade_date]
        i+=1
        #pre_return = globals()[factor + '_groups_return'].loc[trade_date,:]
    #print(globals()[factor + '_groups_return'])
    #print(globals()[factor + '_groups_value'])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5), gridspec_kw={'left':0.06, 'bottom':0.14, 'right':0.965})
    s = 0
    for tp in ['return','value']:
        globals()[factor + f'_groups_{tp}'].plot(ax=axes[s],kind='line')
        axes[s].set_title(factor + f'因子分组回测（因子组合{tp}）')
        s += 1
    fig.suptitle(factor + '因子分组测试')
    plt.show()
    return 0


def factor_zscore_neutral(factor):
    factor_data = globals()[factor].apply(zscore, axis=1)
    #print(factor_data)
    #return_data = globals()['stock_return'].drop(globals()['stock_return'].index[0],axis=0)
    mv_data = globals()['mv'].copy()
    #if factor in ['CurRt', 'CurTotLia']:
    if globals()[factor].shape[1] != mv_data.shape[1]:
        mv_data = mv_data[globals()[factor].columns]
    #regression = pd.DataFrame(index=factor_data.columns,columns=['stock_code','return','factor','mv','industry'])
    factor_neutral = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
    global industry
    industry = pd.read_csv('industry.csv')
    industry.set_index('stock_code',inplace=True)
    industry.drop(columns=['stock_name','sw_industry_code'],inplace=True)
    industry_dummy = pd.get_dummies(globals()['industry']['sw_industry_name'],drop_first=True)
    for trade_date in factor_data.index:
        regression = pd.DataFrame(index=factor_data.columns)
        #regression['return'] = return_data.loc[trade_date].tolist()
        regression['factor'] = factor_data.loc[trade_date].tolist()
        regression['mv'] = mv_data.loc[trade_date].tolist()
        #set1 = set(regression.index)
        regression = pd.concat([regression,industry_dummy],axis=1,join='inner')
        #set2 = set(regression.index)
        #print(set1.symmetric_difference(set2))
        if factor == 'mv':
            X = regression[list(industry_dummy.columns)]
        else:
            X = regression[['mv']+list(industry_dummy.columns)]
        X = sm.add_constant(X)
        y = regression['factor']
        #print(regression)
        model = sm.OLS(y, X.astype(float)).fit()
        #print(model.summary())
        factor_neutral.loc[trade_date] = model.resid.tolist()
    #print(factor_neutral)
    return factor_neutral


def backtest_icir(factor):
    factor_data = factor_zscore_neutral(factor)
    factor_data.drop(factor_data.index[-1],axis=0,inplace=True)
    next_return = globals()['stock_return_next'].drop(globals()['stock_return_next'].index[-1],axis=0)
    #print(factor_data.shape,next_return.shape)
    globals()[factor+'_ic'] = pd.DataFrame(index=factor_data.index,columns=['normal_ic','rank_ic'])
    for trade_date in factor_data.index:
        globals()[factor + '_ic'].loc[trade_date,'normal_ic'], _ = pearsonr(factor_data.loc[trade_date],next_return.loc[trade_date,factor_data.columns])
        globals()[factor + '_ic'].loc[trade_date, 'rank_ic'], _ = spearmanr(factor_data.loc[trade_date],next_return.loc[trade_date,factor_data.columns])
    print(f'【{factor}因子IC分析】')
    for ttype in ['normal', 'rank']:
        print(f'{ttype}_ic均值：'+str(round(globals()[factor+'_ic'][f'{ttype}_ic'].mean(),4))+f'，{ttype}_ic标准差：'+str(round(globals()[factor+'_ic'][f'{ttype}_ic'].std(),4))+f'，{ttype}_icir：'+str(round(globals()[factor+'_ic'][f'{ttype}_ic'].mean()/globals()[factor+'_ic'][f'{ttype}_ic'].std(),4))+'，IC>0占比：'+str(round(len(globals()[factor+'_ic'][f'{ttype}_ic'][globals()[factor+'_ic'][f'{ttype}_ic']>0])/len(globals()[factor+'_ic'][f'{ttype}_ic']),4)))
    #print('rank_ic均值：'+str(round(globals()[factor+'_ic']['rank_ic'].mean(),4))+'，rank_ic标准差：'+str(round(globals()[factor+'_ic']['rank_ic'].std(),4))+'，rank_icir：'+str(round(globals()[factor+'_ic']['rank_ic'].mean()/globals()[factor+'_ic']['rank_ic'].std(),4))+'，IC>0占比：'+str(len(globals()[factor+'_ic']['rank_ic'][globals()[factor+'_ic']['rank_ic']>0])))
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(globals()[factor + '_ic'].index.to_timestamp(),globals()[factor + '_ic']['normal_ic'],color='b')
    ax1.set_xlabel('TRADE_DT')
    ax1.set_ylabel('normal_ic',color='b')
    ax1.axhline(y=0, color='b', linestyle='--')
    ax2 = ax1.twinx()
    ax2.plot(globals()[factor + '_ic'].index.to_timestamp(),globals()[factor + '_ic']['rank_ic'],color='r')
    ax2.set_ylabel('rank_ic',color='r')
    ax2.axhline(y=0, color='r', linestyle='--')
    plt.title(factor+'因子回测（ic时间序列变化图）')
    plt.show()
    return 0


def single_factor_test(factor_list):
    for factor in factor_list:
        backtest_group(factor, 5)
        backtest_icir(factor)


def dual_factor_cross_test(factor1, factor2, bins_n):
    code_list = list(set(globals()[factor1 + '_t'].index.tolist()) & set(globals()[factor2 + '_t'].index.tolist()))
    if factor1 in ['ROETTM', 'CurRt', 'NPPCCutGrRt', 'TotAstTRtTTM', 'NetOCFTOReve', 'CurTotLia', 'OPITPrf']:
        globals()[factor2 + '_t'] = globals()[factor2].T
        globals()[factor2 + '_t'] = globals()[factor2 + '_t'][globals()[factor1 + '_t'].columns]
    if factor2 in ['ROETTM', 'CurRt', 'NPPCCutGrRt', 'TotAstTRtTTM', 'NetOCFTOReve', 'CurTotLia', 'OPITPrf']:
        globals()[factor1 + '_t'] = globals()[factor1].T
        globals()[factor1 + '_t'] = globals()[factor1 + '_t'][globals()[factor2 + '_t'].columns]
    globals()[factor1 + '_t'] = globals()[factor1 + '_t'].loc[code_list]
    globals()[factor1 + '_t'] = globals()[factor1 + '_t'].loc[code_list]
    if factor1 + '_t' not in globals():
        backtest_group(factor1, bins_n)
    if factor2 + '_t' not in globals():
        backtest_group(factor2, bins_n)
    #labels = ['第' + str(i + 1) + '分位' for i in range(bins_n)]
    labels1 = [factor1 + str(i + 1) for i in range(bins_n)]
    labels2 = [factor2 + str(i + 1) for i in range(bins_n)]
    sample_df = pd.DataFrame(index=globals()[factor1 + '_t'].columns, columns=labels2+['基准组合'])
    sample_df1 = sample_df.copy()
    sample_df1.iloc[0] = 1.0000
    globals()['res_return_'+factor1+'_'+factor2] = {x: sample_df.copy() for x in labels1}
    globals()['res_value_'+factor1+'_'+factor2] = {x: sample_df1.copy() for x in labels1}
    #print(globals()['res_return_'+factor1+'_'+factor2])
    #print(globals()['res_value_'+factor1+'_'+factor2])
    #labels_dual = [f'{factor1}' + str(i+1) + f'_{factor2}' + str(j+1) for i in range(bins_n) for j in range(bins_n)]
    #globals()[f'{factor1}&{factor2}' + '_groups_return'] = pd.DataFrame(index=globals()[factor1 + '_t'].columns,columns=labels_dual + ['基准组合'])
    #globals()[f'{factor1}&{factor2}' + '_groups_value'] = pd.DataFrame(index=globals()[factor1 + '_t'].columns,columns=labels_dual + ['基准组合'])
    f1_group = pd.DataFrame(index=globals()[factor1 + '_t'].index)
    i = 0
    #pre_value = {bin: globals()['res_value_'+factor1+'_'+factor2][bin].iloc[0] for bin in list(globals()['res_return_'+factor1+'_'+factor2].keys())}
    for trade_date in globals()[factor1 + '_t'].columns[1:]:
        f1_group[trade_date] = pd.qcut(globals()[factor1 + '_t'][globals()[factor1 + '_t'].columns[i]], bins_n, labels=labels1)
        #stock_for_f1 = {bin: None for bin in labels1}
        for bin in labels1:
            stock_for_bin_f1 = f1_group[f1_group[trade_date]==bin].index.tolist()
            f2_group = pd.DataFrame(index=stock_for_bin_f1)
            #to_cut = globals()[factor2 + '_t'].loc[stock_for_bin_f1]
            f2_group[trade_date] = pd.qcut(globals()[factor2 + '_t'].loc[stock_for_bin_f1,globals()[factor1 + '_t'].columns[i]], bins_n, labels=labels2)
            for bin1 in labels2:
                stock_for_bin_f2 = f2_group[f2_group[trade_date]==bin1].index.tolist()
                globals()['res_return_'+factor1+'_'+factor2][bin].loc[trade_date, bin1] = globals()['stock_return'].loc[trade_date, stock_for_bin_f2].mean()
                globals()['res_value_'+factor1+'_'+factor2][bin].loc[trade_date, bin1] = globals()['res_value_'+factor1+'_'+factor2][bin].loc[globals()[factor1 + '_t'].columns[i], bin1] * (1 + globals()['res_return_'+factor1+'_'+factor2][bin].loc[trade_date, bin1])
            globals()['res_return_'+factor1+'_'+factor2][bin].loc[trade_date, '基准组合'] = globals()['stock_return'].loc[trade_date, stock_for_bin_f1].mean()
            globals()['res_value_'+factor1+'_'+factor2][bin].loc[trade_date, '基准组合'] = globals()['res_value_'+factor1+'_'+factor2][bin].loc[globals()[factor1 + '_t'].columns[i], '基准组合'] * (1 + globals()['res_return_'+factor1+'_'+factor2][bin].loc[trade_date, '基准组合'])
        #pre_value = {bin: globals()['res_value_'+factor1+'_'+factor2][bin].loc[trade_date] for bin in list(globals()['res_return_'+factor1+'_'+factor2].keys())}
        i += 1
    #print(f1_group)
    #print(factor1 + '因子&' + factor2 + '因子交叉分组测试')
    '''for bin in labels1:
        for tp in ['return', 'value']:
            #print(bin + ':' + tp)
            #print(globals()[f'res_{tp}_'+factor1+'_'+factor2][bin])
            globals()[f'res_{tp}_'+factor1+'_'+factor2][bin].plot(kind='line')
            plt.title(factor1 + '&' + factor2 + '因子交叉分组回测（' + bin + f'因子组合{tp}时间序列变化图）')
            plt.show()'''
    fig, axes = plt.subplots(nrows=2, ncols=bins_n, figsize=(10,6), gridspec_kw={'left':0.036, 'bottom': 0.105, 'right':0.983, 'top': 0.897, 'wspace':0.145, 'hspace': 0.455})
    s = 0
    for tp in ['return', 'value']:
        j = 0
        for bin in labels1:
            globals()[f'res_{tp}_' + factor1 + '_' + factor2][bin].plot(ax=axes[s,j],kind='line')
            #axes[s,j].legend()
            axes[s,j].set_title(bin + f'因子组合{tp}时间序列变化图')
            j += 1
        s += 1
    fig.suptitle(factor1 + '因子&' + factor2 + '因子交叉分组测试')
    plt.show()
    return 0

def get_corr(factor1, factor2):
    corr = []
    date_list = list(set(globals()[factor1 + '_t'].columns.tolist()) & set(globals()[factor2 + '_t'].columns.tolist()))
    code_list = list(set(globals()[factor1 + '_t'].index.tolist()) & set(globals()[factor2 + '_t'].index.tolist()))
    for trade_date in date_list:
        corr.append(pearsonr(globals()[factor1 + '_t'].loc[code_list,trade_date], globals()[factor2 + '_t'].loc[code_list,trade_date]))
    globals()['corr_matrix'].loc[factor1,factor2] = np.mean(corr)
    globals()['corr_matrix'].loc[factor2, factor1] = np.mean(corr)
    print(factor1 + '因子&' + factor2 + '因子相关系数: ' + str(round(np.mean(corr), 4)))
    return np.mean(corr)

def test_n_corsstest(factor_to_test, bins_n_single, bins_n_dual):
    for factor in factor_to_test:
        backtest_group(factor, bins_n_single)
        backtest_icir(factor)
    for group in combinations(factor_to_test, 2):
        get_corr(list(group)[0], list(group)[1])
        dual_factor_cross_test(list(group)[0], list(group)[1], bins_n_dual)
        dual_factor_cross_test(list(group)[1], list(group)[0], bins_n_dual)
    globals()['corr_matrix'] = globals()['corr_matrix'].astype(float)
    #print(globals()['corr_matrix'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(globals()['corr_matrix'], annot=True, cmap='YlGnBu', fmt=".2f", square=True)
    plt.title('因子相关系数矩阵')
    plt.xlabel('Factors')
    plt.ylabel('Factors')
    plt.show()


factor_to_test = ['div_12m','pe_ttm','mv','S_RISK_VARIANCE60','ROETTM','NPPCCutGrRt','OPITPrf','NetOCFTOReve','CurTotLia','CurRt','TotAstTRtTTM']
globals()['corr_matrix'] = pd.DataFrame(index=factor_to_test, columns=factor_to_test)
dt = DataConstruct('div_datas/')
dt.print_all()
test_n_corsstest(factor_to_test, 5, 3)
#read_mixed_panel(['RevenueTechnicalFactor1'])
