#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 17:32:27 2018

@author: zhiqiangwu
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
import numpy as np

import data_preprocess as dp
import utils as ut

def cal_score(data):
    """
    计算健康度
    """
    V_WEIGHT = 0.25
    I_WEIGHT = 0.25
    T_WEIGHT = 0.25
    SV_WEIGHT = 0.25
    #filter invalid data
    data = data[~data['score_ca_kwh_mean'].isin([-np.inf, np.inf, np.nan])]
    
    #calculate score
    ##sore_capacity
    tmp = data['score_ca_kwh_mean']
    lower_bound = max(tmp.quantile(0.01), 0)
    upper_bound = min(tmp.quantile(0.99), 500)
    tmp[tmp < lower_bound] = lower_bound
    tmp[tmp > upper_bound] = upper_bound
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    data['score_ca'] = tmp
    ##score_health
    tmp = data['score_ca_health_mean']
    tmp[tmp < 0] = 0
    tmp[tmp > 100] = 100
    tmp = tmp / 100.0
    data['score_health'] = tmp
    ##score_abnormal
    data['isabnormal_v'] = data['bp_v_max'] > data['bi_limitv_mean']
    data['isabnormal_i'] = data['bp_i_max'] > data['bi_limiti_mean']
    data['isabnormal_t'] = data['max_st_max'] > data['bi_limitt_mean']
    data['isabnormal_sv'] = data['max_sv_max'] > data['bi_limitsv_mean']
    
    tmp = (~data['isabnormal_v'] * V_WEIGHT) + (~data['isabnormal_i'] * I_WEIGHT) + \
            (~data['isabnormal_t'] * T_WEIGHT) + (~data['isabnormal_sv'] * SV_WEIGHT)
    data['score_abnormal'] = tmp
    ##score_stable
    stable_col_names = ['std_sv_mean', 'std_sv_max', 'std_st_mean', 'std_st_max']
    for i in ['bp_v', 'bp_i', 'cp_p', 'max_sv', 'mean_sv', 'max_st', 'mean_st']:
        for j in ['_std', '_diff_mean', '_diff_max', '_diffrate_mean', '_diffrate_max']:
            stable_col_names.append(i + j)
    tmp = data[stable_col_names].fillna(tmp.mean())
    for i in tmp.columns:
        lower_bound = max(tmp[i].quantile(0.01), 0)
        upper_bound = tmp[i].quantile(0.99)
        tmp[i][tmp[i] < lower_bound] = lower_bound
        tmp[i][tmp[i] > upper_bound] = upper_bound
        tmp[i] = (tmp[i] - tmp[i].min()) / (tmp[i].max() - tmp[i].min())
    tmp = tmp.sum(axis=1) / tmp.shape[1]
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    data['score_stable'] = 1 - tmp
    ##calculate score
    score_ratio = {'capacity': 0.5, 'stable': 0.3, 'health': 0.1, 'abnormal': 0.1}
    tmp = data['score_ca'] * score_ratio['capacity'] + \
          data['score_stable'] * score_ratio['stable'] + \
          data['score_health'] * score_ratio['health'] + \
          data['score_abnormal'] * score_ratio['abnormal']
    data['score'] = tmp
    
    return data

def cal_feature(data):
    """
    """
    col_names = []
    for i in data.columns:
        for j in ['cp_', 'bp_', '_st_', '_sv_']:
            if j in i:
                col_names.append(i)
                break
    for i in col_names:
        tmp = data[i]
        data['feature_' + i] = tmp
    return data

def select_feature(data_x, data_y, method, feature_num):
    features_chosen = data_x.columns
    featrue_num = min(len(features_chosen), feature_num)
    
    #根据特征工程的方法选择特征参数数量
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression
    
    if method == 'f_regression' or method == 'mutual_info_regression':
        if method == 'f_regression':
            select_model = SelectKBest(f_regression, k=feature_num)
        else:
            select_model = SelectKBest(mutual_info_regression, k=feature_num)
        select_model.fit(data_x.values, data_y.values.ravel())
        feature_mask = select_model.get_support(indices=True)
        feature_chosen = data_x.columns[feature_mask]
        print('feature_chosen: ', feature_chosen)
        data_x = data_x[feature_chosen]
    elif method == 'PCA':
        pca_model = PCA(n_components=feature_num)
        data_x_pc = pca_model.fit(data_x.values).transform(data_x.values)
        data_x = pd.DataFrame(data=data_x_pc,
                      columns=['PCA_' + str(i) for i in range(feature_num)])
    else:
        raise Exception('In select_feature(): invalid parameter method.')
    
    return data_x
    
def build_model(data_dir, data=None, split_mode='test', **kwg):
    """
    标签工程，特征工程，建立模型
    """
    if data is None:
        data = pd.read_csv(open(data_dir + 'data.csv'), encoding='gb18030')
    
    data = cal_score(data)
    data = cal_feature(data)
    data_x = data[[i for i in data.columns if 'feature_' in i]]
    data_x = data_x.replace(np.inf, np.nan)
    data_x = data_x.replace(-np.inf, np.nan)
    data_x = data_x.fillna(data_x.mean())

    data_y = data['score']

    file_name = data_dir + 'data_x0.csv'
    cols_ori = pd.DataFrame(data_x.columns)
    
    if 'feature_method' in kwg and 'feature_num' in kwg:
        # standardize
        data_x = pd.DataFrame(data=preprocessing.scale(data_x.values, axis=0), columns=data_x.columns)

        # select features
        # feature_num: integer that >=0
        # method: ['f_regression', 'mutual_info_regression', 'pca']
        feature_method = kwg['feature_method']
        feature_nums = kwg['feature_num']
        data_x = select_feature(data_x, data_y, method=feature_method, feature_num=feature_nums)
        file_name = data_dir + 'data_x_%s.csv'%feature_method
    elif 'index1' in kwg and 'index2' in kwg:
        index1 = kwg['index1']
        index2 = kwg['index2']
        #drop the selected feature datas
        col_list = list(data_x.columns)[index1:index2]
        data_x = data_x.drop(columns=col_list)
        file_name = data_dir + 'data_x_%s_%s'%(index1, index2)
    #save the feature datas
    writer = pd.ExcelWriter(data_dir + 'data_x_para.xlsx')
    cols_ori.to_excel(writer, '原参数列表')
    data_x.to_csv(file_name, encoding='gb18030', index=False)
    cols_reg = pd.DataFrame(data_x.columns)
    cols_reg.to_excel(writer, '调整参数列表')
    bms_id = data[['bms_id', 'end_time']]
    bms_id.to_excel(writer, 'bms_id')
    writer.save()
    
    #start building model
    np_x = np.nan_to_num(data_x.values)
    np_y = np.nan_to_num(data_y.values)
    print('train_set.shape=%s, test_set.shape=%s' %(np_x.shape, np_y.shape))
    
    pkl_dir = '../data/pkl/'
    res = {}
    if split_mode == 'test':
        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2,
                                                          shuffle=False)
        model = LinearRegression()
        res['lr'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
        model = DecisionTreeRegressor()
        res['dt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=3)
        model = RandomForestRegressor()
        res['rf'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=3)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
    elif split_mode == 'cv':
        model = LinearRegression()
        res['lr'] = ut.cv_model(model, np_x, np_y)
        model = DecisionTreeRegressor()
        res['dt'] = ut.cv_model(model, np_x, np_y)
        model = RandomForestRegressor()
        res['rf'] = ut.cv_model(model, np_x, np_y)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.cv_model(model, np_x, np_y)
    else:
        print('parameter mode=%s unresolved' % (mode))
        
    return res

def main():
    data_dir = '../data/2017大运中心数据/'
    p_data_dir = '../data/处理后数据/'
    result_dir = '../data/结果/'
    """
    data_dict_ori = dp.read_data(data_dir)
    data_dict = dp.clean_data(data_dict_ori)
    #dpp.cut_data(data_dict, 100000, p_data_dir)
    
    #data_dict = dp.read_cut_data(p_data_dir)
    processed_data = dp.process_data(data_dict, p_data_dir)    
    data = dpp.split_data(p_data_dir, processed_data)
    """
    res = build_model(p_data_dir)#, feature_method='f_regression', feature_num=40)
    d = {'lr':'线性回归(LR)', 'dt':'决策树回归', 'rf':'随机森林', 'gbdt':'GBDT',
        'eva':'评估结果'}
    writer = pd.ExcelWriter(result_dir + 'result.xlsx')
    eva = pd.DataFrame()
    for s in res:
        res[s]['train'].to_excel(writer, d[s])
        res[s]['test'].to_excel(writer, d[s], startcol=3)
        eva = eva.append(res[s]['eva'])
    eva = eva[['type', 'EVS', 'MAE', 'MSE', 'R2']]
    eva.to_excel(writer, d['eva'])
    
    
if __name__ == '__main__':
    main()