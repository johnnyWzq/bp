#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:13:25 2018

@author: zhiqiangwu
"""
import pandas as pd
import model as md

def copy_index(data):
    """
    将代表数据序号的index拷贝至一列，为后续保存结果时将index删除做备份
    """
    id_no = list(data.index)
    cols = list(data.columns)
    cols.insert(0, 'id_no')
    data['id_no'] = id_no
    data = data.reset_index(drop=True)
    data = data[cols]
    
    return data

def record_result(output_dir, res, method='normal'):
    d = {'lr':'线性回归(LR)', 'dt':'决策树回归', 'rf':'随机森林', 'gbdt':'GBDT',
        'eva':'评估结果'}
    writer = pd.ExcelWriter(output_dir + 'result_%s.xlsx'%method)
    eva = pd.DataFrame()
    for s in res:
        copy_index(res[s]['train']).to_excel(writer, d[s], index=False)
        copy_index(res[s]['test']).to_excel(writer, d[s], startcol=3, index=False)
        eva = eva.append(res[s]['eva'])
    eva = eva[['type', 'EVS', 'MAE', 'MSE', 'R2']]
    eva.to_excel(writer, d['eva'])
    writer.save()
    return eva

def main(method='normal'):
    data_dir = '../data/2017大运中心数据/'
    p_data_dir = '../data/处理后数据/'
    result_dir = '../data/结果/'
    para_list = ((0, 0),
           (0, 20), (0, 40), (0, 60), (0, 80), (0, 100), (0, 120),
           (20, 40), (20, 60), (20, 80), (20, 100), (20, 120),
           (40, 60), (40, 80), (40, 100), (40, 120),
           (60, 80), (60, 100), (60, 120),
           (80, 100), (80, 120),
           (100, 120)
           )
    eva = []
    i = 0
    if method == 'f_regression' or method == 'mutual_info_regression' or method == 'pca':
        res = md.build_model(p_data_dir, feature_method=method, feature_num=40)
        eva = record_result(result_dir, res, method=method)
    elif method == 'reduce':
        writer = pd.ExcelWriter(result_dir + 'eva_compare.xlsx')
        for xy in para_list:
            print(xy)
            res = md.build_model(p_data_dir, index1=xy[0], index2=xy[1])
            eva = record_result(result_dir, res)
            eva.to_excel(writer, index=True, index_label='p%s:%s-%s'%(str(i),str(xy[0]),str(xy[1])), 
                        startrow = i*9)
            i += 1
        writer.save()
    else:
        res = md.build_model(p_data_dir)
        eva = record_result(result_dir, res, method=method)
if __name__ == '__main__':
    main()