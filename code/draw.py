#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:34:51 2018

@author: zhiqiangwu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

def read_file():
    lr = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name = 0)
    dt = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name = 1)
    rf = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name = 2)
    gbdt = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name = 3)
    lr_pre = lr['test:pred_y']
    lr_pre = lr_pre[lr_pre.notnull()]
    lr_pre = list(lr_pre)
    lr_true = lr['test:true_y']
    lr_true = lr_true[lr_true.notnull()]
    lr_true = list(lr_true)
    dt_pre = dt['test:pred_y']
    dt_pre = dt_pre[dt_pre.notnull()]
    dt_pre = list(dt_pre)
    dt_true = dt['test:true_y']
    dt_true = dt_true[dt_true.notnull()]
    dt_true = list(dt_true)
    rf_pre = rf['test:pred_y']
    rf_pre = rf_pre[rf_pre.notnull()]
    rf_pre = list(rf_pre)
    rf_true = rf['test:true_y']
    rf_true = rf_true[rf_true.notnull()]
    rf_true = list(rf_true)
    gbdt_pre = gbdt['test:pred_y']
    gbdt_pre = gbdt[gbdt.notnull()]
    gbdt_pre = list(gbdt_pre)
    gbdt_true = gbdt['test:true_y']
    gbdt_true = gbdt_true[gbdt_true.notnull()]
    gbdt_true = list(gbdt_true)
    
    draw_all(lr_pre,lr_true,dt_pre,dt_true,rf_pre,rf_true,gbdt_pre,gbdt_true)
    
def draw_all(y1, y2, y3, y4, y5, y6, y7, y8):
    plt.figure(edgecolor='k',figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    gs = GridSpec(2, 2)
    
    x = np.arange(0, len(y1))
    y = np.sqrt(x)

    ax1 = plt.subplot(gs[0, 0])#, facecolor='#383737')
    ax2 = plt.subplot(gs[0, -1])#, facecolor='#383737')
    ax3 = plt.subplot(gs[1, 0])#, facecolor='#383737')
    ax4 = plt.subplot(gs[1, -1])#, facecolor='#383737')

    #ax1.plot(x, y1, 'rs')
    #ax1.plot(x, y2, 'g^')

    #ax1.errorbar(x[0:10], y2[0:10], yerr=z[0:10])
    ax1.set_ylim(0, 1)
    ax1.plot(x[0:50], y1[0:50], 'bs', label='预测健康度')
    ax1.plot(x[0:50], y2[0:50], 'g^', label='实际健康度')
    ax1.plot(x[0:50], y1[0:50], color='b', linestyle='--')
    ax1.plot(x[0:50], y2[0:50], color='g', linestyle='--')
    
    ax1.grid(linestyle=':') #开启网格
    ax1.legend(loc='upper right')
    ax1.set_ylabel('电池健康度0～1')
    ax1.set_title('电池健康度LR预测')  
    
     
    ax2.plot(x[0:50], y3[0:50], 'bs', label='预测健康度')
    ax2.plot(x[0:50], y4[0:50], 'g^', label='实际健康度')
    ax2.plot(x[0:50], y3[0:50], color='b', linestyle='--')
    ax2.plot(x[0:50], y4[0:50], color='g', linestyle='--')
    ax2.set_ylim(0, 1)
    ax2.grid(linestyle=':') #开启网格
    ax2.legend(loc='upper right')
    ax2.set_ylabel('电池健康度0～1')
    ax2.set_title('电池健康度DT预测')  

    ax3.plot(x[0:50], y5[0:50], 'bs', label='预测健康度')
    ax3.plot(x[0:50], y6[0:50], 'g^', label='实际健康度')
    ax3.plot(x[0:50], y5[0:50], color='b', linestyle='--')
    ax3.plot(x[0:50], y6[0:50], color='g', linestyle='--')
    ax3.set_ylim(0, 1)
    ax3.grid(linestyle=':') #开启网格
    ax3.legend(loc='upper right')
    ax3.set_ylabel('电池健康度0～1')
    ax3.set_title('电池健康度随机森林预测')  
    
    ax4.plot(x[0:50], y5[0:50], 'bs', label='预测健康度')
    ax4.plot(x[0:50], y6[0:50], 'g^', label='实际健康度')
    ax4.plot(x[0:50], y5[0:50], color='b', linestyle='--')
    ax4.plot(x[0:50], y6[0:50], color='g', linestyle='--')
    ax4.set_ylim(0, 1)
    ax4.grid(linestyle=':') #开启网格
    ax4.legend(loc='upper right')
    ax4.set_ylabel('电池健康度0～1')
    ax4.set_title('电池健康度GBDT预测') 
    
    
    plt.savefig('../data/结果/result.jpg', dpi=128)
    plt.show()
    
def concat_data(data0, data1, cols):
    s1 = set(cols)
    s2 = set(data0.columns)
    s = s1.issubset(s2)
    if s:
        df1 = data0[cols[0:3]]
        df1 = df1.dropna()
        df2 = data0[cols[3:6]]
        df2 = df2.dropna()
        df2 = df2.rename(columns={'id_no.1': 'id_no',
                                  'test:pred_y': 'train:pred_y',
                                  'test:true_y': 'train:true_y'})
        df1_len = len(df1)
        df2_len = len(df2)
        df1 = df1.append(df2)
        data1['id_no'] = data1.index
        df1 = df1.merge(data1, on=['id_no'], how='left')
        df1 = df1[['bms_id', 'end_time','train:pred_y', 'train:true_y']]
        return df1, df1_len
        
    else:
        print('there are some problems of concating the data!')

def split_data(data, name):
    data_gp = data.groupby(name)
    
    print('The numbers of the data clips is: %d'%len(data_gp.groups))
    point = {}
    for i in data_gp.groups:
        df = data_gp.get_group(i)
        print('bms_id:%s, the length of data is: %d'%(i, len(df)))
        if len(df) >= 10:
            point[i] = df.values[:,1:4]
    return point

def draw_bats_soh():
    print('reading the data...')
    lr = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name=0)
    dt = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name=1)
    rf = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name=2)
    gbdt = pd.read_excel('../data/结果/result_normal.xlsx', sheet_name=3)
    bms_id = pd.read_excel('../data/处理后数据/data_x_para.xlsx', sheet_name='bms_id')
    cols = ['id_no', 'train:pred_y', 'train:true_y', 'id_no.1', 'test:pred_y', 'test:true_y']
    lr_data, lr_len = concat_data(lr, bms_id, cols)
    lr_data.to_excel('../data/结果/lr_data.xlsx')
    dt_data, dt_len = concat_data(dt, bms_id, cols)
    rf_data, rf_len = concat_data(rf, bms_id, cols)
    gbdt_data, gbdt_len = concat_data(gbdt, bms_id, cols)
    
    lr_point = split_data(lr_data, 'bms_id')
    dt_point = split_data(dt_data, 'bms_id')
    rf_point = split_data(rf_data, 'bms_id')
    gbdt_point = split_data(gbdt_data, 'bms_id')
    
    lr_pre = lr_point['010101030613001D'][:,1]
    lr_true = lr_point['010101030613001D'][:,2]
    dt_pre = dt_point['010101030613001D'][:,1]
    dt_true = dt_point['010101030613001D'][:,2]
    rf_pre = rf_point['010101030613001D'][:,1]
    rf_true = rf_point['010101030613001D'][:,2]
    gbdt_pre = gbdt_point['010101030613001D'][:,1]
    gbdt_true = gbdt_point['010101030613001D'][:,2]
    draw_all(lr_pre,lr_true,dt_pre,dt_true,rf_pre,rf_true,gbdt_pre,gbdt_true)
    
def main():
    #read_file()
    draw_bats_soh()

if __name__ == '__main__':
    main()