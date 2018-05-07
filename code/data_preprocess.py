#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:35:40 2018

@author: zhiqiangwu
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser

def read_csv_files(data_dir, keyword):
    temp = []
    
    for file in os.listdir(data_dir):#获取文件夹内所有文件名
        if keyword in file:
            print(file)
            open(data_dir + file)
            temp.append(pd.read_csv(data_dir+file, error_bad_lines=False,
                                    encoding='gb18030'))
    temp = pd.concat(tuple(temp))
    return temp

def read_data(data_dir):
    """
    读取数据
    """
    data_dict = {}
    
    data_dict['charger_stat'] = read_csv_files(data_dir, '充电机充电统计')
    data_dict['charger_process'] = read_csv_files(data_dir, '充电机充电过程')
    data_dict['single_temp'] = read_csv_files(data_dir, '单体电池温度记录')
    data_dict['single_voltage'] = read_csv_files(data_dir, '单体电池电压记录')
    data_dict['battery_stat'] = read_csv_files(data_dir, '电池充电统计')
    data_dict['battery_process'] = read_csv_files(data_dir, '电池充电过程')
    data_dict['battery_info'] = read_csv_files(data_dir, '电池辨识信息')
    
    return data_dict

def clean_data(data_dict_ori):
    """
    对读取对数据进行初步对清洗
    """
    data_dict = {}
    
    #对每一张表进行清洗
    #charger_stat
    print('cleaning charger_stat data...')
    data = data_dict_ori['charger_stat']
    data = data.rename(columns = {'时间': 'time', '编号': 'charger_id',
                                  'kWh': 'cs_kwh', '总时间': 'cs_totaltime'})
    data = data[['time', 'charger_id', 'cs_kwh', 'cs_totaltime']]
    data['time'] = data['time'].apply(str)
    data['charger_id'] = data['charger_id'].apply(str)
    data = data[data['cs_kwh'] >= 0] #删除cs_kwh小于0的行
    data = data[data['cs_totaltime'] >= 0] #删除cs_totaltime小于0的行
    data = data.dropna()
    data_dict['charger_stat'] = data
    
    #charger_process:
    print('cleaning charger_process data...')
    data = data_dict_ori['charger_process']
    data = data.rename(columns={'时间': 'time', '编号': 'charger_id', '电压': 'cp_v',
                                '电流': 'cp_i', '功率': 'cp_p', 'Ah': 'cp_ah',
                                'kWh': 'cp_kwh', '总时间': 'cp_totaltime'})
    data = data[['time', 'charger_id', 'cp_v', 'cp_i', 'cp_p', 'cp_ah', 'cp_kwh',
                 'cp_totaltime']]
    data['time'] = data['time'].apply(str)
    data['charger_id'] = data['charger_id'].apply(str)
    data = data[data['cp_v'] >= 300]
    data = data[data['cp_v'] <= 750]
    data = data[data['cp_i'] >= 0]
    data = data[data['cp_i'] <= 300]
    data = data[data['cp_p'] >= 0]
    data = data[data['cp_ah'] >= 0]
    data = data[data['cp_kwh'] >= 0]
    data = data[data['cp_totaltime'] >= 0]
    data = data.dropna()
    data_dict['charger_process'] = data
    
    # single_temp
    print('cleaning single_temp data...')
    data = data_dict_ori['single_temp']
    cols = ['时间', 'BMS编号'] + [i for i in data.columns if '温度' in i]
    data = data[cols]
    data = data.rename(columns={'时间': 'time', 'BMS编号': 'bms_id'})
    data['time'] = data['time'].apply(str)
    data['bms_id'] = data['bms_id'].apply(str)
    data = data[data >= 0.0]
    data = data.dropna()
    data_dict['single_temp'] = data

    # single_voltage
    print('cleaning single_voltage data...')
    data = data_dict_ori['single_voltage']
    cols = ['时间', 'BMS编号'] + [i for i in data.columns if '电压' in i]
    data = data[cols]
    data = data.rename(columns={'时间': 'time', 'BMS编号': 'bms_id'})
    data['time'] = data['time'].apply(str)
    data['bms_id'] = data['bms_id'].apply(str)
    data = data[data >= 2.5]
    data = data[data <= 4.8]
    data = data.dropna()
    data_dict['single_voltage'] = data

    # battery_stat: 数据质量不够好，直接删掉

    # battery_process
    print('cleaning battery_process data...')
    data = data_dict_ori['battery_process']
    data = data.rename(columns={'时间': 'time', 'BMS编号': 'bms_id', 'SOC': 'bp_soc',
                                '剩余时间': 'bp_lefttime', 'volt': 'bp_v',
                                'cur': 'bp_i'})
    data = data[['time', 'bms_id', 'bp_soc', 'bp_lefttime', 'bp_v', 'bp_i']]
    data['time'] = data['time'].apply(str)
    data['bms_id'] = data['bms_id'].apply(str)
    data = data[data['bp_soc'] >= 0]
    data = data[data['bp_soc'] <= 100]
    data = data[data['bp_lefttime'] >= 0]
    data = data[data['bp_v'] >= 300]
    data = data[data['bp_v'] <= 750]
    data = data[data['bp_i'] >= 0]
    data = data[data['bp_i'] <= 300]
    data = data.dropna()
    # 单体最高/低温度/电压 都不用做操作，因为后期并不会用到，会直接用single表的数据
    data_dict['battery_process'] = data

    # battery_info
    print('cleaning battery_info data...')
    data = data_dict_ori['battery_info']
    data = data[['时间', 'BMS编号', '额定容量', '额定电压', '单体电压限值', '容量限值',
                 '电压限值', '电流限值', '温度限值', '健康度', '充电机编号', '车辆VIN']]
    data = data.rename(columns={'时间': 'time', 'BMS编号': 'bms_id',
                                '额定容量': 'bi_ratedc', '额定电压': 'bi_ratedv',
                                '单体电压限值': 'bi_limitsv', '容量限值': 'bi_limitc',
                                '电压限值': 'bi_limitv', '电流限值': 'bi_limiti',
                                '温度限值': 'bi_limitt', '健康度': 'health',
                                '充电机编号': 'charger_id', '车辆VIN': 'vin'})
    data = data[data['bi_ratedc'] > 0]
    data = data[data['bi_ratedv'] > 0]
    data = data[data['bi_limitsv'] > 0]
    data = data[data['bi_limitc'] > 0]
    data = data[data['bi_limitv'] > 0]
    data = data[data['bi_limiti'] > 0]
    data = data[data['bi_limitt'] > 0]
    data = data[data['health'] >= 0]
    data = data.dropna()
    data_dict['battery_info'] = data
       
    return data_dict

def merge_values_func(data, kind):
    """
    用于处理《单体电池温度》和《单体电池电压》数据，统计出每一时刻每个电池的所有
    """
    #单体电池信息的统计值
    data_gp = data.groupby(['time', 'bms_id'])
    time_lst = []
    bms_lst = []
    min_lst = []
    max_lst= []
    mean_lst = []
    std_lst = []
    median_lst = []
    cnt = 0
    print('The lengths of the data is: %d'%len(data_gp.groups.keys()))
    for k in data_gp.groups.keys():
        v = data_gp.get_group(k)
        val = v.drop(['time', 'bms_id'], 1)
        for i in val.columns:
            val[i] = val[i].apply(float)
        #由于原始数据1s采集4次，但时间精度为1s，因此，需将相同时间的多行合并为1行
        val = val.values.ravel()#二维降为一维
        val = val[~np.isnan(val)]#去除空值
        time_lst.append(k[0])
        bms_lst.append(k[1])
        #求统计值
        min_lst.append(val.min())
        max_lst.append(val.max())
        mean_lst.append(val.mean())
        std_lst.append(val.std())
        median_lst.append(np.median(val))
        
        cnt = cnt + 1
        if cnt % 50 == 0:
            print('%d rows data have been done, finished %.1f%%'%(cnt, float(100*cnt/len(data_gp.groups.keys()))))
    tmp = pd.DataFrame({'time': time_lst, 'bms_id': bms_lst, 'min_' + kind: min_lst,
                            'max_' + kind: max_lst, 'mean_' + kind: mean_lst,
                            'std_' + kind: std_lst, 'median_' + kind: median_lst})
    return tmp

def mean_values_func1(data, id_name):
    """
    用于处理《充电机充电过程》和《电池充电过程》数据
    将同时间对数据行合并，求平均值
    """
    data_gp = data.groupby(['time', id_name])
    cnt = 0
    tmp = []
    print('The lengths of the data is: %d'%len(data_gp.groups.keys()))
    for k in data_gp.groups.keys():
        v = data_gp.get_group(k)
        val = v.drop(['time', id_name], 1).mean()
        val['time'] = k[0]
        val[id_name] = k[1]
        tmp.append(val)
        cnt = cnt + 1
        if cnt % 100 == 0:
            print('%d rows data have been done, finished %.1f%%'%(cnt, float(100*cnt/len(data_gp.groups.keys()))))
    tmp = pd.concat(tuple(tmp), axis=1)
    tmp = tmp.T
    return tmp
    
def process_data(data_dict, p_data_dir):
    #charger_stat
    data_cs = data_dict['charger_stat']

    #single_temp
    print('processing single_temp data...')
    data = data_dict['single_temp']
    data_st = merge_values_func(data, 'st')
    
    #single_voltage
    print('processing single_voltage data...')
    data = data_dict['single_voltage']
    data_sv = merge_values_func(data, 'sv')

    # battery_stat: 数据质量不够好先放着吧
    
    # charger_process: 数据质量不够好先放着吧
    print('processing charger_process data...')
    data = data_dict['charger_process']
    data_cp = mean_values_func1(data, 'charger_id')

    # battery_process
    print('processing battery_process data...')
    data = data_dict['battery_process']
    data_bp = mean_values_func1(data, 'bms_id')
    
    # battery_info
    data_bi = data_dict['battery_info']
    # 拿掉时间列后drop_duplicate后基本上就是每个电池一条数据,都是描述这个电池本身的出厂额定信息的数值
    
    # merge all data together
    # charge_stat 能和其他表merge的数据太少（其他表互相merge能有70%以上重合率，
    # 而charge_stat和其它标merge只有不到5%的重合率
    tmp = data_bi.merge(data_cp, on=['time', 'charger_id'], how='outer')
    tmp = tmp.merge(data_bp, on=['time', 'bms_id'], how='outer')
    tmp = tmp.merge(data_st, on=['time', 'bms_id'], how='outer')
    tmp = tmp.merge(data_sv, on=['time', 'bms_id'], how='outer')
    col_names = list(tmp.columns)
    col_names.remove('time')
    col_names.remove('bms_id')
    col_names.remove('charger_id')
    col_names = ['time', 'bms_id', 'charger_id'] + col_names
    tmp = tmp[col_names]
    tmp.to_csv(p_data_dir + 'processed_data.csv', encoding='gb18030', index=False)
    
    return tmp
    
def split_data(p_data_dir, processed_data=None):
    """
    将预处理后的数据按一次充电过程进行分割合并
    """
    CHARGE_TIMEGAP = 300  # 300 seconds = 5 minutes
    CHARGING_TIMEGAP = 60
    CA_KWH_UB = 350.0
    DROPNA_THRESH = 12
    
    if processed_data is None:
        print('reading processed_data...')
        processed_data = pd.read_csv(open(p_data_dir + 'processed_data.csv'), 
                                     encoding='gb18030')
    
    #filt samples on rows, if a row has too few none-nan value, drop it
    processed_data = processed_data.dropna(thresh=DROPNA_THRESH)
    
    processed_data['time'] = processed_data['time'].apply(str)
    processed_data['time'] = processed_data['time'].apply(lambda x: parser.parse(x))
    processed_data['bms_id'] = processed_data['bms_id'].apply(str)
    processed_data['charger_id'] = processed_data['charger_id'].apply(str)
    processed_data['bp_v'] = processed_data['bp_v'].fillna(processed_data['cp_v'])
    processed_data['bp_i'] = processed_data['bp_i'].fillna(processed_data['cp_i'])
    
     # group by bms_id and sort by time
    processed_data_gp = processed_data.groupby('bms_id')
    #data = pd.DataFrame(columns=['bms_id', 'start_time', 'end_time',
    #                             'charger_id', 'data_num'])
    data = pd.DataFrame()
    cnt = 0
    num = 0
    print('The numbers of the data clips is: %d'%len(processed_data_gp.groups))
    for i in processed_data_gp.groups:
        df = processed_data_gp.get_group(i) #第i组
        print('NO.%d: '%num, i, df.shape, cnt)
        num += 1
        df = df.sort_values('time')
        j_last = 0
        for j in range(1, len(df) + 1):
            if j >= len(df) or (df.iloc[j]['time'] - df.iloc[j - 1]['time']).seconds > CHARGE_TIMEGAP:
                    
                if j >= len(df):
                    cur_df = df.iloc[j_last:]
                elif (df.iloc[j]['time'] - df.iloc[j - 1]['time']).seconds > CHARGE_TIMEGAP:
                    cur_df = df.iloc[j_last:j]
                    #j_last = j

                func = lambda x: x.fillna(method='ffill').fillna(method='bfill').dropna()
                cur_df = func(cur_df)
                
                print('clip %d : j: %d -> %d, the length of cur_df: %d.'
                      %(cnt, j_last, j, len(cur_df)))
                #print('j:', j_last, '->', j, 'len(cur_df):', tmp_len, '->', len(cur_df), 'cnt=', cnt)
                j_last = j
                if len(cur_df) <= 0 or (cur_df['time'].iloc[-1] - cur_df['time'].iloc[0]).seconds < CHARGING_TIMEGAP:
                    continue
                cur_df['score_ca_kwh'] = cur_df['cp_kwh'].diff() / cur_df['bp_soc'].diff() * 100
                cur_df['score_ca_ah'] = cur_df['cp_ah'].diff() / cur_df['bp_soc'].diff() * 100
                cur_df['score_ca_health'] = cur_df['health']
                
                data = data.append(transfer_data(cnt, i, cur_df))
                cnt += 1
                
    data.to_csv(p_data_dir + 'data.csv', encoding='gb18030', index=False)
    return data

def transfer_data(cnt, bms_id, cur_df):
    """
    将2维的df转换为1维
    """
    df = pd.DataFrame(columns=['bms_id', 'start_time', 'end_time',
                                 'charger_id', 'data_num'])
    df.loc[cnt, 'bms_id'] = bms_id
    df.loc[cnt, 'start_time'] = cur_df['time'].iloc[0]
    df.loc[cnt, 'end_time'] = cur_df['time'].iloc[-1]
    df.loc[cnt, 'charger_id'] = cur_df['charger_id'].iloc[0]
    df.loc[cnt, 'data_num'] = len(cur_df)
    
    for col_name in cur_df.columns:
        for fix in ['bi_']:
            if fix in col_name:
                df.loc[cnt, col_name + '_mean'] = cur_df[cur_df[col_name] > 0][col_name].mean(skipna=True)#选择col_name大于0的行并求平均值
        for fix in ['score_']:
            if fix in col_name:
                cal_stat(cnt, cur_df[col_name], col_name, df)
        for fix in ['cp_', 'bp_', '_sv', '_st']:
            if fix in col_name:
                cal_stat(cnt, cur_df[col_name], col_name, df)
                cal_stat(cnt, cur_df[col_name].diff().abs(), col_name + '_diff', df)
                cal_stat(cnt, cur_df[col_name].diff().diff().abs(), col_name + '_diff2', df)
                cal_stat(cnt, cur_df[col_name].diff().abs() / cur_df[col_name], col_name + '_diffrate', df)
    return df

def cal_stat(cnt, ser, col_name, df):
    """
    求统计值
    """
    df.loc[cnt, col_name + '_mean'] = ser.mean(skipna=True)
    df.loc[cnt, col_name + '_min'] = ser.min(skipna=True)
    df.loc[cnt, col_name + '_max'] = ser.max(skipna=True)
    df.loc[cnt, col_name + '_median'] = ser.median(skipna=True)
    df.loc[cnt, col_name + '_std'] = ser.std(skipna=True)
    
def cut_data(data_dict, max_lens, output_dir=None):
    if output_dir:
        writer = pd.ExcelWriter(output_dir + 'cut_data.xlsx')
    for key, value in data_dict.items():
        value = value[:min(len(value), max_lens)]
        print(key, value)
        #保存处理后的数据
        if output_dir:
            value.to_excel(writer, key)
            
def read_cut_data(output_dir):
    print('reading cut_data...')
    return(pd.read_excel(output_dir + 'cut_data.xlsx', sheet_name=None))
           
def main():
    data_dir = '../data/2017大运中心数据/'
    p_data_dir = '../data/处理后数据/'
    """
    data_dict_ori = read_data(data_dir)
    data_dict = clean_data(data_dict_ori)
    #cut_data(data_dict, 10000, p_data_dir)
    data_dict = read_cut_data(p_data_dir)
    processed_data = process_data(data_dict, p_data_dir)
    
    data = split_data(p_data_dir)
    """
if __name__ == '__main__':
    main()