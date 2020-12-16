# -*- coding: utf-8 -*-
import random
from datetime import datetime, timedelta

import dateutil
import joblib
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
mfont = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def show_coor_fig_save_org_quarter(filepath):
    data = pd.read_csv(filepath)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns
    min_max_scaler_model = MinMaxScaler()
    data_to_cal = min_max_scaler_model.fit_transform(data_to_cal)

    for i in range(1, len(columns)):
        y1 = data_to_cal[:, 0]
        y2 = data_to_cal[:, i]
        coor_num = round(np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        plt.title('全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num), fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = plt.plot(xs, y1)
        p2 = plt.plot(xs, y2)
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2], labels=['全国行业销量', columns[i]])
        plt.savefig(
            './result_png/quarter_coor_png/total/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')


def show_coor_fig_save_m1_6_quarter(filepath):
    data = pd.read_csv(filepath)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns
    min_max_scaler_model = MinMaxScaler()
    data_to_cal = min_max_scaler_model.fit_transform(data_to_cal)

    for i in range(1, len(columns)):
        y1 = data_to_cal[:, 0]
        y2 = data_to_cal[:, i]
        coor_num = round(np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        plt.title('全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num), fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = plt.plot(xs, y1)
        p2 = plt.plot(xs, y2)
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2], labels=['全国行业销量', columns[i]])
        if coor_num >= 0.7 and coor_num < 1:
            plt.savefig(
                './result_png/quarter_coor_png/70/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')
        elif coor_num >= 0.6 and coor_num < 0.7:
            plt.savefig(
                './result_png/quarter_coor_png/60/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')
        elif coor_num >= 0.5 and coor_num < 0.6:
            plt.savefig(
                './result_png/quarter_coor_png/50/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')
        else:
            pass


def show_coor_fig_save_org_month(filepath):
    data = pd.read_csv(filepath)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns
    min_max_scaler_model = MinMaxScaler()
    data_to_cal = min_max_scaler_model.fit_transform(data_to_cal)

    for i in range(1, len(columns)):
        y1 = data_to_cal[:, 0]
        y2 = data_to_cal[:, i]
        coor_num = round(np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        plt.title('全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num), fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = plt.plot(xs, y1)
        p2 = plt.plot(xs, y2)
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2], labels=['全国行业销量', columns[i]])
        plt.savefig('./result_png/month_coor_png/total/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')


def show_coor_fig_save_m1_6_month(filepath):
    data = pd.read_csv(filepath)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns
    min_max_scaler_model = MinMaxScaler()
    data_to_cal = min_max_scaler_model.fit_transform(data_to_cal)

    for i in range(1, len(columns)):
        y1 = data_to_cal[:, 0]
        y2 = data_to_cal[:, i]
        coor_num = round(np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        plt.title('全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num), fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = plt.plot(xs, y1)
        p2 = plt.plot(xs, y2)
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2], labels=['全国行业销量', columns[i]])
        if coor_num >= 0.7 and coor_num < 1:
            plt.savefig('./result_png/month_coor_png/70/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')
        elif coor_num >= 0.6 and coor_num < 0.7:
            plt.savefig('./result_png/month_coor_png/60/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')
        elif coor_num >= 0.5 and coor_num < 0.6:
            plt.savefig('./result_png/month_coor_png/50/' + '全国行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png')
        else:
            pass


def cal_huan_tong_month(filepath):
    data = pd.read_csv(filepath)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns

    date = date[12:]

    y = np.asarray(data_to_cal[columns[0]])
    len_y = y.shape[0]
    y_m_1 = y[11:len_y - 1]
    y_m_12 = y[0:len_y - 12]
    y = y[12:len_y]

    for i in range(1, len(columns)):
        xi = np.asarray(data_to_cal[columns[i]])
        len_xi = xi.shape[0]
        xi_m_1 = xi[11:len_xi - 1]
        xi_m_12 = xi[0:len_xi - 12]
        xi = xi[12:len_xi]

        y_xi_combine = np.asarray([y, y_m_1, y_m_12, xi, xi_m_1, xi_m_12])
        y_xi_combine = y_xi_combine.T
        y_xi_combine_df = pd.DataFrame(y_xi_combine)

        y_xi_combine_df_columns = y_xi_combine_df.columns.values
        y_xi_combine_df['y_huan'] = (y_xi_combine_df[y_xi_combine_df_columns[0]] - y_xi_combine_df[
            y_xi_combine_df_columns[1]]) / y_xi_combine_df[y_xi_combine_df_columns[1]] * 100
        y_xi_combine_df['y_tong'] = (y_xi_combine_df[y_xi_combine_df_columns[0]] - y_xi_combine_df[
            y_xi_combine_df_columns[2]]) / y_xi_combine_df[y_xi_combine_df_columns[2]] * 100

        y_xi_combine_df['xi_huan'] = (y_xi_combine_df[y_xi_combine_df_columns[3]] - y_xi_combine_df[
            y_xi_combine_df_columns[4]]) / y_xi_combine_df[y_xi_combine_df_columns[4]] * 100
        y_xi_combine_df['x1_tong'] = (y_xi_combine_df[y_xi_combine_df_columns[3]] - y_xi_combine_df[
            y_xi_combine_df_columns[5]]) / y_xi_combine_df[y_xi_combine_df_columns[5]] * 100


        min_max_scaler_model = MinMaxScaler()
        y_xi_minmax = min_max_scaler_model.fit_transform(y_xi_combine_df)

        plt.clf()
        plt.figure(figsize=(30, 10))
        plt.title('全国行业销量-' + columns[i], fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        xs_de = [d + dateutil.relativedelta.relativedelta(days=8) for d in xs]
        p1 = plt.plot(xs, y_xi_minmax[:, 0], color='red')
        p2 = plt.plot(xs, y_xi_minmax[:, 3], color='green')
        p3 = plt.bar(xs, y_xi_minmax[:, 6], width=8, color='blue')
        p4 = plt.bar(xs_de, y_xi_minmax[:, 8], width=8, color='orange')
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2, p3, p4],
                   labels=['全国行业销量', columns[i], '全国行业销量环比', columns[i] + '环比'])
        plt.savefig('./result_png/month_coor_png/huan/' + '全国行业销量-' + columns[i] + '环比.png')

        plt.clf()
        plt.figure(figsize=(30, 10))
        plt.title('全国行业销量-' + columns[i], fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        xs_de = [d + dateutil.relativedelta.relativedelta(days=8) for d in xs]
        p1 = plt.plot(xs, y_xi_minmax[:, 0], color='red')
        p2 = plt.plot(xs, y_xi_minmax[:, 3], color='green')
        p3 = plt.bar(xs, y_xi_minmax[:, 7], width=8, color='blue')
        p4 = plt.bar(xs_de, y_xi_minmax[:, 9], width=8, color='orange')
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2, p3, p4],
                   labels=['全国行业销量', columns[i], '全国行业销量同比', columns[i] + '同比'])
        plt.savefig('./result_png/month_coor_png/tong/' + '全国行业销量-' + columns[i] + '同比.png')


def cal_huan_tong_quarter(filepath):
    data = pd.read_csv(filepath)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns

    date = date[4:]

    y = np.asarray(data_to_cal[columns[0]])
    len_y = y.shape[0]
    y_m_1 = y[3:len_y - 1]
    y_m_12 = y[0:len_y - 4]
    y = y[4:len_y]

    for i in range(1, len(columns)):
        xi = np.asarray(data_to_cal[columns[i]])
        len_xi = xi.shape[0]
        xi_m_1 = xi[3:len_xi - 1]
        xi_m_12 = xi[0:len_xi - 4]
        xi = xi[4:len_xi]

        y_xi_combine = np.asarray([y, y_m_1, y_m_12, xi, xi_m_1, xi_m_12])
        y_xi_combine = y_xi_combine.T
        y_xi_combine_df = pd.DataFrame(y_xi_combine)

        y_xi_combine_df_columns = y_xi_combine_df.columns.values
        y_xi_combine_df['y_huan'] = (y_xi_combine_df[y_xi_combine_df_columns[0]] - y_xi_combine_df[
            y_xi_combine_df_columns[1]]) / y_xi_combine_df[y_xi_combine_df_columns[1]] * 100
        y_xi_combine_df['y_tong'] = (y_xi_combine_df[y_xi_combine_df_columns[0]] - y_xi_combine_df[
            y_xi_combine_df_columns[2]]) / y_xi_combine_df[y_xi_combine_df_columns[2]] * 100

        y_xi_combine_df['xi_huan'] = (y_xi_combine_df[y_xi_combine_df_columns[3]] - y_xi_combine_df[
            y_xi_combine_df_columns[4]]) / y_xi_combine_df[y_xi_combine_df_columns[4]] * 100
        y_xi_combine_df['x1_tong'] = (y_xi_combine_df[y_xi_combine_df_columns[3]] - y_xi_combine_df[
            y_xi_combine_df_columns[5]]) / y_xi_combine_df[y_xi_combine_df_columns[5]] * 100


        min_max_scaler_model = MinMaxScaler()
        y_xi_minmax = min_max_scaler_model.fit_transform(y_xi_combine_df)

        plt.clf()
        plt.figure(figsize=(30, 10))
        plt.title('全国行业销量-' + columns[i], fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        xs_de = [d + dateutil.relativedelta.relativedelta(days=10) for d in xs]
        p1 = plt.plot(xs, y_xi_minmax[:, 0], color='red')
        p2 = plt.plot(xs, y_xi_minmax[:, 3], color='green')
        p3 = plt.bar(xs, y_xi_minmax[:, 6], width=10, color='blue')
        p4 = plt.bar(xs_de, y_xi_minmax[:, 8], width=10, color='orange')
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2, p3, p4],
                   labels=['全国行业销量', columns[i], '全国行业销量环比', columns[i] + '环比'])
        plt.savefig('./result_png/quarter_coor_png/huan/' + '全国行业销量-' + columns[i] + '环比.png')

        plt.clf()
        plt.figure(figsize=(30, 10))
        plt.title('全国行业销量-' + columns[i], fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        xs_de = [d + dateutil.relativedelta.relativedelta(days=10) for d in xs]
        p1 = plt.plot(xs, y_xi_minmax[:, 0], color='red')
        p2 = plt.plot(xs, y_xi_minmax[:, 3], color='green')
        p3 = plt.bar(xs, y_xi_minmax[:, 7], width=10, color='blue')
        p4 = plt.bar(xs_de, y_xi_minmax[:, 9], width=10, color='orange')
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2, p3, p4],
                   labels=['全国行业销量', columns[i], '全国行业销量同比', columns[i] + '同比'])
        plt.savefig('./result_png/quarter_coor_png/tong/' + '全国行业销量-' + columns[i] + '同比.png')


if __name__ == '__main__':
    # 利用月度m-1至m-6做出相关性分析图
    # show_coor_fig_save_m1_6_month(r'./data/zhuang_index/index_without2feature_interpolate_m1-6_xiaoliang.csv')

    # 利用月度原始数据做的相关性分析图
    # show_coor_fig_save_org_month(r'./data/zhuang_index/index_without2feature_interpolate.csv')

    # 利用季度m-1至m-6做出相关性分析图
    # show_coor_fig_save_m1_6_quarter(r'./data/zhuang_index/quarter_without2feature_interpolate_m1-6.csv')

    # 利用季度原始数据做出相关性分析图
    # show_coor_fig_save_org_quarter(r'./data/zhuang_index/quarter_without2feature_interpolate.csv')

    # 计算月度数据的环比、同比
    # cal_huan_tong_month(r'./data/zhuang_index/index_without2feature_interpolate.csv')

    # 计算月度数据的环比、同比
    # cal_huan_tong_quarter(r'./data/zhuang_index/quarter_without2feature_interpolate.csv')
    
    pass
