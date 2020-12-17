# -*- coding : utf-8-*-
import os

import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import itertools
from datetime import datetime
from matplotlib.font_manager import FontProperties
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
mfont = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def create_interpolate_feature(filename, outputfilename):
    data = pd.read_csv(filename)
    date = data['日期']
    for col in data.columns:
        if col != '日期':
            data[col] = data[col].interpolate()
    data['日期'] = date
    data.to_csv(outputfilename, index=None)
    pass


def get_sarimax_predict(data, start_time, end_time, order, seasonal_order):
    data.columns = ['rtime', 'values']
    data['rtime'] = data['rtime'].apply(lambda x: datetime.strptime(x, '%Y/%m'))
    data = data.set_index('rtime')
    y = data['values']
    y_train = y[:start_time]
    mod = sm.tsa.statespace.SARIMAX(y_train,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime(start_time), end=pd.to_datetime(end_time), dynamic=False)
    pred_ci = pred.conf_int()

    y_real = y[start_time:end_time]
    y_pred = pred.predicted_mean
    return y_pred


def create_arima_feature(filename, outputfilename):
    data = pd.read_csv(filename)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns

    y_pred_total = []
    for i, val in enumerate(columns):
        tempi = data[['日期', val]]
        y_predi = get_sarimax_predict(tempi, '2020-11-01', '2021-12-01', (1, 1, 1), (1, 1, 1, 12))
        y_predi = np.asarray(y_predi)
        y_pred_total.append(y_predi)

    y_pred_total = np.asarray(y_pred_total)
    y_pred_total_np = y_pred_total.T
    y_pred_total_df = pd.DataFrame(y_pred_total_np)
    print(y_pred_total_df)
    pass


def create_m_feature(filename, outputfilename, skip_rows):
    data = pd.read_csv(filename)
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns

    date = date[skip_rows:]
    date = np.asarray(date)

    total_xi = []
    total_column_names = []
    for i in range(0, len(columns)):
        xi = np.asarray(data_to_cal[columns[i]])
        len_xi = xi.shape[0]
        xi_m_1 = xi[skip_rows - 1:len_xi - skip_rows + 1]
        xi_m_2 = xi[0:len_xi - skip_rows]
        xi = xi[skip_rows:len_xi]

        total_xi.append(xi)
        total_xi.append(xi_m_1)
        total_xi.append(xi_m_2)
        total_column_names.append(columns[i])
        total_column_names.append(columns[i] + 'm1')
        total_column_names.append(columns[i] + 'm2')

    total_xi_np = np.asarray(total_xi)
    total_xi_np = total_xi_np.T
    total_xi_df = pd.DataFrame(total_xi_np)
    total_xi_df.columns = total_column_names
    total_xi_df['日期'] = date
    total_xi_df.to_csv(outputfilename, index=None)
    pass


def create_macro_wa_coor_fig(macro_filename, wa_index_name, oufputpath,type='m'):
    macro_index = pd.read_excel(macro_filename, skiprows=5, header=None)
    macro_index = macro_index[:macro_index.shape[0] - 2]
    macro_index_name_strings = macro_filename.split('/')
    macro_index_name_xls = macro_index_name_strings[len(macro_index_name_strings) - 1]
    macro_index_name = macro_index_name_xls[:len(macro_index_name_xls) - 4]

    macro_index.columns = ['日期', macro_index_name]
    if type=='y':
        macro_index['日期'] = macro_index['日期'].apply(lambda x: str(x.year))
        macro_index['日期'] = macro_index['日期'].apply(lambda x: datetime.strptime(x, '%Y'))
    else:
        macro_index['日期'] = macro_index['日期'].apply(lambda x: str(x.year) + '/' + str(x.month))
        macro_index['日期'] = macro_index['日期'].apply(lambda x: datetime.strptime(x, '%Y/%m'))
    macro_index = macro_index.set_index('日期')

    wa_index = pd.read_csv(wa_index_name)
    wa_index['日期'] = wa_index['日期'].apply(lambda x: datetime.strptime(x, '%Y/%m'))
    wa_index = wa_index.set_index('日期')

    wa_macro_index = pd.concat([wa_index[wa_index.columns[0]], macro_index[macro_index.columns[0]]], axis=1)
    wa_macro_index = wa_macro_index.dropna()
    wa_macro_index = wa_macro_index.reset_index()

    wa_macro_index_columns = wa_macro_index.columns
    y = np.asarray(wa_macro_index[wa_macro_index_columns[1]])
    xi = np.asarray(wa_macro_index[wa_macro_index_columns[2]])
    date = wa_macro_index[wa_macro_index_columns[0]]
    skip_rows = 2

    len_y = y.shape[0]
    y = y[skip_rows:len_y]

    len_xi = xi.shape[0]
    xi_m_1 = xi[skip_rows - 1:len_xi - skip_rows + 1]
    xi_m_2 = xi[0:len_xi - skip_rows]
    xi = xi[skip_rows:len_xi]

    date = date[skip_rows:]

    total_xi_np = np.asarray([y, xi, xi_m_1, xi_m_2])
    total_xi_np = total_xi_np.T
    total_xi_df = pd.DataFrame(total_xi_np)
    total_xi_df.columns = [wa_macro_index_columns[1], wa_macro_index_columns[2], wa_macro_index_columns[2] + 'M-1',
                           wa_macro_index_columns[2] + 'M-2']

    min_max_scaler_model = MinMaxScaler()
    total_index = min_max_scaler_model.fit_transform(total_xi_df)

    y1 = total_index[:, 0]
    for i in range(1,len(total_xi_df.columns)):
        y2 = total_index[:, i]
        coor_num = round(
            np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        plt.title(total_xi_df.columns[0] + '与' + total_xi_df.columns[i] + '的相关分析(' + str(coor_num) + ')',
                  fontproperties=mfont)
        plt.xlabel('日期')
        plt.ylabel('值')
        # xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = plt.plot(date, y1)
        p2 = plt.plot(date, y2)
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2], labels=[total_xi_df.columns[0], total_xi_df.columns[i]])

        if abs(coor_num) >= 0.7 and abs(coor_num) < 1:
            plt.savefig(oufputpath + '/70/' +
                        total_xi_df.columns[0] + '与' + total_xi_df.columns[i] + '，相关系数为：' + str(coor_num) + '.png',
                        bbox_inches='tight')
        elif abs(coor_num) >= 0.6 and abs(coor_num) < 0.7:
            plt.savefig(oufputpath + '/60/' +
                        total_xi_df.columns[0] + '与' + total_xi_df.columns[i] + '，相关系数为：' + str(coor_num) + '.png',
                        bbox_inches='tight')
        elif abs(coor_num) >= 0.5 and abs(coor_num) < 0.6:
            plt.savefig(oufputpath + '/50/' +
                        total_xi_df.columns[0] + '与' + total_xi_df.columns[i] + '，相关系数为：' + str(coor_num) + '.png',
                        bbox_inches='tight')
        else:
            plt.savefig(oufputpath + '/other/' +
                        total_xi_df.columns[0] + '与' + total_xi_df.columns[i] + '，相关系数为：' + str(coor_num) + '.png',
                        bbox_inches='tight')

    # y1 = total_index[:, 0]
    # y2 = total_index[:, 1]
    # coor_num = round(
    #     np.corrcoef(y1, y2)[0, 1], 2)
    # plt.clf()
    # plt.title(total_xi_df.columns[0] + '与' + total_xi_df.columns[1] + '的相关分析(' + str(coor_num) + ')',
    #           fontproperties=mfont)
    # plt.xlabel('日期')
    # plt.ylabel('值')
    # # xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
    # p1 = plt.plot(date, y1)
    # p2 = plt.plot(date, y2)
    # plt.gcf().autofmt_xdate()
    # plt.legend([p1, p2], labels=[total_xi_df.columns[0], total_xi_df.columns[1]])
    #
    # if abs(coor_num) >= 0.7 and abs(coor_num) < 1:
    #     plt.savefig(oufputpath + '/70/' +
    #                 total_xi_df.columns[0] + '与' + total_xi_df.columns[1] + '，相关系数为：' + str(coor_num) + '.png',
    #                 bbox_inches='tight')
    # elif abs(coor_num) >= 0.6 and abs(coor_num) < 0.7:
    #     plt.savefig(oufputpath + '/60/' +
    #                 total_xi_df.columns[0] + '与' + total_xi_df.columns[1] + '，相关系数为：' + str(coor_num) + '.png',
    #                 bbox_inches='tight')
    # elif abs(coor_num) >= 0.5 and abs(coor_num) < 0.6:
    #     plt.savefig(oufputpath + '/50/' +
    #                 total_xi_df.columns[0] + '与' + total_xi_df.columns[1] + '，相关系数为：' + str(coor_num) + '.png',
    #                 bbox_inches='tight')
    # else:
    #     plt.savefig(oufputpath + '/other/' +
    #                 total_xi_df.columns[0] + '与' + total_xi_df.columns[1] + '，相关系数为：' + str(coor_num) + '.png',
    #                 bbox_inches='tight')
    pass


if __name__ == '__main__':
    # 预测宏观因子
    # create_arima_feature(r'./data/wa_index/features_month_interpolate.csv',
    #                      r'./data/wa_index/features_month_interpolate_m1-2.csv')

    # 生成m1到m2的宏观因子数据
    # create_m_feature(r'./data/wa_index/features_month_interpolate.csv',
    #                  r'./data/wa_index/features_month_interpolate_m1-2.csv', skip_rows=2)

    # 生成一个相关性图表
    # create_macro_wa_coor_fig(r'./data/wa_index/wa_macro_index/month/PPI按大类分(月)（累计同比）.xls',
    #                          r'./data/wa_index/wa_month.csv',
    #                          r'./data/wa_index/wa_png/month')

    # 按月生成图表
    file_dir = r'./data/wa_index/wa_macro_index/month/'
    wa_index_filename = r'./data/wa_index/wa_month.csv'
    output_filepath = r'./data/wa_index/wa_png/month'
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            one_macro_index = root + file
            create_macro_wa_coor_fig(one_macro_index,
                                     wa_index_filename,
                                     output_filepath,type='m')

    # 按季生成图表
    file_dir = r'./data/wa_index/wa_macro_index/quarter/'
    wa_index_filename = r'./data/wa_index/wa_quarter.csv'
    output_filepath = r'./data/wa_index/wa_png/quarter'
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            one_macro_index = root + file
            create_macro_wa_coor_fig(one_macro_index,
                                     wa_index_filename,
                                     output_filepath,type='m')

    # 按年生成图表
    file_dir = r'./data/wa_index/wa_macro_index/year/'
    wa_index_filename = r'./data/wa_index/wa_year.csv'
    output_filepath = r'./data/wa_index/wa_png/year'
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            one_macro_index = root + file
            create_macro_wa_coor_fig(one_macro_index,
                                    wa_index_filename,
                                    output_filepath,type='y')


    # create_macro_wa_coor_fig(r'./data/wa_index/wa_macro_index/month/PPI按大类分(月).xls',
    #                          r'./data/wa_index/wa_month.csv',
    #                          r'./data/wa_index/wa_png/month',type='m')

    pass
