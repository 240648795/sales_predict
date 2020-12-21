# -*- coding : utf-8-*-
import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = ['AR PL UKai CN']
mfont = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def create_interpolate_feature(filename, outputfilename):
    data = pd.read_csv(filename)
    date = data['日期']
    for col in data.columns:
        if col != '日期':
            data[col] = data[col].interpolate()
    data['日期'] = date
    data.to_csv(outputfilename, index=None)


def get_coorfig_qm(filepath, outputpath, skip_rows):
    data = pd.read_csv(filepath)
    data = data[skip_rows:]
    date = data['日期']
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    columns = data_to_cal.columns
    # min_max_scaler_model = MinMaxScaler()
    # data_to_cal = min_max_scaler_model.fit_transform(data_to_cal)
    data_to_cal = np.asarray(data_to_cal)

    for i in range(1, len(columns)):
        y1 = data_to_cal[:, 0]
        y2 = data_to_cal[:, i]
        coor_num = round(np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        fig, ax1 = plt.subplots()
        plt.title('行业销量与' + columns[i] + '的相关分析(' + str(coor_num) + ')', fontproperties=mfont)
        plt.xlabel('日期')
        # plt.ylabel('值')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = ax1.plot(xs, y1, c='orange')
        ax1.set_ylabel('行业销量')
        ax2 = ax1.twinx()
        p2 = ax2.plot(xs, y2, c='blue')
        ax2.set_ylabel(columns[i])
        plt.legend(p1 + p2, ['行业销量', columns[i]])
        plt.gcf().autofmt_xdate()
        if abs(coor_num) >= 0.7 and abs(coor_num) < 1:
            plt.savefig(outputpath + '/70/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png', bbox_inches='tight')
        elif abs(coor_num) >= 0.6 and abs(coor_num) < 0.7:
            plt.savefig(outputpath + '/60/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png', bbox_inches='tight')
        elif abs(coor_num) >= 0.5 and abs(coor_num) < 0.6:
            plt.savefig(outputpath + '/50/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png', bbox_inches='tight')
        else:
            plt.savefig(outputpath + '/other/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png',
                        bbox_inches='tight')
        plt.close()


def cal_all_feature(filepath, outputpath, type, skip_rows):
    data = pd.read_csv(filepath)

    for i in range(2, len(data.columns)):
        df_each_feature = data[[data.columns[0], data.columns[1], data.columns[i]]]
        get_each_feature_coorfig(df_each_feature, outputpath, type=type, skip_rows=skip_rows)


def get_each_feature_coorfig(df, outputpath, type, skip_rows=3):
    # 获取宏观指标的单位
    macro_index_columnname = df.columns[2]
    macro_index_unit = '_'
    macro_index_unit_value_splite = len(macro_index_columnname)
    if macro_index_columnname.find('（') != -1:
        macro_index_unit_value_splite = macro_index_columnname.find('（')
        macro_index_unit = macro_index_columnname[macro_index_unit_value_splite:]

    # 获取装载机销量的单位
    zhuang_index_columnname = df.columns[1]
    zhuang_index_unit = '_'
    zhuang_index_unit_value_splite = len(zhuang_index_columnname)
    if zhuang_index_columnname.find('（') != -1:
        zhuang_index_unit_value_splite = zhuang_index_columnname.find('（')
        zhuang_index_unit = zhuang_index_columnname[zhuang_index_unit_value_splite:]

    df.columns = ['日期', zhuang_index_columnname[:zhuang_index_unit_value_splite],
                  macro_index_columnname[:macro_index_unit_value_splite]]
    zhuang_macro_index_columns = df.columns

    # print(zhuang_macro_index_columns)
    # print(macro_index_unit,',',zhuang_index_unit)

    y = np.asarray(df[zhuang_macro_index_columns[1]])
    xi = np.asarray(df[zhuang_macro_index_columns[2]])
    date = df[zhuang_macro_index_columns[0]]

    # 计算差分的值，这里只差分两次，即计算到M-2
    skip_rows = 2

    len_y = y.shape[0]
    y = y[skip_rows:len_y]

    len_xi = y.shape[0]

    xi_m_1 = xi[skip_rows - 1:skip_rows - 1 + len_xi]
    xi_m_2 = xi[skip_rows - 2:skip_rows - 2 + len_xi]
    # xi_m_3 = xi[skip_rows - 3:skip_rows - 3 + len_xi]
    xi = xi[skip_rows:skip_rows + len_xi]

    date = date[skip_rows:skip_rows + len_xi]

    total_xi_np = np.asarray([y, xi, xi_m_1, xi_m_2])
    # total_xi_np = np.asarray([y, xi, xi_m_1, xi_m_2,xi_m_3])
    total_xi_np = total_xi_np.T
    total_xi_df = pd.DataFrame(total_xi_np)

    total_xi_df.columns = [zhuang_macro_index_columns[1], zhuang_macro_index_columns[2],
                           zhuang_macro_index_columns[2] + type + '-1',
                           zhuang_macro_index_columns[2] + type + '-2']
    # total_xi_df.columns = [zhuang_macro_index_columns[1], zhuang_macro_index_columns[2],
    #                        zhuang_macro_index_columns[2] + 'M-1',
    #                        zhuang_macro_index_columns[2] + 'M-2',
    #                        zhuang_macro_index_columns[2] + 'M-3']

    # print(total_xi_df)
    data_to_cal = np.asarray(total_xi_df)
    columns = total_xi_df.columns

    # 画图
    for i in range(1, len(columns)):
        y1 = data_to_cal[:, 0]
        y2 = data_to_cal[:, i]
        # 计算相关系数，这里可以替换成其他计算相关系数的方法
        coor_num = round(np.corrcoef(y1, y2)[0, 1], 2)
        plt.clf()
        fig, ax1 = plt.subplots()
        plt.title(
            '行业销量' + zhuang_index_unit + '与' + columns[i] + macro_index_unit + '的相关分析(' + str(
                coor_num) + ')', fontproperties=mfont)
        plt.xlabel('日期')
        xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
        p1 = ax1.plot(xs, y1, c='orange')
        ax1.set_ylabel('行业销量')
        ax2 = ax1.twinx()
        p2 = ax2.plot(xs, y2, c='blue')
        ax2.set_ylabel(columns[i])
        plt.legend(p1 + p2, ['行业销量', columns[i]])
        plt.gcf().autofmt_xdate()
        if abs(coor_num) >= 0.7 and abs(coor_num) < 1:
            plt.savefig(outputpath + '/70/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png', bbox_inches='tight')
        elif abs(coor_num) >= 0.6 and abs(coor_num) < 0.7:
            plt.savefig(outputpath + '/60/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png', bbox_inches='tight')
        elif abs(coor_num) >= 0.5 and abs(coor_num) < 0.6:
            plt.savefig(outputpath + '/50/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png', bbox_inches='tight')
        else:
            plt.savefig(outputpath + '/other/行业销量-' + columns[i] + '，相关系数为：' + str(coor_num) + '.png',
                        bbox_inches='tight')
        plt.close()


def cal_zhuang_coorfig_main():
    # 计算装载机各项月度指标M-1到M-2的相关性图表
    get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_month_m1-2.csv',
                   outputpath=r'./result_png/zhuang/month',
                   skip_rows=2)
    # 计算装载机各项季度指标Q1相关性图表
    get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_quarter_q1.csv',
                   outputpath=r'./result_png/zhuang/quarter',
                   skip_rows=1)
    # 计算装载机各项年度指标相关性图表
    get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_year.csv',
                   outputpath=r'./result_png/zhuang/year',
                   skip_rows=0)
    # 计算装载机去年同月指标相关性图表
    get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_tongqi_month.csv',
                   outputpath=r'./result_png/zhuang/month',
                   skip_rows=12)
    # 计算装载机去年同季指标相关性图表
    get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_tongqi_quarter.csv',
                   outputpath=r'./result_png/zhuang/quarter',
                   skip_rows=4)


if __name__ == '__main__':
    # 计算所有装载机相关性指标
    # cal_zhuang_coorfig_main()

    # 计算装载机各项月度指标M-1到M-2的相关性图表(旧)
    # get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_month_m1-2.csv',
    #                outputpath=r'./result_png/zhuang/month',
    #                skip_rows=2)

    # 计算装载机各项月度指标M-1到M-3的相关性图表(新)
    cal_all_feature(filepath=r'./data/zhuang_index/zhuang_month.csv',
                    outputpath=r'./result_png/zhuang/month',
                    type='M',
                    skip_rows=3)

    # 计算装载机各项季度指标M-1到M-3的相关性图表(新)
    cal_all_feature(filepath=r'./data/zhuang_index/zhuang_quarter.csv',
                    outputpath=r'./result_png/zhuang/quarter',
                    type='Q',
                    skip_rows=3)

    # 计算装载机各项年度指标M-1到M-3的相关性图表(新)
    cal_all_feature(filepath=r'./data/zhuang_index/zhuang_year.csv',
                    outputpath=r'./result_png/zhuang/year',
                    type='Y',
                    skip_rows=3)

    pass
