# -*- coding : utf-8-*-
import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
        ax1.set_ylabel('行业销量值')
        ax2 = ax1.twinx()
        p2 = ax2.plot(xs, y2, c='blue')
        ax2.set_ylabel(columns[i]+'值')


        leg1 = ax1.get_legend()
        leg2 = ax2.get_legend()
        plt.legend([p1, p2], labels=['行业销量1', '行业销量2'])
        # ax1.legend([p1, p2], labels=['行业销量1', '行业销量2'])

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

    # 计算装载机各项月度指标M-1到M-2的相关性图表
    get_coorfig_qm(filepath=r'./data/zhuang_wa/analysis_data/zhuang_month_m1-2.csv',
                   outputpath=r'./result_png/zhuang/month',
                   skip_rows=2)

    pass
