# -*- coding : utf-8-*-
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']
mfont = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def create_interpolate_data(input_file_path, output_file_path):
    data = pd.read_csv(input_file_path)
    for col in data.columns:
        if col != '日期':
            data[col] = data[col].interpolate()
    # data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m')
    data.to_csv(output_file_path, index=False)

def get_lm(row):
    print(row)
    return row
    # d = row['日期'] - relativedelta(months=+1)
    # return d

def create_dislocation_test():
    data = pd.read_csv(r'./data/index/index_forzhuang_notnull.csv')
    data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m')
    data = data[['日期', '全国行业销量']]
    data['Rev'] = data.groupby('日期').apply(
        lambda x: x['全国行业销量'].shift() * (x['日期'].dt.year * 12 + x['日期'].dt.month).diff().eq(1)).replace(0,
                                                                                                        np.nan).values
    print(data)

def get_partial_correlation_one(col1, col2, aim_col):
    data = pd.read_csv(r'./data/zhuang_index/index_forzhuang_notnull_interpolate_main.csv')
    r_ab = data[col1].corr(data[col2])
    r_ac = data[col1].corr(data[aim_col])
    r_bc = data[col2].corr(data[aim_col])
    r_ab_c = (r_ab - r_ac * r_bc) / (((1 - r_ac ** 2) ** 0.5) * ((1 - r_bc ** 2) ** 0.5))
    print(col1, '和', col2, '相对于', aim_col, '的偏相关系数', r_ab_c)

def cal_all_partial_correlation():
    data = pd.read_csv(r'./data/zhuang_index/index_forzhuang_notnull_interpolate_main.csv')
    aim_column = '全国行业销量'
    total_cal_col = data.columns.drop(['日期', '全国行业销量'])
    for cal_col in total_cal_col:
        col1 = cal_col
        temp_cols=total_cal_col.drop([col1])
        for temp_col in temp_cols:
            get_partial_correlation_one(col1,temp_col,aim_column)

if __name__ == '__main__':
    # create_interpolate_data(input_file_path=r'./data/zhuang_index/index_forzhuang_notnull.csv',
    #                         output_file_path='./data/zhuang_index/index_forzhuang_notnull_interpolate2.csv')
    # create_dislocation_test()
    # get_partial_correlation_one('水泥产量','水泥产量M-3','全国行业销量')
    cal_all_partial_correlation()
    pass
