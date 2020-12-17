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


def create_m_feature(filename, outputfilename,skip_rows):
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


if __name__ == '__main__':
    create_m_feature(r'./data/wa_index/features_month_interpolate.csv',
                     r'./data/wa_index/features_month_interpolate_m1-2.csv',skip_rows=2)
    print('eqweqw')
    pass
