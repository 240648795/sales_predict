# -*- coding: utf-8 -*-
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
mfont = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def create_interpolate_data():
    data = pd.read_csv(r'./data/index/temp_data.csv')
    # 不需要第一列
    data = data.drop(data.columns[0], axis=1)
    data = data.drop('开工率', axis=1)
    data = data.drop('全国_采矿业投资额', axis=1)
    data = data.drop('全国道路', axis=1)
    # 选取11年8月（第二十行开始）至17年12（第95行）因为有空行
    data = data[20:]
    for col in data.columns:
        if col != '日期':
            data[col] = data[col].interpolate()
    # print(data)
    data.to_csv('./data/index/data_interpolate.csv')


def getCorrelationMatrix(filename):
    data = pd.read_csv(filename)
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    data_to_cal = data_to_cal.dropna()
    correlation_matrix = np.corrcoef(data_to_cal, rowvar=0)
    print(correlation_matrix[1])


def show_coor_fig(filepath):
    data = pd.read_csv(filepath)
    data = data[['全国行业销量（台）M', '全国价格指数M-2', '去年同期', '社会融资M-2', '铁矿石原矿产量M-1', '日期']]
    data.columns = ['全国行业销量（台）M', '水泥价格M-2', '销量M-12', '社会融资M-2', '铁矿石原矿产量M-1', '日期']
    # data = data[['全国行业销量（台）M', '全国价格指数M-2', '去年同期', '社会融资M-2', '铁矿石原矿产量M-1', '日期']]
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
        plt.title("全国行业销量-" + columns[i], fontproperties=mfont)
        plt.xlabel('x-value')
        plt.ylabel('y-label')
        xs = [datetime.strptime(d, '%Y/%m/%d').date() for d in date]
        p1 = plt.plot(xs, y1)
        p2 = plt.plot(xs, y2)
        plt.gcf().autofmt_xdate()
        plt.legend([p1, p2], labels=['全国行业销量', columns[i]])
        plt.show()


def train():
    data = pd.read_csv(r'./data/index/data_interpolate_forzhuang.csv')
    date = data['日期']
    data = data.drop(data.columns[0], axis=1)
    data_to_cal_col = []
    for val in data.columns.values:
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    # data_to_cal = data_to_cal[['全国行业销量（台）M', '全国价格指数M-2', '去年同期', '社会融资M-2', '铁矿石原矿产量M-1']]
    # data_to_cal = data_to_cal[['全国行业销量（台）M', '去年同期']]
    data_to_cal = data_to_cal[['全国行业销量（台）M', '日本出口_数值', '固定投资额M-2', '全国_原煤价格', '去年同期']]

    min_max_scaler_model = MinMaxScaler()
    data = min_max_scaler_model.fit_transform(data_to_cal)

    X = data[:, 1:]
    y = data[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    print("随机森林回归的评估值r2为：", model.score(x_test, y_test))
    joblib.dump(model, './models/RandomForestRegressor.pkl')
    joblib.dump(min_max_scaler_model, './models/MinMaxScaler.pkl')


def predict():
    data = pd.read_csv(r'./data/index/data_interpolate_forzhuang_test_2020.csv')
    date = data['日期']
    data = data.drop(data.columns[0], axis=1)
    data_real = data
    data_to_predict_col = []
    for val in data.columns.values:
        if val != '日期':
            data_to_predict_col.append(val)
    data_to_predict = data[data_to_predict_col]
    # data_to_predict = data_to_predict[['全国行业销量（台）M', '全国价格指数M-2', '去年同期', '社会融资M-2', '铁矿石原矿产量M-1']]
    # data_to_predict = data_to_predict[['全国行业销量（台）M', '去年同期']]
    data_to_predict = data_to_predict[['全国行业销量（台）M', '日本出口_数值', '固定投资额M-2', '全国_原煤价格', '去年同期']]

    model = joblib.load(r'./models/RandomForestRegressor.pkl')
    min_max_scaler_model = joblib.load(r'./models/MinMaxScaler.pkl')

    data = min_max_scaler_model.fit_transform(data_to_predict)
    X_predict = data[:, 1:]
    y_predict = model.predict(X_predict)
    y_real = data[:, 0]

    real_x_y = np.column_stack((y_real, X_predict))
    predict_x_y = np.column_stack((y_predict, X_predict))

    real_x_y = min_max_scaler_model.inverse_transform(real_x_y)
    predict_x_y = min_max_scaler_model.inverse_transform(predict_x_y)

    out = np.column_stack(
        (real_x_y[:, 0], predict_x_y[:, 0], (real_x_y[:, 0] - predict_x_y[:, 0]) * 100 / real_x_y[:, 0]))
    out_df = pd.DataFrame(out)
    out_df.columns = ['真实值', '预测值', '残差']
    out_df['日期'] = date
    out_df.to_csv('./data/预测准确率.csv')

    plt.title("真实销量和预测销量", fontproperties=mfont)
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    xs = [datetime.strptime(d, '%Y/%m/%d').date() for d in date]
    plt.gcf().autofmt_xdate()
    p1 = plt.plot(xs, real_x_y[:, 0])
    p2 = plt.plot(xs, predict_x_y[:, 0])
    plt.legend(labels=['实际销量', '预测销量'])
    print (predict_x_y[:, 0])
    print (real_x_y[:, 0])
    plt.show()


if __name__ == '__main__':
    # create_interpolate_data()
    # 初步筛选变量
    # getCorrelationMatrix(r'./data/index/data_interpolate_forwa_coor.csv')
    # 添加M-1,M-2筛选变量，'全国价格指数M-2', '去年同期', '社会融资M-2', '铁矿石原矿产量M-1'
    # getCorrelationMatrix(r'./data/index/data_interpolate_forwa_morecoor.csv')
    # 做出相关性分析图
    # show_coor_fig(r'./data/index/data_interpolate_forwa_morecoor.csv')

    train()
    predict()
