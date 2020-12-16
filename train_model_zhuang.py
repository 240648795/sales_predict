# -*- coding: utf-8 -*-
import random
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


def create_interpolate_data(filename, outputfilename):
    # data = pd.read_csv(r'./data/zhuang_index/raw_index_without2feature.csv.csv')
    data = pd.read_csv(filename)
    # 不需要第一列，因为是序号
    data = data.drop(data.columns[0], axis=1)
    for col in data.columns:
        if col != '日期':
            data[col] = data[col].interpolate()
    data.to_csv(outputfilename, index=None)


def get_correlation_matrix(filename, threshold):
    data = pd.read_csv(filename)
    data_to_cal_col = []
    for i, val in enumerate(data.columns.values):
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    data_to_cal = data_to_cal.dropna()
    # print(data_to_cal.columns)

    correlation_matrix = np.corrcoef(data_to_cal, rowvar=0)
    rs_correlation_matrix = correlation_matrix[0]
    # print(rs_correlation_matrix)
    need_columns = np.argwhere(abs(rs_correlation_matrix) > threshold)
    need_columns = need_columns.flatten()
    return_columns = data_to_cal.columns[need_columns].values.tolist()
    # 第一列是总需求，不需要
    return_columns.pop(0)
    return return_columns


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

def train(filename, needcol, savemodel, savedetail):
    data = pd.read_csv(filename)
    date = data['日期']
    data = data.drop(data.columns[0], axis=1)
    data_to_cal_col = []
    for val in data.columns.values:
        if (val != '日期' and val in needcol) or val == '总需求量':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]

    min_max_scaler_model = MinMaxScaler()
    data = min_max_scaler_model.fit_transform(data_to_cal)

    X = data[:, 1:]
    y = data[:, 0]

    # random_state为None用于衡量准确率
    # avg_r2 = []
    # for i in range(0,50):
    #     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)
    #     model = RandomForestRegressor()
    #     model.fit(x_train, y_train)
    #     avg_r2.append(model.score(x_test, y_test))
    #     # print (train_test_split.random_state)
    # avg_mean = np.mean(avg_r2)
    # print("随机森林回归的评估值r2为：", avg_mean)

    # random_state=30
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    print("随机森林回归的评估值r2为：", model.score(x_test, y_test))
    joblib.dump(model, savemodel)
    joblib.dump(min_max_scaler_model, savedetail)


def predict(filename, needcol, savemodel, savedetail, save_png):
    data = pd.read_csv(filename)
    date = data['日期']
    data = data.drop(data.columns[0], axis=1)

    data_to_predict_col = []
    for val in data.columns.values:
        if (val != '日期' and val in needcol) or val == '总需求量':
            data_to_predict_col.append(val)
    data_to_predict = data[data_to_predict_col]

    model = joblib.load(savemodel)
    min_max_scaler_model = joblib.load(savedetail)

    data = min_max_scaler_model.fit_transform(data_to_predict)
    data_len = data.shape[0]
    random_start = random.randint(1, data_len - 13)

    X_predict = data[random_start:random_start + 13, 1:]
    y_predict = model.predict(X_predict)
    y_real = data[random_start:random_start + 13, 0]
    date = date[random_start:random_start + 13]

    real_x_y = np.column_stack((y_real, X_predict))
    predict_x_y = np.column_stack((y_predict, X_predict))

    real_x_y = min_max_scaler_model.inverse_transform(real_x_y)
    predict_x_y = min_max_scaler_model.inverse_transform(predict_x_y)

    # out = np.column_stack(
    #     (real_x_y[:, 0], predict_x_y[:, 0], (real_x_y[:, 0] - predict_x_y[:, 0]) * 100 / real_x_y[:, 0]))
    # out_df = pd.DataFrame(out)
    # out_df.columns = ['真实值', '预测值', '残差']
    # out_df['日期'] = date
    # out_df.to_csv('./data/预测准确率.csv')

    mse = np.mean(abs(real_x_y[:, 0] - predict_x_y[:, 0]))
    print('平均误差:', mse)
    plt.title("真实销量和预测销量", fontproperties=mfont)
    plt.xlabel('x-value')
    plt.ylabel('y-label')
    xs = [datetime.strptime(d, '%Y/%m').date() for d in date]
    plt.gcf().autofmt_xdate()
    p1 = plt.plot(xs, real_x_y[:, 0])
    p2 = plt.plot(xs, predict_x_y[:, 0])
    plt.legend(labels=['实际销量', '预测销量'])
    plt.savefig(save_png)


def predictone(filename, needcol, savemodel, savedetail):
    data = pd.read_csv(filename)
    date = data['日期']
    data = data.drop(data.columns[0], axis=1)
    data_real = data
    data_to_predict_col = []
    for val in data.columns.values:
        if (val != '日期' and val in needcol) or val == '总需求量':
            data_to_predict_col.append(val)
    data_to_predict = data[data_to_predict_col]
    print(data_to_predict)
    model = joblib.load(savemodel)
    min_max_scaler_model = joblib.load(savedetail)

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


if __name__ == '__main__':
    # 数据预处理插值
    # create_interpolate_data(filename=r'./data/zhuang_index/raw_index_without2feature.csv',
    #                         outputfilename='./data/zhuang_index/index_without2feature_interpolate.csv')
    # 初步筛选变量
    # getCorrelationMatrix(r'./data/index/data_interpolate_forwa_coor.csv')
    # 添加M-1,M-2筛选变量，'全国价格指数M-2', '去年同期', '社会融资M-2', '铁矿石原矿产量M-1'
    # getCorrelationMatrix(r'./data/index/data_interpolate_forwa_morecoor.csv')
    # 做出相关性分析图
    # show_coor_fig(r'./data/index/data_interpolate_forwa_morecoor.csv')

    # 这里选择数据
    filepath = r'data/zhuang_index/sichuan_without2feature_interpolate_m1-6.csv'
    savemodel = './models/sichuan_randomforestregressor.pkl'
    savedetail = './models/sichuan_minmaxscaler.pkl'
    savepng = './result_png/sichuan_result.png'
    # 计算相关系数，筛选出相关系数高于0.6的
    return_columns = get_correlation_matrix(filepath, 0.7)
    print(return_columns)
    train(filepath, return_columns, savemodel, savedetail)
    predict(filepath, return_columns, savemodel, savedetail, savepng)

    # predict_path = r'./data/zhuang_index/index_without2feature_interpolate_m1-6_xiaoliang_predict.csv'
    # predictone(predict_path, return_columns, savemodel, savedetail)
