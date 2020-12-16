# -*- coding : utf-8-*-
import warnings
import itertools
from datetime import datetime
from matplotlib.font_manager import FontProperties
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.style.use('fivethirtyeight')
plt.rcParams['font.sans-serif'] = ['SimHei']
mfont = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def cal_pdq_old():
    data = sm.datasets.co2.load_pandas()
    y = data.data
    y = y['co2'].resample('MS').mean()
    y = y.fillna(y.bfill())
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print(seasonal_pdq)

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue


def cal_pdq():
    data = pd.read_csv(r'data/zhuang_index/index_without2feature_interpolate.csv')
    data = data[[data.columns[0], data.columns[1]]]
    data[data.columns[1]] = data[data.columns[1]].interpolate()

    data.columns = ['rtime', 'values']
    data['rtime'] = data['rtime'].apply(lambda x: datetime.strptime(x, '%Y/%m'))
    data = data.set_index('rtime')
    y = data['values']
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    return param, param_seasonal


def get_sarimax_predict(start_time, end_time, order, seasonal_order):
    data = pd.read_csv(r'data/zhuang_index/index_without2feature_interpolate.csv')
    data = data[[data.columns[0], data.columns[1]]]
    data[data.columns[1]] = data[data.columns[1]].interpolate()

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

    print(y_real)
    print(y_pred)
    print(r2_score(y_real, y_pred))
    print(abs(y_real - y_pred).mean())

    p1 = plt.plot(y, label='实际值')
    p2 = plt.plot(y_pred, label='预测值', alpha=.7)
    p3 = plt.fill_between(pred_ci.index,
                          pred_ci.iloc[:, 0],
                          pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.gcf().autofmt_xdate()
    plt.xlabel('时间')
    plt.ylabel('销量')
    plt.show()


if __name__ == '__main__':
    param, param_seasonal = cal_pdq()
    get_sarimax_predict('2018-12-01', '2019-12-01', param, param_seasonal)
