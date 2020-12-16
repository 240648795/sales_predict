import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.utils import plot_model


def data_process(dataset_path, split_rate, infer_seq_length):
    data = pd.read_csv(dataset_path)
    date = data['日期']

    data_to_cal_col = []
    for val in data.columns.values:
        if val != '日期':
            data_to_cal_col.append(val)
    data_to_cal = data[data_to_cal_col]
    data_to_cal = data_to_cal[['全国行业销量（台）M', '日本出口_数值', '固定投资额M-2', '全国_原煤价格', '去年同期']]

    # 插值
    for col in data_to_cal:
        data_to_cal[col] = data_to_cal[col].interpolate()
    # 从11年开始
    data_to_cal = data_to_cal[12:]
    min_max_scaler_model = MinMaxScaler()
    data = min_max_scaler_model.fit_transform(data_to_cal)

    rs = []
    for i in range(data.shape[0] - infer_seq_length):
        rs.append(data[i:i + infer_seq_length + 1].tolist())
    rs = np.array(rs)
    X_train, y_train = rs[:int(rs.shape[0] * split_rate), :-1], rs[:int(rs.shape[0] * split_rate), -1]
    X_test, y_test = rs[int(rs.shape[0] * split_rate):, :-1], rs[int(rs.shape[0] * split_rate):, -1]
    return X_train, y_train, X_test, y_test


def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    epochs = 100
    batch_size = 5
    validation_split = 0.1
    save_path = "./index/ltsm_model.h5"
    X_train, y_train, X_test, y_test = data_process(r'./data/index/data_interpolate.csv', split_rate=0.9,
                                                    infer_seq_length=5)
    print(X_train.shape, y_train.shape)

    print(X_train[0], y_train[0])

    # model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    # model.fit(X_train, y_train, batch_size=batch_size,
    #           epochs=epochs, validation_split=validation_split)
    # model.save(save_path)
