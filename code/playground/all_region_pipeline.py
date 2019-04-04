# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error,mean_absolute_error
#
# from sklearn.preprocessing import StandardScaler
#
#
# import geopandas as gpd
# from shapely.geometry import Point
import pandas as pd

#
# import os
# import re
import numpy as np

#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


new_time_df_new = pd.read_csv('../output/region_by_time_series_scale_new_with_index.csv',index_col=0)

## code snippet: https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/
split_size =0.7
time_window = 3
#creating dataframe
x_train, y_train = np.empty(shape=[0, time_window, 1]),np.empty(shape=[0,1])
X_test, Y_test = np.empty(shape=[0, time_window, 1]),np.empty(shape=[0,1])
print('begin loop')
for i, row in new_time_df_new.iterrows():
    scaled_data = row.T.to_frame()

    labels = scaled_data.index # add label

    #creating train and test sets
    dataset = scaled_data.values

    size = int(len(dataset) * split_size)

    train, test = dataset[0:size], dataset[size:len(dataset)]

    test_label =labels[size:len(labels)]# add label
    #converting dataset into x_train and y_train

    sub_x_train, sub_y_train = [], []
    for i in range(time_window,len(train)):
        sub_x_train.append(train[i-time_window:i])
        sub_y_train.append(train[i])
    sub_x_train, sub_y_train = np.array(sub_x_train), np.array(sub_y_train)
    x_train = np.concatenate((x_train, sub_x_train), axis=0)
    y_train = np.concatenate((y_train, sub_y_train), axis=0)

    # create test dataset predicting values, using past time_window from the train data
    inputs = dataset[len(dataset) - len(test) - time_window:]
    inputs = inputs.reshape(-1,1)

    sub_X_test = []
    sub_Y_test = []
    indexes = list()
    for i in range(time_window,inputs.shape[0]):
        sub_X_test.append(inputs[i-time_window:i])
        sub_Y_test.append(inputs[i])
    #     indexes.append(test_label[i])# add label
    sub_X_test,sub_Y_test = np.array(sub_X_test), np.array(sub_Y_test)

    X_test = np.concatenate((X_test, sub_X_test), axis=0)
    Y_test = np.concatenate((Y_test, sub_Y_test), axis=0)

print('finish loop')

print(x_train.shape)
print(x_train[0,:,:])

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=16, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=16))
model.add(Dense(1))
print('finish add layer on model')
model.compile(loss='mean_squared_error', optimizer='adam')
print('finish complie model')
model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=2)

print('finish train')
predict_speed = model.predict(X_test)
lstm_mse = mean_squared_error(Y_test, predict_speed)
lstm_mae = mean_absolute_error(Y_test, predict_speed)
print('Test RMSE: %.3f' % np.sqrt(lstm_mse))
print('Test MAE: %.3f' % lstm_mae)
