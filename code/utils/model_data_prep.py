from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
import os,sys

sys.path.insert(0,os.getcwd()+'/code/utils/')
from toolkit import load_pickle

def data_prepare_lstm(new_time_df_new,split_size = 0.7, time_window = 12):
    ## code snippet: https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/

    # creating dataframe
    x_train, y_train = np.empty(shape=[0, time_window, 1]), np.empty(shape=[0, 1])
    X_test, Y_test = np.empty(shape=[0, time_window, 1]), np.empty(shape=[0, 1])
    for i, row in new_time_df_new.iterrows():
        scaled_data = row.T.to_frame()

        labels = scaled_data.index  # add label

        # creating train and test sets
        dataset = scaled_data.values

        size = int(len(dataset) * split_size)

        train, test = dataset[0:size], dataset[size:len(dataset)]

        # test_label = labels[size:len(labels)]  # add label
        # converting dataset into x_train and y_train

        sub_x_train, sub_y_train = [], []
        for i in range(time_window, len(train)):
            sub_x_train.append(train[i - time_window:i])
            sub_y_train.append(train[i])
        sub_x_train, sub_y_train = np.array(sub_x_train), np.array(sub_y_train)
        x_train = np.concatenate((x_train, sub_x_train), axis=0)
        y_train = np.concatenate((y_train, sub_y_train), axis=0)

        # create test dataset predicting values, using past time_window from the train data
        inputs = dataset[len(dataset) - len(test) - time_window:]
        inputs = inputs.reshape(-1, 1)

        sub_X_test = []
        sub_Y_test = []
        indexes = list()
        for i in range(time_window, inputs.shape[0]):
            sub_X_test.append(inputs[i - time_window:i])
            sub_Y_test.append(inputs[i])
        #     indexes.append(test_label[i])# add label
        sub_X_test, sub_Y_test = np.array(sub_X_test), np.array(sub_Y_test)

        X_test = np.concatenate((X_test, sub_X_test), axis=0)
        Y_test = np.concatenate((Y_test, sub_Y_test), axis=0)

    return x_train, y_train, X_test, Y_test

def LSTM_base(x_train, y_train, X_test, Y_test, epoch = 20):

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=16, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=16))
    model.add(Dense(1))
    print
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=epoch, validation_split=0.25, batch_size=16, verbose=2)
    predict_speed = model.predict(X_test)
    lstm_rmse = np.sqrt(mean_squared_error(Y_test, predict_speed))
    lstm_mae = mean_absolute_error(Y_test, predict_speed)
    print('Test RMSE: %.3f' % lstm_rmse)
    print('Test MAE: %.3f' % lstm_mae)
    return history, lstm_rmse,lstm_mae

def plot_lstm(history,attr = ''):


    print(history.history.keys())

    # np.sqrt("Loss") = rmse
    rmse = np.sqrt(history.history['loss'])
    val_rmse = np.sqrt(history.history['val_loss'])
    plt.plot(rmse)
    plt.plot(val_rmse)
    plt.title('model RMSE')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('output/lstm_loss%s.png'%(attr))
    plt.show()

    # "MAE"
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('output/lstm_mae%s.png'%(attr))
    plt.show()

def normalize(df, with_std=False):
    # fill na
    df = df.fillna(0)
    # scale data. output: ndarray
    scaler = StandardScaler(with_std=with_std)

    score_scale = scaler.fit_transform(df)
    # turn scaled data into df.
    df_scale = pd.DataFrame(score_scale, index=df.index, columns=df.columns)

    return df_scale

def weighted_avg(criteria_scores,region_list, debug=None):
    """
    for the purpose of evaluating one model one region model(ARIMA/LSTM)
    :param criteria_scores: a list of list of mae/mse/etc scores.
    :return:
    """

    # weighted average
    weights = load_pickle("region_weights.p", path='output/')


    # get ordered subset of weight value
    weight_list = [weights[k] for k in region_list if k in weights]


    if debug:
        weight_list = list(weights.values())[: debug]

    pprint.pprint(weight_list)
    weighted_scores = []
    for score_list in criteria_scores:
        weighted_scores.append(np.average(score_list, weights=weight_list))
    return weighted_scores