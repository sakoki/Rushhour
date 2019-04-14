from sklearn.metrics import mean_squared_error,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import os,sys
import pandas as pd

# from code snippet: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
def predict_time_series_ARIMA(mean_series, title, fig_size, split_size=0.9, full=False):
    labels = mean_series.index  # add label

    X = mean_series.values
    size = int(len(X) * split_size)

    train, test = X[0:size], X[size:len(X)]

    test_label = labels[size:len(labels)]  # add label

    history = [x for x in train]
    predictions = list()
    indexes = list()  # add label
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        indexes.append(test_label[t])  # add label
    #         if t %100 ==0:
    #             print('predicted=%f, expected=%f' % (yhat, obs))
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)
    print('Test RMSE: %.3f' % np.sqrt(mse))
    print('Test MAE: %.3f' % mae)

    # plot
    sns.set(rc={'figure.figsize': fig_size})
    if full:
        plt.plot(labels, X, label='train data')
    #         plt.plot(labels[int(len(labels)/2):],X[int(len(X)/2):])
    elif full == False:
        plt.plot(indexes, test)
    else:#(full == None)
        return mse, mae
    plt.plot(indexes, predictions, color='red', label='validation data')  #
    plt.legend()
    plt.title(title)

    return plt, mse, mae

if __name__=="__main__":

    file_name = sys.argv[1] # 'region_by_time_series.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)

    mse_list, mae_list = [], []
    for region_id, row in new_time_df_new.iterrows():
        mse, mae = predict_time_series_ARIMA(row,  split_size=0.7, full=None)# title=region_id, fig_size=(20, 10),
        mse_list.append(mse)
        mae_list.append(mae)

    plt.plot(mse_list)
    plt.title('model MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['ARIMA_mse'], loc='upper left')
    plt.savefig('output/arima_mse.png')
    plt.show()


    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['ARIMA_mae'], loc='upper left')
    plt.savefig('output/arima_mae.png')
    plt.show()

