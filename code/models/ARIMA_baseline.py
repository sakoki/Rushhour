from sklearn.metrics import mean_squared_error,mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import os,sys
import pandas as pd

sys.path.insert(0,os.getcwd()+'/code/utils/')

# own functions

from model_data_prep import weighted_avg

# from code snippet: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
def predict_time_series_ARIMA(mean_series, title=None, fig_size=(20, 10), split_size=0.25, full=False):
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
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    print('Region id: %s'%(region_id))
    print('Test RMSE: %.3f' % np.sqrt(rmse))
    print('Test MAE: %.3f' % mae)

    if full==None:
        return rmse, mae
    else:
        # plot
        sns.set(rc={'figure.figsize': fig_size})
        if full:
            plt.plot(labels, X, label='train data')
        #         plt.plot(labels[int(len(labels)/2):],X[int(len(X)/2):])
        elif full == False:
            plt.plot(indexes, test)

        plt.plot(indexes, predictions, color='red', label='validation data')  #
        plt.legend()
        plt.title(title)

        return plt, rmse, mae

if __name__=="__main__":

    file_name = sys.argv[1] # 'region_time_series_filtered.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)

    rmse_list, mae_list =[], []
    region_id_list = []
    #TODO: debug
    i=0
    for region_id, row in new_time_df_new.iterrows():
        rmse, mae = predict_time_series_ARIMA(row, title=region_id,fig_size=None, split_size=0.7, full=None)
        region_id_list.append(region_id)
        rmse_list.append(rmse)
        mae_list.append(mae)

        if i == 2:
            break
        i += 1


    # weighted mae, weighted rmse
    weighted_rmse, weighted_mae  = weighted_avg(criteria_scores = [rmse_list, mae_list],region_list=region_id_list)

    print('weighted_rmse:%s\nweighted_mae:%s'%(weighted_rmse,weighted_mae))

    # plt.plot(rmse_list)
    # plt.title('model RMSE')
    # plt.ylabel('RMSE')
    # plt.xlabel('epoch')
    # plt.legend(['ARIMA_rmse'], loc='upper left')
    # plt.savefig('output/arima_rmse.png')
    # plt.show()
    #
    #
    # plt.title('model MAE')
    # plt.ylabel('MAE')
    # plt.xlabel('epoch')
    # plt.legend(['ARIMA_mae'], loc='upper left')
    # plt.savefig('output/arima_mae.png')
    # plt.show()

