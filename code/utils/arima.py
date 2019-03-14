from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error

import seaborn as sns
import matplotlib.pyplot as plt

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
    else:
        plt.plot(indexes, test)
    plt.plot(indexes, predictions, color='red', label='validation data')  #
    plt.legend()
    plt.title(title)

    return plt, mse