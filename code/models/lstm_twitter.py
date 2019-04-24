import pandas as pd
import os,sys
sys.path.insert(0,os.getcwd()+'/code/utils/')

# own functions

from model_data_prep import data_prepare_lstm,LSTM_base,plot_lstm,average_nb_ts
import numpy as np



if __name__ =="__main__":
    split_size = 0.7
    sliding_window = 12
    file_name = sys.argv[1] # 'region_time_series_filtered.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)
    print(new_time_df_new.shape)
    length_col = len(new_time_df_new)
    print(length_col)

    # # generate neighbor time series data
    # nbs_df = average_nb_ts(new_time_df_new)
    # print('finish generate neighbor df')
    
    twitter_df= pd.read_csv('output/tweet_features_by_hour.csv',index_col=0).loc['tweet_count'].to_frame().T
    print(twitter_df.shape)
    twitter_df=pd.concat([twitter_df]*length_col,ignore_index=True)
    print(twitter_df.shape)
    # generate target time series train data
    x_train, y_train, X_test, Y_test = data_prepare_lstm(new_time_df_new, split_size=split_size, time_window=sliding_window)
    x_train_nb, _, X_test_nb, _ = data_prepare_lstm(twitter_df, split_size=split_size, time_window=sliding_window)

    x_train_agg = np.concatenate((x_train, x_train_nb), axis=1)
    y_train_agg = y_train
    X_test_agg = np.concatenate((X_test, X_test_nb), axis=1)
    Y_test_agg = Y_test

    print(x_train_agg.shape)
    history, lstm_rmse,lstm_mae = LSTM_base(x_train_agg, y_train_agg, X_test_agg, Y_test_agg, epoch=8)
    plot_lstm(history,attr='_twitter')