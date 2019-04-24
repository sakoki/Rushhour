import pandas as pd
import os,sys
sys.path.insert(0,os.getcwd()+'/code/utils/')

# own functions

from model_data_prep import data_prepare_lstm,LSTM_base,plot_lstm,weighted_avg,average_nb_ts

if __name__ =="__main__":
    split_size = 0.7
    sliding_window = 12
    file_name = sys.argv[1] # 'region_time_series_filtered.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)
    # # generate neighbor time series data
    # nbs_df = average_nb_ts(new_time_df_new)
    # print('finish generate neighbor df')
    nbs_df= pd.read_csv('output/neighbor_ts.csv',index_col=0)

    rmse_list, mae_list =[], []
    region_id_list = []
    # # for retrieving results:
    # # i=0
    for region_id, row in new_time_df_new.iterrows():
        # # for retrieving results:
        # if i<114:
        #     i+=1
        #     continue
        # print(region_id)

        # targeted region data
        tmp_df = row.to_frame().T
        x_train, y_train, X_test, Y_test = data_prepare_lstm(tmp_df, split_size=split_size,
                                                             time_window=sliding_window)

        #neighbor data
        tmp_df_nb = nbs_df.loc[region_id].to_frame().T
        x_train_nb, _, X_test_nb, _ = data_prepare_lstm(tmp_df_nb, split_size=split_size, time_window=sliding_window)

        # concat together
        x_train_agg = np.concatenate((x_train, x_train_nb), axis=1)
        y_train_agg = y_train
        X_test_agg = np.concatenate((X_test, X_test_nb), axis=1)
        Y_test_agg = Y_test

        _, rmse, mae = LSTM_base(x_train_agg, y_train_agg, X_test_agg, Y_test_agg, epoch=8)
        region_id_list.append(region_id)
        rmse_list.append(rmse)
        mae_list.append(mae)


    # weighted mae, weighted rmse
    weighted_rmse, weighted_mae  = weighted_avg(criteria_scores = [rmse_list, mae_list],region_list=region_id_list)

    print('weighted_rmse:%s\nweighted_mae:%s'%(weighted_rmse,weighted_mae))
