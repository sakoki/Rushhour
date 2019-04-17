import pandas as pd

import os,sys
sys.path.insert(0,os.getcwd()+'/code/utils/')

# own functions

from model_data_prep import weighted_avg,LSTM_base

if __name__ =="__main__":


    file_name = sys.argv[1] # 'region_time_series_filtered.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)

    rmse_list, mae_list =[], []
    region_id_list = []

    for region_id, row in new_time_df_new.iterrows():
        tmp_df = row.to_frame().T
        _, rmse, mae = LSTM_base(tmp_df,split_size = 0.7, time_window = 12,epoch=8)
        region_id_list.append(region_id)
        rmse_list.append(rmse)
        mae_list.append(mae)

    # weighted mae, weighted rmse
    weighted_rmse, weighted_mae  = weighted_avg(criteria_scores = [rmse_list, mae_list],region_list=region_id_list)

    print('weighted_rmse:%s\nweighted_mae:%s'%(weighted_rmse,weighted_mae))