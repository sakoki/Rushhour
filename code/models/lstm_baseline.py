import pandas as pd
import os,sys
sys.path.insert(0,os.getcwd()+'/code/utils/')

# own functions
from decorators import timer

from model_data_prep import plot_lstm,LSTM_base,data_prepare_lstm,data_prepare_lstm_update

if __name__ =="__main__":
    file_name = sys.argv[1] # 'region_time_series_filtered.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)
    region_id = list(new_time_df_new.index)
    x_train, y_train, X_test, Y_test = data_prepare_lstm(new_time_df_new,split_size = 0.7, time_window = 12)
    history, lstm_mse,lstm_mae = LSTM_base(x_train, y_train, X_test, Y_test,epoch=8,region_id=region_id)
    plot_lstm(history,attr='2layerWith16units_batch64_ordered_viz')