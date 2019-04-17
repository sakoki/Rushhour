import pandas as pd
import os,sys
sys.path.insert(0,os.getcwd()+'/code/utils/')

# own functions

from model_data_prep import plot_lstm,LSTM_base

if __name__ =="__main__":
    file_name = sys.argv[1] # 'region_time_series_filtered.csv'
    output_root = os.getcwd()

    new_time_df_new = pd.read_csv(output_root+'/output/'+file_name, index_col = 0)
    history = LSTM_base(new_time_df_new,split_size = 0.7, time_window = 12,epoch=8)
    plot_lstm(history)