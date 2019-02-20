import pandas as pd
import re


def get_fname(path,contains = '2016'):
    # get file name for all subfiles for the whole ukwac corpus.
    file = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    if contains:
        file = [f for f in file if contains in f]
    file = sorted(file)
    return file


def create_time_df(path, columns=[], Y='SPEED', unit='H', drop=True):
    """Take a directory of files and groups it into frequency level time series




    """

    f_names = get_fname(path)
    new_time_df = pd.DataFrame()

    for user_f in f_names:
        data_user = pd.read_csv(path + user_f)
        df_user = convert_date(data_user, columns=columns)

        # select the column that is the Y
        col_inf = [i for i in df_user.columns if Y in i][0]

        # group second data into days.
        weekly = df_user[col_inf].resample(unit).mean()
        # weekly.plot(style = [':','--','-'])

        # turn a series of data into a row(with dataframe type).
        weekly_transposed = weekly.to_frame(name=user_f.strip('.csv')).transpose()

        # add into final result
        new_time_df = pd.concat([new_time_df, weekly_transposed])
    print('finish create_time_df')
    return new_time_df