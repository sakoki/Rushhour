import pandas as pd
from toolkit import get_fname


def generate_fname_wPath(DIR, region_id, attr =False):
    """Generate file name with whole path"""

#     # avoid file name with "'"
#     if "'" in attr:
#         attr = attr.replace("'", "_")
    fname_wPath = '%s/%s_%s.csv'%(DIR,attr,region_id)
    return fname_wPath


def normalize(df):
    #fill na
    df = df.fillna(0)
    #scale data. output: ndarray
    score_scale = scaler.fit_transform(df)
    # turn scaled data into df.
    df_scale = pd.DataFrame(score_scale,index=df.index, columns=df.columns)

    return df_scale


def region_by_time_generator(path, Y='SPEED', unit='H'):
    """Format data into a frequency level time series

    Rows are region_ID and columns are time (based on unit specified).
    Speed os aggregated (currently takes the mean) based on the time granularity specified.
    The output is the input for the function `predict_time_series_ARIMA()`.

    .. todo:: datetime conversion no longer necessary as it is handled by `coordinate_mapper()`
        implement `aggregate_func()`

     :param path: input directory containing files of interest
     :param str Y: specification of name of the column to be treated as Y
     :param str unit: specification for time granularity
     :return: formatted table
     :rtype: DataFrame
    """

    print('begin create_time_df')
    f_names = get_fname(path)
    new_time_df = pd.DataFrame()
    for file_name in f_names:
        data_user = pd.read_csv(path+file_name)

        # df_user = convert_date(data_user,columns=columns)
        df_user = data_user

        # select the column that is the Y 
        # col_inf = [i for i in df_user.columns if Y in i][0]
        col_inf = Y

        print('begin resample')
        # group second data into one time unit.
        unitly_aggr = df_user[col_inf].resample(unit).mean()
        # unitly_aggr.plot(style = [':','--','-'])

        print('finish resample')
        # turn a series of data into a row(with dataframe type).
        weekly_transposed = unitly_aggr.to_frame(name=re.sub("time_series_|\.csv", "", file_name))
        weekly_transposed = weekly_transposed.transpose()
        # add into final result
        new_time_df = pd.concat([new_time_df,weekly_transposed])
    print('finish create_time_df')
    new_time_df_scale = normalize(new_time_df)
    return new_time_df_scale


def prediction_table_generator(data, N):
    """Generate subsets of the data split into training values (x) and prediction value (y)

    The resulting DataFrame will contain 3 columns:
    +-----------+---+---+
    | region_ID | x | y |
    +-----------+---+---+
    region_ID --> unique identifier for census zones
    x --> list of training values (columns 1 to N-1)
    y --> prediction value (column N)

    :param DataFrame data: data to be formatted into training
    :param int N: index of prediction value (y)
    :return: prediction table
    :rtype: DataFrame
    """

    # Create series of Region ID's
    region_id = data.iloc[:, 0]

    # Create column containing list of training values
    input_x = data.iloc[:, list(range(1, N))]
    input_x = input_x.apply(lambda x: x.tolist(), axis=1)
    input_x.rename('x', inplace=True)

    # Create Series of y values
    output_y = data.iloc[:, N]
    output_y.rename('y', inplace=True)

    prediction_table = pd.concat([region_id, input_x, output_y], axis=1)

    # uncomment if you want to save files
    # prediction_table.to_csv('data_up_to_{}.csv'.format(N), index=False)

    return prediction_table
