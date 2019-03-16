import pandas as pd
from toolkit import get_fname


def generate_fname_w_path(root_dir, name, attr):
    """Generate file name with whole path"""

    fname = '{}/{}_{}.csv'.format(root_dir, attr, region_id)
    return fname


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

    file_names = get_fname(path)
    new_time_df = pd.DataFrame()

    for name in file_names:
        data = pd.read_csv(path + name)

        # group second data into one time unit.
        unitly_aggr = data[Y].resample(unit).mean()


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
