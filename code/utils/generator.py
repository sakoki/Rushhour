from toolkit import get_fname


def generate_fname_wPath(DIR, region_id, attr =False):
    """Generate file name with whole path
    """
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


def region_by_time_generator(path, columns=['REPORT_TIME'],Y = 'SPEED',unit = 'H'):
    """take a directory of user files into a frequency level time series.(mean)
    Actually it now returns a pandas series, which is the input of 'def predict_time_series_ARIMA function()'
    
    Inputs:
    path: the output file dir of 'def aggregate_to_region()'
    columns: the list column name string that need to convert to date
    Y: a string of one column that need to be treated as Y
    unit: time granularity. eg, 'H'
    aggregate_func: a function name, specifies how to aggragate data points.
    
    Output:
    new_time_df.iloc[0]: a pandas series with time as index, 
                         mean(or other aggregate_func()) speed within one hour(or other time granularity) as data.
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


def prediction_table_generator(data, x_idx, y_idx):
    """Generate subsets of the data split into training values (x) and prediction value (y)

    The resulting dataframe will contain 3 columns:

    region_ID | x | y |

    Where:
    region_ID --> unique identifier for census zones
    x --> list of training values
    y --> prediction value

    :param data: data to be formatted into training
    :type data: DataFrame
    :param int x_idx: index setting the limit of training values (x)
    :param int y_idx: index of prediction value (y)
    :return: prediction table 
    :rtype: DataFrame
    """

    # Create series of Region ID's
    region_id = data.iloc[:, 0]

    # Create column containing list of training values
    input_x = data.iloc[:, list(range(1, x_idx))]
    input_x = input_x.apply(lambda x: x.tolist(), axis=1)
    input_x.rename('x', inplace=True)

    # Create Series of y values
    if y == x_idx + 1:
        output_y = data.iloc[:, y_idx]
        output_y.rename('y', inplace=True)
    else:
        print('y_idx must be greater than x_idx by 1')
        return

    prediction_table = pd.concat([region_id, input_x, output_y], axis=1)

    prediction_table.to_csv('data_{}_{}.csv'.format(x_idx, y_idx), index=False)

    return prediction_table