from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
import re
import os
from shapely.geometry import Point
from toolkit import get_fname, generate_fname_wPath

#####
# TO DO: Move to pipeline
def SFDATA_file_cleaner_all(input_dir, output_dir, file_name):
    # print('SFDATA_file_cleaner_all:',file_name)
    for fname in file_name:
        SFDATA_file_cleaner(input_dir, output_dir, fname)
#####

def SFDATA_file_cleaner(input_dir, output_dir, file_name):
    """Reads in SFData GPS/AVL speed data and formats them into proper csv files

    :param str input_dir: directory containing files to clean
    :param str output_dir: directory to save cleaned files
    :param str file_name: name of csv file to format
    """

    with open(input_dir + file_name, 'r') as old_file:
        with open(output_dir + file_name, 'w') as new_file:
            # Apply formatting change to first line of file
            first = 0
            for line in old_file.readlines():
                if first == 0:
                    # Separate the header from the data
                    line = re.sub(r'\t|\n|\s+', '', line)
                    header = re.findall(r'[A-Z+?\_?]+', line)
                    data = re.findall(r'[\-?\d+?\/?\.?\:?]+|\,(?=\,)', line)

                    # Split the date and time
                    date = re.search(r'(\d{2}\/){2}\d{4}', data[1]).group(0)
                    time = re.search(r'(\d{2}\:){2}\d{2}', data[1]).group(0)
                    date = date + ' ' + time

                    # Remove the combined date and time and replace with split format
                    data.remove(data[1])
                    data.insert(1, date)

                    # Replace comma with null space
                    data = [x if x != ',' else '' for x in data]
                    header = (',').join(header)
                    data = (',').join(data)
                    new_file.write(header + '\n')
                    new_file.write(data + '\n')
                    first += 1
                else:
                    if line != '\n':
                        line = line.rstrip()
                        new_file.write(line + '\n')

    print('{} cleaned'.format(file_name))

####
# TO DO: move out of utils folder, move to pipeline since all it is only wrapping the function in a for-loop
def coordinate_mapper_all(shp_file, input_dir, output_dir, file_name, columns=list(range(0, 6))):
    """

    :param str shp_file: location of GIS boundary shape file
    :param input_dir:
    :param output_dir:
    :param str file_name: list name of file

    :return:
    """
    # census_zone = shp_file
    # uncomment if we want to accept location of shape file #
    census_zone = gpd.GeoDataFrame.from_file(shp_file)[['geoid10', 'geometry']]
    for fname in file_name:
        coordinate_mapper(census_zone, input_dir, output_dir, fname, columns=list(range(0, 6)))
####

def coordinate_mapper(shp_file, input_dir, output_dir, file_name, columns=list(range(0, 6))):
    """Accepts lat, lon coordinates and maps it to corresponding census polygon

    :param DataFrame shp_file: GIS boundary data table
    :param str input_dir: directory containing input files
    :param str output_dir: directory to save output files
    :param str file_name: name of file
    """

    # dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
    coordinates = pd.read_csv(input_dir + file_name,
                              parse_dates=['REPORT_TIME'],
                              # date_parser=dateparse,
                              usecols=columns,
                              infer_datetime_format=True)

    # Convert lat & lon points ot Point geometry shape and create a new geopandas dataframe
    geom = pd.Series(zip(coordinates['LONGITUDE'], coordinates['LATITUDE'])).apply(Point)
    coordinates = gpd.GeoDataFrame(coordinates, geometry=geom)

    # Check crs of two dataframe match before merging
    coordinates.crs = shp_file.crs

    # specify operation(op) to 'within' to map points that are within polygons
    mapped_coordinates = gpd.sjoin(coordinates, shp_file, op='within')

    mapped_coordinates.to_csv(output_dir + 'mapped_' + file_name, index=False)

    return mapped_coordinates


def aggregate_to_region(input_path, output_path):
    """ this function take a directory of time series data and target output directory string as input,
    get region-based csv file as a preparation for prediction function later.
    for now, this function is based on the output from 'def mapping_function()'
    only need to run once for data preparation.

    input:
    input_path: a string of the path of input, contains one csv file for each day.
                eg:'../output/sf_speed_data_clean/'
    output_path: a string of the path of output, will contains one csv file for each region
                eg:'../output/sf_speed_data_region/'

    output:
    write files to output_path
    """
    out_file_attr = 'time_series_region'
    f_names = get_fname(input_path, contains='2016')
    # make sure whether the output dir exists or not.

    for fname in f_names:
        day_file = pd.read_csv(input_path + fname)

        # loop though data for each region and open only one region file each time to save memory
        for region_id, group_df in day_file.groupby('geoid10'):  # TODO: may need to change column name
            # make sure whether the region file has already exist
            out_fname = generate_fname_wPath(output_path, attr=out_file_attr, region_id=region_id)
            if os.path.exists(out_fname):
                f = open(out_fname, 'a')
            else:
                f = open(out_fname, 'w+')
                f.write(','.join(day_file.columns) + '\n')

            for _, rows in group_df.iterrows():
                f.write(','.join([str(cell) for cell in list(rows)]) + '\n')

            f.close()
        print('finished %s' % (fname))


def generate_fname_w_path(DIR, region_id, attr=False):
    """Generate file name with whole path"""

    fname_w_path = '%s/%s_%s.csv'%(DIR,attr,region_id)
    return fname_w_path


# TO DO: FIll 0's with column means
def normalize(df, with_std=False):
    # fill na
    df = df.fillna(df.mean())
    print(df.columns)

    # scale data. output: ndarray
    scaler = StandardScaler(with_std=with_std)

    score_scale = scaler.fit_transform(df)
    # turn scaled data into df.
    df_scale = pd.DataFrame(score_scale, index=df.index, columns=df.columns)

    return df_scale


def region_by_time_generator(path, columns=['REPORT_TIME'], Y='SPEED', unit='H', with_std=False, outdir='', outfname = False):
    """take a directory of user files into a frequency level time series.(mean)
    Actually it now returns a pandas series, which is the input of 'def predict_time_series_ARIMA function()'

    Inputs:
    :param path: the output file dir of 'def aggregate_to_region()'
    :param columns: the list column name string that need to convert to date
    :param  Y: a string of one column that need to be treated as Y
    :param unit: time granularity. eg, 'H'
    :param aggregate_func: a function name, specifies how to aggragate data points.

    Output:
    :return: new_time_df.iloc[0]: a pandaframe with time as column,
                         mean(or other aggregate_func()) speed within one hour(or other time granularity) as data.
    """
    print('begin create_time_df')
    f_names = get_fname(path,contains='')
    new_time_df = pd.DataFrame()
    for file_name in f_names:
        data_user = pd.read_csv(path + file_name, parse_dates=columns)
        #         df_user = convert_date(data_user,columns=columns)
        data_user.index = data_user[columns[0]]
        df_user = data_user
        # select the column that is the Y
        # col_inf = [i for i in df_user.columns if Y in i][0]
        col_inf = Y

        # group second data into one time unit.
        unitly_aggr = df_user[col_inf].resample(unit).mean()
        # unitly_aggr.plot(style = [':','--','-'])
        # turn a series of data into a row(with dataframe type).
        weekly_transposed = unitly_aggr.to_frame(name=re.sub("time_series_|\.csv", "", file_name))
        weekly_transposed = weekly_transposed.transpose()

        # add into final result
        new_time_df = pd.concat([new_time_df, weekly_transposed])

    print('finish create_time_df')
    print('df shape:',new_time_df.shape)
    # normalize the data frame
    new_time_df_new = normalize(new_time_df,with_std = with_std)
    if outfname:

        new_time_df_new .to_csv(outdir+outfname, index=True)
        print('successfully output csv file')


    return new_time_df_new


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

