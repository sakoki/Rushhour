import geopandas as gpd
import pandas as pd
import os
import csv
import re
from shapely.geometry import Point
from pytz import timezone
from toolkit import get_fname


def time_zone_converter(date, zone):
    """Convert the datetime object to a specified timezone
    
    :param date: datetime 
    :type date: pandas._libs.tslibs.timestamps.Timestamp
    :param str zone: desired timezone 
    :return: datetime in specified timezone 
    :rtype: pandas._libs.tslibs.timestamps.Timestamp
    """
    
    date_format = '%Y-%m-%d %H:%M:%S %Z'
    date.strftime(date_format)
    date = date.astimezone(timezone(zone))
    return date.strftime('%Y-%m-%d %H:%M:%S')


def tweet_coordinate_mapper(shp_file, input_dir, output_dir, file_name, columns, col_time, zone=None):
    """Accepts lat, lon coordinates and maps it to corresponding census Polygon

    :param DataFrame shp_file: GIS boundary data table
    :param str input_dir: directory containing input files
    :param str output_dir: directory to save output files
    :param str file_name: name of file
    :param columns: name of columns to keep
    :type columns: list of str
    :param str col_time: column containing datetime object
    :param str zone: specify the timezone to convert datetime object, default None
    :return: table of tweets mapped to corresponding Polygon
    """

    # dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
    coordinates = pd.read_csv(input_dir + file_name,
                              parse_dates=[col_time],
                              # date_parser=dateparse,
                              usecols=columns,
                              infer_datetime_format=True)

    # Match time zone to specific timezone
    if zone != None:
        coordinates.loc[:, col_time] = coordinates.apply(lambda row: time_zone_converter(date=row[col_time],
                                                                                         zone=zone), axis=1)

    # Convert lat & lon points ot Point geometry shape and create a new geopandas DataFrame
    geom = pd.Series(zip(coordinates['lon'], coordinates['lat'])).apply(Point)
    coordinates = gpd.GeoDataFrame(coordinates, geometry=geom)

    # Check crs of two dataframe match before merging
    coordinates.crs = shp_file.crs

    # Specify operation(op) to 'within' to map points that are within polygons
    mapped_coordinates = gpd.sjoin(coordinates, shp_file, op='within')

    print("Mapping completed. Size before: {}, Size after: {}".format(coordinates.shape[0],
                                                                      mapped_coordinates.shape[0]))

    mapped_coordinates.to_csv(output_dir + 'mapped_' + file_name, index=False)

    return mapped_coordinates


def tweet_aggregate_by_region(input_dir, output_dir, base_fname):
    """Aggregate files by cooresponding region ID (geoid)

    For each file, the script will partition data by region ID.
    Each region ID file will get updated every time a new file is read.
    Each of the resulting files will contain all data pertaining to a region ID.

    :param str input_dir: directory containing input files
    :param str output_dir: directory to save output files
    :param str base_fname: base name of the output file
    :return: table of all data for one region
    :rtype: DataFrame
    """

    file_names = get_fname(input_dir)

    for file in file_names:
        data = pd.read_csv(input_dir + file)

        # Loop though data for each region
        # open only one region file at a time to save memory
        for region_id, grouped_data in data.groupby('geoid10'):
            output_fname = '{}/{}_{}.csv'.format(output_dir, base_fname, region_id)
            # Check that file already exists
            if os.path.exists(output_fname):
                f = open(output_fname, "a")
                w = csv.writer(f)
            else:
                f = open(output_fname, "w")
                w = csv.writer(f)
                w.writerow(data.columns)
            for i in grouped_data.itertuples():
                line = list(i)[1:]
                # line[3] = re.escape(line[3])
                line[3] = re.sub("\\n", "", line[3])
                w.writerow(line)
            f.close()
        print('Finished partitioning {}'.format(file))


def tweet_region_by_time_generator(path, columns, Y, unit='H', usecols=None):
    """Takes all regional time series data from a directory and aggregates them into one time series at desired time
    frequency

    Where the resulting DataFrame will contain the following columns:
    +-----------+----+----+-----+----+
    | region_ID | T1 | T2 | ... | TN |
    +-----------+----+----+-----+----+

    Each element in the columns T1...TN will be the averaged speed of all speeds recorded in a region at a specific time
    point.

    :param str path: input directory containing files of interest
    :param list columns: name of column to be converted to datetime
    :param str Y: name of column to be treated as the Y
    :param unit: specification of time granularity
    :param list usecols: specification of columns to read
    :return: formatted table
    :rtype: DataFrame
    """

    print("Reading files from directory: {}".format(path))
    file_names = get_fname(path, contains='')
    new_time_df = pd.DataFrame()

    for name in file_names:
        region_data = pd.read_csv(path + name, parse_dates=columns, infer_datetime_format=True, usecols=usecols)
        region_data.index = region_data[columns[0]]

        # Convert tweets to count
        if Y == 'text':
            region_data['text'] = 1

        # group second data into one time unit.
        unit_aggregate = region_data[Y].resample(unit).mean()

        # turn a series of data into a row(with dataframe type).
        unit_aggregate = unit_aggregate.to_frame(name=re.sub("filtered_|time_series_|\.csv", "", name))
        unit_aggregate = unit_aggregate.transpose()

        # add into final result
        new_time_df = pd.concat([new_time_df, unit_aggregate])

    return new_time_df