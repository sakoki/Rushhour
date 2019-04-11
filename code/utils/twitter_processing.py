import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


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


def tweet_coordinate_mapper(shp_file, input_dir, output_dir, file_name, columns, col_time, col_coords, zone=None):
    """Accepts lat, lon coordinates and maps it to corresponding census Polygon

    :param DataFrame shp_file: GIS boundary data table
    :param str input_dir: directory containing input files
    :param str output_dir: directory to save output files
    :param str file_name: name of file
    :param columns: name of columns to keep
    :type columns: list of str
    :param str col_time: column containing datetime object
    :param col_coords: columns containing lon,lat coordinates
    :type cold_coords: Tuple of (str, str)
    :param str zone: specify the timezone to convert datetime object, default None
    :return: table of tweets mapped to corresponding Polygon
    """

    # dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
    coordinates = pd.read_csv(input_dir + file_name,
                              parse_dates=[col_time],
                              # date_parser=dateparse,
                              usecols=columns,
                              infer_datetime_format=True)
    
    # Match time zone
    if zone != None:
        coordinates.loc[:, col_time] = coordinates.apply(lambda row: time_zone_converter(date=row[col_time], zone=zone), axis=1)
    

    # Convert lat & lon points ot Point geometry shape and create a new geopandas dataframe
    geom = pd.Series(zip(coordinates['LONGITUDE'], coordinates['LATITUDE'])).apply(Point)
    coordinates = gpd.GeoDataFrame(coordinates, geometry=geom)

    # Check crs of two dataframe match before merging
    coordinates.crs = shp_file.crs

    # Specify operation(op) to 'within' to map points that are within polygons
    mapped_coordinates = gpd.sjoin(coordinates, shp_file, op='within')

    mapped_coordinates.to_csv(output_dir + 'mapped_' + file_name, index=False)

    return mapped_coordinates