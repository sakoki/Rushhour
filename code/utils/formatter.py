import geopandas as gpd
import pandas as pd
import re
from shapely.geometry import Point


# Specify the path for input and output data
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


def coordinate_mapper(shp_file, input_dir, output_dir, file_name, columns=list(range(0,6))):
    """Accepts lat, lon coordinates and maps it to corresponding census polygon

    :param str shp_file: location of GIS boundary shape file
    :param str input_dir: directory containing input files
    :param str output_dir: directory to save output files
    :param str file_name: name of file
    """

    # census_zone = shp_file
    # uncomment if we want to accept location of shape file #
    census_zone = gpd.GeoDataFrame.from_file(shp_file)[['geoid10', 'geometry']]

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
    coordinates.crs = census_zone.crs

    # specify operation(op) to 'within' to map points that are within polygons
    mapped_coordinates = gpd.sjoin(coordinates, census_zone, op='within')

    mapped_coordinates.to_csv(output_dir + 'mapped_' + file_name, index=False)

    return mapped_coordinates
