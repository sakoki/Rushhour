import geopandas as gpd
import pandas as pd
import re, os
from shapely.geometry import Point

# own functions
from utils.toolkit import get_fname, generate_fname_wPath

def SFDATA_file_cleaner_all(input_dir, output_dir, file_name):
    print('SFDATA_file_cleaner_all:',file_name)
    for fname in file_name:
        SFDATA_file_cleaner(input_dir, output_dir, fname)


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
    print('finish clean file and saved in',output_dir + file_name)

def coordinate_mapper_all(shp_file, input_dir, output_dir, file_name, columns=list(range(0,6))):
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

def coordinate_mapper(census_zone, input_dir, output_dir, file_name, columns=list(range(0,6))):
    """Accepts lat, lon coordinates and maps it to corresponding census polygon

    :param str shp_file: location of GIS boundary shape file
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
    geom = pd.Series(list(zip(coordinates['LONGITUDE'], coordinates['LATITUDE']))).apply(Point)
    coordinates = gpd.GeoDataFrame(coordinates, geometry=geom)
    
    # Check crs of two dataframe match before merging
    coordinates.crs = census_zone.crs

    # specify operation(op) to 'within' to map points that are within polygons
    mapped_coordinates = gpd.sjoin(coordinates, census_zone, op='within')

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
