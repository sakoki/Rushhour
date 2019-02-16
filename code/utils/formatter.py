import geopandas as gpd
import pandas as pd
import json
import re
import os
from shapely.geometry import Point
from decorators import timer


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


def coordinate_mapper(shp_file, input_dir, output_dir, file_name, how='within'): 
    """Accepts lat, lon coordinates and maps it to corresponding polygon 
    
    :param str shp_file: location of shape file
    :param str input_dir: directory containing input files
    :param str output_dir: directory to save output files
    :param str file_name: name of file 
    :param str how: how to perform the joining
    """
    census_zone = gpd.GeoDataFrame.from_file(shp_file)
        
    # Parse datetime 
    dateparse = lambda dates: [pd.datetime.strptime(d, '%m/%d/%Y %H:%M:%S') for d in dates]
    coordinates = pd.read_csv(input_dir + filename, parse_dates=['REPORT_TIME'], date_parser=dateparse)
    
    # Convert lat & lon points to Point geometry shape 
    geom = coordinates.apply(lambda x: Point(x['LONGITUDE'], x['LATITUDE']), axis=1)
    coordinates = gpd.GeoDataFrame(coordinates, geometry=geom)
    
    # Check crs of two dataframe match before merging
    coordinates.crs = census_zone.crs
    
    # Map coordinates to to census zones 
    # specify operation(op) to 'within' maps points that are within polygons 
    mapped_coordinates gpd.sjoin(coordinates, census_zone, op='within')

    mapped_coordinates.to_csv(output_dir + file_name, index=False)
    
    
