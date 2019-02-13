import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import urllib
import json
import re
import os
from shapely.geometry import Point
from decorators import timer

# Specifiy directory containing shape file
census_zone = '../raw_data/census_zones/sf/geo_export_8ddce19a-2ba8-4da5-8c6c-28ee960b9bc6.shp'
census_zone = gpd.GeoDataFrame.from_file(census_zone)

# Specify the path for input and output data
data_intput = '../raw_data/sf_speed_data/'
data_output = '../output/sf_speed_data_clean/'

def file_cleaner(input_dir, output_dir, file_name):
    '''Reads in SF speed data and formats them into proper csv files

    :params str input_dir: directory containing files to format
    :params str output_dir: directory to store formatted files
    :params str file_name: name of csv file to format
    '''

    with open(data_intput + file_name, 'r') as old_file:
        with open(data_output + file_name, 'w') as new_file:
            first = 0
            for line in old_file.readlines():
                # If it is the first line, separate the header from the data
                if first == 0:
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

for file_name in os.listdir(data_intput):
	file_cleaner(data_intput, data_output, file_name)

traffic = pd.read_csv(data_output+file_name, dtype={'REPORT_TIME': str})

geom = traffic.apply(lambda x: Point(x['LONGITUDE'], x['LATITUDE']), axis=1)
traffic = gpd.GeoDataFrame(traffic, geometry=geom)

# Make sure crs matches before merging
sf_census_zones.crs = traffic.crs
mapped_traffic = gpd.sjoin(traffic, sf_census_zones, op='within')

mapped_traffic.to_csv(data_output + file_name, index=False)



