import os
import geopandas as gpd
from utils.traffic_processing import SFDATA_file_cleaner, coordinate_mapper, aggregate_to_region
from utils.toolkit import get_fname, check_dir_exist
from utils.traffic_processing import region_by_time_generator
from utils.arima import predict_time_series_ARIMA

"""
Standard input on Flux:
/output/Dec2016


"""

# Global Variable
YEAR = '2016'

# DEBUG begin
MONTH=''
# === comment begin
#MONTH = '02' # TODO: comment for debug locally
# === DEBUG ends

# get parent directory
DIR_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print('DIR_ROOT:',DIR_ROOT)
INPUT_DIR = DIR_ROOT + '/raw_data/sf_speed_data/test_data_2016/'
OUTPUT_DIR = DIR_ROOT + '/output/'

def cleaning_step(INPUT_DIR,OUTPUT_DIR):
    '''
    # 1. conduct cleaning step
    '''
    clean_input_dir = INPUT_DIR
    clean_output_dir = OUTPUT_DIR + '/clean_%s_%s/'%(MONTH,YEAR)
    check_dir_exist(clean_output_dir)
    print('cleaning_step: ',clean_input_dir, MONTH+YEAR)
    file_name = get_fname(clean_input_dir,contains= MONTH+YEAR) # only filtered files within specific month and year
    SFDATA_file_cleaner_all(clean_input_dir, clean_output_dir, file_name)

    return clean_output_dir

def conduct_mapping(DIR_ROOT,OUTPUT_DIR,clean_output_dir):
    '''
    # 2. conduct mapping function
    '''

    SHP_FILE = DIR_ROOT + '/raw_data/census_zones/sf/geo_export_8c14e8b6-2d3c-4109-af6b-f6e25cb69f0c.shp'
    mapped_out_dir = OUTPUT_DIR + '/mapped_%s_%s/'%(MONTH,YEAR)
    file_name = get_fname(clean_output_dir)
    check_dir_exist(mapped_out_dir) # make sure directory
    coordinate_mapper_all(SHP_FILE, clean_output_dir, mapped_out_dir,file_name)
    return mapped_out_dir

def aggregate_into_region(OUTPUT_DIR,mapped_out_dir):
    '''
    # 3. Aggregate raw data into region seperated files

    '''
    region_output_dir = OUTPUT_DIR+'/region_%s_%s/'%(MONTH,YEAR)
    check_dir_exist(region_output_dir)
    aggregate_to_region(mapped_out_dir,region_output_dir)
    return region_output_dir

def sample_time_series_(region_output_dir,OUTPUT_DIR,Out_ts_path_fname):
    '''
    # 4. sample_time_series = region_by_time_generator(region_dir,columns=['REPORT_TIME'],Y = 'SPEED',unit = 'H')
    '''
    sample_time_series = region_by_time_generator(region_output_dir,columns=['REPORT_TIME'],Y = 'SPEED',unit = 'H',outdir = OUTPUT_DIR, outfname=Out_ts_path_fname)
    sample_time_series = normalize(sample_time_series, with_std=False)
    return sample_time_series
# '''
# # 5. conduct arima baseline
# '''
# predict_time_series_ARIMA()
def main():


    # change_INPUT_DIR = input(
    #     "Default INPUT_DIR is %s. sub dir 'sf_speed_data_clean/' and 'sfmta_regions/' will be created under it. Wanna change it? (y/n)" % (
    #         INPUT_DIR))
    # if change_INPUT_DIR == 'y':
    #     INPUT_DIR = DIR_ROOT + input("New INPUT_DIR: %s" % (DIR_ROOT))
    # else:
    #     pass
    #
    # # get directory
    # DIR_ROOT = os.getcwd()
    # OUTPUT_DIR = DIR_ROOT + '/output/'
    # change_INPUT_DIR = input("Default OUTPUT_DIR is %s. Wanna change it? (y/n)" % (OUTPUT_DIR))
    # if change_INPUT_DIR == 'y':
    #     OUTPUT_DIR = DIR_ROOT + input("New OUTPUT_DIR: %s" % (DIR_ROOT))
    # else:
    #     pass

    # create output dir if not exist

    check_dir_exist(OUTPUT_DIR)

    # ask if need to generate time series region dataframe
    gen_ts = input("Do you want to generate time series region dataframe? (y/n)")

    if gen_ts == 'y':
        Out_ts_path_fname = input("dataframe fname with path(please include .csv suffix): %s" % (OUTPUT_DIR))
    else:
        Out_ts_path_fname = False
    return Out_ts_path_fname

#####
# TO DO: Move to pipeline
def SFDATA_file_cleaner_all(input_dir, output_dir, file_name):
    # print('SFDATA_file_cleaner_all:',file_name)
    for fname in file_name:
        SFDATA_file_cleaner(input_dir, output_dir, fname)
#####

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
if __name__ =="__main__":
    # Out_ts_path_fname = main()
    # clean_output_dir = cleaning_step(INPUT_DIR, OUTPUT_DIR)
    # mapped_out_dir = conduct_mapping(DIR_ROOT,OUTPUT_DIR,clean_output_dir)
    # region_output_dir = aggregate_into_region(OUTPUT_DIR, mapped_out_dir)

    #--- DEBUG begins
    Out_ts_path_fname = 'test0316.csv'
    region_output_dir = OUTPUT_DIR + '/region_%s_%s/' % (MONTH, YEAR)
    print(region_output_dir)
    sample_time_series = sample_time_series_(region_output_dir, OUTPUT_DIR, Out_ts_path_fname)
