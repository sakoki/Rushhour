import os
from utils.formatter import SFDATA_file_cleaner_all, coordinate_mapper, aggregate_to_region
from utils.toolkit import get_fname, check_dir_exist
from utils.generator import region_by_time_generator
from utils.arima import predict_time_series_ARIMA

"""
Standard input on Flux:
/output/Dec2016


"""

# Global Variable
YEAR = '2016'
MONTH = '02'
# get directory
DIR_ROOT = os.getcwd()
INPUT_DIR = DIR_ROOT + '/raw_data/'
OUTPUT_DIR = DIR_ROOT + '/output/'

def cleaning_step(INPUT_DIR,OUTPUT_DIR):
    '''
    # 1. conduct cleaning step
    '''
    clean_input_dir = INPUT_DIR
    clean_output_dir = OUTPUT_DIR + '/clean_%s_%s/'%(MONTH,YEAR)
    check_dir_exist(clean_output_dir)
    file_name = get_fname(clean_input_dir,contains= MONTH+YEAR) # only filtered files within specific month and year
    SFDATA_file_cleaner_all(clean_input_dir, clean_output_dir, file_name)

    return clean_output_dir

def conduct_mapping(DIR_ROOT,OUTPUT_DIR,clean_output_dir):
    '''
    # 2. conduct mapping function
    '''
    SHP_FILE = DIR_ROOT + 'raw_data/census_zones/sf/geo_export_8c14e8b6-2d3c-4109-af6b-f6e25cb69f0c.shp'
    mapped_out_dir = OUTPUT_DIR + '/mapped_%s_%s/'%(MONTH,YEAR)
    check_dir_exist(mapped_out_dir) # make sure directory
    coordinate_mapper(SHP_FILE, clean_output_dir, mapped_out_dir)
    return mapped_out_dir

def aggregate_into_region(OUTPUT_DIR,mapped_out_dir):
    '''
    # 3. Aggregate raw data into region seperated files

    '''
    region_output_dir = OUTPUT_DIR+'/region_%s_%s/'%(MONTH,YEAR)
    aggregate_to_region(mapped_out_dir,region_output_dir)
    return region_output_dir

def sample_time_series_(region_output_dir,OUTPUT_DIR,Out_ts_path_fname):
    '''
    # 4. sample_time_series = region_by_time_generator(region_dir,columns=['REPORT_TIME'],Y = 'SPEED',unit = 'H')
    '''
    sample_time_series = region_by_time_generator(region_output_dir,columns=['REPORT_TIME'],Y = 'SPEED',unit = 'H',outdir = OUTPUT_DIR, outfname=Out_ts_path_fname)

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
        Out_ts_path_fname = input("dataframe fname with path(please include .csv suffix): %s" % (DIR_ROOT))
    else:
        Out_ts_path_fname = False
    return Out_ts_path_fname


if __name__ =="__main__":
    Out_ts_path_fname = main()
    clean_output_dir = cleaning_step(INPUT_DIR, OUTPUT_DIR)
    mapped_out_dir = conduct_mapping(DIR_ROOT,OUTPUT_DIR,clean_output_dir)
    region_output_dir = aggregate_into_region(OUTPUT_DIR, mapped_out_dir)
    sample_time_series = sample_time_series_(region_output_dir, OUTPUT_DIR, Out_ts_path_fname)
