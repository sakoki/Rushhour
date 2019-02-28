import os
from utils.formatter import SFDATA_file_cleaner_all, coordinate_mapper
from utils.toolkit import get_fname

"""
Standard input on Flux:
/output/Dec2016


"""

# get directory
DIR_ROOT = os.getcwd()
INPUT_DIR = DIR_ROOT + '/raw_data/'

change_INPUT_DIR = input(
    "Default INPUT_DIR is %s. sub dir 'sf_speed_data_clean/' and 'sfmta_regions/' will be created under it. Wanna change it? (y/n)" % (
    INPUT_DIR))
if change_INPUT_DIR == 'y':
    INPUT_DIR = DIR_ROOT + input("New INPUT_DIR: %s" % (DIR_ROOT))
else:
    pass

# get directory
DIR_ROOT = os.getcwd()
OUTPUT_DIR = DIR_ROOT + '/output/'
change_INPUT_DIR = input("Default OUTPUT_DIR is %s. Wanna change it? (y/n)" % (OUTPUT_DIR))
if change_INPUT_DIR == 'y':
    OUTPUT_DIR = DIR_ROOT + input("New OUTPUT_DIR: %s" % (DIR_ROOT))
else:
    pass

# create output dir if not exist
if os.path.isdir(OUTPUT_DIR):
    pass
else:
    print('Creating new directory...')
    command = 'mkdir -p {}'.format(OUTPUT_DIR)
    os.system(command)

# ask if need to generate time series region dataframe
gen_ts = input("Do you want to generate time series region dataframe? (y/n)")

if gen_ts == 'y':
    Out_ts_path_fname = input("dataframe fname with path: %s" % (DIR_ROOT))
else:
    Out_ts_path_fname = False


# 1. conduct cleaning step
clean_input_dir = INPUT_DIR
clean_output_dir = OUTPUT_DIR
file_name = get_fname(clean_input_dir)
SFDATA_file_cleaner_all(clean_input_dir, clean_output_dir, file_name)
SHP_FILE = DIR_ROOT + 'raw_data/census_zones/sf/geo_export_8c14e8b6-2d3c-4109-af6b-f6e25cb69f0c.shp'

# 2. conduct mapping function
# coordinate_mapper(SHP_FILE)

