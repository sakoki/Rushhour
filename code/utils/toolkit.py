import pandas as pd
import re
import os



def get_fname(path,contains = '2016'):
    # get file name for all subfiles for the whole ukwac corpus.
    file = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    if contains:
        file = [f for f in file if contains in f]
    file = sorted(file)
    return file


def check_dir_exist(path):
    if os.path.isdir(path):
        pass
    else:
        print('Creating new directory...')
        command = 'mkdir {}'.format(path)
        os.system(command)