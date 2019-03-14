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
        print('directory %s exists'%(path))
    else:
        print('Creating new directory...')
        command = 'mkdir -p {}'.format(path)
        os.system(command)

def generate_fname_wPath(DIR, region_id, attr =False):
    """Generate file name with whole path
    """
#     # avoid file name with "'"
#     if "'" in attr:
#         attr = attr.replace("'", "_")
    fname_wPath = '%s/%s_%s.csv'%(DIR,attr,region_id)
    return fname_wPath
