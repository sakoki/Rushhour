import re
import os
import pickle

def get_fname(path, contains=False):
    file = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    if contains:
        file = [f for f in file if contains in f]
    file = sorted(file)
    return file


def check_dir_exist(path):
    if os.path.isdir(path):
        print('directory %s exists'%(path))
    else:
        print('Creating new directory: %s'%(path))
        command = 'mkdir -p {}'.format(path)
        os.system(command)


def generate_fname_wPath(DIR, attr1, attr2=False):
    """Generate file name with whole path"""
    fname_wPath = '%s/%s_%s.csv'%(DIR,attr2,attr1)
    return fname_wPath


def save_pickle(variable, f_name,path=False):
    pickle.dump(variable, open("%s%s"%(path,f_name),"wb"))
    print('successfully dump %s'%(f_name))

def load_pickle(f_name,path=False):
    variable=pickle.load(open(r"%s%s"%(path,f_name), "rb"))
    print('successfully load %s'%(f_name))
    return variable