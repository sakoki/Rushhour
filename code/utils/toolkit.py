

def get_fname(path,contains = '2016'):
    # get file name for all subfiles for the whole ukwac corpus.
    file = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    if contains:
        file = [f for f in file if contains in f]
    file = sorted(file)
    return file
