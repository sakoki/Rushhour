import re
import os



def get_fname(path, contains=None):
    """Retrieve names of all csv files in directory

    :param str path: input directory containing files of interest
    :param str contains: specify key word to identify file of interest
    :return: list of sorted file names
    :rtype: list of str
    """

    file_list = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    if contains:
        file_list = [f for f in file_list if contains in f]
    file_list = sorted(file_list)

    return file_list


def create_time_table(path, columns=[], Y='SPEED', unit='H', drop=True):
    """Take a directory of files and groups it into frequency level time series

    :param str path:
    :param list columns:
    :param str y:
    :param str unit:
    :param bool drop:
    :return: formatted table
    :rtype: DataFrame
    """

    f_names = get_fname(path)
    new_time_df = pd.DataFrame()

    for user_f in f_names:
        data_user = pd.read_csv(path + user_f)
        df_user = convert_date(data_user, columns=columns)

        # select the column that is the Y
        col_inf = [i for i in df_user.columns if Y in i][0]

        # group second data into days.
        weekly = df_user[col_inf].resample(unit).mean()
        # weekly.plot(style = [':','--','-'])

        # turn a series of data into a row(with dataframe type).
        weekly_transposed = weekly.to_frame(name=user_f.strip('.csv')).transpose()

        # add into final result
        new_time_df = pd.concat([new_time_df, weekly_transposed])
    print('finish create_time_df')
    return new_time_df

def check_dir_exist(path):
    if os.path.isdir(path):
        print('directory %s exists'%(path))
    else:
        print('Creating new directory: %s'%(path))
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
