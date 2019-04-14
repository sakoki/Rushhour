import requests
import requests_ftp
import re


def get_sfmta_data(base_url, local_dir):
    """Scrapes 2016 data from sfmta website and saves it in specified directory

    :param str base_url: url of website
    :param str local_dir: path to download files
    """

    # Make connection
    requests_ftp.monkeypatch_session()
    s = requests.Session()
    resp = s.list(base_url)

    # Scrape content
    content = resp.content.decode()

    c = re.split(' |\ |\n |\r |-r', content)
    c = list(filter(None, c))

    list_names = []
    for fname in c:
        if 'sfmtaAVLRawData' in fname:
            # specify years of file to keep in the '[]'
            if re.search(r"201[6](?=.csv)", fname):
                print("Downloading {}".format(fname))
                fname = fname.rstrip()
                list_names.append(fname)
                r = s.get(base_url + fname)
                open(local_dir + fname, 'wb').write(r.content)

    return list_names

# Uncomment to run directly in
# get_sfmta_data('ftp://avl-data.sfmta.com/avl_data/avl_raw/', '../../raw_data/sf_speed_data/')

def get_distance(start, stop):
    """Find the distance between two locations via Google distance matrix API

    :param start: starting location
    :param stop: ending location
    :return: distance between the two locations in meters
    :rtype: float
    """
    pass

