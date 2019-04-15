import requests
import requests_ftp
import re
import urllib
from secret import APP_ID, APP_CODE


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


def reverse_geocode(lon, lat, location='address'):
    """Reverse geocode lon and lat coordinates using the Geocoder API

    The following API is used
    https://developer.here.com/api-explorer/rest/geocoder/reverse-geocode

    :param flaot lon: longitude coordinate
    :param float lat: latitude coordinate
    :param str location: specify country, state, or address (default address)
    ;return: specified information about location
    :rtype: str
    """

    # Encode parameters
    coordinates = str(lat) + ',' + str(lon)
    params = urllib.parse.urlencode({'prox': coordinates,
                                     'mode': 'retrieveAddresses',
                                     'maxresults': 1,
                                     'gen': 9,
                                     'app_id': APP_ID,
                                     'app_code': APP_CODE,
                                     })

    # Contruct request URL
    # url = 'https://geo.fcc.gov/api/census/area?' + params
    url = 'https://reverse.geocoder.api.here.com/6.2/reversegeocode.json?' + params

    # Get response from API
    response = requests.get(url)

    # Parse json in response
    data = response.json()

    if location == 'country':
        return data["Response"]["View"][0]["Result"][0]["Location"]["Address"]["Country"]
    elif location == 'state':
        return data["Response"]["View"][0]["Result"][0]["Location"]["Address"]["State"]
    return data["Response"]["View"][0]["Result"][0]["Location"]["Address"]['Label']


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


