import requests
import requests_ftp
import re

def get_sfmta_data(base_url,local_dir = './'):

    requests_ftp.monkeypatch_session()
    s = requests.Session()
    resp = s.list(base_url)

    content = resp.content.decode()

    c = re.split(' |\ |\n |\r |-r',content)
    c = list(filter(None, c))
    
    print('begin scraping ...')

    list_names = []
    i=0
    for fname in c:
        if 'sfmtaAVLRawData' in fname:

            fname = fname.rstrip()
            list_names.append(fname)
            r = s.get(base_url+fname)
            open(local_dir+fname, 'wb').write(r.content)
            i+=1
            print('%s write file %s'%(i, fname))
    return list_names

base_url = 'ftp://avl-data.sfmta.com/avl_data/avl_raw/'
save_dir = '../raw_data/sfmta/'
#This will automatically download the file into local directory
list_names = get_sfmta_data(base_url,local_dir = save_dir)
