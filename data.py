import requests as rq
import json


def download_data(pair, path='', save=True, since=1, interval=60):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since}"
    r = rq.get(url)
    jj = json.loads(r.text)
    data = jj['result'][list(jj['result'].keys())[0]]
    last = jj['result']['last']
    since = int(last)
    if save:
        if path == '':
            path = f'ohlc_json/{pair}_{interval}'
        open(path, 'w').write(json.dumps(data))
    return data


def get_pairs(curr='usd', path='', save=False):
    url = "https://api.kraken.com/0/public/AssetPairs"
    r = rq.get(url)
    ret = [x for x in list(json.loads(r.text)['result'].keys()) if x[-3:].lower() == curr.lower()]
    if save:
        open(path, 'w').write(json.dumps(ret))
    return ret


def download_all():
    for pair in get_pairs():
        download_data(pair, f'data/{pair}.json')

def local_data(pair, interval=60): 
    return json.loads(open(f'ohlc_json/{pair}_{interval}', 'r').read())


def local_pairs(): 
    return json.loads(open(f'pairs_json/kraken', 'r').read())