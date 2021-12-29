import ccxt
from ccxt import Exchange
import json


print('ccxt version:', ccxt.__version__)


class Api:
    def __init__(self, exchange='kucoin', key='key.json', verbose=False, use_proxy=True) -> None:
        key = json.loads(open(key, 'r').read())
        config = {
            'apiKey': key['apiKey'],
            'secret': key['secret'],
            'passphrase': key['passphrase'],
            'password': key['passphrase'],
            'timeout': 50000,
            'enableRateLimit': True,
            'verbose': False
        }
        if use_proxy:
            import requests as rq
            proxies = json.loads(rq.get(
                'http://62.171.165.127:4000/api/working?n=10').text)
            proxy = proxies[1][1]
            config['proxies'] = {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}',
            }
        for ex in ccxt.exchanges:
            if exchange == ex:
                _exchange = getattr(ccxt, ex)
                break
        if verbose:
            print(config)
        self.ex: Exchange = _exchange(config=config)

    def get_pairs(self, curr='usdt') -> list:
        return [x for x in list(self.ex.load_markets().keys()) if curr.lower() in x.split('/')[1].lower()]

    def get_accounts(self) -> dict:
        return self.ex.fetch_accounts()

    def get_ohlcv(self, pair, timeframe, since, limit, params={}) -> list:
        print('GET OHLCV!!')
        return self.ex.fetch_ohlcv(pair, timeframe, since, limit, params)

    def get_ticker(self, pair):
        return self.ex.fetch_ticker(pair)

    def parse_tf(self, timeframe) -> int:
        return self.ex.parse_timeframe(timeframe)
