import ccxt
import json


print('ccxt version:', ccxt.__version__)


class Api:
    def __init__(self, exchange='kucoin', key='key.json', verbose=False) -> None:
        key = json.loads(open(key, 'r').read())
        config = {
            'apiKey': key['apiKey'],
            'secret': key['secret'],
            'passphrase': key['passphrase'],
            'password': key['passphrase'],
            'timeout': 50000,
            'enableRateLimit': True,
            'verbose': verbose,
        }
        for ex in ccxt.exchanges:
            if exchange == ex:
                _exchange = getattr(ccxt, ex)
                break
        if verbose:
            print(config)
        self.ex = _exchange(config=config)

    def get_pairs(self, curr='usdt') -> list:
        return [x for x in list(self.ex.load_markets().keys()) if curr.lower() in x.split('/')[1].lower()]

    def get_accounts(self) -> dict:
        return self.ex.fetch_accounts()

    def parse_tf(self, timeframe) -> int:
        return self.ex.parse_timeframe(timeframe)
