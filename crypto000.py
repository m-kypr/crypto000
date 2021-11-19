from queue import Queue
from time import sleep, time
from threading import Thread

import numpy as np

import ccxt
import json
import time
import os

class Trader: 
    def __init__(self, exchange) -> None:
        key = json.loads(open('key.json', 'r').read())
        args = {
            'apiKey': key['apiKey'],
            'secret': key['secret'],
            'passphrase': key['passphrase'],
            'password': key['passphrase'],
            'timeout': 50000,
            'enableRateLimit': True,
        }
        self.ssid = int(time.time() * 1000)
        self.ex = exchange(config=args)
        self.ticker_interval = 5
        self.ticker_queues = dict()
        print('exchange: ', exchange, '\nticker_interval: ', self.ticker_interval)
    

    def set_ticker_interval(self, interval:float) -> None:
        self.ticker_interval = interval
    

    @staticmethod
    def ema(x, n):
        alpha = 2 /(n + 1.0)
        alpha_rev = 1-alpha
        n = x.shape[0]

        pows = alpha_rev**(np.arange(n+1))

        scale_arr = 1/pows[:-1]
        offset = x[0]*pows[1:]
        pw0 = alpha*alpha_rev**(n-1)

        mult = x*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        return out


    def populate_signal_queue(self, b:int, e:int, pair: str, signal_queue: Queue, ydata=list(), offset=0) -> None:
        lastts = 0
        if not pair in self.ticker_queues:
            raise Exception(f'No ticker queue for pair {pair}') 
        q = self.ticker_queues[pair]
        while True:
            if not q.empty():
                n = q.get()
                ts = n[0]
                if ts == lastts:
                    continue
                lastts = ts
                now = time.time() * 1000
                """ current price = (buy + ask) / 2
                """
                ydata.append([n[0], 0, n[1], n[2], (n[4] + n[5]) / 2, n[6]]) 
                y = np.array([n[4] for n in ydata])
                emabase = Trader.ema(y, b)
                emaY = Trader.ema(y, e)
                emadiff = emaY - emabase
                emasigndiff = np.diff(np.sign(emadiff))
                sell = ((emasigndiff < 0) * 1).astype('float')
                buy = ((emasigndiff > 0) * 1).astype('float')
                sell[sell == 0] = np.nan
                buy[buy == 0] = np.nan
                # print(y[-3:], buy[-3:], sell[-3:])
                print(f'[{pair}] ticker is', int(now - ts), 'ms in the past || ',y[-1], emadiff[-1])
                if buy[-1] == 1.0:
                    signal_queue.put([ts, 'BUY', pair, y[-1]])
                if sell[-1] == 1.0: 
                    signal_queue.put([ts, 'SELL', pair, y[-1]])



    def execute_trades_on_queue(self, q:Queue, stake=0, profit=0) -> None:
        while True: 
            if not q.empty():
                sgnl = q.get()
                now = time.time() * 1000
                sgnl_age = now - sgnl[0]
                print(f'[{sgnl[2]}]', sgnl[1], 'signal is', int(sgnl_age), 'ms in the past')
                if sgnl_age < 5000:
                    if sgnl[1] == 'BUY':
                        if stake == 0:
                            print(f'[{sgnl[2]}] buy at', sgnl[3])
                            open(os.path.join('past_trades', f'trades-{self.ssid}.txt'), 'a').write(f'buy at {sgnl[2]}\n')
                            stake = sgnl[2]
                    if sgnl[2] == 'SELL':
                        if stake > 0: 
                            pp = sgnl[2] - stake
                            if pp > 0: 
                                stake = 0
                                profit += pp
                                print('sell at', sgnl[2], 'new profit', profit)
                            else:
                                print('no sell because profit negative!!!')


    

    def populate_ticker_queue(self, pair: str) -> None:
        if not pair in self.ticker_queues: 
            self.ticker_queues[pair] = Queue()
        q = self.ticker_queues[pair]
        while True:
            # o = self.ex.fetch_order_book(pair)
            tkk = self.ex.fetch_ticker(pair)
            tk = tkk['info']
            n = [tk['time'], float(tk['high']), float(tk['low']), float(tk['averagePrice']), float(tk['buy']), float(tk['sell']), float(tk['vol']), float(tk['takerFeeRate']), float(tk['makerFeeRate'])]
            q.put(n)
            time.sleep(self.ticker_interval)
    

    def get_pairs(self, curr='usdt') -> list:
        return [x for x in list(self.ex.load_markets().keys()) if curr.lower() in x.split('/')[1].lower()] 


    def get_ohlc(self, pair, limit=10, try_local=True, ohlc_dir='ohlc_json') -> list: 
        pair_filesafe = pair.replace('/', '-')
        pp = os.path.join(ohlc_dir, f'{pair_filesafe}.json')
        if try_local: 
            if os.path.isfile(pp):
                return json.loads(open(pp, 'r').read())
        ohlc = self.ex.fetchOHLCV(pair, limit=limit)
        open(pp, 'w').write(json.dumps(ohlc))
        return ohlc 


if __name__ == '__main__': 
    trader = Trader(ccxt.kucoin)
    pairs = trader.get_pairs()
    numberofpairs = 30
    trader.set_ticker_interval(5.0)
    signal_queue = Queue()
    threads = list()
    try:
        for pair in pairs[:numberofpairs]: 
            t1 = Thread(target=trader.populate_ticker_queue, args=(pair, ))
            t1.daemon = True
            t1.start()
            threads.append(t1)
        sleep(1)
        for pair in pairs[:numberofpairs]:
            ohlc = trader.get_ohlc(pair)
            t2 = Thread(target=trader.populate_signal_queue, 
                args=(10, 45, pair, signal_queue, ohlc, ))
            t2.daemon = True
            t2.start()
            threads.append(t2)
        t3 = Thread(target=trader.execute_trades_on_queue, args=(signal_queue, ))
        t3.daemon = True
        t3.start()
    except KeyboardInterrupt:
        pass
    
    for th in threads:
        th.join()
    t3.join()
