from queue import Queue, LifoQueue
from threading import Thread

import datetime
import json
import time
import os

import ccxt.kucoin
import numpy as np


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
        self.past_trades = []
        self.log = []
        self.signals = []
        self.ssid = int(time.time() * 1000)
        self.ex = exchange(config=args)
        self.ticker_interval = 5
        self.ticker_queues = dict()
        self.number_of_pairs = 1
        self.saving_batch_size = 32
        self.latency_logging = True
        self.latency_queue = LifoQueue()
        self.stakes = dict()
        self.dirs = {}
        base = os.path.dirname(os.path.realpath(__file__))
        self.dirs['log'] = os.path.join(base, 'log')
        for v in self.dirs.values():
            if not os.path.isdir(v):
                os.mkdir(v)
        self.dirs['base'] = base
        self.signals_path = os.path.join(self.dirs['log'], f'signals-{self.ssid}.txt')
        if not os.path.isfile(self.signals_path):
            open(self.signals_path, 'w').write("")
        

    def set_ticker_interval(self, interval:float) -> None:
        self.ticker_interval = interval


    def set_number_of_pairs(self, number_of_pairs:int) -> None:
        self.number_of_pairs = number_of_pairs
    

    def set_saving_batch_size(self, saving_batch_size:int) -> None:
        self.saving_batch_size = saving_batch_size

    def set_latency_logging(self, latency_logging:bool) -> None:
        self.latency_logging = latency_logging
    

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
        latencies = list()
        while True:
            if not q.empty():
                n = q.get()
                ts = n[0]
                if ts == lastts:
                    continue
                lastts = ts
                now = time.time() * 1000
                latencies.append(int(now - ts))
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
                self.log.append(f'[{pair}]+{latencies[-1]}ms || {y[-1]} {emadiff[-1]}')
                # print(f'[{pair}]+{latencies[-1]}ms ||',y[-1], emadiff[-1])
                self.latency_queue.put([ts, int(sum(latencies)/len(latencies))])
                if buy[-1] >= .9:
                    self.signals.append([ts, 'BUY', pair, y[-1]])
                    # open(self.signals_path, 'a').write(json.dumps([ts, 'BUY', pair, y[-1]]) + "\n")
                    signal_queue.put([ts, 'BUY', pair, y[-1]])
                if sell[-1] >= .9: 
                    self.signals.append([ts, 'SELL', pair, y[-1]])
                    # open(self.signals_path, 'a').write(json.dumps([ts, 'SELL', pair, y[-1]]) + "\n")
                    signal_queue.put([ts, 'SELL', pair, y[-1]])


    def do_buy(self, timestamp: int, pair: str, price: int) -> None: 
        msg = f'{pair} buy at {price}'
        self.past_trades.append(f'[{datetime.datetime.fromtimestamp(timestamp / 1000)}] {msg}\n')
        # past_trades_path = os.path.join('past_trades', f'trades-{self.ssid}.txt')
        # open(past_trades_path, 'a').write(f'[{datetime.datetime.fromtimestamp(timestamp / 1000)}] {msg}\n')
        self.stakes[pair] = price


    def do_sell(self, timestamp: int, pair: str, price: int, profit: int) -> None: 
        msg = f'{pair} sell at {price}'
        # past_trades_path = os.path.join('past_trades', f'trades-{self.ssid}.txt')
        self.past_trades.append(f'[{datetime.datetime.fromtimestamp(timestamp / 1000)}] {msg}\n')
        # open(past_trades_path, 'a').write(f'[{datetime.datetime.fromtimestamp(timestamp / 1000)}] {msg}\n')
        # profits_path = os.path.join('past_trades', f'profits-{self.ssid}.txt')
        # open(profits_path, 'a').write(f'{pair} {profit}\n')

    
    def do_buy_live(self) -> None:
        pass


    def do_sell_live(self) -> None:
        pass


    def execute_trades_on_queue(self, q:Queue, stake=0, profit=0) -> None:
        self.profit = 0
        while True: 
            if not q.empty():
                sgnl = q.get()
                now = time.time() * 1000
                sgnl_age = now - sgnl[0]
                print(f'[{sgnl[2]}]', sgnl[1], 'signal is', int(sgnl_age), 'ms in the past')
                if sgnl_age < 5000:
                    if sgnl[2] not in self.stakes:
                        self.stakes[sgnl[2]] = 0
                    stake = self.stakes[sgnl[2]]
                    if sgnl[1] == 'BUY':
                        if stake == 0:
                            self.do_buy(sgnl[0], sgnl[2], sgnl[3])
                    if sgnl[1] == 'SELL':
                        if stake > 0: 
                            pp = sgnl[3] - stake
                            if pp > 0: 
                                self.stakes[sgnl[2]] = 0
                                self.do_sell(sgnl[0], sgnl[2], sgnl[3], pp)
                                self.profit += pp
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


    def latency_bookkeeper(self) -> None: 
        X, Y = [], []
        while True:
            if not self.latency_queue.empty():
                if len(X) < self.saving_batch_size: 
                    avg = self.latency_queue.get()
                    X.append(avg[0])
                    Y.append(avg[1])
                else:
                    latencies_path = os.path.join(self.dirs['log'], f'latencies-{self.ssid}.npz')
                    if os.path.isfile(latencies_path):
                        loaded = np.load(latencies_path)
                        oldX = loaded['a'].tolist()
                        oldY = loaded['b'].tolist()
                        X = oldX + X
                        Y = oldY + Y
                    np.savez_compressed(latencies_path, a=np.array(X), b=np.array(Y))
                    X, Y = [], []
        


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

    def get_profit_per_second(self) -> int:
        t = int(time.time() * 1000) - self.ssid
        return self.profit / t

    def __call__(self) -> None:
        print(self.__dict__)
        signal_queue = Queue()
        pairs = self.get_pairs()
        threads = list()
        try:
            for pair in pairs[:self.number_of_pairs]: 
                t1 = Thread(target=self.populate_ticker_queue, args=(pair, ))
                t1.daemon = True
                t1.start()
                threads.append(t1)
            time.sleep(1)
            for pair in pairs[:self.number_of_pairs]:
                ohlc = self.get_ohlc(pair)
                t2 = Thread(target=self.populate_signal_queue, 
                    args=(10, 45, pair, signal_queue, ohlc, ))
                t2.daemon = True
                t2.start()
                threads.append(t2)
            t3 = Thread(target=self.execute_trades_on_queue, args=(signal_queue, ))
            t3.daemon = True
            t3.start()
            if self.latency_logging:
                t4 = Thread(target=self.latency_bookkeeper)
                t4.daemon = True
                t4.start()
        except KeyboardInterrupt:
            pass
        for th in threads:
            th.join()
        t3.join()
        t4.join()



def server(trader: Trader): 
    from flask import Flask, send_from_directory, jsonify
    app = Flask(__name__)
    import logging
    llg = logging.getLogger('werkzeug')
    llg.disabled = True
    
    @app.route("/js/update.js")
    def javascript():
        return send_from_directory('.', 'update.js')
    @app.route("/")
    def hello_world():
        return """
        <h2>
        <a href="/pps">pps</a><br><br>
        <a href="/log">log</a><br><br>
        <a href="/trades">trades</a><br><br>
        <a href="/profit">profit</a><br><br>
        <a href="/signals">signals</a><br><br>
        </h2>"""
    @app.route("/pps")
    def pps():
        return jsonify(trader.get_profit_per_second())
    @app.route("/log")
    def log():
        return jsonify(trader.log)
    @app.route("/trades")
    def trades():
        return jsonify(trader.past_trades)
    @app.route("/profit")
    def profit():
        return jsonify(trader.profit)
    @app.route("/signals")
    def signals():
        return jsonify(trader.signals)
    app.run(host='0.0.0.0', port=5566)



if __name__ == '__main__': 
    trader = Trader(ccxt.kucoin)
    trader.set_ticker_interval(5.0)
    trader.set_number_of_pairs(5)
    trader.set_saving_batch_size(64)
    trader.set_latency_logging(False)
    s = Thread(target=server, args=(trader, ))
    s.start()
    trader()
    s.join()