import builtins as __builtin__

import time
import math
from threading import Thread
from queue import Queue

from crypto000.api import Api

import ccxt
import pymongo


def ohlcv_to_dict(ohlcv: list) -> dict:
    return [{'T': x[0], 'O': x[1], 'H': x[2], 'L': x[3], 'C': x[4], 'V': x[5]} for x in ohlcv]


def dict_to_ohlcv(dd: dict) -> list:
    return [dd['T'], dd['O'], dd['L'], dd['H'], dd['C'], dd['V']]


MAX_LIMIT = 1500


print('pymongo version:', pymongo.__version__)


class Database:
    def __init__(self, host: str, db_name: str, username: str, password: str, api: Api) -> None:
        con = pymongo.MongoClient(f'mongodb://{host}:27017', username=username,
                                  password=password, authSource=db_name, authMechanism='SCRAM-SHA-256')
        self.db = con[db_name]
        self.latencies = {}
        self.queues = {'latency': Queue()}
        print('connected to database:', self.db)
        self.api = api

    def init_coll(self, pair, timeframe, limit):
        coll_name = f'{pair}_{timeframe}'
        tf = self.api.parse_tf(timeframe)
        if coll_name in self.db.list_collection_names():
            print(f'Cant init collection that already exists: {coll_name}')
            return
        print(f'init collection {coll_name} with {limit} values')
        coll = self.db[coll_name]
        ohlcv = self.api.get_ohlcv(
            pair, timeframe, since=time.time()*1000-(tf+1)*1000*limit, limit=limit)
        coll.insert_many(ohlcv_to_dict(ohlcv))
        return coll

    def get_coll(self, pair, timeframe):
        return self.db[f'{pair}_{timeframe}']

    def average_latency(self):
        l = list(self.latencies.values())
        if len(l) > 0:
            return sum(l)/len(l)
        return 0

    def fetch_ticker(self, pair, latency=True):
        start = time.time()
        tk = self.api.get_ticker(pair)
        end = time.time()
        self.queues['latency'].put((end, end-start))
        # self.latencies[end] = end - start
        return tk

    def data(self, pair, timeframe, limit):
        coll = self.db[f'{pair}_{timeframe}']
        cc = coll.estimated_document_count()
        if limit > cc:
            prepend = limit - cc
            print(
                f'the requested {limit} exceeds database size of {cc} by {prepend}, downloading more data')
            if prepend > MAX_LIMIT:
                mod = prepend % MAX_LIMIT
                for _ in range((prepend - mod) // MAX_LIMIT):
                    self.builder(pair, timeframe,
                                 prepend=MAX_LIMIT, check_db=False)
                prepend = mod
            self.builder(pair, timeframe, prepend=prepend, check_db=False)
        return list(reversed(list(coll.find().sort([('T', pymongo.DESCENDING)]).limit(limit))))

    def bookkeeper(self, pair, timeframe, logQ=None):
        def print(msg, *args):
            if not logQ:
                __builtin__.print(f'[{pair}_{timeframe}] {msg}', *args)
            else:
                logQ.put((msg, *args))

        ticker_to_ohlc_map = {'T': 'timestamp', 'O': 'open',
                              'L': 'low', 'H': 'high', 'C': 'close', 'V': 'baseVolume'}
        coll = self.get_collf(pair, timeframe)
        tf = self.api.parse_tf(timeframe)
        while True:
            last = coll.find_one(sort=[('T', pymongo.DESCENDING)])
            last_t = last['T']
            next_t = last_t + tf * 1000
            now = time.time()
            delta = now-next_t/1000
            # print(delta)
            if delta > tf:
                print(
                    f'{round(delta-(delta%tf), 4)}s is missing, attempting ohlcv fetch')
                limit = math.ceil(delta/tf)
                print(f'requesting {limit}')
                try:
                    ohlcv = self.api.get_ohlcv(
                        pair, timeframe, since=next_t, limit=limit)
                except ccxt.errors.RateLimitExceeded:
                    print('rate limit exceeded, sleeping 30s')
                    time.sleep(30)
                    continue
                if len(ohlcv) == 0:
                    print('no ohlcv data returned, retrying')
                    continue
                if ohlcv[0][0] - next_t != 0:
                    print(ohlcv[0][0] - next_t)
                    print('ohlcv fetch is out of sync, retrying')
                    continue
                coll.insert_many(ohlcv_to_dict(ohlcv))
            else:
                time.sleep(1)
                # wait = abs(delta) - 3 - self.average_latency() * 2
                # if wait > 0:
                #     print(
                #         f'waiting {round(wait, 4)}s latency is {round(self.average_latency()*1000, 4)}ms')
                #     time.sleep(wait)
                # _tk = last
                # while True:
                #     tk = self.fetch_ticker(pair)
                #     for k, v in ticker_to_ohlc_map.items():
                #         tk[k] = tk[v]
                #     if tk['T'] == next_t:
                #         break
                #     if tk['T'] > next_t:
                #         for word in ticker_to_ohlc_map.keys():
                #             x1, y1 = _tk['T'], _tk[word]
                #             x2, y2 = tk['T'], tk[word]
                #             x = next_t
                #             y = y1 * (1 - (x - x1) / (x2 - x1)) + \
                #                 y2 * (1 - (x2 - x) / (x2 - x1))
                #             # print(f'{word} li: {y1, y, y2}')
                #             tk[word] = y
                #         tk['T'] = next_t
                #         break
                #     _tk = tk
                # new_tk = {}
                # for k in ticker_to_ohlc_map.keys():
                #     new_tk[k] = tk[k]
                # coll.insert_one(new_tk)

            time.sleep(.2)

    def builder(self, pair, timeframe, prepend=0, check_db=True):
        coll = self.get_coll(pair, timeframe)
        tf = self.api.parse_tf(timeframe)
        first = coll.find_one(sort=[('T', pymongo.ASCENDING)])
        if check_db:
            error = 0
            a = list(coll.find().sort([('T', pymongo.ASCENDING)]))
            for i in range(len(a) - 1):
                it = a[i]['T']
                nt = a[i+1]['T']
                if nt - it != tf * 1000:
                    error += 1
                    since = it + tf*1000
                    limit = int((nt-it)/(tf*1000))-1
                    print(
                        f'missing {limit} data points between {it} and {nt}, repairing')
                    while True:
                        ohlcv = self.api.get_ohlcv(
                            pair, timeframe, since=since, limit=limit)
                        if len(ohlcv) == limit:
                            break
                        print('incomplete data, retrying')
                    coll.insert_many(ohlcv_to_dict(ohlcv))
            print(f'{error} database erros')
        if prepend > 0:
            print(f'attempting prepend of {prepend} data points')
            since = first['T'] - tf*prepend*1000
            while True:
                try:
                    ohlcv = self.api.get_ohlcv(
                        pair, timeframe, since=since, limit=prepend)
                except ccxt.errors.RateLimitExceeded:
                    print('rate limit exceeded, sleeping')
                    time.sleep(30)
                    continue
                if len(ohlcv) == prepend:
                    break
                print('incomplete data, retrying')
            coll.insert_many(ohlcv_to_dict(ohlcv))
            if (first['T'] - coll.find_one(sort=[('T', pymongo.ASCENDING)])['T']) / 60000 == prepend:
                print('prepend success')

    def loop(self):
        lq = self.queues['latency']
        while True:
            if not lq.empty():
                o = lq.get()
                self.latencies[o[0]] = o[1]
            time.sleep(.2)


if __name__ == '__main__':

    host = '62.171.165.127'
    username = 'admin1'
    password = 'kbq6v=d%3xk@MD2*js6w'
    db_name = 'crypto000'
    db = Database(host, db_name, username, password, Api())
    tt = []
    for pair in db.api.get_pairs()[:3]:
        db.init_coll(pair, '1m', 100)
        bkq = Queue()
        db.queues[pair] = {'bookkeeper_log': bkq}
        t1 = Thread(target=db.bookkeeper, args=(pair, '1m', ))
        t1.daemon = True
        tt.append(t1)
    try:
        for t in tt:
            t.start()
        db.loop()
    except KeyboardInterrupt:
        input()
    for t in tt:
        t.join()
