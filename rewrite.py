import os
import json
from queue import Queue
import shutil
import math
import time
from typing import Any

from util import _ewma
from database import Database, ohlcv_to_dict

import ccxt
import numpy as np

print('ccxt version:', ccxt.__version__)
DBASE = os.path.dirname(os.path.realpath(__file__))

VERBOSE = False


class Crypto000:
    def __init__(self, exchange: ccxt.Exchange, key: dict) -> None:
        self.DOHLC = os.path.join(DBASE, 'ohlc')
        self.DCYP = os.path.join(DBASE, 'cyp')
        print(self.DCYP)
        # quit()
        # shutil.rmtree(self.DOHLC)
        for d in [self.DOHLC, self.DCYP]:
            if not os.path.isdir(d):
                os.mkdir(d)

        config = {
            'apiKey': key['apiKey'],
            'secret': key['secret'],
            'passphrase': key['passphrase'],
            'password': key['passphrase'],
            'timeout': 50000,
            'enableRateLimit': True,
            'verbose': VERBOSE,
        }
        self.ex = exchange(config=config)
        accs = self.ex.fetch_accounts()
        print('Accounts:', accs)
        self.timeframe = '1m'

    def get_pairs(self, curr='usdt') -> list:
        from util import get_pairs as __get_pairs
        return __get_pairs(self.ex)

    # def get_pairs(self, curr='usdt') -> list:
    #     return [x for x in list(self.ex.load_markets().keys()) if curr.lower() in x.split('/')[1].lower()]

    def get_ohlc(self, pair, limit=100, try_local=True, write_local=True) -> Any:
        pair_path = os.path.join(self.DOHLC, pair.replace('/', '-'))
        if try_local:
            if os.path.isdir(pair_path):
                for tf in os.listdir(pair_path):
                    # TODO: Local caching
                    if int(tf) == self.ex.parse_timeframe(self.timeframe):
                        timeframe_path = os.path.join(pair_path, tf)
                        for f in os.listdir(timeframe_path):
                            ts, l = [int(x) for x in f[:-5].split('-')]
                            te = ts + int(tf) * l
                            last = te - int(tf)
                            data = json.loads(
                                open(os.path.join(timeframe_path, f), 'r').read())
                            # xdata = [x[0] for x in data]
                            X, O, H, L, C, V = map(list, zip(*data))
                            x = list(
                                range(since * 1000, last * 1000, int(tf) * 1000))
                            Q = [O, H, L, C, V]
                            for i in range(len(Q)):
                                Q[i] = np.interp(x, X, Q[i])
                            ohlc = list(map(list, zip(X, *Q)))
                            print(len(ohlc))
                            return

        params = {}
        if self.ex.__class__ == ccxt.kucoin:
            params['endAt'] = 0

        tfparse = self.ex.parse_timeframe(self.timeframe)
        while True:
            # TODO Find interval that was ticked (not all are ticked according to https://docs.kucoin.com/#get-klines)
            # while True:
            #     tk = self.ex.fetch_ticker(pair)
            #     temp = self.ex.fetch_ohlcv(pair, timeframe=self.timeframe, since=math.floor(now - tfparse) * 1000, limit=1, params=params)
            #     print(now - tk['timestamp'] / 1000, len(temp))
            #     if bool(len(temp)): break
            # print(f'found ticked interval {now}')
            now = time.time()
            padding = 1
            since = math.floor(now - (limit + padding) * tfparse)
            ohlc = self.ex.fetch_ohlcv(
                pair, timeframe=self.timeframe, since=since * 1000, limit=limit + padding, params=params)
            if len(ohlc) < limit:
                print(f'ohlc data incomplete {len(ohlc)}<{limit}, retrying...')
                continue
            if len(ohlc) > limit:
                print(f'ohlc data overflow, removing {len(ohlc) - limit}')
                c = len(ohlc) - limit
                ohlc = ohlc[c:]
            break
        if write_local:
            if not os.path.isdir(pair_path):
                os.mkdir(pair_path)
            timeframe_path = os.path.join(pair_path, str(
                self.ex.parse_timeframe(self.timeframe)))
            if not os.path.isdir(timeframe_path):
                os.mkdir(timeframe_path)
            s = int(ohlc[0][0] / 1000)
            # e = int(ohlc[-1][0] / 1000)
            open(os.path.join(timeframe_path,
                 f'{s}-{len(ohlc)}.json'), 'w').write(json.dumps(ohlc))
        return ohlc

    def init_db(self):
        host = '62.171.165.127'
        username = 'admin1'
        password = 'kbq6v=d%3xk@MD2*js6w'
        db_name = 'crypto000'
        self.db = Database(host, db_name, username, password)
        self.db.init_ex(self.ex)

    def test(self, pair, B, E):
        K = 10
        LIMIT = 1500
        q = self.db.data(pair, '1m', LIMIT * K)
        roi = 0
        trades = 0
        for k in range(K):
            _q = q[k*1500:(k+1)*1500]
            X = np.array([x['T'] for x in _q])
            Y = np.array([x['C'] for x in _q])
            b = _ewma(Y, B)
            e = _ewma(Y, E)
            d = b - e
            _t = 0
            for i in range(len(b) - 1):
                if d[i] > 0 and d[i + 1] < 0:
                    if _t == 0:
                        _t = Y[i]
                if d[i] < 0 and d[i+1] > 0:
                    if _t != 0:
                        net = Y[i] - _t
                        roi += net / _t
                        trades += 1
        print(B, E, roi, trades)

    def learn2(self, pair):
        frames = 200
        frame_size = 100
        fee = 0.01
        q = self.db.data(pair, '1m', frame_size * frames)
        X = np.array([x['T'] for x in q])
        Y = np.array([x['C'] for x in q])
        LEARN = {}
        total_frames = frames
        try:
            for fn in os.listdir(self.DCYP):
                if fn.startswith(f'{pair[:-5]}_{frame_size}_'):
                    LEARN = json.loads(
                        open(os.path.join(self.DCYP, fn), 'r').read())
                    loaded_frames = int(fn[-7:-5])
                    total_frames += loaded_frames
                    print(f'loaded learn data, frames={loaded_frames}')
                    break
        except Exception as e:
            print(e)
        BMAX = frame_size // 2
        print(BMAX)
        for f in range(frames):
            print(f'frame={f}')
            DATA = {}
            _Y = Y[f*frame_size:(f+1)*frame_size]
            for B in range(6, BMAX):
                # for i in range(1, len(Y)):
                #     if (B, f) not in DATA:
                if B not in DATA:
                    DATA[B] = _ewma(_Y, B)
                b = DATA[B]
                EMAX = B // 2
                # print(B)
                # print(f'\r{round((B/BMAX)*100, 2)}% {EMAX} ', end='')
                for E in range(3, EMAX):
                    if f'{B},{E}' not in LEARN:
                        LEARN[f'{B},{E}'] = {'roi': 0, 'trades': 1}
                    if E not in DATA:
                        DATA[E] = _ewma(_Y, E)
                    e = DATA[E]
                    d = b - e
                    _t = 0
                    for i in range(len(d)):
                        if d[i-1] > 0 and d[i] < 0:
                            # print(i-1)
                            if _t == 0:
                                _t = Y[i]
                        if d[i-1] < 0 and d[i] > 0:
                            if _t != 0:
                                # print(i-1)
                                net = _Y[i] - _t - _t * fee - _Y[i] * fee
                                LEARN[f'{B},{E}']['roi'] += (net / _t)
                                LEARN[f'{B},{E}']['trades'] += 1
                                _t = 0
            # print(sorted([x['roi']
            #       for x in list(LEARN.values())], reverse=False))
            s = {k: v for k, v in sorted(
                LEARN.items(), key=lambda item: item[1]['roi'])}
            best = list(s.keys())[-3:]
            for x in best:
                print(x, s[x])
            print(s[x]['roi'] / frames)
            print()
        path = os.path.join(
            self.DCYP, f'{pair[:-5]}_{frame_size}_{total_frames}.json')
        print('output:', path)
        open(path, 'w').write(json.dumps(LEARN))
        # worst = list(s.keys())[:10]
        # for x in best[-1:]:
        #     print('best', x, s[x])
        # for x in worst[:1]:
        #     print('worst', x, s[x])
        # print(best[-5:])
        # for x in best:
        #     print(x, s[x], x[1] / x[0])
        # DUMB.append((x, s[x], x[0] / x[1]))
        # best = list(s.keys())[-5:]
        # for x in best:
        #     print(x, s[x], x[1] / x[0])
        # WISE.append((x, s[x], x[0] / x[1]))
        # for x in WISE:
        #     print(x[0][0], 'R:', x[2], 'roi:', x[1]['roi'])
        # print()
        # for x in DUMB:
        #     print(x[0], 'R:', x[2], 'roi:', x[1]['roi'])
        # return WISE[-1][0][0], WISE[-1][0][1]
        # print([x[0][0], x[2] for x in WISE])
        # import matplotlib.pyplot as plt
        # plt.plot(X, DATA[x[0]])
        # plt.plot(X, DATA[x[1]])
        # plt.plot(X, Y)
        # plt.show()

    def learn(self, pairs=1) -> None:
        """Bruteforce best window sizes for EWMA (Exponentially weighted moving average) with OHLC data.
        """
        self.init_db()
        for pair in self.get_pairs()[:pairs]:
            # self.db.builder(pair, '1m')
            self.learn2(pair)
            # self.test(pair, B, E)
            continue
            ohlc = self.get_ohlc(pair, limit=1500, try_local=True)
            DATA = {}
            EWMA = {}
            print(f'Learning B, E values for {pair}')
            for B in range(4, 17):
                DATA[B] = {}

                _y = [x[4] for x in ohlc]
                _x = [x[0] for x in ohlc]

                y = np.array(_y)
                x = np.array(_x)

                _f = B + 1
                _t = B * 45
                print(f'B={B}, trying E from {_f} to {_t}')

                import matplotlib.pyplot as plt
                # f = plt.figure()
                # ax = f.add_subplot(111)

                for E in range(_f, _t + 1):
                    # ax.cla()
                    bsgnl = np.full_like(y, np.nan)
                    ssgnl = np.full_like(y, np.nan)

                    # ax.plot(x, y, 'b-')

                    def b(i):
                        if B not in EWMA:
                            EWMA[B] = {}
                        if i not in EWMA[B]:
                            EWMA[B][i] = _ewma(y[:i], B)
                        return (x[:i], EWMA[B][i])

                    def e(i):
                        if E not in EWMA:
                            EWMA[E] = {}
                        if i not in EWMA[E]:
                            EWMA[E][i] = _ewma(y[:i], E)
                        return (x[:i], EWMA[E][i])

                    H = E + 1
                    # lineB, = ax.plot(*b(H), 'g-')
                    # lineE, = ax.plot(*e(H), 'r-')
                    # scatterBuy = ax.scatter(x, bsgnl)
                    # scatterSell = ax.scatter(x, ssgnl)
                    status = False
                    bought = 0
                    trades = []
                    profit = 0
                    roi = 0
                    for i in range(H, len(y)):
                        bdata = b(i)
                        edata = e(i)
                        _b = bdata[1]
                        _e = edata[1]

                        s = _b[-1] > _e[-1]
                        if i > 0:
                            if s != status:
                                if s:
                                    bsgnl[i] = y[i]
                                    if bought == 0:
                                        bought = y[i]
                                        trades.append((x[i], y[i], 0))
                                else:
                                    ssgnl[i] = y[i]
                                    if bought != 0:
                                        net = y[i] - bought
                                        # if net > 0:
                                        profit += net
                                        roi += net / bought
                                        trades.append((x[i], y[i], 1))
                                        bought = 0
                            status = s
                        # lineB.set_data(*bdata)
                        # lineE.set_data(*edata)
                        # scatterBuy.set_offsets(np.c_[x, bsgnl])
                        # scatterSell.set_offsets(np.c_[x, ssgnl])
                        # f.canvas.draw()
                        # plt.pause(.0000001)
                        # f.canvas.flush_events()
                    DATA[B][E] = (roi, len(trades))
                    # print(B, E, roi, len(trades))

                v = list(DATA[B].values())
                k = list(DATA[B].keys())
                # best E by roi_per_trade
                EE = k[v.index(max(v, key=lambda x: x[0]/x[1]))]
                print('EE:', EE, 'ratio:', EE / B, DATA[B][EE])
                DATA[B]['EE'] = (EE, DATA[B][EE])
                # ratio~10.2       profit 0.34          neg trades allowed
                # 18.7<ratio<19.8  profit 0.35-0.36  no neg trades allowed
                # for ratios < 7 no neg trades is better
                #     r < 19 neg trades is better
                #     r < 19.8 no neg trades is way better
                #     r > 19.8 neg trades is better
                #     r > 45 its almost the same
                # s = json.dumps([[x/B for x in k], [x[0] for x in v]])
                # open(os.path.join(self.DCYP, f'{pair[:-5]}-with-negative.json'), 'w').write(s)
                # plt.plot([x/B for x in k], [x[0] for x in v])
                # plt.show()

            _DATA = {}
            for B in DATA.keys():
                EE, ROI = DATA[B]['EE']
                _DATA[B] = (EE, *ROI)
            v = list(_DATA.values())
            k = list(_DATA.keys())
            BB = k[v.index(max(v, key=lambda x: x[1]))]
            print(BB, _DATA[BB])
            open(os.path.join(self.DCYP, f'{pair[:-5]}.json'), 'w').write(
                json.dumps({'B': BB, 'E': _DATA[BB][0], 'ROI': _DATA[BB][1]}))

    def live(self) -> None:
        # Order book to Buy and sell signals
        # Partial computation of ewma only for recent
        # Rate limit Order book 30 request/s
        # latency Logging infos

        from queue import Queue
        from threading import Thread

        def data(dq: Queue, pair: str):
            # Prepare 50 y data and send to signal q

            limit = 250
            ohlc = self.get_ohlc(
                pair, limit=limit, try_local=False, write_local=True)
            frame = [(x[0], x[4]) for x in ohlc]
            print('Frame length:', len(frame))
            dq.put(frame)
            next = ohlc[-1][0] + 60000
            avg_latency = 2500
            while True:
                wait = (next - time.time() * 1000 - avg_latency * 3) / 1000
                print(f'waiting for engage: {int(wait)}s')
                # while next - time.time() * 1000 > avg_latency * 3:
                #     time.sleep(.1)
                time.sleep(wait)
                _tk = {'timestamp': frame[-1][0], 'close': frame[-1][1]}
                while True:
                    tk = self.ex.fetch_ticker(pair)
                    if tk['timestamp'] == next:
                        frame = frame[1:] + [(tk['timestamp'], tk['close'])]
                        break
                    if tk['timestamp'] > next:
                        x1, y1 = _tk['timestamp'], _tk['close']
                        x2, y2 = tk['timestamp'], tk['close']
                        x = next
                        y = y1 * (1 - (x - x1) / (x2 - x1)) + \
                            y2 * (1 - (x2 - x) / (x2 - x1))
                        print(f'linear interpolation: {y1, y, y2}')
                        frame = frame[1:] + [(x, y)]
                        break
                    _tk = tk
                dq.put(frame)
                next += 60000

        def signal(dq: Queue, pair: str, sq: Queue):
            # weights = json.loads(open(os.path.join(self.DCYP, f"{pair[:-5]}.json"), 'r').read())
            # B = weights['B']
            # E = weights['E']
            B = 7
            E = 2 * B
            while True:
                if not dq.empty():
                    while not dq.empty():
                        frame = dq.get()
                    # print(frame[-1], len(frame))
                    y = np.array([x[1] for x in frame])
                    b = _ewma(y, B)
                    e = _ewma(y, E)
                    sq.put((frame, b, e))
                    print(b[-1] > e[-1])
                    # s = b[-1] > e[-1]
                    # if s != status:
                    #     if s:
                    #         bsgnl[i] = y[i]
                    #         if bought == 0:
                    #             bought = y[i]
                    #             trades.append((x[i], y[i], 0))
                    #     else:
                    #         ssgnl[i] = y[i]
                    #         if bought != 0:
                    #             net = y[i] - bought
                    #             # if net > 0:
                    #             profit += net
                    #             roi += net / bought
                    #             trades.append((x[i], y[i], 1))
                    #             bought = 0
                    # status = s
                time.sleep(.001)

        def execute(sq: Queue):
            while True:
                if not sq.empty():
                    s = sq.get()
                    print(f'execute {s}')
                    time.sleep(5)

        host = '62.171.165.127'
        username = 'admin1'
        password = 'kbq6v=d%3xk@MD2*js6w'
        db_name = 'crypto000'
        db = Database(host, db_name, username, password)
        db.init_ex()
        pair = self.get_pairs()[0]
        print(db.ex.rateLimit, db.ex.rateLimitMaxTokens,
              db.ex.rateLimitTokens, db.ex.rateLimitUpdateTime)
        ob = db.ex.fetch_order_book(pair)
        print(ob.keys())
        a = ob['asks'][0]
        b = ob['bids'][0]
        print(a, b)
        tk = db.ex.fetch_ticker(pair)
        print(tk.keys())
        print(tk.values())
        # frame = db.data(pair, '1m', 10000)
        # import datetime as dt
        # print(dt.datetime.fromtimestamp(frame[0]['T']//1000))
        # print(len(frame))
        # db.builder(pair, '1m')

        # pairs = self.get_pairs()
        # sq = Queue()
        # data_q = Queue()
        # pair = pairs[1]
        # data_t = Thread(target=data, args=(data_q, pair, ))
        # data_t.setDaemon(True)
        # signal_t = Thread(target=signal, args=(data_q, pair, sq, ))
        # signal_t.setDaemon(True)
        # exec_t = Thread(target=execute, args=(sq, ))
        # exec_t.setDaemon(True)
        # data_t.start()
        # signal_t.start()
        # import matplotlib.pyplot as plt
        # plt.ion()
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # Yp, bp, ep = None, None, None
        # while True:
        #     if not sq.empty():
        #         frame, b, e = sq.get()
        #         X, Y = zip(*frame)
        #         print(X[-5:], Y[-5:], b[-5:], e[-5:])
        #         if not Yp:
        #             Yp, = ax.plot(X, Y, 'b-')
        #         else:
        #             Yp.set_data(X, Y)
        #         if not bp:
        #             bp, = ax.plot(X, b, 'g-')
        #         else:
        #             bp.set_data(X, b)
        #         if not ep:
        #             ep, = ax.plot(X, e, 'r-')
        #             f.show()
        #         else:
        #             ep.set_data(X, e)
        #     f.canvas.draw_idle()
        #     f.canvas.flush_events()
        #     time.sleep( 1)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == '-v':
            VERBOSE = True

    key = json.loads(open(os.path.join(DBASE, 'key.json'), 'r').read())
    c = Crypto000(exchange=ccxt.kucoin, key=key)

    if len(sys.argv) == 2:
        if sys.argv[1].lower() == 'plot':
            pairs = os.listdir(c.DOHLC)
            i = int(input('\n'.join([f'{i}. {pairs[i]}' for i in list(
                range(len(pairs)))]) + f'\n[0-{len(pairs)-1}]: '))
            path = os.path.join(c.DOHLC, pairs[i])
            y = [x[4] for x in json.loads(
                open(os.path.join(path, os.listdir(path)[0]), 'r').read())]
            import matplotlib.pyplot as plt
            plt.title(pairs[i])
            plt.plot(y)
            plt.show()
            quit()
    try:
        c.learn()
    except KeyboardInterrupt:
        quit()
