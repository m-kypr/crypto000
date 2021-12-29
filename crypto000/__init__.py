import os
import json
import time
from queue import Queue
from threading import Thread
import builtins

from crypto000.util import _ewma
from crypto000.api import Api
from crypto000.database import Database

import numpy as np

DBASE = os.path.dirname(os.path.realpath(__file__))


class Crypto000:
    def __init__(self, datadir='data', key='key.json', verbose=False) -> None:
        if not datadir.startswith('/'):
            self.DCYP = os.path.join(DBASE, datadir)
        else:
            self.DCYP = datadir
        for d in [x for x in dir(self) if x.startswith('D')]:
            dirname = getattr(self, d)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)

        self.api = Api(key=key, verbose=verbose)
        print('Accounts:', self.api.get_accounts())
        self.timeframe = '1m'
        self.verbose = verbose

    def init_db(self):
        host = '62.171.165.127'
        username = 'admin1'
        password = 'kbq6v=d%3xk@MD2*js6w'
        db_name = 'crypto000'
        self.db = Database(host, db_name, username, password, self.api)

    def websocket(self, pair: str, out_q: Queue):
        def ping(ws, interval):
            last_ping = time.time()
            ws.send(json.dumps({'id': int(last_ping), 'type': 'ping'}))
            while True:
                now = time.time()
                if now - last_ping + 1 > interval:
                    ws.send(json.dumps({'id': int(now), 'type': 'ping'}))
                    last_ping = now
                time.sleep(.1)

        from websocket import WebSocket
        pub = self.api.ex.fetch2('bullet-public', api='public', method='POST')
        ws = pub['data']['instanceServers'][0]
        ws_url = ws['endpoint']
        token = pub['data']['token']
        ping_interval = ws['pingInterval']
        ping_interval_sec = ping_interval / 1000
        print(ws)
        ws = WebSocket()
        connectId = 'hallo1'
        ws.connect(f'{ws_url}?token={token}&[connectId={connectId}]')
        print(ws.recv())
        ping_thread = Thread(target=ping, args=(ws, ping_interval / 1000, ))
        # ping_thread.daemon = True
        ping_thread.start()
        ws.send(json.dumps({'id': int(time.time()), 'type': 'subscribe',
                'topic': f'/market/ticker:{pair.replace("/", "-")}', 'privateChannel': False, 'response': True}))
        while True:
            recv = json.loads(ws.recv())
            if recv['type'] == 'message':
                out_q.put(recv['data'])

    def test(self, pair, timeframe, queues):
        def print(msg, *args):
            if queues['log']:
                queues['log'].put(f'{msg} {args}')
            else:
                builtins.print(msg, *args)
        B, E = map(int, json.loads(open(os.path.join(self.DCYP, 'best.json'), 'r').read())[
            pair[:-5]].split(','))
        print(pair, f"B={B}, E={E}")
        tf = self.api.parse_tf(timeframe)
        _t = 0
        fee = 0.01
        roi = 0
        trades = 0
        dq = Queue()
        ws_thread = Thread(target=self.websocket, args=(pair, dq, ))
        # ws_thread.daemon = True
        ws_thread.start()
        prev_t = 0
        r = dq.get(block=True)
        r_age_sum = 0
        r_age_count = 0
        while True:
            now = time.time()
            q = self.db.data(pair, timeframe, 250)
            while True:
                if dq.empty():
                    break
                r = dq.get(block=True)
            r_t = r['time']
            last_t = q[-1]['T']
            r_age = now*1000 - r_t
            last_age = now*1000 - last_t
            r_age_sum += r_age
            r_age_count += 1
            # print(r_t - last_t)
            Y = np.array([x['C'] for x in q])
            # print(r)
            if _t == 0:
                buy_price = float(r['bestBid'])
                # print(buy_price)
                Y = np.append(Y, buy_price)
                d = _ewma(Y, B) - _ewma(Y, E)
                if d[-2] > 0 and d[-1] < 0:
                    _t = Y[-1]
                    trades += 1
                    queues['trades'].put(
                        {'type': 'BUY', 'time': time.time(), 'data': {'price': Y[-1]}})
            else:
                sell_price = float(r['bestAsk'])
                Y = np.append(Y, sell_price)
                d = _ewma(Y, B) - _ewma(Y, E)
                if d[-2] < 0 and d[-1] > 0:
                    net = Y[-1] - _t - fee * _t - fee * Y[-1]
                    roi += net / _t
                    trades += 1
                    _t = 0
                    queues['trades'].put({'type': 'SELL', 'time': time.time(), 'data': {
                                         'oldPrice': _t, 'price': Y[-1], 'roi': roi}})

            print(now, r_t - last_t, roi, trades, '     ', d[-1], d[-2])
            time.sleep(.1)

    def learn(self, pair, timeframe, frame_size, frames, sell_negative=False, write_out=False):
        """Bruteforce best window sizes for EWMA (Exponentially weighted moving average) with OHLC data.
        """
        fee = 0.00
        q = self.db.data(pair, timeframe, frame_size * frames)
        X = np.array([x['T'] for x in q])
        Y = np.array([x['C'] for x in q])
        LEARN = {}
        total_frames = frames
        # try:
        #     for fn in os.listdir(self.DCYP):
        #         if fn.startswith(f'{pair[:-5]}_{frame_size}_'):
        #             LEARN = json.loads(
        #                 open(os.path.join(self.DCYP, fn), 'r').read())
        #             loaded_frames = int(fn[-7:-5])
        #             total_frames += loaded_frames
        #             print(f'loaded learn data, frames={loaded_frames}')
        #             break
        # except Exception as e:
        #     print(e)
        BMAX = frame_size // 2
        BMIN = 10
        print('frame size:', frame_size)
        for f in range(frames):
            print(f'frame={f+1}/{frames}')
            DATA = {}
            _Y = Y[f*frame_size:(f+1)*frame_size]
            for B in range(BMIN, BMAX):
                if B not in DATA:
                    DATA[B] = _ewma(_Y, B)
                b = DATA[B]
                EMAX = B // 2
                EMIN = BMIN // 2
                for E in range(EMIN, EMAX):
                    if f'{B},{E}' not in LEARN:
                        LEARN[f'{B},{E}'] = {'roi': 0, 'trades': 1}
                    if E not in DATA:
                        DATA[E] = _ewma(_Y, E)
                    d = b - DATA[E]
                    _t = 0
                    for i in range(B, len(d)):
                        if d[i-1] > 0 and d[i] < 0:
                            if _t == 0:
                                _t = _Y[i]
                        if d[i-1] < 0 and d[i] > 0:
                            if _t != 0:
                                fee_t = _t * fee - _Y[i] * fee
                                net = _Y[i] - _t
                                net -= fee_t
                                if not sell_negative and net < 0:
                                    continue
                                LEARN[f'{B},{E}']['roi'] += (net / _t)
                                LEARN[f'{B},{E}']['trades'] += 1
                                _t = 0
            s = {k: v for k, v in sorted(
                LEARN.items(), key=lambda item: item[1]['roi'])}
            best = list(reversed(list(s.keys())[-3:]))
            for x in best:
                print(x, s[x], s[x]['roi'] / s[x]['trades'])
            worst = reversed(list(s.keys())[:3])
            for x in worst:
                print(x, s[x])
            if len(best) > 0:
                x = best[0]
            # print('roi per frame:', s[x]['roi'] / frames)
            print()
        path = os.path.join(
            self.DCYP, f'{pair[:-5]}_{frame_size}_{total_frames}.json')
        if write_out:
            print('output:', path)
            open(path, 'w').write(json.dumps(LEARN))
        best_path = os.path.join(self.DCYP, 'best.json')
        if not os.path.isfile(best_path):
            open(best_path, 'w').write("{}")
        with open(best_path, 'r+') as f:
            f.seek(0)
            old = json.loads(f.read())
            old[pair[:-5]] = x
            f.seek(0)
            f.write(json.dumps(old))
            f.truncate()

    def learn2(self, pair, timeframe, frames):
        # coll = self.db.get_coll(pair, timeframe)
        _DATA = self.db.data(pair, timeframe, 1500*frames)
        # _DATA = list(coll.find().sort(
        #     [('T', pymongo.ASCENDING)]).limit(1500*frames))
        data = {}
        for f in range(frames):
            DATA = _DATA[1500*f:(f+1)*1500]
            Y = np.array([x['C'] for x in DATA])
            # X = np.arange(len(Y))
            # Y_avg = sum(Y)/len(Y)
            fee = .01
            bmax = 200
            emin = 3
            ewma = {}
            for b in range(emin + 1, bmax + 1):
                if b not in ewma:
                    ewma[b] = _ewma(Y, b)
                B = ewma[b]
                print(f'\rframe={f+1} {b}/{bmax} ', end='', flush=True)
                for e in range(emin, b - 3):
                    if (b, e) not in data:
                        data[(b, e)] = 0
                    if e not in ewma:
                        ewma[e] = _ewma(Y, e)
                    E = ewma[e]
                    D = B - E
                    # D_sum = np.array([D[b]] * b)
                    # for i in range(b, len(Y)):
                    #     D_sum = np.append(D_sum, (D_sum[-1]+D[i]))
                    M = D
                    sell, buy = [], []
                    s = 0
                    roi = 0
                    for i in range(b, len(Y)):
                        if M[i-1] > 0 and M[i] < 0:
                            if s == 0:
                                s = Y[i]
                                buy.append(i)
                        if M[i-1] < 0 and M[i] > 0:
                            if s != 0:
                                fee_price = s * fee + Y[i] * fee
                                net = Y[i] - s - fee_price
                                if net > 0:
                                    roi += net / s
                                    s = 0
                                    sell.append(i)
                    if s != 0:
                        fee_price = s * fee - Y[i] * fee
                        net = Y[i] - s
                        net -= fee_price
                        roi += net / s
                        sell.append(i)

                    data[(b, e)] += roi
                    # print(roi)
            sort = sorted(data.items(), key=lambda x: x[1])
            print()
            print([(round(x[0][0]/x[0][1], 2), *x) for x in sort[-3:]])
            # print([(round(x[0][0]/x[0][1], 2), *x) for x in sort[:3]])

    def learns(self, timeframe, frame_size, frames, pairs=1, sell_neg=False, write_out=True) -> None:
        self.init_db()
        pairs_list = self.api.get_pairs()
        # pairs_list = ['SNX/USDT']
        for pair in pairs_list[:pairs]:
            self.db.init_coll(pair, timeframe, 50)
            self.learn(pair, timeframe, frame_size, frames,
                       sell_negative=sell_neg, write_out=write_out)

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

    def tests(self, timeframe='1m', pairs=1) -> None:
        self.init_db()
        queues = {'log': Queue(), 'trades': Queue(), }
        for pair in self.api.get_pairs()[:pairs]:
            t = Thread(target=self.test, args=(pair, timeframe, queues, ))
            # t.daemon = True
            t.start()
        from crypto000.server import server
        server('0.0.0.0', 4000, queues, verbose=self.verbose)
