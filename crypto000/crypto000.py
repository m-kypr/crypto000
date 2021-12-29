from logging import log
import os
import json
import time
from queue import Queue
from threading import Thread
import builtins

from util import _ewma
from api import Api
from database import Database

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
        # print(self.api.ex.fetch_accounts())
        # print('Accounts:', self.api.get_accounts())
        self.timeframe = '1m'
        self.verbose = verbose

    def init_db(self):
        host = '62.171.165.127'
        username = 'admin1'
        password = 'kbq6v=d%3xk@MD2*js6w'
        db_name = 'crypto000'
        self.db = Database(host, db_name, username, password)
        self.db.init_ex(self.api.ex)

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
        ping_thread.daemon = True
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
        B, E = map(int, json.loads(open(os.path.join(self.DCYP, 'best.json'), 'r'))[
            pair[:-5]].split(','))
        print(pair, f"B={B}, E={E}")
        tf = self.api.parse_tf(timeframe)
        _t = 0
        fee = 0.01
        roi = 0
        trades = 0
        dq = Queue()
        ws_thread = Thread(target=self.websocket, args=(pair, dq, ))
        ws_thread.daemon = True
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
        fee = 0.01
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
        print(BMAX)
        for f in range(frames):
            print(f'frame={f}')
            DATA = {}
            _Y = Y[f*frame_size:(f+1)*frame_size]
            for B in range(BMIN, BMAX):
                # for i in range(1, len(Y)):
                #     if (B, f) not in DATA:
                if B not in DATA:
                    DATA[B] = _ewma(_Y, B)
                b = DATA[B]
                EMAX = B // 2
                EMIN = BMIN // 2
                # print(B)
                # print(f'\r{round((B/BMAX)*100, 2)}% {EMAX} ', end='')
                for E in range(EMIN, EMAX):
                    if f'{B},{E}' not in LEARN:
                        LEARN[f'{B},{E}'] = {'roi': 0, 'trades': 1}
                    if E not in DATA:
                        DATA[E] = _ewma(_Y, E)
                    d = b - DATA[E]
                    _t = 0
                    for i in range(frame_size >> 2, len(d)):
                        if d[i-1] > 0 and d[i] < 0:
                            if _t == 0:
                                _t = Y[i]
                        if d[i-1] < 0 and d[i] > 0:
                            if _t != 0:
                                fee_t = _t * fee - _Y[i] * fee
                                net = _Y[i] - _t
                                net -= fee_t
                                if sell_negative and net < 0:
                                    continue
                                LEARN[f'{B},{E}']['roi'] += (net / _t)
                                LEARN[f'{B},{E}']['trades'] += 1
                                _t = 0
            s = {k: v for k, v in sorted(
                LEARN.items(), key=lambda item: item[1]['roi'])}
            best = list(s.keys())[-3:]
            for x in best:
                print(x, s[x])
            worst = reversed(list(s.keys())[:3])
            for x in worst:
                print(x, s[x])

            # LEARN['BEST'] = x
            print(s[x]['roi'] / frames)
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

    def learns(self, timeframe, frame_size, frames, pairs=1, sell_neg=False) -> None:
        self.init_db()
        # pairs_list = self.api.get_pairs()
        pairs_list = ['SNX/USDT']
        for pair in pairs_list[:pairs]:
            self.learn(pair, timeframe, frame_size, frames,
                       sell_negative=sell_neg, write_out=True)

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
            t.daemon = True
            t.start()
        from server import server
        server('0.0.0.0', 4000, queues, verbose=self.verbose)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Crypto000 bot.')
    parser.add_argument('-v', '--verbose',
                        help='Enable verbose output', action='store_true')
    parser.add_argument('-k', '--keyfile', help='Key.json file',
                        type=str, default='key.json')
    parser.add_argument('-d', '--datadir', help='Data directory',
                        type=str, default='data')
    parser.add_argument(
        '-l', '--learn', help='Learn values', action='store_true')

    args = parser.parse_args()

    c = Crypto000(datadir=args.datadir, key=args.keyfile, verbose=args.verbose)

    try:
        if args.learn:
            c.learns('1m', 250, 40, 1, True)
        else:
            c.tests('1m', 1)
    except KeyboardInterrupt:
        quit()
