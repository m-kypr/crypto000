import ccxt
# import matplotlib.pyplot as plt
import numpy as np
# import scipy.signal as scipy

def rmean(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    return ret, len(ret)
    # return ret[n - 1:] / n


def some_shit():
    pairs = get_pairs()
    pair = pairs[0]
    data =download_data(pair, f'ohlc_json/{pair}')
    close = np.array([float(x[4]) for x in data])
    close_orig = close.copy()
    close = close[:int(len(close) / 2)]
    factor = 3
    n1 = 11
    n2 = n1 + 2 * factor
    assert (n1 - 1) % 2 == 0
    assert (n2 - n1) % 2 == 0
    plt.axvline(x=len(close), color='k', linestyle='--')
    close_extra = np.append(close, close[-n1:])
    X = np.arange(0, len(close_extra), 1)
    print(n1, n2)
    rm1, len1 = rmean(close_extra, n1)
    rm2, len2 = rmean(close_extra, n2)
    print(len1, len2)
    # for j in range(200):
    #     for i in range(int((n2 - n1) / 2)):
    #         rm2 = np.insert(rm2, 0, rm2[0], axis=0)
    #         rm2 = np.insert(rm2, -1, rm2[-1], axis=0)
    #     rmdiff = rm2 - rm1
    #     asign = np.sign(rmdiff)
    #     signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    #     for i in range(int((len(close) - len(rmdiff)) / 2)):
    #         rmdiff = np.insert(rmdiff, 0, rmdiff[0], axis=0)
    #         rmdiff = np.insert(rmdiff, -1, rmdiff[-1], axis=0)
    #     print(np.count_nonzero(signchange == 1), j)
    #     X.append(j)
    #     Y.append(np.count_nonzero(signchange == 1))
    # rmdiff = rm1[8:-8] - rm2
    # rolling_mean = np.append([0.0], rolling_mean)
    # rolling_mean = np.insert(rolling_mean, 0, rolling_mean[0], axis=0)
    # rolling_mean = np.insert(rolling_mean, -1, rolling_mean[-1], axis=0)
    # arr = np.where(signchange == 1, close[2:-2], np.nan)
    # plt.scatter(X[2:-2], arr, color='purple')
    # plt.fill_between(X[2:-2], rm1, rm2, color='red', where=(rm1 > rm2))
    # plt.fill_between(X[2:-2], rm1, rm2, color='green', where=(rm1 < rm2))
    plt.plot(X, close_orig[:len(close_extra)])
    plt.plot(X[len(close):], close_extra[len(close):], color='red', linestyle='dotted')
    _n1idx = int((n1 - 1) / 2)
    _n2idx = _n1idx + factor
    Xn1 = X[_n1idx:-_n1idx]
    Xn2 = X[_n2idx:-_n2idx]
    print(len(rm2))
    print(_n1idx, _n2idx)
    # quit()
    plt.plot(Xn1, rm1, label=f'{n1}')
    plt.plot(Xn2, rm2, label=f'{n2}')
    rmdiff = rm2 - rm1[factor:-factor]
    # plt.fill_between(X[6:-6], rm1[4:-4], rm2, color='red', where=(rm1[4:-4] > rm2))
    # plt.fill_between(X[6:-6], rm1[4:-4], rm2, color='green', where=(rm1[4:-4] < rm2))
    # arr = np.where(rm1[4:-4] > rm2, rm2, np.nan)
    # print(arr)
    # arr = argrelextrema(arr, np.greater)[0]
    # print(np.sign(rmdiff))
    zero_crossings = np.where(np.diff(np.sign(rmdiff)))[0]
    zero_crossings_pos = zero_crossings[::2] + _n2idx
    zero_crossings_neg = zero_crossings[1::2] + _n2idx
    # plt.scatter(zero_crossings[::2] + _n2idx, np.zeros_like(zero_crossings[::2]) + 2.5, color='red')
    # plt.scatter(zero_crossings[1::2] + _n2idx, np.zeros_like(zero_crossings[1::2]) + 2.5, color='green')
    plt.scatter(zero_crossings_pos, close_extra[zero_crossings_pos], color='red')
    plt.scatter(zero_crossings_neg, close_extra[zero_crossings_neg], color='green')

    localmax = scipy.argrelextrema(rmdiff, np.greater)[0]
    localmin = scipy.argrelextrema(rmdiff, np.less)[0]
    print('last max: ', len(X) - localmax[-1] - _n2idx - n1)
    print('last min: ', len(X) - localmin[-1] - _n2idx - n1)

    plt.plot(Xn2, rmdiff + 2.5)
    plt.plot(Xn2, np.zeros_like(rmdiff) + 2.5, "--", color="gray")
    plt.scatter(localmax + _n2idx, rm2[localmax], color='yellow')
    plt.scatter(localmin + _n2idx, rm2[localmin], color='yellow')
    # plt.scatter(localmin + _n2idx, rm2[localmin], color='black')
    plt.legend()
    plt.show()



def api():
    import json
    def get_pairs(exchange, path='', save=True, curr='usdt'):
        ret = [x for x in list(exchange.load_markets().keys()) if curr.lower() in x.split('/')[1].lower()] 
        if save: 
            if path == '': 
                path = f'pairs_json/kucoin_{curr}.json'
            open(path, 'w').write(json.dumps(ret))
        return ret
    def local_pairs(path='', curr='usdt'):
        if path == '':
            path = f'pairs_json/kucoin_{curr}.json'
        return json.loads(open(path, 'r').read())

    def worker(pair):
        # print(pair)
        clock = int(time.time() * 1000) # milliseconds unix time
        
        q = ex.fetchOHLCV(pair)
        Y = [x[4] for x in q]
        avg = sum(Y) / len(Y)
        t = (max(Y) - avg) / 10
        interval = 60 # seconds
        sleep_delay = 1 # seconds
        buys, sells = [], []
        cY = None
        i = 0
        filesafe_pair = pair.replace('/', '.')
        open(f'log/{filesafe_pair}', 'w').write(f'start: {clock}\n\n')
        while True: 
            tk = ex.fetch_ticker(pair)
            timestamp = tk['timestamp']
            clockdiff = timestamp - clock
            # print('ticker time - clock time:', clockdiff, 'ms')
            if 'close' in tk:
                price = tk['close']
            else:
                price = tk['buy']
            if not cY: 
                if price < avg:
                    open(f'log/{filesafe_pair}', 'a').write(f'\n\nbuying at price {price}\n\n')
                    # print()
                    # print('buying at price', price)
                    cY = price
                    buys.append(timestamp)
            else: 
                if price > cY + t: 
                    # print()
                    # print('selling at price', price)    
                    profit = price - cY
                    open(f'log/{filesafe_pair}', 'a').write(f'\n\nselling at price {price}   profit={profit}\n\n')
                    cY = None
                    sells.append(timestamp)
            if cY:
                l = 44
                im = i % l
                lines = open(f'log/{filesafe_pair}', 'r').readlines()
                lines[-1] = '[' + ('=' * im) + '>' + ' ' * (l - im) + f']: current {price} | target {cY + t} | missing {(cY + t) - price} '
                open(f'log/{filesafe_pair}', 'w').writelines(lines)
            # print('\r[' + ('=' * im) + '>' + ' ' * (l - im) + f']: current {price} | target {cY + t} | missing {(cY + t) - price} ', end='')

            # if clockdiff < 0:
            #     time.sleep(abs(2 + clockdiff / 1000))
            # else:
            time.sleep(sleep_delay)
            clock = tk['timestamp']
            i += 1

    key = json.loads(open('key.json', 'r').read())
    args = {
    'apiKey': key['apiKey'],
    'secret': key['secret'],
    'passphrase': key['passphrase'],
    'timeout': 50000,
    'enableRateLimit': True,
    }
    ex = ccxt.kucoin(config=args)
    # r = get_pairs(ex)
    # X = np.arange(0, len(Y), 1)
    from os.path import isfile
    if isfile('pairs_json/kucoin_usdt.json'):
        r = local_pairs()
    else:
        r = get_pairs(ex)

    import time
    from threading import Thread
    threads = []
    r = r[:10]

    for pair in r:
        threads.append(Thread(target=worker, args=(pair,)))
    print(f'starting {len(threads)} threads...')
    for t in threads: 
        t.start()
    print('Trading pairs\n%s' % ' '.join(r))
    for t in threads: 
        t.join()

    quit()
    cY = Y[0]
    profit = 0
    buys, sells = [], []
    buys.append(s)
    for i in range(s, X[-1]): 
        y = Y[i]
        if cY == 0:
            if Y[sells[-1]] - y > t:
                cY = y
                buys.append(i)
        else:
            if y - cY > t:
                profit += y - cY
                cY = 0
                # print(i)
                sells.append(i)

    plt.plot(X, Y, color='blue')
    plt.plot(np.unique(X), np.poly1d(np.polyfit(X, Y, 1))(np.unique(X)), linestyle='-.', color='red')
    plt.plot(X, np.full_like(Y, avg), linestyle='--')
    plt.plot(X, np.full_like(Y, t + avg), linestyle='--', color='green')
    plt.show()
    




if __name__ == '__main__': 
    api()
    quit()
    # pairs = get_pairs(path='pairs_json/kraken', save=True)
    # data =download_data(pair, f'ohlc_json/{pair}')
    pair = local_pairs()[0]
    # download_data(pair, interval=60)
    # quit()
    Y = np.array([float(x[4]) for x in local_data(pair, interval=60)])
    print(len(Y))
    X = np.arange(0, len(Y), 1)
    X2, Y2 = [], []
    for j in range(1, int(len(Y) / 10)):
        s = j
        avg = sum(Y[:s]) / s
        t = (max(Y[:s]) - avg) / 2
        # print(t)
        cY = Y[s]
        profit = 0
        buys, sells = [], []
        buys.append(s)
        for i in range(s, X[-1]): 
            y = Y[i]
            if cY == 0:
                if Y[sells[-1]] - y > t:
                    cY = y
                    buys.append(i)
            else:
                if y - cY > t:
                    profit += y - cY
                    cY = 0
                    # print(i)
                    sells.append(i)
        X2.append(s)
        Y2.append((profit / len(Y)) * 10000)
        print(f'{s} {int((profit / len(Y)) * 10000)} {int((profit / avg) * 10000) / 100}% ')

    Y2 = np.array(Y2)
    Y3, _ = rmean(Y2, 5)
    plt.plot(X, Y)
    # plt.plot(X2, Y2)
    # plt.plot(X2[2:-2], Y3)
    plt.scatter(buys, Y[buys], color='green')
    plt.scatter(sells, Y[sells], color='red')
    plt.show()