
if __name__ == '__main__':
    import argparse
    from crypto000 import Crypto000
    parser = argparse.ArgumentParser(description='Crypto000 bot.')
    parser.add_argument('-v', '--verbose',
                        help='Enable verbose output', action='store_true')
    parser.add_argument('-k', '--keyfile', help='Key.json file',
                        type=str, default='key.json')
    parser.add_argument('-d', '--datadir', help='Data directory',
                        type=str, default='data')
    parser.add_argument('-p', '--port', help='Port for Api',
                        type=int, default=4001)
    # parser.add_argument('-proxy', '--use-proxy', help='Port for Api',
    #                     type=int, default=4001)
    parser.add_argument(
        '-l', '--learn', help='Learn values', action='store_true')

    args = parser.parse_args()

    c = Crypto000(datadir=args.datadir,
                  verbose=args.verbose, port=args.port)
    c.init_api(key=args.keyfile)

    try:
        if args.learn:
            c.learns(frames=100)
            # c.learns('1m', 50, 500, 10, sell_neg=True, write_out=False)
        else:
            c.tests('1m', 1)
    except KeyboardInterrupt:
        quit()


# def init():
#     host = '62.171.165.127'
#     username = 'admin1'
#     password = 'kbq6v=d%3xk@MD2*js6w'
#     db_name = 'crypto000'
#     db = Database(host, db_name, username, password, Api())
#     pair = 'SNX/USDT'
#     c = db.get_coll(pair, '1m')
#     learn2(c, frames=100)
#     # _DATA = list(c.find().sort(
#     #     [('T', pymongo.ASCENDING)]).limit(1500*12))[-1500:]
#     # plot('D', 0.01, True, 10, 5, 0.5, np.array([x['C'] for x in _DATA]))
