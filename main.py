
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
    parser.add_argument(
        '-l', '--learn', help='Learn values', action='store_true')

    args = parser.parse_args()

    c = Crypto000(datadir=args.datadir, key=args.keyfile, verbose=args.verbose)

    try:
        if args.learn:
            c.learns('1m', 100, 140, 1, True)
        else:
            c.tests('1m', 1)
    except KeyboardInterrupt:
        quit()
