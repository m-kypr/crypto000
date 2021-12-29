from flask import Flask, send_from_directory, jsonify, abort, Response
import os


DBASE = os.path.dirname(os.path.realpath(__file__))
MAX_LOG_LEN = 1500


def server(host: str, port: int, queues: dict, verbose=False):
    if not verbose:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
    app = Flask(__name__)
    api = {'log': {'data': [], 'max_length': 1500}, 'trades': {'data': []}}

    @app.route('/js/<path:text>')
    def javascript(text):
        if not text.endswith('.js'):
            return abort(Response('Not a JS file'))
        js_dir = os.path.join(DBASE, 'js')
        for fn in os.listdir(js_dir):
            if fn == text:
                return send_from_directory(js_dir, text)

    @app.route("/api/log")
    def loggy():
        data = api['log']['data']
        log_q = queues['log']
        while not log_q.empty():
            data.append(log_q.get())
        if len(data) > api['log']['max_length']:
            data = data[-api['log']['max_length']:]
        api['log']['data'] = data
        return jsonify(list(reversed([str(x) for x in data])))

    @app.route("/api/trades")
    def trades():
        data = api['trades']['data']
        trades_q = queues['trades']
        while not trades_q.empty():
            data.append(trades_q.get())
        api['trades']['data'] = data
        return jsonify(data)

    @app.route('/')
    def index():
        return '<body></body><script src="js/index.js"></script>'

    # if not verbose:
    # import sys
    # cli = sys.modules['flask.cli']
    # cli.show_server_banner = lambda *x: None
    # print(f'Running Flask on http://{host}:{port}')
    app.run(host, port)
