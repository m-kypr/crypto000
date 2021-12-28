from flask import Flask, send_from_directory, jsonify, abort, Response
import os

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

DBASE = os.path.dirname(os.path.realpath(__file__))


def server(host, port, log_q):
    app = Flask(__name__)
    logs = {'1': []}

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
        log = logs['1']
        while not log_q.empty():
            log.append(log_q.get())
        logs['1'] = log
        return jsonify(list(reversed([str(x) for x in log])))

    @app.route('/')
    def index():
        return '<script src="js/index.js"></script>'

    import sys
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    print(f'Running Flask on http://{host}:{port}')
    app.run(host, port)
