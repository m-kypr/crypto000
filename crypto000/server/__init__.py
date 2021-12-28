from flask import Flask, send_from_directory
import os

DBASE = os.path.dirname(os.path.realpath(__file__))


def server(host, port, log_q):
    app = Flask(__name__)
    logs = {'1': []}

    @app.route('/js/<path:text>')
    def javascript(text):
        js_dir = os.path.join(DBASE, 'js')
        for fn in os.listdir(js_dir):
            if fn == text:
                return send_from_directory(os.path.join(js_dir, text))

    @app.route("/")
    def hello_world():
        log = logs['1']
        while not log_q.empty():
            log.append(log_q.get())
        logs['1'] = log
        return '<script src="js/update.js"></script><br>' + '<br>'.join(list(reversed([str(x) for x in log])))
    app.run(host, port)
