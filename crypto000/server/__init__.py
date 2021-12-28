from flask import Flask, send_from_directory


def server(log_q):
    app = Flask(__name__)
    logs = {'1': []}

    @app.route('/js/<path:text>')
    def javascript(text):
        import os
        print(os.listdir('js'))
        for fn in os.listdir('js'):
            if fn == text:
                return send_from_directory(os.path.join('js', text))

    @app.route("/")
    def hello_world():
        log = logs['1']
        while not log_q.empty():
            log.append(log_q.get())
        logs['1'] = log
        return '<script src="js/update.js"></script><br>' + '<br>'.join(list(reversed([str(x) for x in log])))
    print('lol')
    app.run('0.0.0.0', 1234)
