# from concurrent.futures import TimeoutError
import asyncio
from asyncio.exceptions import TimeoutError
import websocket
from api import Api
import requests as rq
import json
import time
from threading import Thread, Lock, Event
from queue import Queue

lock = Lock()
event = Event()

out_queue = Queue()


def ping(ws):
    now = time.time()
    ws.send(json.dumps({'id': int(now), 'type': 'ping'}))
    return now


api = Api(verbose=False)
ex = api.ex()
pairs = api.get_pairs()
pair = pairs[0]
pub = ex.fetch2('bullet-public', api='public', method='POST')
ws = pub['data']['instanceServers'][0]
ws_url = ws['endpoint']
token = pub['data']['token']
ping_interval = ws['pingInterval']
ping_interval_sec = ping_interval / 1000
print(pub)
ws = websocket.WebSocket()
connectId = 'hallo1'
ws.connect(f'{ws_url}?token={token}&[connectId={connectId}]')
print(ws.recv())


async def coroutine(ws):
    return ws.recv()


async def q(ws):
    tick = .1
    last_ping = ping(ws)
    print(ws.recv())
    ws.send(json.dumps({'id': int(time.time()), 'type': 'subscribe',
            'topic': f'/market/ticker:{pair.replace("/", "-")}', 'privateChannel': False, 'response': True}))

    # ws.send(json.dumps({'id': int(time.time()), 'type': 'subscribe',
    #         'topic': f'/market/candles:{pair.replace("/", "-")}_1min', 'privateChannel': False, 'response': True}))
    print(ws.recv())
    while True:
        now = time.time()
        # print(now - last_ping)
        if now - last_ping >= ping_interval_sec - tick:
            last_ping = ping(ws)
            pong = ws.recv()
            print('ping:', pong)
        try:
            timeout = ping_interval_sec - now + last_ping
            # print(timeout)
            r = await asyncio.wait_for(coroutine(ws), timeout)
        except TimeoutError as e:
            continue
        print(r)
        # data = json.loads(r)['data']
        # olhcv = [float(x) for x in data['candles']]
        # print(time.time() - olhcv[0], olhcv[4])
        await asyncio.sleep(tick)
    ws.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(q(ws))
loop.run_forever()
