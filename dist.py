import tensorflow as tf
import time
import matplotlib.pyplot
import tensorflow_datasets as tfds
import json
import socketserver
import sys
import argparse
import socket
import urllib
import threading
import numpy
import enum
import statistics
import io
import collections
import sklearn.utils
import Pipeline as pl

def build_dense_model(input_shape, input_units, layers, outputs, optimizer):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=input_units, input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    for layer in layers:
        model.add(tf.keras.layers.Dense(units=layer, activation='relu'))
    model.add(tf.keras.layers.Dense(units=outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
            metrics=['accuracy'])
    return model

def export_stats(times, times_net, losses, accuracy, val_accuracy):
    matplotlib.pyplot.ylabel('training loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.plot(losses, '-or')
    matplotlib.pyplot.savefig('training_loss.png')
    matplotlib.pyplot.clf()

    matplotlib.pyplot.ylabel('training error')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.plot(accuracy, '-og')
    matplotlib.pyplot.savefig('training_error.png')
    matplotlib.pyplot.clf()

    matplotlib.pyplot.ylabel('validation error')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.plot(val_accuracy, '-ob')
    matplotlib.pyplot.savefig('validation_error.png')
    matplotlib.pyplot.clf()

    with open('times.csv', 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in times:
            w.writerow([epoch, row])
            epoch += 1
    with open('times_net.csv', 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in times_net:
            w.writerow([epoch, row])
            epoch += 1
    with open('losses.csv', 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in lossess:
            w.writerow([epoch, row])
            epoch += 1
    with open('accuracy.csv', 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in accuracy:
            w.writerow([epoch, row])
            epoch += 1
    with open('val_accuracy.csv', 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in val_accuracy:
            w.writerow([epoch, row])
            epoch += 1

def load_data():
    X, Y = pl.import_data()
    X_tr, Y_tr, X_te, Y_te = X, Y, X, Y
    Data = collections.namedtuple('Data', 'x_tr y_tr x_te y_te')
    return Data(X_tr, Y_tr, X_te, Y_te)

class SGDTrainer:

    def __init__(self, config, idx, model, data):
        self.config = config
        self.idx = idx
        self.data = data
        self.model = model
        self.times = []

    def train(self):
        start = time.perf_counter()
        history = self.model.fit(data.x_tr, data.y_tr, validation_split=self.config['validation_split'],
            batch_size=self.config['batch_size'], epochs=self.config['epochs'])
        end = time.perf_counter()
        self.times.append(end - start)
        self.losses = history.history['loss']
        self.accuracy = list(map(lambda x: 1 - x, history.history['acc']))
        self.val_accuracy = list(map(lambda x: 1 - x, history.history['val_acc']))


    def get_stats(self):
        return self.times, [], self.losses, self.accuracy, self.val_accuracy


class SimuParallelSGDTrainer:

    def __init__(self, config, idx, server, model, data):
        self.config = config
        self.idx = idx
        self.server = server
        self.data = data
        self.model = model
        self.times = []
        self.times_net = []
        self.losses = []
        self.accuracy = []
        self.val_accuracy = []

    def train(self):
        for e in range(self.config['epochs']):
            time_net = 0
            start = time.perf_counter()
            size = len(self.data.x_tr)
            worker_size = size // len(self.config['workers'])
            itr_size = worker_size // self.config['sync_iterations']
            for i in range(self.config['sync_iterations']):
                start = (self.idx * worker_size) + (i * itr_size)
                end = start + itr_size
                x_itr = self.data.x_tr[start:end]
                y_itr = self.data.y_tr[start:end]
                # sklearn.utils.shuffle(x_itr, y_itr, random_state=0)
                history = self.model.fit(x_itr, y_itr, validation_split=0.1,
                    batch_size=self.config['batch_size'], epochs=1)

                start_net = time.perf_counter()
                self.server.send_weights(0, numpy.array(self.model.get_weights()))
                if self.idx == 0:
                    all_weights = self.server.wait_and_consume_weights(len(self.config['workers']))
                    weights = numpy.mean(all_weights, axis=0)
                    for j in range(1, len(self.config['workers'])):
                        self.server.send_weights(j, weights)
                else:
                    weights = self.server.wait_and_consume_weights(1)[0]
                self.model.set_weights(weights)
                end_net = time.perf_counter()
                time_net += end_net - start_net
            end = time.perf_counter()
            self.times.append(end - start)
            self.times_net.append(time_net)
            self.losses.extend(history.history['loss'])
            self.accuracy.extend(list(map(lambda x: 1 - x, history.history['acc'])))
            self.val_accuracy.extend(list(map(lambda x: 1 - x, history.history['val_acc'])))

    def get_stats(self):
        return self.times, self.times_net, self.losses, self.accuracy, self.val_accuracy


def recv_weights(sock, server):
    while server.is_running:
        print('waiting to recv weights...')
        b = sock.recv(4)
        if len(b) == 0:
            break
        n = int.from_bytes(b, byteorder='big')
        print('receiving %d bytes' % n)
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        bio = io.BytesIO(data)
        weights = numpy.load(bio, allow_pickle=True)
        server.received_weights(weights)

    print('thread done')

class WeightReceiverHandler(socketserver.BaseRequestHandler):

    def handle(self):
        b = self.request.recv(4)
        idx = int.from_bytes(b, byteorder='big')
        self.server.add_connection(idx, self.request)
        print('opened connection from %d' % idx)
        recv_weights(self.request, self.server)

class Server(socketserver.ThreadingTCPServer):

    def __init__(self, config, idx, connections):
        self.config = config
        self.is_running = True
        self.idx = idx
        host, port = self.config['workers'][self.idx].split(':')
        port = int(port)
        super().__init__((host, port), WeightReceiverHandler)
        self.conn_cv = threading.Condition()
        self.connections = {}
        for i in range(len(connections)):
            self.connections[i] = connections[i]

        self.l_cv = threading.Condition()
        self.lists = []

    def start(self):
        self.serve_forever()

    def wait_for_connections(self):
        with self.conn_cv:
            while len(self.connections) < len(self.config['workers']) - 1:
                self.conn_cv.wait()

    def read_data(self, connection):
        while not self.done:
            connection.recv(4)
        print('closing client sock')
        connection.close()

    def add_connection(self, i, conn):
        with self.conn_cv:
            self.connections[i] = conn
            self.conn_cv.notify_all()

    def received_weights(self, l):
        with self.l_cv:
            self.lists.append(l)
            print(len(self.lists))
            self.l_cv.notify_all()

    def wait_and_consume_weights(self, n):
        with self.l_cv:
            while len(self.lists) < n:
                self.l_cv.wait()
            lists = [l for l in self.lists]
            self.lists = []
        return lists

    def send_weights(self, idx, weights):
        if self.idx == idx:
            self.received_weights(weights)
        else:
            with self.conn_cv:
                bs = io.BytesIO()
                numpy.save(bs, weights)
                print(len(bs.getbuffer()))
                b = len(bs.getbuffer()).to_bytes(4, byteorder='big')
                print('sending to %d' % idx)
                self.connections[idx].sendall(b)
                self.connections[idx].sendall(bs.getbuffer())

    def end(self):
        for i, c in self.connections.items():
            c.shutdown(socket.SHUT_WR)
        self.shutdown()

def connect_to_worker(config, i, idx):
    host, port = config['workers'][i].split(':')
    port = int(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    b = idx.to_bytes(4, byteorder='big')
    sock.sendall(b)

    return sock

def main(args):
    with open(args.config_file) as f:
        config = json.load(f)

        connections = []
        for i in range(args.worker_idx):
            connections.append(connect_to_worker(config, i, args.worker_idx))

        Server.allow_reuse_address = True
        server = Server(config, args.worker_idx, connections)

        threads = []
        for i in range(args.worker_idx):
            ti = threading.Thread(target=recv_weights, args=(connections[i], server))
            ti.start()
            threads.append(ti)

        t = threading.Thread(target=server.start)
        t.start()
        threads.append(t)

        server.wait_for_connections()

        #plug in here.
        data = load_data()
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['alpha'], momentum=config['beta'], nesterov=False)
        model = pl.new_model(optim=optimizer, lo=pl.crps)

        tr = SimuParallelSGDTrainer(config, args.worker_idx, server, model, data)
        tr.train()

        print('done training')

        server.is_running = False
        server.end()
        for ti in threads:
            ti.join()

        export_stats(*tr.get_stats())

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Distributed SGD worker.')
    parser.add_argument('--config_file', help='path to json config file')
    parser.add_argument('--worker_idx', type=int, help='index of this worker in config')
    args = parser.parse_args()
    main(args)
