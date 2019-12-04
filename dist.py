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
import csv
import os

# ["128.84.167.136:8000", "128.84.167.131:8001"],
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

def export_stats(config, times, times_net, losses, accuracy, val_accuracy):
    os.makedirs(config['out_directory'], exist_ok=True)
    matplotlib.pyplot.ylabel('training loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.plot(losses, '-or')
    matplotlib.pyplot.savefig(os.path.join(config['out_directory'], 'training_loss.png'))
    matplotlib.pyplot.clf()

    matplotlib.pyplot.ylabel('training error')
    matplotlib.pyplot.xlabel('epoch')

    matplotlib.pyplot.plot(list(map(lambda x: 1 - x, accuracy)), '-og')
    matplotlib.pyplot.savefig(os.path.join(config['out_directory'], 'training_error.png'))
    matplotlib.pyplot.clf()

    matplotlib.pyplot.ylabel('validation error')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.plot(list(map(lambda x: 1 - x, val_accuracy)), '-og')
    matplotlib.pyplot.savefig(os.path.join(config['out_directory'], 'validation_error.png'))
    matplotlib.pyplot.clf()

    with open(os.path.join(config['out_directory'], 'times.csv'), 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in times:
            w.writerow([epoch, row])
            epoch += 1
    with open(os.path.join(config['out_directory'], 'times_net.csv'), 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in times_net:
            w.writerow([epoch, row])
            epoch += 1
    with open(os.path.join(config['out_directory'], 'losses.csv'), 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in losses:
            w.writerow([epoch, row])
            epoch += 1
    with open(os.path.join(config['out_directory'], 'accuracy.csv'), 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in accuracy:
            w.writerow([epoch, row])
            epoch += 1
    with open(os.path.join(config['out_directory'], 'val_accuracy.csv'), 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in val_accuracy:
            w.writerow([epoch, row])
            epoch += 1

def load_data(config):
    X, Y = pl.import_data(config)
    if config['data_size'] != -1:
        X = X[0:config['data_size']]
        Y = Y[0:config['data_size']]
    test_indices = numpy.random.choice(X.shape[0], config['test_size'])

    X_te, Y_te = X[test_indices], Y[test_indices]
    X_tr, Y_tr = numpy.delete(X, test_indices, axis = 0), numpy.delete(Y, test_indices, axis =0)

    Data = collections.namedtuple('Data', 'x_tr y_tr x_te y_te')
    return Data(X_tr, Y_tr, X_te, Y_te)

class SGDTrainer:

    def __init__(self, config, model, data):
        self.config = config
        self.data = data
        self.model = model
        self.times = []

    def train(self):
        start = time.time()
        history = self.model.fit(self.data.x_tr, self.data.y_tr,
            validation_data=(self.data.x_te, self.data.y_te),
            batch_size=self.config['batch_size'], epochs=self.config['epochs'])
        end = time.time()
        self.times.append(end - start)
        self.losses = history.history['loss']
        self.accuracy = history.history['accuracy']
        self.val_accuracy = history.history['val_accuracy']

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
        self.val_losses = []
        self.val_accuracy = []

    def train(self):
        size = len(self.data.x_tr)
        worker_size = size // len(self.config['workers'])
        itr_size = worker_size // self.config['sync_iterations']
        ev = self.model.evaluate(self.data.x_tr, self.data.y_tr)
        self.losses.append(ev[0])
        self.accuracy.append(ev[1])
        ev = self.model.evaluate(self.data.x_te, self.data.y_te)
        self.val_losses.append(ev[0])
        self.val_accuracy.append(ev[1])
        for e in range(self.config['epochs']):
            time_net = 0
            start = time.time()
            for i in range(self.config['sync_iterations']):
                start = (self.idx * worker_size) + (i * itr_size)
                end = start + itr_size
                x_itr = self.data.x_tr[start:end]
                y_itr = self.data.y_tr[start:end]
                # sklearn.utils.shuffle(x_itr, y_itr, random_state=0)
                history = self.model.fit(x_itr, y_itr,
                    batch_size=self.config['batch_size'], epochs=1)
                start_net = time.time()
                self.server.send_weights(0, numpy.array(self.model.get_weights()))
                if self.idx == 0:
                    all_weights = self.server.wait_and_consume_weights(len(self.config['workers']))
                    weights = numpy.mean(all_weights, axis=0)
                    for j in range(1, len(self.config['workers'])):
                        self.server.send_weights(j, weights)
                else:
                    weights = self.server.wait_and_consume_weights(1)[0]
                self.model.set_weights(weights)
                end_net = time.time()
                time_net += end_net - start_net
            end = time.time()
            self.times.append(end - start)
            self.times_net.append(time_net)
            ev = self.model.evaluate(self.data.x_tr, self.data.y_tr)
            self.losses.append(ev[0])
            self.accuracy.append(ev[1])
            ev = self.model.evaluate(self.data.x_te, self.data.y_te)
            self.val_losses.append(ev[0])
            self.val_accuracy.append(ev[1])

    def get_stats(self):
        return [sum(self.times)], [sum(self.times_net)], self.losses, self.accuracy, self.val_accuracy


def recv_weights(sock, server):
    while server.is_running:
        b = sock.recv(4)
        if len(b) == 0:
            break
        n = int.from_bytes(b, byteorder='big')
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        bio = io.BytesIO(data)
        weights = numpy.load(bio, allow_pickle=True)
        server.received_weights(weights)

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

    def add_connection(self, i, conn):
        with self.conn_cv:
            self.connections[i] = conn
            self.conn_cv.notify_all()

    def received_weights(self, l):
        with self.l_cv:
            self.lists.append(l)
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
                b = len(bs.getbuffer()).to_bytes(4, byteorder='big')
                self.connections[idx].sendall(b)
                self.connections[idx].sendall(bs.getbuffer())

    def end(self):
        for i, c in self.connections.items():
            try:
                c.shutdown(socket.SHUT_WR)
            except OSError:
                print('Socket %d already closed.' % i)
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

        if not args.local:
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
        data = load_data(config)
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['alpha'], momentum=config['beta'], nesterov=False)
        model = pl.new_model(optim=optimizer, lo=pl.crps)
        #model = pl.new_model(optim=optimizer)

        if args.local:
            tr = SGDTrainer(config, model, data)
        else:
            tr = SimuParallelSGDTrainer(config, args.worker_idx, server, model, data)
        tr.train()

        if not args.local:
            server.is_running = False
            server.end()
            for ti in threads:
                ti.join()

        if args.local or args.worker_idx == 0:
            export_stats(config, *tr.get_stats())

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Distributed SGD worker.')
    parser.add_argument('--config_file', help='path to json config file')
    parser.add_argument('--worker_idx', type=int, help='index of this worker in config')
    parser.add_argument('--local', default=False, action='store_true', help='train with local SGD')
    args = parser.parse_args()
    main(args)
