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
import sys

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

def export_stats(config, times, times_net, losses, accuracy, val_losses):
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

    matplotlib.pyplot.ylabel('validation loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.plot(val_losses, '-og')
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
    with open(os.path.join(config['out_directory'], 'val_losses.csv'), 'w') as f:
        w = csv.writer(f)
        epoch = 0
        for row in val_losses:
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

class EarlyStoppingTime(tf.keras.callbacks.Callback):

    def __init__(self, monitor='val_loss', target_time=0):
        super(EarlyStoppingTime, self).__init__()
        self.target_time = target_time
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.start = time.time()
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        if (now - self.start) > self.target_time:
            print('Stopping after %d seconds.' % int(now - self.start))
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

class EarlyStoppingMonitorThreshold(tf.keras.callbacks.Callback):

    def __init__(self, monitor='val_loss', target_value=0, verbose=0):
        super(EarlyStoppingMonitorThreshold, self).__init__()

        self.monitor = monitor
        self.target_value = target_value
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if current < self.target_value:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s',
                self.monitor, ','.join(list(logs.keys())))
        return monitor_value

class SGDTrainer:

    def __init__(self, config, model, data, scheduler):
        self.config = config
        self.data = data
        self.model = model
        self.times = []
        self.scheduler = scheduler
        self.losses = []
        self.accuracy = []
        self.val_losses = []

    def train(self):
        cb = [self.scheduler]
        if self.config['early_stopping_loss'] != -1:
            cb.append(EarlyStoppingMonitorThreshold(target_value=self.config['early_stopping_loss']))
        if self.config['early_stopping_time'] != -1:
            cb.append(EarlyStoppingTime(target_time=self.config['early_stopping_time']))
        ev = self.model.evaluate(self.data.x_tr, self.data.y_tr)
        self.losses.append(ev[0])
        self.accuracy.append(ev[1])
        start = time.time()
        history = self.model.fit(self.data.x_tr, self.data.y_tr,
            validation_data=(self.data.x_te, self.data.y_te), callbacks=cb,
            batch_size=self.config['batch_size'], epochs=self.config['epochs'])
        end = time.time()
        self.times.append(end - start)
        self.losses.extend(history.history['loss'])
        self.accuracy.extend(history.history['accuracy'])
        self.val_losses.extend(history.history['val_loss'])

    def get_stats(self):
        return self.times, [], self.losses, self.accuracy, self.val_losses


class SimuParallelSGDTrainer:

    def __init__(self, config, idx, server, model, data, scheduler):
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
        self.scheduler = scheduler

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
        train_start = time.time()
        for e in range(self.config['epochs']):
            time_net = 0
            start_time = time.time()
            for i in range(self.config['sync_iterations']):
                start = (self.idx * worker_size) + (i * itr_size)
                end = start + itr_size
                x_itr = self.data.x_tr[start:end]
                y_itr = self.data.y_tr[start:end]
                # sklearn.utils.shuffle(x_itr, y_itr, random_state=0)
                history = self.model.fit(x_itr, y_itr, initial_epoch=e, callbacks=[self.scheduler],
                    batch_size=self.config['batch_size'], epochs=e+1)
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
            self.times.append(end - start_time)
            self.times_net.append(time_net)
            ev = self.model.evaluate(self.data.x_tr, self.data.y_tr)
            self.losses.append(ev[0])
            self.accuracy.append(ev[1])
            ev = self.model.evaluate(self.data.x_te, self.data.y_te)
            self.val_losses.append(ev[0])
            self.val_accuracy.append(ev[1])
            if self.config['early_stopping_loss'] != -1 and ev[0] < self.config['early_stopping_loss']:
                break
            now = time.time()
            if self.config['early_stopping_time'] != -1 and now - train_start > self.config['early_stopping_time']:
                break

    def get_stats(self):
        return [sum(self.times)], [sum(self.times_net)], self.losses, self.accuracy, self.val_losses


def recv_weights(sock, server):
    while server.is_running:
        try:
            b = sock.recv(4)
            if not b or len(b) == 0:
                break
            n = int.from_bytes(b, byteorder='big')
            data = bytearray()
            while len(data) < n:
                packet = sock.recv(n - len(data))
                if not packet or len(packet) == 0:
                    break
                data.extend(packet)
            if len(data) > 0:
                bio = io.BytesIO(data)
                weights = numpy.load(bio, allow_pickle=True)
                server.received_weights(weights)
        except:
            break

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
                c.shutdown(socket.SHUT_RDWR)
                print('Closed socket %d.' % i)
            except OSError:
                print('Socket %d already closed.' % i)
        self.shutdown()
        print("Done shutting down.")

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
        print('Done loading data.')
        optimizer = tf.keras.optimizers.SGD(learning_rate=config['alpha'], momentum=config['beta'], nesterov=False)
        model = pl.new_model(optim=optimizer, lo=pl.crps)
        #model = pl.new_model(optim=optimizer)

        def scheduler(epoch):
            return config['alpha'] * (0.96**epoch)
        sched = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if args.local:
            tr = SGDTrainer(config, model, data, sched)
        else:
            tr = SimuParallelSGDTrainer(config, args.worker_idx, server, model, data, sched)
        tr.train()

        print("Done training.")

        if not args.local:
            server.is_running = False
            server.end()
            for c in connections:
                try:
                    c.shutdown(socket.SHUT_RDWR)
                    print('Closed socket %d.' % i)
                except OSError:
                    print('Socket %d already closed.' % i)
            i = 0
            for ti in threads:
                ti.join(timeout=1)
                if ti.is_alive():
                    print("Failed to join thread %d." % i)
                else:
                    print("Successfully joined thread %d." % i)
                i += 1

        if args.local or args.worker_idx == 0:
            print("Exporting stats.")
            export_stats(config, *tr.get_stats())

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Distributed SGD worker.')
    parser.add_argument('--config_file', help='path to json config file')
    parser.add_argument('--worker_idx', type=int, help='index of this worker in config')
    parser.add_argument('--local', default=False, action='store_true', help='train with local SGD')
    args = parser.parse_args()
    main(args)
    print("Done.")
