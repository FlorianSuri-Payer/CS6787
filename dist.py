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

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
ALPHA = 0.001
BETA = 0.99

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

def train(dataset, model, epochs=1):
    history = model.fit(x=dataset, epochs=epochs)
    return history

def export_stats(history):
    #matplotlib.pyplot.ylabel('training loss')
    #matplotlib.pyplot.xlabel('epoch')
    #matplotlib.pyplot.plot(training_loss, '-or')
    #matplotlib.pyplot.savefig('training_loss.png')
    #matplotlib.pyplot.clf()

    #matplotlib.pyplot.ylabel('training error')
    #matplotlib.pyplot.xlabel('epoch')
    #matplotlib.pyplot.plot(training_error, '-og')
    #matplotlib.pyplot.savefig('training_error.png')
    #matplotlib.pyplot.clf()

    #matplotlib.pyplot.ylabel('validation error')
    #matplotlib.pyplot.xlabel('epoch')
    #matplotlib.pyplot.plot(validation_error, '-ob')
    #matplotlib.pyplot.savefig('validation_error.png')
    #matplotlib.pyplot.clf()
    pass

def load_data():
    mnist = tf.keras.datasets.mnist
    (X_tr, Y_tr), (X_te, Y_te) = mnist.load_data()
    X_tr = X_tr.reshape(X_tr.shape[0], 28, 28, 1) / 255.0
    X_te = X_te.reshape(X_te.shape[0], 28, 28, 1) / 255.0
    Y_tr = tf.keras.utils.to_categorical(Y_tr)
    Y_te = tf.keras.utils.to_categorical(Y_te)
    return X_tr, Y_tr, X_te, Y_te

def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

class SimuParallelSGDTrainer:

    def __init__(self, config, idx, server):
        self.config = config
        self.idx = idx
        self.server = server
        self.X_tr, self.Y_tr, self.X_te, self.Y_te = load_data()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=ALPHA, momentum=BETA, nesterov=False)
        self.model = build_dense_model((28, 28, 1), 25, [25, 25, 25], 10, self.optimizer)
        self.model.summary()
        grad_tensor = tf.keras.backend.gradients(self.model.output, self.model.input)
        self.grad_fn = tf.keras.backend.function([self.model.input], grad_tensor)

    def get_sample(self, k):
        rows = numpy.random.choice(self.X_tr.shape[0], k)
        return self.X_tr[rows, :]

    def get_partitioned_sample(self):
        return self.get_sample(len(self.X_tr) // len(self.config['workers']))

    def aggregate_gradients(self):
        self.server.add_list(self.idx, self.grad_fn([self.get_partitioned_sample()]))
        grads = self.server.wait_and_consume_gradients()
        # Sum up gradients to compute exact global gradient
        grad = grads[0][0]
        for i in range(len(grads)):
            for j in range(len(grads[i])):
                if i == 0 and j == 0:
                    continue
                grad[j] += grads[i][j]
        return grad

    def train(self):
        size = len(self.X_tr)
        worker_size = size // len(self.config['workers'])
        itr_size = worker_size // self.config['sync_iterations']
        for i in range(self.config['sync_iterations']):
            start = (self.idx * worker_size) + (i * itr_size)
            end = start + itr_size
            self.model.fit(self.X_tr[start:end], self.Y_tr[start:end],
                validation_split=0.1, batch_size=self.config['batch_size'],
                epochs=1)

            self.server.send_weights(0, numpy.array(self.model.get_weights()))
            if self.idx == 0:
                all_weights = self.server.wait_and_consume_weights(len(self.config['workers']))
                weights = numpy.mean(all_weights, axis=0)
                for j in range(1, len(self.config['workers'])):
                    self.server.send_weights(j, weights)
            else:
                weights = self.server.wait_and_consume_weights(1)[0]
                print(weights)
            self.model.set_weights(weights)

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

        tr = SimuParallelSGDTrainer(config, args.worker_idx, server)
        tr.train()

        print('done training')

        server.is_running = False
        server.end()
        for ti in threads:
            ti.join()
        #export_stats(history)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Distributed SGD worker.')
    parser.add_argument('--config_file', help='path to json config file')
    parser.add_argument('--worker_idx', type=int, help='index of this worker in config')
    args = parser.parse_args()
    main(args)

