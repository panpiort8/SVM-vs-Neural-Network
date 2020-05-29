import numpy as np
from sklearn import preprocessing


def vectorize(data):
    vect = []
    for x, y in data:
        y = [1, 0] if y == -1 else [0, 1]
        vect.append((x, y))
    return vect


def load_data(path, scaling=False):
    data_pos, data_neg, Xs, ys = [], [], [], []
    with open(path, 'r') as file:
        for line in file:
            line = line.split(',')
            Xs.append(np.array([float(x) for x in line[:-1]]))
            ys.append(int(line[-1]))

    Xs = np.array(Xs)
    if scaling:
        Xs = preprocessing.scale(Xs)

    for x, y in zip(Xs, ys):
        if y == 1:
            data_pos.append((x, y))
        else:
            data_neg.append((x, y))
    return data_pos, data_neg


def split_data(data_pos, data_neg, p=0.6):
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    len_pos = int(len(data_pos) * p)
    len_neg = int(len(data_neg) * p)
    training = data_pos[:len_pos] + data_neg[:len_neg]
    test = data_pos[len_pos:] + data_neg[len_neg:]
    np.random.shuffle(training)
    np.random.shuffle(test)
    return training, test


def simple_split(data, p=0.25):
    np.random.shuffle(data)
    size = int(len(data) * p)
    return data[:size], data[size:]


def accuracy(model, data):
    ok = 0
    for x, y in data:
        y0 = model.classify(x)
        if y0 == y:
            ok += 1
    return ok / len(data)


def error_rate(model, data):
    return 1 - accuracy(model, data)


def feed_with_data(measure, data):
    def func(model):
        return measure(model, data)

    return func
