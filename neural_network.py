import numpy as np
import random


def sigmoid(z):
    # print(1.0/(1.0+np.exp(-z)))
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_derivative(A, Y):
    return (A - Y)


class Network:

    def __init__(self, sizes, labels):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.labels = labels
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.w_acc = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def next_batch(self, training_set, batch_size):
        m = len(training_set)
        for i in np.arange(0, m, batch_size):
            yield training_set[i:i + batch_size]

    def calc_cost(self, a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def avg_cost(self, data):
        cost = 0.0
        for (x, y) in data:
            x = np.array(x, ndmin=2).transpose()
            y = np.array(y, ndmin=2).transpose()
            a = self.feedforward(x)
            cost += self.calc_cost(a, y)
        return cost / len(data)

    def classify(self, x):
        x = np.array(x, ndmin=2).transpose()
        return self.labels[np.argmax(self.feedforward(x))]

    def fit(self, training_set, epochs, batch_size, eta, lmbda=0.0, mi=0.9, validation_set=None, validation_points=200):
        n = len(training_set)
        history = {"validation_acc": [], "training_cost": [], "validation_cost": []}
        for j in range(epochs):
            try:
                random.shuffle(training_set)
                for i, batch in enumerate(self.next_batch(training_set, batch_size)):
                    self.update_batch(batch, eta, lmbda, mi, n)
                    if i % validation_points == 0 and validation_set:
                        validation_cost = self.avg_cost(validation_set)
                        training_cost = self.avg_cost(training_set)
                        print("Epoch %d %d/%d" % (j, i, n))
                        print(' training_cost:      %.4f' % training_cost)
                        print(' validation_cost:    %.4f' % validation_cost)
                        print()
                        history["validation_cost"].append(validation_cost)
                        history["training_cost"].append(training_cost)
            except KeyboardInterrupt:
                return history, j+1

        return history, epochs

    def update_batch(self, mini_batch, eta, lmbda, mi, n):
        X = np.asarray([x[0] for x in mini_batch]).transpose()
        Y = np.asarray([x[1] for x in mini_batch]).transpose()
        nabla_b, nabla_w = self.backprop(X, Y)
        for w, wacc, nw in zip(self.weights, self.w_acc, nabla_w):
            wacc = mi * wacc - (eta / len(mini_batch)) * nw
            w *= (1 - eta * (lmbda / n))
            w += wacc
        for b, nb in zip(self.biases, nabla_b):
            b -= (eta / len(mini_batch)) * nb

    def backprop(self, X, Y):
        nabla_b = [np.zeros((b.shape[0], X.shape[1],)) for b in self.biases]
        nabla_w = [np.zeros((w.shape[0], X.shape[1],)) for w in self.weights]
        activation = X
        activations = [X]
        Zs = []
        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, activation) + b
            Zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)
        self.crnt_cost = np.sum(self.calc_cost(activation[-1], Y)) / X.shape[1]

        delta = cost_derivative(activations[-1], Y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            Z = Zs[-l]
            SP = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * SP
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        nabla_b = [np.expand_dims(np.sum(nb, axis=1), axis=1) for nb in nabla_b]
        return nabla_b, nabla_w
