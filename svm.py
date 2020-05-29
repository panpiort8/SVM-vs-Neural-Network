# from http://cs229.stanford.edu/materials/smo.pdf

import numpy as np

class SVM:

    def __init__(self, training_set):
        self.X = [x for x, y in training_set]
        self.Y = [y for x, y in training_set]
        self.m = len(training_set)
        self.alpha = [0]*self.m
        self.b = 0

    def accuracy(self, data):
        oks = 0
        for (x, y) in data:
            if self.classify(x) == y:
                oks += 1
        return oks / len(data)

    def classify(self, x):
        return 1 if self.f(x) >= 0 else -1

    def kernel(self, x, z):
        return np.exp(-np.linalg.norm(x-z, ord=2)/32)

    def f(self, x):
        f = 0
        for a, z, y in zip(self.alpha, self.X, self.Y):
            f += a*y*self.kernel(x, z)
        return f+self.b

    def fast_f(self, i):
        pass

    def fit(self, epochs, C=2, validation_set=None, validation_point=200, tol=10e-5,):
        history = {"validation_acc": []}
        for epoch in range(epochs):
            num_changed = 0
            try:
                for i in range(self.m):
                    print("{}/{}".format(i, self.m))
                    if i % validation_point == 0 and validation_set:
                        print("Epoch %d %d/%d" % (epoch, i, self.m))
                        print(' changed alphas      %d' % num_changed)
                        validation_acc = self.accuracy(validation_set)
                        history["validation_acc"].append(validation_acc)
                        print(' validation_acc:     %.2f' % (100 * validation_acc))
                        print()
                    yi, ai, xi = self.Y[i], self.alpha[i], self.X[i]
                    Ei = self.f(xi) - yi
                    if (yi*Ei < -tol and ai < C) or (yi*Ei > tol and ai > 0):
                        j = np.random.choice(np.setdiff1d(range(self.m), i))
                        yj, aj, xj = self.Y[j], self.alpha[j], self.X[j]
                        Ej = self.f(xj) - yj
                        if yi != yj:
                            L = max(0, aj-ai)
                            H = min(C, C+aj-ai)
                        else:
                            L = max(0, aj + ai - C)
                            H = min(C, aj + ai)
                        if L == H:
                            continue
                        mi = 2*self.kernel(xi, xj) - self.kernel(xi, xi) - self.kernel(xj, xj)
                        if mi >= 0:
                            continue

                        new_aj = aj - ((yj*(Ei-Ej))/mi)
                        new_aj = min(new_aj, H)
                        new_aj = max(new_aj, L)

                        if abs(aj-new_aj) < tol:
                            continue

                        new_ai = ai + yj*yi*(aj-ai)

                        b1 = self.b - Ei - yi*(new_ai - ai)*self.kernel(xi, xi) - yj*(new_aj-aj)*self.kernel(xi, xj)
                        b2 = self.b - Ej - yi*(new_ai - ai)*self.kernel(xi, xj) - yj*(new_aj-aj)*self.kernel(xj, xj)
                        if 0 < ai < C:
                            new_b = b1
                        elif 0 < aj < C:
                            new_b = b2
                        else:
                            new_b = (b1+b2)/2

                        self.b = new_b
                        self.alpha[i] = new_ai
                        self.alpha[j] = new_aj

                        num_changed += 1
            except KeyboardInterrupt:
                return history, epoch+1

        return history, epochs
