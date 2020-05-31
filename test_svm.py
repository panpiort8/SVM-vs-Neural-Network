import os
import json
import matplotlib.pyplot as plt
from svm import *
from tools import *

data_pos, data_neg = load_data("phishing.data")
training_set, validation_set = split_data(data_pos, data_neg)
validation_set, test_set = simple_split(validation_set)

validation_point = 1000
epochs = 200
C = 2
t = 4


def kernel(x, z):
    return np.exp(-np.linalg.norm(x - z, ord=2) / (2 * t * t))


svm = SVM(training_set=training_set, kernel=kernel)
history, epochs_done = svm.fit(
    epochs=epochs, C=C,
    validation_set=validation_set[:500],
    validation_point=validation_point
)

print("Accuracy on test_set: %.4f" % (100*accuracy(svm, test_set)))

h = len(history["validation_acc"])
samples = [(epochs_done / h) * (i + 1) for i in range(h)]
print(len(samples))
plt.figure()
plt.plot(samples, history['validation_acc'], label='validation_acc')
plt.title("Validation accuracy for SVM")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(os.path.join("graphs", name + ".png"))
plt.show()
