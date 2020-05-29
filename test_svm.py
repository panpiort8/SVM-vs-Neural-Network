import os
import matplotlib.pyplot as plt
from svm import *
from tools import *

data_pos, data_neg = load_data("phishing.data", scaling=True)
training_set, validation_set = split_data(data_pos, data_neg)
validation_set, test_set = simple_split(validation_set)

validation_point = 200
epochs = 200
C = 4

svm = SVM(training_set=training_set)
history, epochs_done = svm.fit(
    epochs=epochs, C=C,
    validation_set=validation_set[:100],
    validation_point=validation_point
)

# print("Accuracy on test_set: %.4f" % (100*accuracy(network, test_set)))

h = len(history["validation_acc"])
samples = [(epochs_done/h)*(i+1) for i in range(h)]
print(len(samples))
plt.figure()
plt.plot(samples, history['validation_acc'], label='validation_acc')
# plt.plot(partial_triggers, bayes_history, 'bo-', label="bayes")
plt.title("Validation accuracy for SVM")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
name = "svm_{}_{}.png".format(epochs_done, C)
plt.savefig(os.path.join("graphs", name))
plt.show()
