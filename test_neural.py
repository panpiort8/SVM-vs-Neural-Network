import os
import matplotlib.pyplot as plt
from neural_network import *
from tools import *

data_pos, data_neg = load_data("phishing.data", scaling=True)
training_set, validation_set = split_data(data_pos, data_neg)
validation_set, test_set = simple_split(validation_set)

validation_point = 1000
epochs = 200
eta = 0.01
lmbda = 0.1
mi = 0.0

network = Network([30, 100, 2], [-1, 1])
history, epochs_done = network.fit(
    vectorize(training_set),
    epochs=epochs, batch_size=1, eta=eta, lmbda=lmbda, mi=mi,
    validation_set=vectorize(validation_set),
    validation_points=validation_point
)

print("Accuracy on test_set: %.4f" % (100*accuracy(network, test_set)))

h = len(history['validation_cost'])
samples = [(epochs_done/h)*(i+1) for i in range(h)]
plt.figure()
plt.plot(samples, history['validation_cost'], label='validation_cost')
plt.plot(samples, history['training_cost'], label='training_cost')
# plt.plot(partial_triggers, bayes_history, 'bo-', label="bayes")
plt.title("Cost (loss) function of Neural Network")
plt.xlabel("Epochs")
plt.ylabel("Cost (loss)")
plt.legend()
name = "neural_{}_{}_{}_{}.png".format(epochs_done, eta, lmbda, mi)
plt.savefig(os.path.join("graphs", name))
plt.show()
