# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mnist_loader
import network
import time

(training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()
print "Load data finished."


network = network.Network([784, 30, 10])
print "Network initialize completed."

start = time.clock()
network.SGD(training_data, epochs = 30, mini_batch_size = 10, eta = 3.0, 
            test_data = validation_data)
elapsed = (time.clock() - start)
print("Training Time used:",elapsed)

print "Evaluate: {0} / {1}".format(network.evaluate(test_data), len(test_data))
