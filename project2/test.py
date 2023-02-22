import numpy as np
import sys

from project2 import *

learningRate = 0.1
testCNN = NeuralNetwork(input_size=[3,3], loss='bce', learning_rate=0.01, verbose=True)
testCNN.addLayer('convolutional', 2, 'linear', kernel_size=2)
print("CNN Layer")
weights = testCNN.layers[0].weights
print(weights)
testCNN.addLayer('maxpooling',kernel_size=2)



test_input = [[1,1,1],[1,1,1],[1,1,1]]
output = testCNN.calculate(test_input)

print(output.shape)
print(output)

print('training network')
#print(test.architecture)

# res = test.calculate(x)
# print(res)
