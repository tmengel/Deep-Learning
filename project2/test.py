import numpy as np
import sys

from project2 import *

learningRate = 0.1
testCNN = NeuralNetwork(input_size=[4,4], loss='bce', learning_rate=0.01, verbose=True)
testCNN.addLayer('convolutional', 1, 'linear', kernel_size=1)
print("CNN Layer")
weights = testCNN.layers[0].weights
print(weights)
testCNN.addLayer('maxpooling',kernel_size=2)
testCNN.addLayer('flatten')


test_input = [[1,8,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# test_dloss = [[0.5,0.5,],[0.5,0.5]]
# test_dloss = [[0.5,0.5,],[0.5,0.5]]
test_dloss_flat = [0.5,0.5,0.5,0.5]
for i in range(10):
    test_output = testCNN.calculate(test_input)
    print("test_output", test_output)
    testCNN.backpropagate(test_dloss_flat)
    
output = testCNN.calculate(test_input)
testflat = testCNN.layers[2].calculatewdeltas(test_dloss_flat)
testmax = testCNN.layers[1].calculatewdeltas(testflat)
testConv = testCNN.layers[0].calculatewdeltas(testmax)
print("testflat", testflat)
print("testmax", testmax)
print("testConv", testConv)
