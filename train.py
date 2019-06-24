import numpy as np
from model import *
from tensorflow.examples.tutorials.mnist import input_data  # 通过tensorflow调用mnist数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

np.random.seed(1)
LAYERS_DIMS = [784, 10]  # 2-layer model
LEARNING_RATE = 0.0075
NUM_ITERATION = 500
ROLL = 1000
PRINT_COST = True

X = mnist.train.images.T
Y = mnist.train.labels.T
X_t = mnist.test.images.T
Y_t = mnist.test.labels.T

parameters = L_layer_model(X, Y, LAYERS_DIMS, LEARNING_RATE, NUM_ITERATION, ROLL,  PRINT_COST)
np.save('parameters_{}'.format(ROLL), parameters)

p = predict(X, Y, parameters)   # 计算整体精度
print('train dataset accuracy: ', p)
p = predict(X_t, Y_t, parameters)
print('test dataset accuracy: ', p)
