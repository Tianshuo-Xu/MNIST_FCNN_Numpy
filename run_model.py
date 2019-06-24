import numpy as np
from model import *
from tensorflow.examples.tutorials.mnist import input_data  # 通过tensorflow调用mnist数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

X = mnist.train.images.T
Y = mnist.train.labels.T
X_t = mnist.test.images.T
Y_t = mnist.test.labels.T

parameters = np.load('parameters_500.npy', allow_pickle=True).item()    # 加载模型参数

p = predict(X_t, Y_t, parameters)   # 计算测试集精度
print('Accuracy: ', p)

# 抽看一张图片及预测结果
x = X_t[..., 100].reshape(784, 1)
y = Y_t[..., 100].reshape(10, 1)
show_predict_image(x, y, parameters)
