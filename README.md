# MNIST_FCNN_Model

通过numpy实现 基于FCNN（全连接神经网络）的手写数字识别
-
## 项目环境
Python 3.6
Tensorflow 1.13.1  (用于载入MNIST数据集）
numpy 1.16.4
matplotlib 3.1.0
opencv-python 4.1.0.25
-
## 项目文件介绍
MNIST_data: MNIST数据集，内容不需要解压，通过tensorflow调用即可；
model.py: 存储了项目所需要使用的函数：
-
- sigmoid(Z): 激活函数Sigmoid: A = 1 / (1 + exp(-Z))
- relu(Z): 激活函数ReLU: A = maximum(0, Z)
- relu_backward(dA, cache): 反向传播中ReLU后项的梯度: dZ[Z<=0] = 0
- sigmoid_backward(dA, cache): 反向传播sigmoid后项的梯度: dZ = dA * sigmoid(Z) * sigmoid(1 - Z)
- initialize_parameters(layers_dims): 初始化模型
- linear_forward(A, W, b): 前向传播线性部分：Z = W * A + b
- linear_activation_forward(A_prev, W, b, activation): 前向传播激活函数部分, A = sigmoid(Z) | A = relu(Z)
- L_model_forward(X, parameters): 前向传播搭建，LINEAR->RELU]* (L-1)->LINEAR->SIGMOID->AL
- compute_cost(AL, Y): 计算cost损失值，cost = 1/m * (Y * (AL) - ((1 - Y) * (1 - AL)))
- linear_backward(dZ, cache): 线性反向传播，计算dW, db, dA_prev(激活函数前)
-- dW = 1/m * (dZ*A_prev.T)
-- db = 1/m * dZ
- linear_activation_backward(dA, cache, activation): 激活函数反向传播，计算dW, db, dA_prev(激活函数后)
-- 'relu': dA_prev, dW, db = linear_backward(relu_backward(dA))
-- 'sigmoid': dA_prev, dW, db = linear_backward(sigmoid_backward(dA))
- L_model_backward(AL, Y, caches): 整合前两项，建立所有的参数的反向传播模型:
- update_parameters(parameters, grads, learning_rate): 更新所有的W, b参数：
-- W = W - α * dW   (α: 学习率)
-- b = b - α * db
- L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=1000, roll=1000, print_cost=True): 对训练过程进行封装
- predict(X, y, parameters): 输入训练或者测试数据，获得数据前向传播后的预测值，返回模型精确度：
-- 精确度 = 1/m * (正确项-错误项*8) / 10     (每一项有10个单位)
- show_predict_image(X, y, parameters): 输入一张图片，显示图片和模型预测值以及真实值

train.py: 训练模型，并存储训练结果参数（详见 train）
run_model.py: 读取参数文件，测试模型预测准确率（详见 run）
-
## train
原始数据在MNIST_data目录下，数据来自[http://yann.lecun.com/exdb/mnist/]
训练代码: train.py:
-
- X：训练数据集图片数组（784, 60000）of （28*28, example）
- Y：训练数据集标签   (10, 60000)   of （10, example）
- X_t：测试数据集图片数组  (784, 10000)
- Y_t：测试数据集标签  (10, 10000)

- 随机种子：np.random.seed(1)
- FCNN层数和每层对应的神经元数量：LAYERS_DIMS = []    # 第一层和最后一层分别与X，Y的shape对应
- 学习率：LEARNING_RATE
- 训练轮数：NUM_ITERATION
- 训练集迭代数量：ROLL
- 打印参数：PRINT_COST

- 训练并返回训练后的参数：parameters = L_layer_model(X, Y, LAYERS_DIMS, LEARNING_RATE, NUM_ITERATION, ROLL,  PRINT_COST)
- 保存：np.save('parameters_{}'.format(ROLL), parameters)

## run
运行代码：run_model.py
-
- X：训练数据集图片数组（784, 60000）of （28*28, example）
- Y：训练数据集标签   (10, 60000)   of （10, example）
- X_t：测试数据集图片数组  (784, 10000)
- Y_t：测试数据集标签  (10, 10000)
- parameters: W1, W2, W3, ...; b1, b2, b3, ...

- 加载预先训练的参数：parameters = np.load('parameters_500.npy', allow_pickle=True).item()
- 用此参数测试测试集精确度：
-- p = predict(X_t, Y_t, parameters)   # 计算测试集精度
-- print('Accuracy: ', p)

- 查看其中一张图片的结果（第100张）：
-- x = X_t[..., 100].reshape(784, 1)
-- y = Y_t[..., 100].reshape(10, 1)
-- show_predict_image(x, y, parameters)

## 参考文献
- [1] Andrew Ng (instructor), Kian Katanforoosh (Head TA), Younes Bensouda Mourri (TA). Deeplearing. [https://www.coursera.org/learn/neural-networks-deep-learning]
