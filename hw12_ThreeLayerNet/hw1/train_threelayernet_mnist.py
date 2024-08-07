# coding: utf-8
import sys, os

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from three_layer_net import ThreeLayerNet
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(0)

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

network = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=100, output_size=10)
network.make_layer()

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('iter: (%d:%d)' % (i, iters_num), 'train acc: ', train_acc, ' test acc: ', test_acc)

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 파라미터 저장
path_dir = './ckpt'
file_name = "threelayernet_params.pkl"
if not os.path.isdir(path_dir):
    os.mkdir(path_dir)

network.save_params(os.path.join(path_dir, file_name))
