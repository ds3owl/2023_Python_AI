# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from three_layer_net import ThreeLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=100, output_size=10)

# 파라미터 로드
path_dir = './ckpt'
file_name = "threelayernet_params.pkl"
network.load_params(os.path.join(path_dir, file_name))

network.make_layer()

test_acc = network.accuracy(x_test, t_test)
print("test acc | " + str(test_acc))
