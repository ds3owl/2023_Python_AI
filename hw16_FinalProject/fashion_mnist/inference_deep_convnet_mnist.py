# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.fashion_mnist import load_fashionMNIST
from model.deep_convnet import DeepConvNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_fashionMNIST(normalize=True, flatten=False, one_hot_label=True)

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = DeepConvNet()

# 매개변수 가져오기
file_name = "deep_convnet_params.pkl"
network.load_params(os.path.join(file_name))
print("Parameter Load Complete!")

test_acc = network.accuracy(x_test, t_test)
print("test acc | ", format(test_acc*100, ".2f"), '%')