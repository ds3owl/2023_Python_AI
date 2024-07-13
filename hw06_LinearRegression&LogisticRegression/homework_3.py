import numpy as np
from matplotlib import pyplot as plt

# CSV 파일 경로
csv_file_path = "C:\\2023_python_ai\\myproject\\week06_lab\\lab_03_homework\\chdage.csv"

# CSV 파일을 NumPy 배열로 읽어오기
# delimiter=','는 데이터 구분자로 쉼표를 사용하겠다는 설정
# skip_header=1은 CSV 파일의 첫 번째 행을 무시하겠다는 설정
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# text file input/output

# x = data[:, 0] / 100
x = data[:, 1]
y = data[:, 2]

def sigmoid(x):  # 시그모이드 함수 정의
    return 1/(1+np.exp(-x))

w = np.random.uniform(low=0, high=20)
b = np.random.uniform(low=-20, high=10)
print('w: ', w, 'b: ', b)

num_epoch = 1000000

learning_rate = 0.001

costs = []

eps = 1e-5

for epoch in range(num_epoch):
    hypothesis = sigmoid(w * x + b)

    cost = y * np.log(hypothesis + eps) + (1 - y) * np.log(1 - hypothesis + eps)
    cost = -1 * cost
    cost = cost.mean()

    if cost < 0.0005:
        break

    # reference : https://nlogn.in/logistic-regression-and-its-cost-function-detailed-introduction/
    w = w - learning_rate * ((hypothesis - y) * x).mean()
    b = b - learning_rate * (hypothesis - y).mean()

    costs.append(cost)

    if epoch % 5000 == 0:
        print("{0:2} w = {1:.5f}, b = {2:.5f} error = {3:.5f}".format(
            epoch, w, b, cost))

print("----" * 15)
print("{0:2} w = {1:.5f}, b = {2:.5f} error = {3:.5f}".format(epoch, w, b, cost))


# 예측
x = 20 # True : 0
pred_y = sigmoid(w * x + b)
print("20세일 때 심장병 발병 확률 :", pred_y*100, "%")

x = 80 # True : 1
pred_y = sigmoid(w * x + b)
print("80세일 때 심장병 발병 확률 :", pred_y*100, "%")

x = data[:, 1]
y = data[:, 2]

org_x = np.linspace(0, 100, 100)
pred_y = sigmoid(w * (org_x) + b)

plt.scatter(x, y)
plt.title("Heart Disease by Age")
plt.xlabel("Age")
plt.ylabel("Heart Disease O/X")
plt.plot(org_x, pred_y, 'r')
plt.show()