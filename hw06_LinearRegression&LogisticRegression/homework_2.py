import numpy as np
from matplotlib import pyplot as plt

# CSV 파일 경로
csv_file_path = "C:\\2023_python_ai\\myproject\\week06_lab\\lab_03_homework\\student_info.csv"

# CSV 파일을 NumPy 배열로 읽어오기
# delimiter=','는 데이터 구분자로 쉼표를 사용하겠다는 설정
# skip_header=1은 CSV 파일의 첫 번째 행을 무시하겠다는 설정
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)

# 정규화
x = data[:, 2] / 100  # weight 컬럼을 정규화
y = data[:, 3] / 100  # height 컬럼을 정규화

# 최대 반복 횟수
num_epoch = 2000

# 학습율 (leaning_rate)
learning_rate = 0.2

costs = []
# random 한 값으로 w, b를 초기화합니다.
w = np.random.uniform(low=1, high=5)
b = np.random.uniform(low=0, high=5)
print('w: ', w, 'b: ', b)

for epoch in range(num_epoch):
    y_hat = w * x + b
    error = ((y_hat - y) ** 2)
    cost = error.mean()

    if cost < 0.0005:
        break

    w = w - learning_rate * ((y_hat - y) * x).mean()
    b = b - learning_rate * (y_hat - y).mean()

    costs.append(cost)

    if epoch % 5 == 0:
        print("{0:2} w = {1:.5f}, b = {2:.5f} error = {3:.5f}".format(epoch, w, b, cost))

print("----" * 15)
print("{0:2} w = {1:.5f}, b = {2:.5f} error = {3:.5f}".format(epoch, w, b, cost))

# 예측
x = 50
y_predict = x / 100 * w + b
print("50kg일 때의 예측키 :",y_predict * 100)

x = 60
y_predict = x / 100 * w + b
print("60kg일 때의 예측키 :",y_predict * 100)

x = 70
y_predict = x / 100 * w + b
print("70kg일 때의 예측키 :",y_predict * 100)

x = 80
y_predict = x / 100 * w + b
print("80kg일 때의 예측키 :",y_predict * 100)

# 예측
org_x = np.linspace(43, 110, 100)
pred_x = org_x / 100  # 정규화
pred_y = w * pred_x + b
pred_y = pred_y * 100  # 정규화 반대 과정

plt.scatter(data[:, 2], data[:, 3])
plt.title("Weight / Height")
plt.xlabel("Weight (kg)")
plt.ylabel("Height (cm)")
plt.plot(org_x, pred_y)
plt.axis([43, 110, 132, 198])
plt.show()