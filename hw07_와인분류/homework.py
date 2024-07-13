import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 데이터 획득하기
wine_data = np.loadtxt('wine.csv', delimiter=",", skiprows=1, dtype=np.float32)

x = np.array(wine_data[:, 1:])
y = np.array(wine_data[:, 0])

# 2 데이터 시각화
sns.set(style="ticks", color_codes=True)
wine_df = pd.DataFrame(data=wine_data, columns=["Class", "Alcohol", "Malic.acid", "Ash", "ACl", "Mg", "Phenols", "Flavanoids", "Nonflavanoid.phenols", "Proanth", "Color.int", "Hue", "OD", "Proline"])
g = sns.pairplot(wine_df, hue="Class", palette=["#ff0000", "#00ff00", "#0000ff"])
plt.show()

# 3-1. 정규화
x_max = x.max(axis=0)
x_normal = x / x_max

x = x_normal.copy()

label = []
for ans in wine_data[:, 0]:
    if ans == 1:
        label.append(0)
    elif ans == 2:
        label.append(1)
    else:
        label.append(2)

# 3-2. 원 핫 인코딩 (one-hot encoding)
num = np.unique(label, axis=0)
num = num.shape[0] # 와인의 종류의 수인 3이 저장됨

encoding = np.eye(num)[label]
y = np.array(label)
y_hot = encoding.copy()

# 4. 가설(Hypothesis) 설정
dim = 13
nb_classes = 3

print('x shape: ', x.shape, 'y shape: ', y.shape)

w = np.random.normal(size=[dim, nb_classes])
b = np.random.normal(size=[nb_classes])

print('w shape: ', w.shape, 'b shape: ', b.shape)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

hypothesis = softmax(np.dot(x, w) + b)
print('hypothesis: ', hypothesis.shape)

# hypothesis:  (178, 3)

# 5. 비용함수 설정
eps = 1e-7

num_epoch = 50000
learning_rate = 1000
costs = []

m, n = x.shape

# 6. 경사하강법을 통한 w,b 훈련
for epoch in range(num_epoch):
    z = np.dot(x, w) + b
    hypothesis = softmax(z)

    cost = y_hot * np.log(hypothesis + eps)
    cost = -cost.mean()

    if cost < 0.0000005:
        break

    w_grad = (1 / m) * np.dot(x.T, (hypothesis - y_hot))
    b_grad = (1 / m) * np.sum(hypothesis - y_hot)

    w = w - learning_rate * w_grad
    b = b - learning_rate * b_grad

    costs.append(cost)

    if epoch % 50 == 0:
        print("{0:2} error = {1:.5f}".format(epoch, cost))

print("----" * 15)
print("{0:2} error = {1:.5f}".format(epoch, cost))

np.savetxt('weights.txt', w, delimiter=',')
np.savetxt('bias.txt', b, delimiter=',')

plt.figure(figsize=(10, 7))
plt.plot(costs)
plt.xlabel('Epochs')
plt.ylabel('Costs')
plt.show()

# 7. 훈련된 w,b를 이용하여 예측해보기
def predict(x, w, b):
    t = np.dot(x, w) + b
    z = softmax(t)
    prediction = np.argmax(z, axis=1)
    return prediction

predictions = predict(x, w, b)

for i in range(len(predictions)):
    print("데이터 {}의 예측 결과: 클래스 {}, 실제 클래스 {}".format(i, predictions[i], int(y[i])))

# 8. 예측된 결과로 정확도(Accuracy) 측정하기
correct_predictions = np.sum(predictions == y)
accuracy = correct_predictions / x.shape[0]
print("Accuracy: {:.2f}%".format(accuracy * 100))