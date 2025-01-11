import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# พารามิเตอร์
N, step, n_train = 100, 11, 70
t = np.arange(N)
y = np.sin(0.5 * t) * 20 + 10 + np.random.randn(N) * 5

# ฟังก์ชันแปลงข้อมูล
def convert_to_matrix(data, step):
    X = [data[i:i+step] for i in range(len(data) - step)]
    return np.array(X), np.array(data[step:])

# แบ่งข้อมูลและปรับรูปแบบ
x_train, y_train = convert_to_matrix(y[:n_train], step)
x_test, y_test = convert_to_matrix(y[n_train:], step)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# โมเดล RNN
model = Sequential([SimpleRNN(32, input_shape=(step, 1), activation="relu"), Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# การพยากรณ์
input_seq, predicted = y[:step], []
for _ in range(len(y) - step):
    next_val = model.predict(input_seq[-step:][np.newaxis, ..., np.newaxis], verbose=0)
    predicted.append(next_val[0, 0])
    input_seq = np.append(input_seq, next_val)

# การ Plot
plt.figure(figsize=(10, 6))
plt.plot(y, label="Original", color="blue")
plt.plot(np.arange(step, N), predicted, label="Predict", color="red", linestyle="--")
plt.axvline(x=n_train, color="magenta", linestyle="-", linewidth=2)
plt.ylim(-22, 39)
plt.xticks(np.arange(0, 101, 20))
plt.title("Comparison of Original and Predicted Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
