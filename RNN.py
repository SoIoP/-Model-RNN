import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

# Step 1: Generate Sine Wave with Noise
def generate_sine_wave(seq_len=50, num_samples=1000):
    x = np.linspace(0, 2 * np.pi * num_samples / seq_len, num_samples)
    sine_wave = np.sin(x)  # Pure sine wave
    noise = np.random.normal(0, 0.1, num_samples)  # Add some noise
    noisy_signal = sine_wave + noise
    return noisy_signal, sine_wave

# Step 2: Prepare Data
def prepare_data(signal, seq_len=50):
    X, y = [], []
    for i in range(len(signal) - seq_len):
        X.append(signal[i:i + seq_len])
        y.append(signal[i + seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Step 3: Build RNN Model
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, activation='tanh', input_shape=input_shape),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Step 4: Train Model
seq_len = 50
noisy_signal, sine_wave = generate_sine_wave(seq_len=seq_len, num_samples=2000)
X, y = prepare_data(noisy_signal, seq_len=seq_len)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]

model = build_rnn_model((seq_len, 1))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Step 5: Evaluate and Plot
def plot_results(noisy_signal, sine_wave, predicted_signal, seq_len):
    plt.figure(figsize=(14, 6))
    plt.plot(noisy_signal, label='Noisy Signal', alpha=0.5)
    plt.plot(sine_wave, label='Original Sine Wave', alpha=0.8)
    plt.plot(range(seq_len, len(predicted_signal) + seq_len), predicted_signal, label='Predicted Signal', alpha=0.8)
    plt.legend()
    plt.title('Sine Wave Prediction')
    plt.show()

# Predict
predicted_signal = []
input_seq = X_test[0]
for _ in range(len(X_test)):
    pred = model.predict(input_seq[np.newaxis, ...])
    predicted_signal.append(pred[0, 0])
    input_seq = np.roll(input_seq, -1, axis=0)
    input_seq[-1] = pred

# Plot results
plot_results(noisy_signal, sine_wave, predicted_signal, seq_len)
