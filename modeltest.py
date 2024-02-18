import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from joblib import load
from tensorflow.keras.utils import to_categorical
import featurex

# Load the trained model
model = load_model('model/pump_predictions.keras', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Load the fitted scaler
scaler = load('data/scaler.joblib')  # Adjust path as needed

# Load and prepare data
data = pd.read_csv('data/sensor.csv')[featurex.columns].dropna(axis=1, how='all')
sensor_columns = [col for col in data.columns if col.startswith('sensor_')]
data[featurex.columns] = scaler.transform(data[featurex.columns])  # Normalize sensor data

# Map machine status to numerical values
status_mapping = {'NORMAL': 0, 'BROKEN': 1, 'RECOVERING': 2}
data['machine_status'] = data['machine_status'].map(status_mapping)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)][sensor_columns].values
        y = data.iloc[i + seq_length]['machine_status']
        xs.append(x)
        ys.append(y)
        print('create_sequences', i)
    return np.array(xs), np.array(ys)

def create_broken_sequences(data, seq_length):
    broken_indices = data[data['machine_status'] == 1].index
    xs = []
    ys = []

    for idx in broken_indices:
        if idx >= seq_length:  # Ensure there are enough previous data points
            sequence = data.iloc[idx - seq_length: idx]
            x = sequence[sensor_columns].values
            y = data.iloc[idx]['machine_status']
            xs.append(x)
            ys.append(y)
            print('create_broken_sequences', idx)

    return np.array(xs), np.array(ys)

seq_length = 10  # Number of time steps to look back 
X, y = create_sequences(data, seq_length)
X_test, y_test = X[int(len(X) * 0.8):], y[int(len(X) * 0.8):]

# Predict on a subset of test data
test_subset = X_test[:10]
predictions = model.predict(test_subset)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test[:10]

# Evaluate predictions
print("Predicted classes:", predicted_classes)
print("Actual classes:", actual_classes)
print(classification_report(actual_classes, predicted_classes))


# Create sequences from the 'BROKEN' data
X_broken, y_broken = create_broken_sequences(data, seq_length)

if np.isnan(X_broken).any():
    print("NaN values found in the input data")

print("Shape of X_broken:", X_broken.shape)
print("Shape of a single sequence in X_broken:", X_broken[0].shape)

single_sequence = X_broken[0:1]  # Selecting the first sequence
print("Predicting on a single sequence:", model.predict(single_sequence))

# Predict on the 'BROKEN' data
predictions_broken = model.predict(X_broken)
predicted_classes_broken = np.argmax(predictions_broken, axis=1)

# Evaluate predictions
print("Predicted classes (BROKEN):", predicted_classes_broken)
print("Actual classes (BROKEN):", y_broken)

# Detailed classification report
print(classification_report(y_broken, predicted_classes_broken))