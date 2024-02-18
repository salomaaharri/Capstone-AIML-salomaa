import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from joblib import load

# Assuming featurex is a module containing definitions for engineered features
import featurex

# Load the trained model
model = load_model('model/pump_predictionsv1.keras', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the fitted scaler
scaler = load('data/scaler.joblib')

# Load and prepare the sensor data
sensordata = pd.read_csv('data/sensor_engineered.csv')
danger_zones_df = pd.read_csv('data/danger_zones.csv')
danger_zones = danger_zones_df.to_dict(orient='records')
sensordata = sensordata[featurex.columns]
status_mapping = {'NORMAL': 0, 'BROKEN': 1}
sensordata['machine_status'] = sensordata['machine_status'].map(status_mapping)

# Normalize sensor data using a modified approach for batch processing
sensor_columns = [col for col in sensordata.columns if col.startswith('sensor_')]
sensordata[sensor_columns] = scaler.transform(sensordata[sensor_columns])

def create_sequences(data, seq_length=10):
    """Generates sequences for LSTM prediction."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)][sensor_columns].values
        y = data.iloc[i + seq_length]['machine_status']
        xs.append(x)
        ys.append(y)
        if i % 10000 == 0:  # Check if i is a multiple of 10000
            print('create_sequences', i)
    return np.array(xs), np.array(ys)

# Generate sequences for testing
X, y = create_sequences(sensordata)
X_test, y_test = X[int(len(X) * 0.8):], y[int(len(X) * 0.8):]

# Predict and evaluate on a subset of test data
test_subset = X_test[:10]
predictions = model.predict(test_subset)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = y_test[:10]

print("Predicted classes:", predicted_classes)
print("Actual classes:", actual_classes)

