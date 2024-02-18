import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from joblib import dump
from joblib import load
import numpy as np
import featurex

# Load and clean the data
print('Loading and cleaning the data')
data = pd.read_csv('data/sensor_engineered.csv')

# Combine 'BROKEN' and 'RECOVERING' into a single 'BROKEN' class
# data['machine_status'] = data['machine_status'].replace(['RECOVERING'], 'BROKEN')

# Map machine status to numerical values: 'NORMAL' to 0 and 'BROKEN' to 1
status_mapping = {'NORMAL': 0, 'BROKEN': 1}
data['machine_status'] = data['machine_status'].map(status_mapping)

# Filter the data to include only the relevant sensor columns
data = data[featurex.sensors + ['machine_status']]

# Fill NaNs with mean or median of the respective columns
for col in featurex.sensors:
    data[col].fillna(data[col].mean(), inplace=True)

# Scaling the data (Assuming you have a scaler fitted, else fit a new scaler)
scaler = load('data/scaler.joblib')  # Adjust the path if necessary
data[featurex.sensors] = scaler.transform(data[featurex.sensors])

# Define your sequence creation function here (create_sequences)
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        if i % 10000 == 0:  # Check if i is a multiple of 10000
            print('create_sequences', i)
        x = data[i:(i + seq_length)][featurex.sensors].values
        y = data.iloc[i + seq_length]['machine_status']
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


# Create sequences from the data
seq_length = 10  # Number of time steps to look back 
X, y = create_sequences(data, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to binary (2 classes) instead of categorical
y_train_binary = to_categorical(y_train, num_classes=2)
y_test_binary = to_categorical(y_test, num_classes=2)

# LSTM model definition
print('Defining the LSTM model')
n_features = X_train.shape[2]
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(seq_length, n_features)))
model.add(Dense(2, activation='softmax'))  # 2 classes

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
earlystopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Train the model
print('Training the model')
model.fit(X_train, y_train_binary, epochs=100, verbose=1, 
          callbacks=[tensorboard_callback, earlystopping_callback], 
          validation_data=(X_test, y_test_binary), 
          class_weight=class_weights_dict)

# Make predictions
print('Making predictions')
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Evaluate the model
print('Evaluating the model')
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Save the model
print('Saving the model')
model.save('model/pump_predictionsv1.keras')
