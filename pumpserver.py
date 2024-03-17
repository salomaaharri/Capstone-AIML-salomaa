from urllib3.exceptions import InsecureRequestWarning
import warnings

# Suppress only the single InsecureRequestWarning from urllib3 needed
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

import asyncio
import websockets
import pandas as pd
from pandas import json_normalize
import json
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load
import featurex

# https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

# Load sensor data and danger zones from CSV files. 
# The sensor data has been pre-processed and features engineered for prediction.
sensordata = pd.read_csv('data/sensor_engineered.csv')
danger_zones_df = pd.read_csv('data/danger_zones.csv')
danger_zones = danger_zones_df.to_dict(orient='records')

# Filter sensor data to include only specified columns and map machine status to numerical values.
sensordata = sensordata[featurex.columns]
status_mapping = {'NORMAL': 0, 'BROKEN': 1}
sensordata['machine_status'] = sensordata['machine_status'].map(status_mapping)

print(sensordata['machine_status'].unique())

# Initialize the current index for processing sensor data sequentially.
current_index = 0

# Find the index of the first occurrence where machine_status is 'BROKEN' (1)
# Optionally, subtract a certain number to start from a bit earlier
broken_index = sensordata[sensordata['machine_status'] == 1].index[0] - 10  # start 20 readings before the first broken status

# Ensure the index is not negative
current_index = max(broken_index, 0)

# Set to track connected WebSocket clients.
connected = set()

# Load the model (adjust the path to where your model is saved)
print('loading model...')
model = load_model('model/pump_predictionsv1.keras', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the fitted scaler
scaler = load('data/scaler.joblib')

def normalize_sensor_data(sensor_data, scaler):
    # Reshape the data to match the scaler's expected input
    sensor_data_reshaped = sensor_data.reshape(1, -1)
    sensor_data_reshaped_df = pd.DataFrame(sensor_data_reshaped, columns=featurex.sensors)
    normalized_data = scaler.transform(sensor_data_reshaped_df)
    return normalized_data.flatten()  # Flatten back to original shape

def smart_round(value):
    if value > 1000:  # Large values are rounded to the nearest hundred
        return round(value, -3)
    if value > 100:  # Large values are rounded to the nearest hundred
        return round(value, -2)
    elif value > 10:  # Medium values are rounded to the nearest ten
        return round(value, -1)
    else:  # Smaller values are rounded normally
        return round(value)

def preprocess_row(row, seq_length=10):
    # Assuming `row` is a 1D NumPy array of shape (6,)
    # We need to reshape it to (1, seq_length, 6)
    # Since you're processing one row at a time, you might need to use padding or a different approach

    # If you're sending one row at a time, you might need to pad the sequence
    # Here's an example of padding with zeros
    padded_sequence = np.zeros((seq_length, len(row)))
    padded_sequence[-1, :] = row  # Place the row at the end of the padded sequence

    # Reshape for the model and convert to float32
    processed_row = np.array([padded_sequence], dtype=np.float32)

    return processed_row

def update_buffer(buffer, new_row):
    # Remove the oldest row and add the new one
    buffer[:-1] = buffer[1:]
    buffer[-1] = new_row
    return buffer

# Determine the correct number of features
n_features = len(featurex.sensors)  # Number of sensor features

# Initialize a buffer to hold the most recent sensor readings
seq_length = 10  # This should match the sequence length used during training
buffer = np.zeros((seq_length, n_features))  # Adjust the number of sensors accordingly

async def sensor_data_job(websocket, path):
    """Coroutine to handle WebSocket connections and send sensor data and predictions."""
    global current_index, sensordata, buffer  # Add 'buffer' to the global declaration

    # Pre-fill the buffer with the first 'seq_length' data points
    for i in range(seq_length):
        sensor_data = sensordata.iloc[i][featurex.sensors].values
        normalized_sensor_data = normalize_sensor_data(sensor_data, scaler)
        buffer[i] = normalized_sensor_data

    print(f"Client connected: {websocket.remote_address}")
    connected.add(websocket)
    try:
        while current_index < len(sensordata):
            # Extract and preprocess the current row for prediction
            current_row = sensordata.iloc[current_index]
            sensor_data = current_row[featurex.sensors].values  # Only sensor data
            # Normalize the sensor data
            normalized_sensor_data = normalize_sensor_data(sensor_data, scaler)
            buffer = update_buffer(buffer, normalized_sensor_data)
            max_values = sensordata[featurex.sensors].max()
            smart_rounded_max_values = {col: smart_round(max_values[col]) for col in featurex.sensors}

            # Reshape the buffer to match the expected input shape of the model
            buffer_reshaped = buffer.reshape(1, seq_length, n_features)  # Use n_features instead of hardcoded value
            # print('buffer', buffer_reshaped, buffer_reshaped.shape)
            # Make a prediction
            # prediction = model.predict(np.array([buffer_reshaped], dtype=np.float32))
            prediction = model.predict(buffer_reshaped)
            # print('model.predict', prediction)
            prediction = prediction.tolist()  # Convert NumPy array to Python list
            prediction = [[float(x) if not np.isnan(x) else None for x in pred] for pred in prediction]
            sensor_data_with_names = [(name, current_row[name]) for name in featurex.sensors]

            # print('sensordata', current_row['timestamp'], sensor_data_with_names)

            actual_status = 'NORMAL' if int(current_row['machine_status']) == 0 else 'BROKEN'
            predicted_status = 'NORMAL' if prediction[0][0] > prediction[0][1] else 'BROKEN'

            # print(current_row['machine_status'], predicted_status)

            # Prepare the data to be sent
            data_to_send = {
                'timestamp': current_row['timestamp'],
                'machine_status': actual_status,
                'sensor_data': sensor_data_with_names,  # Convert NumPy array to Python list
                'prediction': predicted_status,
                'max_values': smart_rounded_max_values,  # Include smart-rounded max values
                'danger_zones': danger_zones
            }
            # print(f"Data to send: {data_to_send}")
            # Increment the index for the next iteration
            current_index += 1
            # Send data
            if websocket.open:
                # print('sending data to socket client...', data_to_send)
                await websocket.send(json.dumps(data_to_send))

            await asyncio.sleep(10)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    finally:
        connected.remove(websocket)


async def main():
    # This function sets up the WebSocket server and awaits connections.
    print('waiting for socket client...')
    # The websockets.serve function creates a WebSocket server at the specified address and port.
    # The 'sensor_data_job' coroutine is passed as the handler for incoming WebSocket connections.
    async with websockets.serve(sensor_data_job, "localhost", 8765):
        # The server runs indefinitely, waiting for and handling WebSocket connections.
        # await asyncio.Future() keeps the server running forever by creating a never-ending task.
        await asyncio.Future()  # run forever

# The __name__ == "__main__" check is Python's way of running code only when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    # asyncio.run(main()) starts the asynchronous event loop and runs the 'main' coroutine.
    # It's the entry point to start the WebSocket server.
    asyncio.run(main())