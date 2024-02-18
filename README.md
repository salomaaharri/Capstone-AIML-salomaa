# Capstone Project: Predictive Maintenance Modeling for Industrial Pumps Using Machine Learning

## Introduction

In my project, I address the critical challenge of industrial pump failures, which can lead to significant operational disruptions and financial losses across various sectors. By leveraging machine learning (ML), specifically Long Short-Term Memory (LSTM) networks, I aim to predict such failures before they occur, enabling timely maintenance actions that can prevent downtime and extend equipment lifespan.

## Methodology

### Data Acquisition and Preparation

#### **Data Cleanup**

The project began with a comprehensive Kaggle dataset, which included detailed sensor readings, timestamps, and machine statuses. My initial steps involved:

- **Timestamp Conversion**: I transformed timestamp data into a datetime format to facilitate time-series analysis.
- **Column Removal**: I removed irrelevant columns, such as 'Unnamed: 0' and 'sensor_15', which did not contribute to the analysis.
- **Missing Value Handling**: I employed interpolation to fill missing values, ensuring a continuous dataset for training the model. Where interpolation was not applicable, I used backward filling as a secondary approach.
- **Status Consolidation**: The 'BROKEN' and 'RECOVERING' statuses were merged into a single 'BROKEN' category to simplify the prediction model's output.

#### **Data Engineering**

To enhance the dataset for ML modeling, I performed several engineering steps:

- **Feature Generation**: I created temporal features using rolling statistics (mean, standard deviation, minimum, and maximum) over different time windows, along with lagged features to capture previous states of sensors.
- **Rate of Change**: Calculating the rate of change for sensor readings added another layer of information, potentially indicative of emerging failures.

#### **Data Scaling**

Normalization was crucial for the LSTM model's performance. I scaled the sensor data to a \[0, 1\] range using MinMaxScaler, ensuring that the LSTM inputs were appropriately normalized to facilitate efficient training and prediction accuracy.

### Model Development and Training

#### **LSTM Model Architecture**

I chose a Sequential model with LSTM layers due to their ability to remember information over long periods, which is vital for time-series data like sensor readings. The model included:

- **LSTM Layers**: To capture temporal dependencies in the data.
- **Dense Output Layer**: With softmax activation for binary classification (NORMAL vs. BROKEN).

#### **Training Process**

The model was trained using:

- **Adam Optimizer**: With a specified learning rate and clipnorm to manage gradient explosion.
- **Dropout**: For regularization to prevent overfitting.
- **Early Stopping**: To halt training when validation loss ceased to decrease, ensuring the model did not overfit the training data.
- **SMOTE**: To address class imbalance by oversampling minority classes in the training data.

### Model Evaluation

I evaluated the model on a held-out test dataset, measuring accuracy, precision, recall, and F1-score. The evaluation confirmed the model's effectiveness, demonstrating its potential to accurately predict pump failures.

## Implementation and Visualization

### Pump Server Integration

The trained model was deployed on a pump server, enabling real-time prediction capabilities. This server continuously receives sensor data, processes it through the model, and outputs predictions regarding the pump's status.

### Data Flow to WebSocket Client

A Next.js client application was developed to receive data and predictions via WebSockets. This setup ensures real-time monitoring, allowing users to visualize sensor trends, current pump status, and receive alerts for potential failures.

## Model Testing Process

### Preparing Data for Testing

I prepared the data for testing by:

- **Loading and Scaling**: Normalizing the sensor data using the previously fitted MinMaxScaler.
- **Mapping Machine Status**: Converting categorical statuses into numerical values for processing.

### Generating Sequences

The creation of sequences was crucial for the LSTM model, as it requires a series of data points to make predictions. I generated sequences that represented the sensor data over time, leading up to each data point's current status.

### Prediction and Evaluation

I conducted predictions on subsets of the data, including sequences directly before failure events, to test the model's predictive capability in critical scenarios. The model's predictions were then evaluated against actual outcomes, providing a detailed understanding of its performance through metrics such as accuracy and a classification report.

## Conclusion

This project demonstrates the applicability and effectiveness of LSTM networks in predictive maintenance for industrial pumps. Through detailed data preparation, careful model training, and rigorous testing, I've developed a system capable of predicting failures with high accuracy. Future enhancements will focus on refining the model with additional data and exploring more sophisticated architectures to improve predictive performance further.

The integration of ML models with operational technology, as demonstrated in this project, signifies a significant advancement in maintenance strategies, offering a proactive approach to preventing equipment failures.

This detailed report expands on each phase of the project, from data preparation through model training and evaluation,