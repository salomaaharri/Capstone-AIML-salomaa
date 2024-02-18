# Capstone Project: Predictive Maintenance Modeling for Industrial Pumps Using Machine Learning

[Pump NextJS Dashboard talking to Python Pumpserver via WebSocket!](https://youtu.be/Vm6gt0G4SXg)

## Introduction

In my project, I address the critical challenge of industrial pump failures, which can lead to significant operational disruptions and financial losses across various sectors. By leveraging machine learning, specifically Long Short-Term Memory (LSTM) networks, I aim to predict such failures before they occur, enabling timely maintenance actions that can prevent downtime and extend equipment lifespan.

## Methodology

### Data Acquisition and Preparation

#### **Data Cleanup**

The project began with a comprehensive Kaggle dataset, which included detailed sensor readings, timestamps, and machine statuses. My initial steps involved:

- **Timestamp Conversion**: I transformed timestamp data into a datetime format to facilitate time-series analysis.
- **Column Removal**: I removed irrelevant columns, such as 'Unnamed: 0' and 'sensor_15', which did not contribute to the analysis.
- **Missing Value Handling**: I employed interpolation to fill missing values, ensuring a continuous dataset for training the model. Where interpolation was not applicable, I used backward filling as a secondary approach.
- **Status Consolidation**: The 'BROKEN' and 'RECOVERING' statuses were merged into a single 'BROKEN' category to simplify the prediction model's output.
- **Git large file support**: I needed to install git large file support for all data files: 1: brew install git-lfs. 2: git lfs install 3: git lfs track "*.csv"

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

```
Training the model
Epoch 1/100
5508/5508 [==============================] - 12s 2ms/step - loss: 0.0266 - accuracy: 0.9944 - val_loss: 0.0145 - val_accuracy: 0.9962
Epoch 2/100
5508/5508 [==============================] - 11s 2ms/step - loss: 0.0139 - accuracy: 0.9970 - val_loss: 0.0055 - val_accuracy: 0.9988
Epoch 3/100
5508/5508 [==============================] - 11s 2ms/step - loss: 0.0129 - accuracy: 0.9971 - val_loss: 0.0097 - val_accuracy: 0.9967
Epoch 4/100
5508/5508 [==============================] - 11s 2ms/step - loss: 0.0117 - accuracy: 0.9974 - val_loss: 0.0165 - val_accuracy: 0.9951
Epoch 5/100
5503/5508 [============================>.] - ETA: 0s - loss: 0.0108 - accuracy: 0.9975Restoring model weights from the end of the best epoch: 2.
5508/5508 [==============================] - 11s 2ms/step - loss: 0.0108 - accuracy: 0.9975 - val_loss: 0.0090 - val_accuracy: 0.9966
Epoch 5: early stopping
```

### Model Evaluation

I evaluated the model on a held-out test dataset, measuring accuracy, precision, recall, and F1-score. The evaluation confirmed the model's effectiveness, demonstrating its potential to accurately predict pump failures.

```
Evaluating the model
Accuracy: 0.9987517588852072
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     41239
         1.0       0.99      0.99      0.99      2823

    accuracy                           1.00     44062
   macro avg       0.99      1.00      0.99     44062
weighted avg       1.00      1.00      1.00     44062
```


**Initial Performance**: The training started with an initial loss of 0.0266 and an accuracy of 99.44% on the training set. The validation loss was 0.0145 with a validation accuracy of 99.62% in the first epoch.

**Improvement Over Epochs**: The model showed consistent improvement in the first couple of epochs, achieving its best validation loss of 0.0055 and a validation accuracy of 99.88% by the end of the second epoch. This marks the peak performance of the model during training.

**Subsequent Epochs**: In epochs 3 and 4, while the training loss and accuracy continued to improve slightly, the validation loss increased compared to its best at epoch 2, indicating the beginning of overfitting.

**Early Stopping**: The training utilized early stopping, which halted the training at epoch 5 when the model's performance on the validation set did not improve. The model weights were restored to those of the best epoch, which is epoch 2 in this case, to prevent overfitting and ensure the model retains its best generalization ability.

**Final Model Evaluation**: Upon evaluating the model on the test set, it achieved an impressive accuracy of 99.88%. The precision, recall, and F1-score for classifying the 'NORMAL' state (0.0) were all approximately 100%, and for the 'BROKEN' state (1.0), these metrics were around 99%, indicating a highly effective model in distinguishing between the two states.

**Summary**: The training process was efficient, with the model quickly reaching a high level of accuracy. The use of early stopping effectively prevented overfitting, ensuring the model maintained its generalization capabilities. The final evaluation metrics demonstrate the model's excellent performance in accurately classifying the two states of the pumps, with nearly perfect accuracy, precision, recall, and F1-scores across both classes.


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

Pump dashboard seems to visualize (predict) pump failure few minutes before pumps real 'Broken' status. This is good and pump can be shut down to prevent total failure.

This project demonstrates the applicability and effectiveness of LSTM networks in predictive maintenance for industrial pumps. Through detailed data preparation, careful model training, and rigorous testing, I've developed a system capable of predicting failures with high accuracy. Future enhancements will focus on refining the model with additional data and exploring more sophisticated architectures to improve predictive performance further.

The integration of ML models with operational technology, as demonstrated in this project, signifies a significant advancement in maintenance strategies, offering a proactive approach to preventing equipment failures.

This detailed report is WIP can be updated as long model is being improved and tested

