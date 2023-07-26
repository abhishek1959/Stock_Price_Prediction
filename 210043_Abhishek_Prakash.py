#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')

    actual_close = np.loadtxt('sample_close.txt')

    pred_close = predict_func(df)

    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close - pred_close))

    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close

    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr) - np.array(pred_prev)
    actual_dir = np.array(actual_curr) - np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir * actual_dir) > 0) * 100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')


def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples using an LSTM model.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    # Preprocess the data
    df = data.copy()
    df['Close'] = df['Close'].fillna(method='ffill')
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Prepare the input sequence
    sequence_length = 2  # Length of input sequence
    input_sequence = scaled_data[-sequence_length:].reshape(1, -1, 1)

    # Load the trained LSTM model
    model = load_model('lstm_model_weights.h5')

    # Make predictions for the next 2 samples
    predictions = []
    for _ in range(2):
        predicted_value = model.predict(input_sequence)
        predictions.append(predicted_value[0][0])
        input_sequence = np.append(input_sequence[:, 1:, :], predicted_value.reshape(1, 1, 1), axis=1)

    # Inverse transform the predictions
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(1, -1))
    print(predictions.tolist()[::-1])
    return predictions.flatten().tolist()



if __name__ == "__main__":
    evaluate()


# In[ ]:




