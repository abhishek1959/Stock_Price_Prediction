
# Stock Price Prediction

Stock Price Prediction using machine learning helps us to discover the future value of company stock and other financial assets traded on an exchange. The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. 
I have used a Long Short Term Memory Network (LSTM) for building our model to predict the stock prices.
LSTM (Long Short-Term Memory) models are a type of recurrent neural network (RNN) that are particularly useful for stock price prediction and time series forecasting tasks.
### Closing Price: 
The closing price is the last price at which the stock is traded during the regular trading day. A stock’s closing price is the standard benchmark used by investors to track its performance over time. 

### Problem: Write a python program that will predict the closing price of the given index for the next 2 days, given the information of 50 past days

## Installation of Libraries

Installed different required libraries in Jupyter notebook by using the following command in the code cell: 

```bash
  !pip install numpy
```
```bash
  !pip install pandas
```
```bash
  !pip install tensorflow
```
```bash
  !pip install scikit-learn
```
## Import and use of installed libraries

```bash
  import numpy as np
```
```bash
  import pandas as pd
```
```bash
  from tensorflow.keras.models import Sequential, load_model
```
```bash
  from tensorflow.keras.layers import LSTM, Dense
```
```bash
  from sklearn.preprocessing import MinMaxScaler
```
    
## Files and Dataset

- submit_input.csv --> It is a csv file taken as input to predict the closing prices of next two days 
- submit_close.txt --> Given closing prices 
- STOCK_INDEX.csv --> Taken as input to train the model 
- train_LSTM_model.py --> It is a python file code of trained lstm model
- train_LSTM_model.ipynb --> It is IPython Notebook format (.ipynb) of the above python file
- lstm_model_weights.h5 --> Saved model/parameter file
- 210043_Abhishek_Prakash.py  --> modified python file from sample_test.py
- 210043_Abhishek_Prakash.ipynb --> It is IPython Notebook format (.ipynb) of the above python file


## Instructions
Here are the brief steps involved in the code snippet provided for training the LSTM model:
- Import the required above mentioned libraries.
- Read the stock index data from the 'STOCK_INDEX.csv' file. Scale the closing prices using MinMaxScaler to bring the values within a specific range.
- Set the sequence_length equal to 30 to define the number of past time steps to use for predicting the next closing price.
- Create input sequences (X) and corresponding target values (y).
- Split the data into training and testing sets.
- Reshape the input sequences (X_train and X_test) to match the LSTM model's input shape, which expects a 3D array with dimensions (samples, time steps, features).
- Build and train the LSTM model --> 1. Create a sequential model (model) and add LSTM layers with the specified number of units (128 and 64 in this case) 2. Compile the model with the mean squared error loss and Adam optimizer 3. Fit the model to the training data (X_train and y_train) using a defined number of epochs (30) and batch size (32).
- Evaluate the model's performance on the testing data (X_test and y_test) using the evaluate method.
- Use the trained model to make predictions on the testing data (X_test) using the predict method.
- Save the trained model weights to a file called 'lstm_model_weights.h5' using the save method of the model.



Here are the brief steps involved in the code snippet for predicting the next two days' closing prices using an LSTM model:
- After importing above mentioned libraries read the sample input data from the 'sample_input.csv' file using 'pd.read_csv'.
- Load the actual closing prices from the 'sample_close.txt' file using 'np.loadtxt'.
- Call the 'predict_func' function to get the predicted closing prices for the next two samples.
- Calculate the mean squared error (MSE) and directional accuracy using the actual and predicted closing prices. Print the MSE and directional accuracy.
- Define the 'predict_func' function --> 1. Preprocess the input data by filling null values in the 'Close' column with the previous non-null value using 'fillna'.
- Prepare the input sequence by taking the last sequence_length scaled closing prices from the data.
- Load the trained LSTM model from the 'lstm_model_weights.h5' file using 'load_model'.
- Inverse transform the predicted prices to obtain the actual values using 'scaler.inverse_transform'.
- Print the predicted closing prices. Return the predicted closing prices as a flattened list.
- Check if the script is being run directly by checking if ' __name__ == "__main__" '.
- Call the evaluate function to perform the evaluation and prediction.


