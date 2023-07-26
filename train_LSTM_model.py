#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('STOCK_INDEX.csv')


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


df = df.fillna(method='ffill')


# In[5]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))


# In[6]:


sequence_length = 30  # Length of input sequence
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X = np.array(X)
y = np.array(y)


# In[7]:


split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]


# In[8]:


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=30, batch_size=32)


# In[10]:


loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


# In[11]:


predicted_values = model.predict(X_test)


# In[13]:


model.save('lstm_model_weights.h5')


# In[ ]:




