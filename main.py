
import numpy as np
import tensorflow as tf
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
import datetime

import streamlit as st

def create_dataset(X, y, time_steps=1):
    """
    Preprocess data
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def plot_loss(history):
	"""Plot training & validation loss values"""
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

def build_model(timesteps, num_features):
    """Build LSTM model for prediction"""
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(timesteps, num_features)),
        RepeatVector(timesteps),
        LSTM(128, activation='relu', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(num_features))                 
    ])

    # This is not a classification problem, but a regression problem. We can't use accuracy. 
    # We have to use mean squared error on each prediction.
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    return model


def load_model(model_file_name):
    print("[INFO] loading {} model".format(model_file_name))
    json_file = open(model_file_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_file_name + ".h5")
    loaded_model.compile(loss='mae', optimizer='adam')
    return loaded_model

def save_model(model, model_file_name):
    print("[INFO] Saving {} model".format(model_file_name))
    model_json = model.to_json()
    with open(model_file_name +".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_file_name + ".h5")

def detect_anomaly(test, time_steps):
    THRESHOLD = 0.55

    test_score_df = pd.DataFrame(test[time_steps:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['close'] = test[time_steps:].close


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].date, y=test_score_df.loss,
                        mode='lines',
                        name='Test Loss'))
    fig.add_trace(go.Scatter(x=test[time_steps:].date, y=test_score_df.threshold,
                        mode='lines',
                        name='Threshold'))
    fig.update_layout(showlegend=True)
    fig.show()


    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies.head()

    scaler = StandardScaler()
    scaler = scaler.fit(train[['close']])

    train['close'] = scaler.transform(train[['close']])
    test['close'] = scaler.transform(test[['close']])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].date, y=scaler.inverse_transform(test[time_steps:].close),
                        mode='lines',
                        name='Close Price'))
    fig.add_trace(go.Scatter(x=anomalies.date, y=scaler.inverse_transform(anomalies.close),
                        mode='markers',
                        name='Anomaly'))
    fig.update_layout(showlegend=True)
    fig.show()

def get_and_plot_data(stock="AAPL", start="2010-03-01", end="2021-01-01"):
    """
    Get and plot data using yfinance library
    """
    stock_obj = yf.Ticker(stock)
    stock_historical = stock_obj.history(start=start, end=end, interval="1d")

    plt.plot(stock_historical)
    plt.show()

    # start_date = st.slider('Enter start date:', value = datetime.datetime(2020,1,1,9,30))
    # end_date = st.slider('Enter end date:', value = datetime.datetime(2020,1,1,9,30))
    return stock_historical

def buy_and_hold(stock="AAPL", amt=1, start=datetime.date(2010, 2, 7), end=datetime.date(2021, 2, 7), rounding=False, interval="1d"):
	'''
	
	'''
	stock_obj = yf.Ticker(stock)
	start_end = datetime.date(start.year, start.month, start.day + 10)
	end_end = datetime.date(end.year, end.month, end.day + 10)
	stock_history_start = stock_obj.history(start=start, end=start_end, interval=interval, rounding=rounding)
	stock_history_end = stock_obj.history(start=end, end=end_end, interval=interval, rounding=rounding)
	close_delta = stock_history_end.iloc[len(stock_history_end) - 1, 3] - stock_history_start.iloc[0, 3]
	return amt * close_delta


def calc_max_profit(price_list):
	'''
	For LSTM predictions
	'''
	max_profit = 0
	for i in range(len(price_list) - 1):
		if price_list[i + 1] > price_list[i]:
			max_profit += price_list[i + 1] - price_list[i]
	return max_profit

if __name__ == "__main__":

    # read data
    df = pd.read_csv(os.path.join('data', 'S&P_500_Index_Data.csv'), parse_dates=['date'])

    # adapted from https://github.com/Tekraj15/AnomalyDetectionTimeSeriesData/blob/master/Anomaly_Detection_Time_Series_Keras.ipynb
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.close,
                        mode='lines',
                        name='close'))
    fig.update_layout(showlegend=True)
    fig.show()

    # preprocess
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)

    # Go back 30 days for now
    time_steps = 30

    X_train, y_train = create_dataset(train[['close']], train.close, time_steps)
    X_test, y_test = create_dataset(test[['close']], test.close, time_steps)

    print(X_train.shape)

    timesteps = X_train.shape[1]
    num_features = X_train.shape[2]

    # Fit model with early stopping
    model = build_model(time_steps, num_features)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

    history = model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=32,
        validation_split=0.1,
        callbacks = [es],
        shuffle=False
    )

    plot_loss(history)

    X_train_pred = model.predict(X_train)

    train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])
    print("train_mae_loss", train_mae_loss)
    print(model.evaluate(X_test, y_test))

    sns.distplot(train_mae_loss, bins=50, kde=True)

    X_test_pred = model.predict(X_test)

    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    print("test_mae_loss", test_mae_loss)
    sns.distplot(test_mae_loss, bins=50, kde=True)
    plt.show()

    detect_anomaly(test, time_steps)