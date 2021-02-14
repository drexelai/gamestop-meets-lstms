
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
from datetime import date, datetime
import yfinance as yf
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

def detect_anomaly(model, train, test, X_train, X_test, time_steps=1, close_column_name="Adj Close"):
    THRESHOLD = 0.55

    test_mae_loss = calculate_mae_loss(model, X_test)
    print("[INFO] mae loss:", test_mae_loss)

    test_score_df = pd.DataFrame(test[time_steps:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df[close_column_name] = test[time_steps:][close_column_name]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].Date, y=test_score_df.loss,
                        mode='lines',
                        name='Test Loss'))
    fig.add_trace(go.Scatter(x=test[time_steps:].Date, y=test_score_df.threshold,
                        mode='lines',
                        name='Threshold'))
    fig.update_layout(showlegend=True)
    
    # Plot mean squared error
    plot_mae_loss(model, X_train)
    plot_mae_loss(model, X_test)

    fig.show()

    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies.head()

    scaler = StandardScaler()
    scaler = scaler.fit(train[[close_column_name]])

    train[close_column_name] = scaler.transform(train[[close_column_name]])
    test[close_column_name] = scaler.transform(test[[close_column_name]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].Date, y=scaler.inverse_transform(test[time_steps:][close_column_name]),
                        mode='lines',
                        name='Close Price'))
    fig.add_trace(go.Scatter(x=anomalies.Date, y=scaler.inverse_transform(anomalies[close_column_name]),
                        mode='markers',
                        name='Anomaly'))
    fig.update_layout(showlegend=True)
    fig.show()
    return scaler

def fetch_data(ticker, start_date, end_date):
    """
    Downloads and writes the stock price data to csv.
    If the csv data already exists, read from it.
    """
    file_path = os.path.join("data", ticker+".csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)
    start_date_object = to_datetime(start_date)
    end_date_object = to_datetime(end_date)
    stock_price = yf.download(ticker, start_date=start_date_object, end_date=end_date_object)
    stock_price.to_csv(file_path)
    return pd.read_csv(file_path, index_col=0)

def buy_and_hold(stock_price, start_date, end_date):
    """Calculate profit from buy and hold"""
    return stock_price.loc[end_date, 'Adj Close'] - stock_price.loc[start_date, 'Open']


def to_datetime(date_str):
    temp = datetime.strptime(date_str, '%Y-%M-%d')
    return date(temp.year, temp.month, temp.day)

def calc_max_profit(price_list):
	'''
	For LSTM predictions
	'''
	max_profit = 0
	for i in range(len(price_list) - 1):
		if price_list[i + 1] > price_list[i]:
			max_profit += price_list[i + 1] - price_list[i]
	return max_profit

def calculate_mae_loss(model, X):
    X_pred = model.predict(X)
    mae_loss = np.mean(np.abs(X_pred - X), axis=1)
    print("[INFO] mae loss:", mae_loss)
    return mae_loss

def plot_mae_loss(model, X_train, bins=50, kde=True):
    X_train_pred = model.predict(X_train)
    train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])
    sns.distplot(train_mae_loss, bins=bins, kde=kde)
    plt.show()

def evaluate_model(model, X_test, y_test):
    print("[INFO] Model evaluation results:")
    print(model.evaluate(X_test, y_test))

def train_model(X_train, y_train, epochs=1):
    """Train model and plot loss and accuracy"""
    time_steps = X_train.shape[1]
    num_features = X_train.shape[2]

    # Fit model with early stopping
    model = build_model(time_steps, num_features)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks = [es],
        shuffle=False
    )

    plot_loss(history)
    return model

def plot_and_prepare_data(df, close_column_name="Adj Close"):

    # adapted from https://github.com/Tekraj15/AnomalyDetectionTimeSeriesData/blob/master/Anomaly_Detection_Time_Series_Keras.ipynb
    fig = go.Figure()
    print(df.head())
    fig.add_trace(go.Scatter(x=df.index, y=df[close_column_name],
                        mode='lines',
                        name=close_column_name))
    fig.update_layout(showlegend=True)
    fig.show()

    # creating a DataFrame 
    my_df = {'Date': df.index,  
            'Adj Close': df[close_column_name]} 
    df = pd.DataFrame(my_df) 
    df.reset_index(drop=True, inplace=True)
    print(df.head())

    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)

    scaler = StandardScaler()
    scaler = scaler.fit(train[[close_column_name]])

    train[close_column_name] = scaler.transform(train[[close_column_name]])
    test[close_column_name] = scaler.transform(test[[close_column_name]])

    time_steps = 30

    X_train, y_train = create_dataset(train[[close_column_name]], train[close_column_name], time_steps)
    X_test, y_test = create_dataset(test[[close_column_name]], test[close_column_name], time_steps)

    print(X_train.shape)
    return X_train, y_train, X_test, y_test, train, test

def trade_pure_lstm_predictions(model, train, X_test, time_steps=30, close_column_name="Adj Close"):
    # TODO: Fix IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
    scaler = StandardScaler()
    scaler = scaler.fit(test[[close_column_name]])
    X_test_pred = model.predict(X_test)
    inversed_test = scaler.inverse_transform(test[time_steps:][close_column_name])
    print(inversed_test)
    
def run_pipeline(ticker="GME", start_date="2002-02-13", end_date = "2021-02-12"):
    """For a given stock ticker, 
    i)  calculate and report profit from buy and hold (see buy_and_hold), 
    ii) calculate and report from LSTM predictions-based trading (see trade_pure_lstm_predictions)
    iii) calculate and report profit from anomaly detection-based trading (see detect_anomaly)
    No more than 1 share is bought/sold at a time.
    No shorting allowed. i.e we can only sell if we have a share.
    """
    print("[INFO] Pipeline started for Ticker:{} Start date:{} End date:{}".format(ticker, start_date, end_date))

    # Load data
    # stock_df = get_and_plot_data(ticker, start_date=start_date, end_date=end_date)
    # print(stock_df)
    
    stock_price = fetch_data(ticker, start_date, end_date)
    buy_and_hold_profit = buy_and_hold(stock_price, start_date, end_date)
    print("[INFO] Buy and Hold profit is ${}".format(buy_and_hold_profit))
    X_train, y_train, X_test, y_test, train, test = plot_and_prepare_data(stock_price)

    # Train model
    model = train_model(X_train, y_train)

    trade_pure_lstm_predictions(model, train, X_test, time_steps=30)

    # Evaluate model
    # evaluate_model(model, X_test, y_test)

    # Detect anomaly
    # detect_anomaly(model, train, test, X_train, X_test, time_steps = 30)

if __name__ == "__main__":
    # GME start_date=2002-02-13, end_date=2021-02-12
    run_pipeline(ticker="SPY", start_date="2002-02-13", end_date = "2021-02-12")
    
    # S&P start_date=2002-02-13, end_date=2021-02-12
    # run_pipeline(ticker="SPY", start_date="2010-01-01", end_date = "2020-01-01")
    


   