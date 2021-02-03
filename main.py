
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



df = pd.read_csv(os.path.join('data', 'S&P_500_Index_Data.csv'), parse_dates=['date'])
df.head()
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

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# Go back 30 days for now
time_steps = 30

X_train, y_train = create_dataset(train[['close']], train.close, time_steps)
X_test, y_test = create_dataset(test[['close']], test.close, time_steps)

print(X_train.shape)

timesteps = X_train.shape[1]
num_features = X_train.shape[2]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

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


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks = [es],
    shuffle=False
)

print(history.history)
def plot_loss(history):
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

plot_loss_and_accuracy(history)

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


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].date, y=scaler.inverse_transform(test[time_steps:].close),
                        mode='lines',
                        name='Close Price'))
    fig.add_trace(go.Scatter(x=anomalies.date, y=scaler.inverse_transform(anomalies.close),
                        mode='markers',
                        name='Anomaly'))
    fig.update_layout(showlegend=True)
    fig.show()


detect_anomaly(test, time_steps)