import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

# Create a function to plot a graph when a menu option is selected
def plot_graph():
    selected_option = var.get()
    
    # Clear any previous plot
    ax.clear()
    
    # Define your data and plotting logic based on the selected option
    if selected_option == "ARKX":
        make_predictions('ARKX', ax)
    elif selected_option == "UFO":
        make_predictions('UFO', ax)
    elif selected_option == "ROKT":
        make_predictions('ROKT', ax)
    
    canvas.draw()

# Create the main window
root = tk.Tk()
root.title("Stock Price Predictor")

# Create a label
label = tk.Label(root, text="Please select an ETF from ['ARKX', 'UFO', 'ROKT']: ")
label.pack()

# Create a variable to store the selected option
var = tk.StringVar()

# Create a dropdown menu
menu = ttk.Combobox(root, textvariable=var)
menu['values'] = ("ARKX", "UFO", "ROKT")
menu.pack()

# Create a label or text widget to display the prediction information
prediction_label = tk.Label(root, text="", wraplength=400)
prediction_label.pack()

# Create a button to plot the graph
plot_button = tk.Button(root, text="Plot Graph", command=plot_graph)
plot_button.pack()

# Create a Matplotlib figure and a canvas to embed the plot
fig = Figure(figsize=(6, 4))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Set the window size (adjust the width and height as needed)
root.geometry("600x400")

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Open', 'High', 'Low', 'Adj Close']]

def prepare_data(data, n_past=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(n_past, len(scaled_data)):
        X.append(scaled_data[i-n_past:i])
        y.append(scaled_data[i, -1])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_and_train_model(X, y):
    model = build_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
    return model

def maxProfit_future(prices):
    minPrice = prices[0]
    maxProfit = 0
    buy_date = 0
    sell_date = 0
    potential_buy_date = 0

    for i in range(1, len(prices)):
        if prices[i] < minPrice:
            minPrice = prices[i]
            potential_buy_date = i

        current_profit = prices[i] - minPrice

        if current_profit > maxProfit:
            maxProfit = current_profit
            sell_date = i
            buy_date = potential_buy_date

    if buy_date == sell_date:
        return None, None, None

    percent_return = (prices[sell_date] - prices[buy_date]) / prices[buy_date] * 100
    return buy_date, sell_date, percent_return

def make_predictions(etf_selected, ax):
    model = model_dict[etf_selected]
    scaler = scaler_dict[etf_selected]
    today_date = datetime.now().strftime('%Y-%m-%d')
    data = get_data(etf_selected, start_date='2022-05-01', end_date=today_date)
    scaled_data = scaler.transform(data)
    predicted_for_test = []
    predicted_future = []

    # Use 60-day windows for historical prediction
    for i in range(60, len(scaled_data)):
        historical_data = scaled_data[i-60:i].reshape(1, 60, 4)
        pred = model.predict(historical_data)

        inversed_prediction = scaler.inverse_transform(
            np.concatenate([np.zeros((1, 3)), pred], axis=1))[:, -1]
        predicted_for_test.append(inversed_prediction[0])

    input_data = scaled_data[-60:]
    current_date = datetime.strptime(today_date, '%Y-%m-%d')
    predicted_future = []
    predicted_dates = []

    for i in range(14):  # Loop to predict the next 14 days
        if input_data.shape == (60, 4):
            future_data = input_data.reshape(1, 60, 4)
        else:
            print("Insufficient data for prediction. Expected shape (60, 4), got ", input_data.shape)
            break
        pred = model.predict(future_data)

        inversed_prediction = scaler.inverse_transform(
            np.concatenate([np.zeros((1, 3)), pred], axis=1))[:, -1]
        predicted_future.append(inversed_prediction[0])

        pred_compatible = np.zeros((1, 4))
        pred_compatible[:, -1] = pred

        input_data = np.concatenate((input_data[1:], pred_compatible.reshape(1, 4)), axis=0)

        current_date += timedelta(days=1)
        while current_date.weekday() in [5, 6]:
            current_date += timedelta(days=1)
        predicted_dates.append(current_date)

    buy_date, sell_date, percent_return = maxProfit_future(predicted_future)

    if buy_date is not None and sell_date is not None:
        prediction_info = f"Buy Date: {predicted_dates[buy_date].strftime('%Y-%m-%d')}, " \
                          f"Sell Date: {predicted_dates[sell_date].strftime('%Y-%m-%d')}, " \
                          f"Percent Return: {percent_return:.2f}%"
    else:
        prediction_info = "This is not the best time to buy and sell the stock. " \
                          "There is no predicted opportunity for profit within the next 60 market days."

    # Plot the prediction graph on the specified Matplotlib axis (ax)
    prediction_label.config(text=prediction_info)
    ax.plot(data.index, data['Adj Close'], label='True')
    ax.plot(data.index[60:], predicted_for_test, label='Historical Predicted')  # Align with the starting index
    future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=14, freq='B')
    ax.plot(future_dates, predicted_future, label='Future Predicted')
    ax.legend()
    ax.set_title(f"{etf_selected} Predictions")

# Start the GUI event loop
etf_list = ['ARKX', 'UFO', 'ROKT']
model_dict = {}
scaler_dict = {}

for etf in etf_list:
    data = get_data(etf, start_date='2022-01-01', end_date='2023-09-15')
    X, y, scaler = prepare_data(data)
    model = build_and_train_model(X, y)
    model_dict[etf] = model
    scaler_dict[etf] = scaler

root.mainloop()