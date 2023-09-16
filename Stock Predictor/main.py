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
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

# Create a FigureCanvasTkAgg widget to display the plot in the GUI
#fig = Figure(figsize=(16, 6))
#ax = fig.add_subplot(111)
#canvas = FigureCanvasTkAgg(fig, master=root)
#canvas.get_tk_widget().pack()

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
    return data['Close']  # Only keep the 'Close' prices

def prepare_data(data, n_past=14):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(n_past, len(scaled_data)):
        X.append(scaled_data[i-n_past:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def build_model(look_back):
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_and_train_model(X, y):
    model = build_model(X.shape[1])
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    return model

def maxProfit_future(prices):
    minPrice = prices[0]
    maxProfit = 0
    buy_date = None
    sell_date = None

    for i in range(1, len(prices)):
        if prices[i] < minPrice:
            minPrice = prices[i]
            potential_sell_date = i

        elif prices[i] - minPrice > maxProfit:
            maxProfit = prices[i] - minPrice
            sell_date = i
            buy_date = potential_sell_date

    if buy_date is None or sell_date is None:
        return None, None, None

    percent_return = (prices[sell_date] - prices[buy_date]) / prices[buy_date] * 100
    return buy_date, sell_date, percent_return

def make_predictions(etf_selected, ax):
    model = model_dict[etf_selected]
    scaler = scaler_dict[etf_selected]
    today_date = datetime.now().strftime('%Y-%m-%d')
    data = get_data(etf_selected, start_date='2022-05-01', end_date=today_date)
    data = data.values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    # Predicting the historical data
    predicted_for_test = []
    for i in range(14, len(scaled_data)):
        historical_data = scaled_data[i-14:i].reshape(1, -1, 1)
        pred = model.predict(historical_data)
        predicted_for_test.append(scaler.inverse_transform(pred).flatten()[0])

    # Future Prediction
    input_data = scaled_data[-14:]
    current_date = datetime.strptime(today_date, '%Y-%m-%d')
    predicted_future = []
    predicted_dates = []

    for i in range(14):
        pred = model.predict(input_data.reshape(1, -1, 1))
        predicted_future.append(scaler.inverse_transform(pred).flatten()[0])

        input_data = np.append(input_data.flatten()[1:], pred).reshape(-1, 1)
        current_date += timedelta(days=1)
        while current_date.weekday() in [5, 6]:
            current_date += timedelta(days=1)

        predicted_dates.append(current_date)

    # Max profit calculation
    buy_date, sell_date, percent_return = maxProfit_future(predicted_future)

    if buy_date is not None and sell_date is not None:
        prediction_info = f"Buy Date: {predicted_dates[buy_date].strftime('%Y-%m-%d')}, " \
                          f"Sell Date: {predicted_dates[sell_date].strftime('%Y-%m-%d')}, " \
                          f"Percent Return: {percent_return:.2f}%"
    else:
        prediction_info = "This is not the best time to buy and sell the stock. " \
                          "There is no predicted opportunity for profit within the next 14 market days."

    # Update the label with the prediction information
    prediction_label.config(text=prediction_info)

    # Plot and update the embedded plot in the GUI
    ax.plot(data, label='True')
    ax.plot(np.arange(14, len(data)), predicted_for_test, label='Historical Predicted')
    ax.plot(np.arange(len(data), len(data) + len(predicted_future)), predicted_future, label='Future Predicted')
    ax.legend()
    ax.set_title(f"{etf_selected} Predictions")
    canvas.draw()

# Start the GUI event loop
etf_list = ['ARKX', 'UFO', 'ROKT']
model_dict = {}
scaler_dict = {}

for etf in etf_list:
    data = get_data(etf, start_date='2022-01-01', end_date='2023-09-15')
    X, y, scaler = prepare_data(data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to (batch_size, timesteps, features)
    model = build_and_train_model(X, y)
    model_dict[etf] = model
    scaler_dict[etf] = scaler

root.mainloop()