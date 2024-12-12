import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_dataset(dataset, time_step=1, output=1):
    X, Y = [], []

    for i in range(len(dataset) - time_step - output):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        
        b = dataset[(i + time_step):(i + time_step + output), 0]  
        Y.append(b)
    
    return np.array(X), np.array(Y)

def history_plot(history):
    plt.plot(history.history['loss'],label='Training loss')
    plt.plot(history.history['val_loss'],label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()


def plot_data(data, title, x_label, y_label):
    plt.title(title)
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def model_plot(actual_data, predict_data):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data, label='Actual Data', color='blue')
    plt.plot(predict_data, label='Predicted Data', color='red')
    plt.title('Comparison of Actual and Predicted Values') 
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.grid()
    plt.show()

def read_data(file_path, debug = False, columns_to_drop = ['DPZ', 'BTC', 'NFLX']):
    df = pd.read_csv('portfolio_data.csv')

    if debug:
        print(df.head(10))
        print('----------------------------------------------------')
    

    
    df['Date'] = pd.to_datetime(df['Date'])

    clean_df = df.drop(columns=columns_to_drop)

    clean_df.sort_values(by='Date', inplace=True)
    if debug:
        print(clean_df.head(10))

    clean_df.set_index('Date', inplace=True)

    result = np.array(clean_df.iloc[:])

    return clean_df, result

def prepare_train_test_data(data, time_step_in, time_step_out, test_size=0.4):
    X, Y = create_dataset(data, time_step_in, time_step_out)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,shuffle=False)

    scaler = MinMaxScaler(feature_range=(0,1))
    x_train = scaler.fit_transform(X_train)
    y_train = scaler.transform(Y_train)
    x_test = scaler.transform(X_test)
    y_test = scaler.transform(Y_test)

    print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')
    return x_train, x_test, y_train, y_test, scaler