

# **Time Series Forecasting with LSTM for Amazon Stock Price Prediction**
![Forecasting Graph](./télécharger.jpg)
## **Project Overview**
This project demonstrates the use of Long Short-Term Memory (LSTM) networks for time series forecasting. The main objective is to predict future values of Amazon's stock price based on its historical data. Although the dataset includes data for Netflix, Bitcoin, and Amazon, the focus of this project is on Amazon stock price prediction. LSTM, a type of recurrent neural network (RNN), is particularly effective for modeling sequences and time-series data, which makes it ideal for stock price forecasting.

### **Key Features:**
- Data preprocessing for handling multiple assets: Netflix, Bitcoin, Amazon and ,...
- **Four distinct models** developed and tested:
  1. **Dense Model with 20 Input Features and 1 Output**
  2. **Dense Model with 20 Input Features and 7 Outputs**
  3. **LSTM Model with 20 Input Features and 1 Output**
  4. **LSTM Model with 20 Input Features and 7 Outputs**
- Random sequence generation for evaluating model performance.
- Forecasting for multiple future time steps of Amazon's stock price.

## **Technologies Used**
- **Python** (Programming Language)
- **NumPy** (Numerical Computing)
- **Matplotlib** (Visualization)
- **TensorFlow/Keras** (Machine Learning Framework)
- **Scikit-Learn** (Machine Learning Libraries for preprocessing)


## **Data Preparation**

1. **Dataset**:
   - [dataset](portfolio_data.csv)
   - Although the dataset contains data for multiple assets, the model is trained and tested only on **Amazon's stock price**.
   
2. **Preprocessing**:
   - The data is normalized to a range of [0, 1] using MinMaxScaler.
   - The data is reshaped into sequences of time steps to be fed into the LSTM model for forecasting.

## **Model Architectures**

The project implements four distinct models for stock price prediction:

1. **Dense Model with 20 Input Features and 1 Output**
   - This model uses a dense (fully connected) neural network with 20 input features and predicts a single value as the output.

2. **Dense Model with 20 Input Features and 7 Outputs**
   - This dense network architecture predicts 7 output values based on 20 input features, designed to forecast the next 7 days of Amazon's stock price.

3. **LSTM Model with 20 Input Features and 1 Output**
   - This model is based on Long Short-Term Memory (LSTM) layers and forecasts a single value using 20 input features.

4. **LSTM Model with 20 Input Features and 7 Outputs**
   - An LSTM model designed to predict 7 output values based on 20 input features, useful for multi-step forecasting.

### **Model Training**
- Each model is trained using the training dataset of Amazon stock prices. The training loop includes evaluating the model's performance on the validation set and adjusting the weights based on the loss function (Mean Squared Error).
  
## **Forecasting with LSTM**

After training the models, we can forecast future values of Amazon's stock price. The forecast is performed by predicting one value at a time, appending it to the original data, and then using it as input for subsequent predictions.

### **Main Forecasting Function:**
```python
def calculate(n_random, model, x_train, n_step_forecast=1):
    # Forecasts future n_step_forecast steps for a random sequence from the dataset
    # Appends the predictions to the original data and visualizes the results
```

## **Visualization**

The results are visualized using **Matplotlib**. The following is a typical plot comparing the original time series of Amazon's stock price and the forecasted values:

```python
plt.figure(figsize=(10, 6))
plt.plot(original_data, label='Actual Amazon Stock Data', color='blue')
plt.plot(predicted_result, label='Predicted Amazon Stock Data', color='red')
plt.title('Comparison of Actual and Predicted Amazon Stock Values')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

This allows us to visually evaluate how well the model performs in forecasting future Amazon stock prices.



## Results

Here are the evaluation metrics for each model:

### **Dense Model (20 inputs, 1 output)**

- **Mean Squared Error (MSE):** 0.05290762967736506  
- **Mean Absolute Error (MAE):** 0.003832602127925256

### **LSTM Model (20 inputs, 1 output)**

- **Mean Squared Error (MSE):** 0.00015751762190681102  
- **Mean Absolute Error for Train Data:** 0.009029541205325843  
- **Mean Absolute Error for Test Data:** 0.044071174719926234

### **LSTM Model (20 inputs, 7 outputs)**

- **Mean Squared Error (MSE):** 0.00028420048792367485  
- **Mean Absolute Error for Train Data:** 0.012037574119517632  
- **Mean Absolute Error for Test Data:** 0.04765177213495433



## **Contributing**

Feel free to contribute to this project by:
1. Forking the repository.
2. Making changes to improve the model or code.
3. Submitting a pull request with your changes.

