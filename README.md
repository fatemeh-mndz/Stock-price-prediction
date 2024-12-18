

## **Project Overview**
![Forecasting Graph](./télécharger.jpg)
This project demonstrates the use of Long Short-Term Memory (LSTM) networks for time series forecasting. The main objective is to predict Amazon's future stock prices based on its historical data. While the dataset includes information for other assets like Netflix and Bitcoin, the focus is solely on predicting Amazon's stock price. LSTM, a type of recurrent neural network (RNN), is particularly effective for modeling time-series data, making it ideal for stock price forecasting.

### **Key Features:**
- **Data Preprocessing**: Handling multiple assets (Netflix, Bitcoin, Amazon, etc.) while focusing on Amazon's stock.
- **Four Models Developed**:
  1. **Dense Model with 20 Input Features and 1 Output**
  2. **Dense Model with 20 Input Features and 7 Outputs**
  3. **LSTM Model with 20 Input Features and 1 Output**
  4. **LSTM Model with 20 Input Features and 7 Outputs**
- **Sequential Data**: The model forecasts future prices using 30-day input sequences.
- **Multiple Time Steps Forecasting**: Predictions made for 30 future days.

## **Technologies Used**
- **Python** (Programming Language)
- **NumPy** (Numerical Computing)
- **Matplotlib** (Data Visualization)
- **TensorFlow/Keras** (Deep Learning Framework)
- **Scikit-Learn** (Machine Learning Library for preprocessing)

## **Data Preparation**

1. **Dataset**:
   - The dataset includes historical stock prices of Amazon and other assets. [Link to dataset](portfolio_data.csv).
   - The focus of the model is solely on **Amazon's stock price**.

2. **Preprocessing**:
   - Data is normalized to the range [0, 1] using MinMaxScaler.
   - Data is reshaped into sequences to be fed into the LSTM model for forecasting.

### **Model Training**
Each model is trained on Amazon's stock price data. During training, the model performance is evaluated on the validation set using the Mean Squared Error (MSE) loss function. The weights are adjusted accordingly to minimize the error.

## **Results**
Here are the evaluation metrics for each model:

### **Dense Model (20 Inputs - 1 Output)**:
- **Test Loss**: 0.0066
- **Test MAPE**: 0.0478
- **Test Accuracy**: 95.21%

### **Dense Model (20 Inputs - 7 Outputs)**:
- **Test Loss**: 0.0119
- **Test MAPE**: 0.0712
- **Test Accuracy**: 92.87%

### **LSTM Model (20 Inputs - 1 Output)**:
- **Test Loss**: 0.0145
- **Test MAPE**: 0.1045
- **Test Accuracy**: 89.54%

### **LSTM Model (20 Inputs - 7 Outputs)**:
- **Test Loss**: 0.0165
- **Test MAPE**: 0.1388
- **Test Accuracy**: 86.11%

## **Contributing**
Feel free to contribute by:
1. Forking the repository.
2. Making improvements to the code or models.
3. Submitting a pull request with your changes.
