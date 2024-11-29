# EMD-LSTM Stock Price Prediction

## Overview

Predicting future values of various financial data holds great importance for diverse business applications. However, these financial datasets often exhibit a high degree of non-linearity and non-stationarity, posing challenges to the accuracy of predictive models. Employing the empirical mode decomposition (EMD) algorithm for pre-processing raw data has shown promising results in enhancing predictive model accuracy.

This paper introduces a hybrid approach that incorporates EMD before applying predictive models to forecast one-step future stock values of Apple Inc. Various models, including the feed-forward neural network (ANN), LSTM recurrent neural network, auto-regressive integrated moving average (ARIMA), Support Vector Regression (SVR), and Gaussian Process Regression (GP), have been explored as potential predictive models in this study.

This Python script implements a stock price prediction methodology using Empirical Mode Decomposition (EMD) and Long Short-Term Memory (LSTM) neural networks. The project focuses on analyzing and predicting stock price movements by decomposing time series data and applying machine learning techniques.

## Features

- Data preprocessing using Empirical Mode Decomposition (EMD)
- LSTM-based prediction model
- Multiple stock dataset support (IBM, Indian Stock Exchange, Apple)
- Performance metrics calculation
- Visualization of predictions

## Dependencies

- NumPy
- Matplotlib
- TensorFlow (1.x)
- PyEMD
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emd-lstm-stock-prediction.git
```

2. Install required dependencies:
```bash
pip install numpy matplotlib tensorflow==1.x pyemd scikit-learn
```

## Usage

### Configuration

Before running the script, set up your:
- `datadirectory`: Path to input stock data files
- `save_dir`: Directory for saving output plots

### Available Datasets

The script includes predefined datasets:
- `ibmstockdata.txt`
- `indianstockexchange.txt`
- `applestockbiggest.txt`

### Running the Script

```bash
python emd_lstm_prediction.py
```

## Methodology

### Data Preprocessing
1. Read stock data from text files
2. Calculate time series differences
3. Apply Empirical Mode Decomposition (EMD)

### LSTM Prediction
- Input sequence length: 5 steps
- LSTM layer size: 128
- Dropout rate: 0.9
- Training epochs: 130

## Functions

### Key Functions
- `read_data()`: Load stock datasets
- `differenceTimeSeries()`: Calculate time series differences
- `run_emd()`: Apply Empirical Mode Decomposition
- `lstm_prediction()`: LSTM neural network prediction
- `model_validation()`: Calculate prediction metrics
- `plot_predictions()`: Visualize prediction results

## Performance Metrics

The script calculates:
- Mean Squared Error (MSE)
- Binary classification accuracy
- Precision
- Recall
- F1 Score

## Visualization

Generates plots for:
- Original stock data
- Difference time series
- IMF components
- Prediction comparisons

## Limitations

- Designed for TensorFlow 1.x
- Specific to stock price prediction
- Performance varies with different datasets

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - youremail@example.com

Project Link: [https://github.com/yourusername/emd-lstm-stock-prediction](https://github.com/yourusername/emd-lstm-stock-prediction)
