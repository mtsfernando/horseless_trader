# Horseless Trader 🐎📈

A Streamlit web application for predicting stock prices using LSTM neural networks.

## Overview

Horseless Trader is a user-friendly web application that allows you to predict the future stock prices of major companies. It utilizes the `yfinance` library to fetch historical stock data, `keras` with `tensorflow` for building and training an LSTM model, and `streamlit` for creating an interactive web interface.

## Features

- **Stock Data Retrieval:** Fetches historical stock data for selected companies using `yfinance`.
- **Data Preprocessing:** Calculates technical indicators such as SMA, EMA, RSI, Bollinger Bands, Momentum, and Volatility.
- **LSTM Model:** Builds and trains an LSTM neural network to predict future stock prices.
- **Prediction Visualization:** Displays predicted stock prices in a table and a Plotly chart.
- **Interactive Interface:** Uses Streamlit to create a seamless and intuitive user experience.
- **Lottie Animations:** Uses Lottie animations to enhance the visual appeal.
- **Caching:** Caches the trained model to improve performance.

## Getting Started

### Prerequisites

- Python 3.6+
- pip

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download Lottie animations and place them in an `assets` folder in the root directory. You will need `horse-icon.json`, `among-us.json`, and `gradient-loader.json`.

4.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

5.  Open your browser and navigate to the URL displayed in the terminal.

### Usage

1.  Select a stock from the dropdown menu.
2.  Click the "Predict" button to start the prediction process.
3.  View the predicted stock prices in a table and a Plotly chart.

## Code Structure

- `app.py`: Contains the main Streamlit application code.
- `requirements.txt`: Lists the required Python packages.
- `assets/`: Contains Lottie animation files.

## Dependencies

- `yfinance`
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `keras`
- `scikit-learn`
- `streamlit-lottie`

## Functions

- `load_lottie_file(file_path: str)`: Loads a Lottie animation file.
- `fetch_data(symbol)`: Fetches historical stock data from `yfinance`.
- `preprocess_data(df)`: Preprocesses the stock data by calculating technical indicators.
- `normalize_data(df)`: Normalizes the stock data using `MinMaxScaler`.
- `prepare_data(scaled_data, time_steps=60)`: Prepares the data for the LSTM model.
- `build_model(input_shape)`: Builds and compiles the LSTM model.
- `build_predictions_df(predictions)`: Creates a DataFrame for the predicted stock prices.
- `plot_predictions(prediction_df)`: Plots the predicted stock prices using Plotly.
- `run_predictions(data)`: Runs the prediction process and returns the predicted stock prices.
- `update_selection()`: updates session state when a new stock is selected.
- `footer()`: Adds a footer to the streamlit app.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.

## Author

Created by MTS | [GitHub](https://github.com/mtsfernando)

## Acknowledgements

- Powered by `yfinance`, `streamlit`, `keras`, and other amazing open-source libraries.
- Lottie animations from LottieFiles.
