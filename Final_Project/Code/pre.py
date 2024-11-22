import os
import numpy as np
import pandas as pd
import pickle  # For efficient binary file handling
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load Datasets
def load_data(china_path, us_path):
    """Loads and returns the datasets."""
    # Parse and set datetime index for the China dataset
    china_data = pd.read_csv(
        china_path,
        parse_dates=['Time(year-month-day h:m:s)'],  # Parse datetime column
        index_col='Time(year-month-day h:m:s)'  # Set as index
    )
    # Parse and set datetime index for the US dataset
    us_data = pd.read_csv(
        us_path,
        parse_dates=['datetime'],  # Parse datetime column
        index_col='datetime'  # Set as index
    )
    return china_data, us_data

# Filter Data
def filter_data(china_data, us_data):
    """Filters data for January 2020."""
    china_time_range = china_data.index[0], china_data.index[-1]
    us_time_range = us_data.index[0], us_data.index[-1]

    # Print the time ranges
    print(f"China Data Time Range: {china_time_range[0]} to {china_time_range[1]}")
    print(f"US Data Time Range: {us_time_range[0]} to {us_time_range[1]}")

    # Filter data for January 2020
    china_jan_2020 = china_data.loc["2020-01-01 00:00:00":"2020-01-31 23:00:00"]
    us_jan_2020 = us_data.loc["2020-01-01 00:00:00":"2020-01-31 23:00:00"]
    return china_jan_2020, us_jan_2020

# Preprocess Data
def preprocess_china_data(china_data):
    """Normalize China dataset."""
    china_data= china_data.resample('1H').mean()
    scaler = MinMaxScaler()
    china_features = [
        'Wind speed at height of 10 meters (m/s)',
        'Wind direction at height of 10 meters (˚)',
        'Air temperature  (°C) ',
        'Atmosphere (hpa)',
        'Relative humidity (%)'
    ]
    china_data[china_features] = scaler.fit_transform(china_data[china_features])
    return china_data

def preprocess_us_data(us_data, us_config):
    """Normalize and preprocess US dataset."""
    # Normalize all plant output columns
    scaler = MinMaxScaler()
    us_data[us_data.columns] = scaler.fit_transform(us_data)

    # Integrate metadata if needed
    metadata = us_config.set_index("plant_code_unique")
    return us_data, metadata

# Save Filtered Data
def save_filtered_data(china_jan_2020, us_jan_2020, 
                       china_pickle_file="china_jan_2020.pkl", 
                       us_pickle_file="us_jan_2020.pkl", 
                       china_csv_file="china_jan_2020.csv", 
                       us_csv_file="us_jan_2020.csv"):
    """Saves the filtered data in Pickle and CSV formats."""
    with open(china_pickle_file, 'wb') as f:
        pickle.dump(china_jan_2020, f)
    with open(us_pickle_file, 'wb') as f:
        pickle.dump(us_jan_2020, f)
    print(f"Filtered data saved as Pickle files: '{china_pickle_file}' and '{us_pickle_file}'.")

    china_jan_2020.to_csv(china_csv_file)
    us_jan_2020.to_csv(us_csv_file)
    print(f"Filtered data saved as CSV files: '{china_csv_file}' and '{us_csv_file}'.")

# Load Filtered Data
def load_filtered_data(china_pickle_file="china_jan_2020.pkl", us_pickle_file="us_jan_2020.pkl"):
    """Loads filtered data from Pickle files."""
    with open(china_pickle_file, 'rb') as f:
        china_jan_2020 = pickle.load(f)
    with open(us_pickle_file, 'rb') as f:
        us_jan_2020 = pickle.load(f)
    print(f"Filtered data loaded from Pickle files: '{china_pickle_file}' and '{us_pickle_file}'.")
    return china_jan_2020, us_jan_2020

# Main Process
if __name__ == "__main__":
    # File paths for raw data
    china_path = "Wind_capacity_99MW.csv"
    us_path = "wind_gen_cf_2020.csv"
    us_config_path = "eia_wind_configs.csv"

    # File paths for filtered data
    china_pickle_file = "china_jan_2020.pkl"
    us_pickle_file = "us_jan_2020.pkl"
    china_csv_file = "china_jan_2020.csv"
    us_csv_file = "us_jan_2020.csv"

    
    # Load raw data
    china_data, us_data = load_data(china_path, us_path)
    us_config = pd.read_csv(us_config_path)

    # Filter data
    china_jan_2020, us_jan_2020 = filter_data(china_data, us_data)

    # Preprocess data
    china_jan_2020 = preprocess_china_data(china_jan_2020)
    us_jan_2020, us_metadata = preprocess_us_data(us_jan_2020, us_config)

    # Save filtered data
    save_filtered_data(china_jan_2020, us_jan_2020, china_pickle_file, us_pickle_file, china_csv_file, us_csv_file)

