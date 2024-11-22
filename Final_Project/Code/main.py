# import numpy as np
# import pickle
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tqdm import tqdm

# # Load Preprocessed Data
# def load_filtered_data(china_pickle_file="china_jan_2020.pkl", us_pickle_file="us_jan_2020.pkl"):
#     with open(china_pickle_file, 'rb') as f:
#         china_jan_2020 = pickle.load(f)
#     with open(us_pickle_file, 'rb') as f:
#         us_jan_2020 = pickle.load(f)
#     print(f"Filtered data loaded from Pickle files: '{china_pickle_file}' and '{us_pickle_file}'.")
#     return china_jan_2020, us_jan_2020

# # Prepare Sliding Window Data
# def create_sliding_window(data, input_features, target_feature, window_size=24):
#     X, y = [], []
#     for i in range(len(data) - window_size):
#         X.append(data[input_features].iloc[i:i + window_size].values)
#         y.append(data[target_feature].iloc[i + window_size])
#     return np.array(X), np.array(y)

# # Build LSTM Model
# def build_lstm(input_shape):
#     model = Sequential([
#         LSTM(64, input_shape=input_shape, return_sequences=False),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
#     return model

# # Expand features to match a given number of features
# def expand_features_to_match(data, target_feature_count):
#     current_feature_count = data.shape[2]
#     if current_feature_count < target_feature_count:
#         zeros_to_add = target_feature_count - current_feature_count
#         zeros = np.zeros((data.shape[0], data.shape[1], zeros_to_add))
#         data = np.concatenate([data, zeros], axis=2)
#     return data

# # Evaluate model performance
# def evaluate_model(model, test_X, test_y):
#     predictions = model.predict(test_X)
#     rmse = np.sqrt(mean_squared_error(test_y, predictions))
#     mae = mean_absolute_error(test_y, predictions)
#     return rmse, mae

# if __name__ == "__main__":
#     china_pickle_file = "china_jan_2020.pkl"
#     us_pickle_file = "us_jan_2020.pkl"
#     china_data, us_data = load_filtered_data(china_pickle_file, us_pickle_file)

#     # Define features and targets
#     china_features = ['Wind speed at height of 10 meters (m/s)', 'Wind direction at height of 10 meters (˚)', 'Air temperature  (°C) ', 'Atmosphere (hpa)', 'Relative humidity (%)']
#     china_target = 'Power (MW)'
#     us_features = ['508_1']
#     us_target = '508_1'

#     # Create sliding window data
#     china_X, china_y = create_sliding_window(china_data, china_features, china_target, window_size=24)
#     us_X, us_y = create_sliding_window(us_data, us_features, us_target, window_size=24)

#     # Split data into training and testing
#     split_idx_china = int(0.7 * len(china_X))
#     china_train_X, china_test_X = china_X[:split_idx_china], china_X[split_idx_china:]
#     china_train_y, china_test_y = china_y[:split_idx_china], china_y[split_idx_china:]

#     split_idx_us = int(0.7 * len(us_X))
#     us_train_X, us_test_X = us_X[:split_idx_us], us_X[split_idx_us:]
#     us_train_y, us_test_y = us_y[:split_idx_us], us_y[split_idx_us:]

#     # Expand US data to match feature count
#     us_X_expanded = expand_features_to_match(us_X, 5)
#     us_train_X_expanded, us_test_X_expanded = us_X_expanded[:split_idx_us], us_X_expanded[split_idx_us:]

#     # Build models
#     china_model = build_lstm((china_train_X.shape[1], china_train_X.shape[2]))
#     us_model_separate = build_lstm((us_train_X.shape[1], us_train_X.shape[2]))
#     us_model_expanded = build_lstm((us_train_X_expanded.shape[1], us_train_X_expanded.shape[2]))

#     # Early stopping callback
#     early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     # Train China model
#     china_model.fit(china_train_X, china_train_y, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop])

#     # Train US models
#     us_model_separate.fit(us_train_X, us_train_y, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop])
#     us_model_expanded.fit(us_train_X_expanded, us_train_y, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop])

#     # Evaluate models
#     rmse_china, mae_china = evaluate_model(china_model, china_test_X, china_test_y)
#     rmse_us_separate, mae_us_separate = evaluate_model(us_model_separate, us_test_X, us_test_y)
#     rmse_us_expanded, mae_us_expanded = evaluate_model(us_model_expanded, us_test_X_expanded, us_test_y)

#     # Print results
#     print(f"China Model RMSE: {rmse_china}, MAE: {mae_china}")
#     print(f"US Model Separate RMSE: {rmse_us_separate}, MAE: {mae_us_separate}")
#     print(f"US Model Expanded RMSE: {rmse_us_expanded}, MAE: {mae_us_expanded}")

from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt


# Load Preprocessed Data
def load_filtered_data(china_pickle_file="china_jan_2020.pkl", us_pickle_file="us_jan_2020.pkl"):
    with open(china_pickle_file, 'rb') as f:
        china_jan_2020 = pickle.load(f)
    with open(us_pickle_file, 'rb') as f:
        us_jan_2020 = pickle.load(f)
    print(f"Filtered data loaded from Pickle files: '{china_pickle_file}' and '{us_pickle_file}'.")
    return china_jan_2020, us_jan_2020


# Prepare Sliding Window Data
def create_sliding_window(data, input_features, target_feature, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[input_features].iloc[i:i + window_size].values)
        y.append(data[target_feature].iloc[i + window_size])
    return np.array(X), np.array(y)


# Build LSTM Model
def build_lstm(input_shape, name_prefix=""):
    model = Sequential(name=f"{name_prefix}_model")
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False, name=f"{name_prefix}_lstm"))
    model.add(Dense(32, activation='relu', name=f"{name_prefix}_dense1"))
    model.add(Dense(1, name=f"{name_prefix}_output"))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# Transfer weights from one model to another based on layer names
def transfer_weights(source_model, target_model):
    for target_layer in target_model.layers:
        if 'lstm' not in target_layer.name:  # Skip LSTM layer due to input shape dependency
            base_layer_name = '_'.join(target_layer.name.split('_')[1:])  # Remove prefix
            source_layer_name = 'china_' + base_layer_name
            if source_model.get_layer(source_layer_name):
                weights = source_model.get_layer(source_layer_name).get_weights()
                target_layer.set_weights(weights)
                print(f"Transferred weights to {target_layer.name}")
            else:
                print(f"No matching layer found in source model for {target_layer.name}")


# Evaluate model performance
# Evaluate model performance with AUC and AP
from sklearn.metrics import r2_score

def evaluate_model(model, test_X, test_y):
    predictions = model.predict(test_X).flatten()
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    return rmse, mae, r2, predictions



# Plot predictions vs actual values
def plot_predictions(actual, predicted, title, xlabel, ylabel):
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# Train and evaluate the models
def train_and_evaluate(data, features, target, model_name, transfer_model=None, window_size=24, epochs=50):
    X, y = create_sliding_window(data, features, target, window_size)
    split_idx = int(0.7 * len(X))
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]

    model = build_lstm((train_X.shape[1], train_X.shape[2]), name_prefix=model_name)

    if transfer_model:
        transfer_weights(transfer_model, model)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_X, train_y, validation_split=0.2, epochs=epochs, batch_size=32)

    rmse, mae, r2, predictions = evaluate_model(model, test_X, test_y)
    return model, rmse, mae, r2, test_y, predictions



if __name__ == "__main__":
    # Load datasets
    china_data = pd.read_csv('china_jan_2020.csv')
    us_data = pd.read_csv('us_jan_2020.csv')

    # Define features and targets
    china_features = ['Wind speed at height of 10 meters (m/s)', 'Wind direction at height of 10 meters (˚)', 
                      'Air temperature  (°C) ', 'Atmosphere (hpa)', 'Relative humidity (%)']
    china_target = 'Power (MW)'
    us_features = ['508_1']
    us_target = '508_1'

    # Train and evaluate the China model
    china_model, china_rmse, china_mae, china_r2, china_actual, china_predictions = train_and_evaluate(
        china_data, china_features, china_target, model_name="china"
    )

    # Train and evaluate the US model with weight transfer
    us_model, us_rmse, us_mae, us_r2, us_actual, us_predictions = train_and_evaluate(
        us_data, us_features, us_target, model_name="us", transfer_model=china_model
    )

    # Print results
    print(f"China Model RMSE: {china_rmse}, MAE: {china_mae}, R2: {china_r2}")
    print(f"US Model RMSE: {us_rmse}, MAE: {us_mae}, R2: {us_r2}")


    # Plot results
    plot_predictions(china_actual, china_predictions, 'China Model: Actual vs Predicted Power', 'Time', 'Power (MW)')
    plot_predictions(us_actual, us_predictions, 'US Model: Actual vs Predicted Power', 'Time', 'Power (MW)')
