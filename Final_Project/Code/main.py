import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


# Train and evaluate the models for multiple trials
def train_and_evaluate_multiple_times(data, features, target, model_name, transfer_model=None, window_size=24, epochs=50, num_trials=10):
    # Initialize lists to store evaluation metrics for each trial
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    last_model = None  # Store the model from the last trial
    
    for trial in range(num_trials):
        print(f"\nRunning trial {trial + 1}/{num_trials}...")
        
        # Train and evaluate the model for each trial
        model, rmse, mae, r2, test_y, predictions = train_and_evaluate(data, features, target, model_name, transfer_model, window_size, epochs)
        
        # Store the metrics for this trial
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        # Update the last trained model
        last_model = model
    
    # Calculate mean and standard deviation for each metric
    rmse_mean = np.mean(rmse_scores)
    rmse_std = np.std(rmse_scores)
    
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    
    # Return the metrics and the last trained model
    return {
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'model': last_model  # Return the model from the last trial
    }


# Train and evaluate the model for a single trial
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

    # Train and evaluate the China model with multiple trials
    china_results = train_and_evaluate_multiple_times(
        china_data, china_features, china_target, model_name="china", num_trials=50
    )

    # Train and evaluate the US model with multiple trials and weight transfer
    us_results = train_and_evaluate_multiple_times(
        us_data, us_features, us_target, model_name="us", transfer_model=china_results['model'], num_trials=50
    )

    # Print results for China model
    print(f"China Model RMSE Mean: {china_results['rmse_mean']}, RMSE Std: {china_results['rmse_std']}")
    print(f"China Model MAE Mean: {china_results['mae_mean']}, MAE Std: {china_results['mae_std']}")
    print(f"China Model R2 Mean: {china_results['r2_mean']}, R2 Std: {china_results['r2_std']}")

    # Print results for US model
    print(f"US Model RMSE Mean: {us_results['rmse_mean']}, RMSE Std: {us_results['rmse_std']}")
    print(f"US Model MAE Mean: {us_results['mae_mean']}, MAE Std: {us_results['mae_std']}")
    print(f"US Model R2 Mean: {us_results['r2_mean']}, R2 Std: {us_results['r2_std']}")

    # You can plot results for the last trial (or mean predictions) here if desired
