import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load Preprocessed Data
def load_filtered_data(china_pickle_file="china_jan_2020.pkl"):
    """
    Loads preprocessed filtered data.
    """
    with open(china_pickle_file, 'rb') as f:
        china_data = pickle.load(f)
    print(f"Filtered data loaded from Pickle file: '{china_pickle_file}'.")
    return china_data


# Prepare Sliding Window Data
def create_sliding_window(data, input_features, target_feature, window_size=24):
    """
    Create input-output pairs using sliding windows.
    Args:
    - data: DataFrame with the features and target.
    - input_features: List of column names to use as features.
    - target_feature: Name of the target column.
    - window_size: Number of time steps in the input sequence.

    Returns:
    - X: Input features array (shape: samples x window_size x features).
    - y: Target values array (shape: samples,).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[input_features].iloc[i:i + window_size].values)
        y.append(data[target_feature].iloc[i + window_size])
    return np.array(X), np.array(y)


# Build LSTM Model
def build_lstm(input_shape):
    """
    Build and return an LSTM model.
    Args:
    - input_shape: Tuple representing the input shape (time steps, features).

    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# Evaluate model performance
def evaluate_model(model, test_X, test_y):
    """
    Evaluate the model and return RMSE and MAE.
    """
    predictions = model.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    return predictions, rmse, mae


# Plot Actual vs Predicted
def plot_actual_vs_predicted(actual, predicted, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Power (MW)", color="blue", linewidth=2)
    plt.plot(predicted, label="Predicted Power (MW)", color="orange", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Power (MW)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load China data
    china_pickle_file = "china_jan_2020.pkl"
    china_data = load_filtered_data(china_pickle_file)

    # Define features and target
    china_features = ['Wind speed at height of 10 meters (m/s)',
                      'Wind direction at height of 10 meters (˚)',
                      'Air temperature  (°C) ',
                      'Atmosphere (hpa)',
                      'Relative humidity (%)']
    china_target = 'Power (MW)'

    # Prepare sliding windows
    window_size = 24  # Number of time steps
    china_X, china_y = create_sliding_window(china_data, china_features, china_target, window_size=window_size)

    # Split data into training and testing sets
    split_idx = int(0.7 * len(china_X))
    train_X, test_X = china_X[:split_idx], china_X[split_idx:]
    train_y, test_y = china_y[:split_idx], china_y[split_idx:]

    # Build and train the model
    input_shape = (train_X.shape[1], train_X.shape[2])
    lstm_model = build_lstm(input_shape)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training the LSTM model...")
    lstm_model.fit(train_X, train_y, validation_split=0.2, epochs=200, batch_size=32, callbacks=[early_stop])

    # Evaluate the model
    print("Evaluating the model on the test set...")
    predicted, rmse, mae = evaluate_model(lstm_model, test_X, test_y)

    print(f"Evaluation Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")

    # Plot actual vs predicted values
    plot_actual_vs_predicted(test_y, predicted, title="China Model: Actual vs Predicted Power")
