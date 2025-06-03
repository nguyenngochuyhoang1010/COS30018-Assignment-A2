import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

class LSTMTrafficPredictor:
    """
    Implements a simple LSTM model for traffic flow prediction.
    """
    def __init__(self, input_dim, units=50, output_dim=1, learning_rate=0.001):
        """
        Initializes the LSTM model.

        Args:
            input_dim (int): The number of input features per timestep (e.g., 18).
            units (int): The number of units in the LSTM layer.
            output_dim (int): The dimension of the final prediction output (e.g., 1 for traffic volume).
            learning_rate (float): Learning rate for the Adam optimizer.
        """
        self.input_dim = input_dim
        self.timesteps = 1 # We'll reshape our 2D data to be 1 timestep per sample
        self.units = units
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.model = None

        self._build_model()

    def _build_model(self):
        """
        Builds the sequential LSTM model.
        """
        print(f"Building LSTM Model with Input Shape: ({self.timesteps}, {self.input_dim})")
        self.model = Sequential([
            # Input layer for LSTM expects (timesteps, features_per_timestep)
            Input(shape=(self.timesteps, self.input_dim), name='lstm_input'),
            LSTM(self.units, activation='relu', name='lstm_layer'),
            Dense(self.output_dim, activation='linear', name='output_layer') # Linear activation for regression
        ], name='lstm_traffic_predictor')

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        print("\nLSTM Model Summary:")
        self.model.summary()

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        """
        Trains the LSTM model.

        Args:
            X_train (np.array): Training features. Must be 3D: (samples, timesteps, features).
            y_train (np.array): Training target.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training data to be used as validation data.
        """
        if self.model is None:
            print("Model not built. Call _build_model() or initialize the class.")
            return None

        print("\nTraining the LSTM Traffic Predictor model...")
        history = self.model.fit(X_train, y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_split=validation_split,
                               verbose=1)
        print("\nLSTM model training complete.")
        return history

    def predict(self, X_test):
        """
        Makes predictions using the trained LSTM model.

        Args:
            X_test (np.array): Test features. Must be 3D: (samples, timesteps, features).

        Returns:
            np.array: Predicted traffic volumes.
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """
        Saves the trained LSTM model.
        """
        if self.model:
            self.model.save(filepath)
            print(f"LSTM Model saved to {filepath}")
        else:
            print("No LSTM model to save.")

    def load_model(self, filepath):
        """
        Loads a pre-trained LSTM model.
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"LSTM Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading LSTM model from {filepath}: {e}")
            self.model = None

# Example usage (for testing this module directly)
if __name__ == "__main__":
    input_dim = 18 # Based on your DataLoader's feature_columns
    timesteps = 1 # Example: one timestep per sample
    
    # Dummy data for demonstration - must be 3D
    X_train_dummy = np.random.rand(1000, timesteps, input_dim)
    y_train_dummy = np.random.rand(1000, 1) * 100 # Example target

    lstm_predictor = LSTMTrafficPredictor(input_dim=input_dim)
    
    # Train the model
    lstm_predictor.train(X_train_dummy, y_train_dummy, epochs=10)

    # Make a prediction
    X_test_dummy = np.random.rand(10, timesteps, input_dim)
    predictions = lstm_predictor.predict(X_test_dummy)
    print("\nSample Predictions:")
    print(predictions)

    # Save and load example
    lstm_predictor.save_model('test_lstm_model.h5')
    loaded_lstm = LSTMTrafficPredictor(input_dim=input_dim) # Re-initialize for loading
    loaded_lstm.load_model('test_lstm_model.h5')
    loaded_predictions = loaded_lstm.predict(X_test_dummy)
    print("\nSample Predictions from loaded model:")
    print(loaded_predictions)