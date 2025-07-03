import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

class GRUTrafficPredictor:
    """
    Implements a GRU model for traffic flow prediction.
    """
    def __init__(self, input_dim, units=50, output_dim=1, learning_rate=0.001):
        """
        Initializes the GRU model.
        Args:
            input_dim (int): The number of input features per timestep.
            units (int): The number of units in the GRU layer.
            output_dim (int): The dimension of the final prediction output.
            learning_rate (float): Learning rate for the Adam optimizer.
        """
        self.input_dim = input_dim
        self.timesteps = 1 # Reshape 2D data to 3D with 1 timestep
        self.units = units
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Builds the sequential GRU model."""
        print(f"Building GRU Model with Input Shape: ({self.timesteps}, {self.input_dim})")
        model = Sequential([
            Input(shape=(self.timesteps, self.input_dim), name='gru_input'),
            GRU(self.units, activation='relu', name='gru_layer'),
            Dense(self.output_dim, activation='linear', name='output_layer')
        ], name='gru_traffic_predictor')

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        model.summary()
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Trains the GRU model."""
        print("\nTraining the GRU Traffic Predictor model...")
        history = self.model.fit(X_train, y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_split=validation_split,
                               verbose=1)
        print("\nGRU model training complete.")
        return history

    def predict(self, X_test):
        """Makes predictions using the trained GRU model."""
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """Saves the trained GRU model."""
        self.model.save(filepath)
        print(f"GRU Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads a pre-trained GRU model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"GRU Model loaded from {filepath}")
