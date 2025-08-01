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
        self.input_dim = input_dim
        self.timesteps = 1
        self.units = units
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.timesteps, self.input_dim), name='gru_input'),
            GRU(self.units, activation='relu', name='gru_layer'),
            Dense(self.output_dim, activation='linear', name='output_layer')
        ], name='gru_traffic_predictor')
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        model.summary()
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        """Trains the GRU model."""
        print("\nTraining the GRU Traffic Predictor model...")
        self.history = self.model.fit(X_train, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=validation_data,
                                      verbose=1)
        print("\nGRU model training complete.")
        return self.history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"GRU Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        print(f"GRU Model loaded from {filepath}")
