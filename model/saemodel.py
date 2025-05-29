import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

class StackedAutoencoder:
    """
    Implements a Stacked Autoencoder (SAE) for feature learning,
    followed by a regression layer for traffic flow prediction.
    """
    def __init__(self, input_dim, encoding_dims, regression_output_dim=1, learning_rate=0.001):
        """
        Initializes the Stacked Autoencoder model.

        Args:
            input_dim (int): The number of input features.
            encoding_dims (list): A list of integers, where each integer represents
                                  the number of neurons in the hidden layer of an autoencoder.
                                  E.g., [128, 64, 32] for three autoencoders.
            regression_output_dim (int): The dimension of the final regression output (e.g., 1 for traffic volume).
            learning_rate (float): Learning rate for the Adam optimizer.
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.regression_output_dim = regression_output_dim
        self.learning_rate = learning_rate
        self.autoencoders = [] # Stores individual autoencoder models
        self.encoder_model = None # The stacked encoder part
        self.full_model = None # The full SAE + regression model

        self._build_stacked_autoencoders()
        self._build_full_model()

    def _build_stacked_autoencoders(self):
        """
        Builds individual autoencoders and stacks them to form the encoder part.
        This method trains each autoencoder sequentially (greedy layer-wise training).
        """
        current_input_dim = self.input_dim
        current_input_layer = Input(shape=(self.input_dim,)) # For the first autoencoder

        print("Building Stacked Autoencoders (greedy layer-wise training)...")

        for i, encoding_dim in enumerate(self.encoding_dims):
            print(f"   Building Autoencoder Layer {i+1} with encoding dim: {encoding_dim}")

            # Encoder part
            encoder_layer = Dense(encoding_dim, activation='relu', name=f'encoder_h{i+1}')
            # Decoder part
            decoder_layer = Dense(current_input_dim, activation='sigmoid', name=f'decoder_h{i+1}') # Sigmoid for reconstruction

            # Connect layers for the current autoencoder
            encoded = encoder_layer(current_input_layer)
            decoded = decoder_layer(encoded)

            # Define the current autoencoder model
            autoencoder = Model(inputs=current_input_layer, outputs=decoded, name=f'autoencoder_{i+1}')
            autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
            self.autoencoders.append(autoencoder)

            # Update input for the next autoencoder: output of the current encoder
            current_input_layer = encoded # The output of the current encoder becomes input for the next
            current_input_dim = encoding_dim # The dimension of the next input

        # After building all individual autoencoders, create the full encoder model
        # The encoder model takes the original input and outputs the final encoded representation
        x = Input(shape=(self.input_dim,))
        current_output = x
        for i, encoding_dim in enumerate(self.encoding_dims):
            # Extract the encoder layer from the trained autoencoder
            # Ensure the weights are transferred correctly (they are by get_layer)
            encoder_layer = self.autoencoders[i].get_layer(f'encoder_h{i+1}')
            current_output = encoder_layer(current_output)

        self.encoder_model = Model(inputs=x, outputs=current_output, name='stacked_encoder')
        print("\nStacked Encoder Model Summary:")
        self.encoder_model.summary()

    def _build_full_model(self):
        """
        Builds the full model by attaching a regression layer on top of the
        pre-trained stacked encoder.
        """
        if self.encoder_model is None:
            print("Encoder model not built. Call _build_stacked_autoencoders() first.")
            return

        print("\nBuilding Full Model (SAE + Regression Layer)...")
        full_model_input = Input(shape=(self.input_dim,))

        # --- FIX STARTS HERE ---
        # Pass the input through the entire pre-trained encoder model directly.
        # This treats self.encoder_model as a single, callable block.
        encoded_output = self.encoder_model(full_model_input)
        # --- FIX ENDS HERE ---

        # Add a regression output layer
        regression_output = Dense(self.regression_output_dim, activation='linear', name='regression_output')(encoded_output)

        self.full_model = Model(inputs=full_model_input, outputs=regression_output, name='sae_traffic_predictor')
        self.full_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse') # MSE for regression

        print("\nFull SAE Traffic Predictor Model Summary:")
        self.full_model.summary()

    def pretrain_autoencoders(self, X_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Pre-trains each autoencoder in a greedy layer-wise fashion.

        Args:
            X_train (np.array): Training data for pre-training.
            epochs (int): Number of epochs for pre-training each autoencoder.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training data to be used as validation data.
        """
        print("\nStarting greedy layer-wise pre-training of autoencoders...")
        current_data = X_train

        for i, autoencoder in enumerate(self.autoencoders):
            print(f"\nPre-training Autoencoder {i+1}/{len(self.autoencoders)}...")
            print(f"   Input shape for this autoencoder: {current_data.shape}")
            autoencoder.fit(current_data, current_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            verbose=1)

            # Get the encoded representation to use as input for the next autoencoder
            # The encoder layer is the first Dense layer in each autoencoder model
            encoder_output = autoencoder.layers[1].output
            encoder_model_for_next_layer = Model(inputs=autoencoder.input, outputs=encoder_output)
            current_data = encoder_model_for_next_layer.predict(current_data)
            print(f"   Output shape from encoder for next layer: {current_data.shape}")

        print("\nStacked Autoencoders pre-training complete.")

    def train_full_model(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1):
        """
        Trains the full SAE + regression model.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training target.
            epochs (int): Number of epochs for training the full model.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training data to be used as validation data.
        """
        if self.full_model is None:
            print("Full model not built. Please call _build_full_model() or initialize the class.")
            return None # Return None to indicate failure

        print("\nTraining the full SAE Traffic Predictor model...")
        history = self.full_model.fit(X_train, y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_split=validation_split,
                               verbose=1)
        print("\nFull model training complete.")
        return history # Return history for plotting

    def predict(self, X_test):
        """
        Makes predictions using the trained full model.

        Args:
            X_test (np.array): Test features.

        Returns:
            np.array: Predicted traffic volumes.
        """
        if self.full_model is None:
            print("Full model not trained. Please train the model first.")
            return None
        return self.full_model.predict(X_test)

    def save_model(self, filepath):
        """
        Saves the full trained model.

        Args:
            filepath (str): Path to save the model (e.g., 'sae_model.h5').
        """
        if self.full_model:
            self.full_model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No full model to save.")

    def load_model(self, filepath):
        """
        Loads a pre-trained model.

        Args:
            filepath (str): Path to the saved model.
        """
        try:
            self.full_model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            self.full_model = None

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Dummy data for demonstration
    # In a real scenario, X_train would come from your DataLoader
    input_dim = 20 # Example number of features
    X_train_dummy = np.random.rand(1000, input_dim)
    y_train_dummy = np.random.rand(1000, 1) * 100 # Example target

    # Initialize SAE with encoding dimensions
    # For instance, two autoencoder layers with 10 and 5 neurons respectively
    sae = StackedAutoencoder(input_dim=input_dim, encoding_dims=[10, 5])

    # Pre-train autoencoders
    sae.pretrain_autoencoders(X_train_dummy, epochs=10)

    # Train the full model
    sae.train_full_model(X_train_dummy, y_train_dummy, epochs=20)

    # Make a prediction
    X_test_dummy = np.random.rand(10, input_dim)
    predictions = sae.predict(X_test_dummy)
    print("\nSample Predictions:")
    print(predictions)

    # Save and load example
    sae.save_model('test_sae_model.h5')
    loaded_sae = StackedAutoencoder(input_dim=input_dim, encoding_dims=[10, 5]) # Re-initialize for loading
    loaded_sae.load_model('test_sae_model.h5')
    loaded_predictions = loaded_sae.predict(X_test_dummy)
    print("\nSample Predictions from loaded model:")
    print(loaded_predictions)