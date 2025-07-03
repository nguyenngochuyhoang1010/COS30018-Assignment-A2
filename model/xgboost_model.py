import xgboost as xgb
import numpy as np

class XGBoostTrafficPredictor:
    """
    Implements an XGBoost model for traffic flow prediction using a Scikit-Learn-like API.
    """
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Initializes the XGBoost model.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = self._build_model()
        # Create a dummy history object to mimic Keras API
        self.history = type('obj', (object,), {'history': {'loss': [0], 'val_loss': [0]}})()


    def _build_model(self):
        """Builds the XGBoost regressor model."""
        print("Building XGBoost Regressor Model...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1  # Use all available CPU cores
        )
        return model

    def train(self, X_train, y_train, epochs=None, batch_size=None, validation_split=None):
        """
        Trains the XGBoost model. The epochs and batch_size arguments are ignored
        to maintain a consistent API with Keras models but are not used by XGBoost.
        """
        print("\nTraining the XGBoost Regressor model...")
        self.model.fit(X_train, y_train, verbose=False)
        print("\nXGBoost model training complete.")
        # Since XGBoost doesn't have epochs, we can't generate a real loss history.
        # We return a dummy object for API consistency with the frontend.
        return self.history


    def predict(self, X_test):
        """Makes predictions using the trained XGBoost model."""
        # XGBoost predict does not require reshaping
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """Saves the trained XGBoost model."""
        self.model.save_model(filepath)
        print(f"XGBoost Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads a pre-trained XGBoost model."""
        self.model.load_model(filepath)
        print(f"XGBoost Model loaded from {filepath}")