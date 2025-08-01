import xgboost as xgb
import numpy as np

class XGBoostTrafficPredictor:
    """
    Implements an XGBoost model for traffic flow prediction using a Scikit-Learn-like API.
    """
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = self._build_model()
        self.history = {}

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
            n_jobs=-1,
            early_stopping_rounds=10 # Add early stopping
        )
        return model

    def train(self, X_train, y_train, eval_set=None, **kwargs):
        """
        Trains the XGBoost model.
        """
        print("\nTraining the XGBoost Regressor model...")
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        # Store history in a Keras-like format
        results = self.model.evals_result()
        self.history = {'loss': [], 'val_loss': []}

        # FIX: Safely access evaluation results to prevent KeyErrors.
        # The server passes the test set as the first (and only) evaluation set.
        if results and 'validation_0' in results:
            # We'll assign the validation loss to both loss and val_loss
            # to ensure the frontend chart can render without errors.
            validation_loss = results['validation_0']['rmse']
            self.history['val_loss'] = validation_loss
            self.history['loss'] = validation_loss # Use the same data for training loss for consistency
        
        print("\nXGBoost model training complete.")
        return self.history

    def predict(self, X_test):
        """Makes predictions using the trained XGBoost model."""
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """Saves the trained XGBoost model."""
        self.model.save_model(filepath)
        print(f"XGBoost Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads a pre-trained XGBoost model."""
        self.model.load_model(filepath)
        print(f"XGBoost Model loaded from {filepath}")