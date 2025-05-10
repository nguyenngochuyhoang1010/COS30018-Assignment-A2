from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_mlp(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(64, 32, 64), max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse