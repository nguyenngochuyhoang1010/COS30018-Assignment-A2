from utils.dataloader import load_traffic_data, scale_data, create_sequences
from model.mlpmodel import train_mlp, evaluate_model
from sklearn.model_selection import train_test_split
import joblib

# Load and prepare data
df = load_traffic_data("data/traffic_boroondara.csv")
df, scaler = scale_data(df)
X, y = create_sequences(df['volume'].values)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_mlp(X_train, y_train)

# Evaluate
mse = evaluate_model(model, X_test, y_test)
print("Model MSE:", mse)

# Save model and scaler
joblib.dump(model, "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")