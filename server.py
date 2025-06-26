from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import traceback
import uuid # Used for creating unique temporary filenames

# Import your existing classes
from utils.dataloader import TrafficDataLoader
from model.saemodel import StackedAutoencoder
from model.lstm_model import LSTMTrafficPredictor

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global variables to hold state (in a real app, this might be a database or cache) ---
data_loader = None
active_model = None
scaler = None
X_train_columns = None # Store column order for prediction

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page."""
    try:
        with open("index.html") as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Error: index.html not found. Make sure the HTML file is in the same directory as server.py.", 404


@app.route('/get-headers', methods=['POST'])
def get_headers():
    """
    Safely reads the header row of an uploaded CSV.
    This function is wrapped in a try-except block to guarantee a JSON response.
    """
    # --- FIX: The entire function logic is now wrapped in a try-except block ---
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file was selected."}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Please upload a CSV."}), 400

        temp_filename = f"temp_{uuid.uuid4().hex}.csv"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            file.save(temp_filepath)
            header_row = int(request.form.get('header_row', 0))
            
            # Read headers from the saved file. This is where errors often occur.
            headers = pd.read_csv(temp_filepath, header=header_row, nrows=0).columns.str.strip().tolist()
            
            return jsonify({"headers": headers})
            
        finally:
            # Ensure the temporary file is always cleaned up
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    except Exception as e:
        # This will catch any error, including file parsing errors, and return a specific message.
        traceback.print_exc() # This will print the full error to your Flask console for debugging.
        return jsonify({"error": f"A server error occurred: {str(e)}"}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Handles data uploading, mapping, and preprocessing."""
    global data_loader, scaler, X_train_columns
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        column_mapping = request.form.get('column_mapping')
        header_row = int(request.form.get('header_row', 0))
        
        if not column_mapping:
            return jsonify({"error": "Column mapping is required"}), 400
            
        import json
        column_mapping = json.loads(column_mapping)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)
        file.save(filepath)

        data_loader = TrafficDataLoader(filepath, column_mapping, header_row)
        data_loader.load_and_initial_clean()

        if data_loader.is_long_format:
            data_loader.process_long_format()
        else:
            data_loader.transform_to_long_format()

        data_loader.engineer_features()
        X_scaled, y, fitted_scaler = data_loader.prepare_for_model()
        
        if X_scaled is None:
            raise ValueError("Data preprocessing failed to produce features.")
        
        scaler = fitted_scaler
        X_train_columns = X_scaled.columns.tolist()

        return jsonify({"message": "Data preprocessed successfully!", "rows_processed": len(data_loader.df_final)})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during preprocessing: {e}"}), 500

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """Trains the selected model."""
    global active_model, data_loader
    
    if not data_loader or data_loader.df_final is None:
        return jsonify({"error": "Please preprocess data first."}), 400
        
    try:
        data = request.get_json()
        model_name = data.get('model', 'lstm')
        epochs = int(data.get('epochs', 50))
        
        X_scaled, y, _ = data_loader.prepare_for_model()
        X_train, _, y_train, _ = data_loader.chronological_train_test_split(X_scaled, y)
        
        X_train_data = X_train.values
        y_train_data = y_train.values

        if model_name == 'lstm':
            model_instance = LSTMTrafficPredictor(input_dim=X_train_data.shape[1])
            X_train_reshaped = X_train_data.reshape(X_train_data.shape[0], 1, X_train_data.shape[1])
            model_instance.train(X_train_reshaped, y_train_data, epochs=epochs)
        elif model_name == 'sae':
            model_instance = StackedAutoencoder(input_dim=X_train_data.shape[1], encoding_dims=[128, 64, 32])
            model_instance.train_full_model(X_train_data, y_train_data, epochs=epochs)
        else:
            return jsonify({"error": "Invalid model type specified"}), 400
            
        active_model = model_instance
        
        return jsonify({"message": f"{model_name.upper()} model trained successfully."})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during training: {e}"}), 500
        
@app.route('/predict-csv', methods=['POST'])
def predict_batch():
    """Makes predictions on a batch of data from an uploaded CSV."""
    global active_model, scaler, X_train_columns, data_loader
    
    if not all([active_model, scaler, X_train_columns, data_loader]):
        return jsonify({"error": "Please preprocess data and train a model first."}), 400
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        pred_df = pd.read_csv(file, header=data_loader.header_row)
        pred_df.columns = pred_df.columns.str.strip()
        
        rename_dict = {v: k for k, v in data_loader.mapping.items() if k != 'volume_columns'}
        pred_df.rename(columns=rename_dict, inplace=True)
        if len(data_loader.mapping['volume_columns']) == 1:
            pred_df.rename(columns={data_loader.mapping['volume_columns'][0]: 'Traffic_Volume'}, inplace=True)
        
        pred_df['Date_Time'] = pd.to_datetime(pred_df['date'], errors='coerce')
        pred_df['hour'] = pred_df['Date_Time'].dt.hour
        pred_df['minute'] = pred_df['Date_Time'].dt.minute
        # Simplified feature engineering for prediction. A robust solution would replicate
        # the full feature engineering from the dataloader.
        for col in X_train_columns:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        pred_features = pred_df[X_train_columns]
        pred_features_scaled = scaler.transform(pred_features)
        
        input_for_model = pred_features_scaled
        if isinstance(active_model, LSTMTrafficPredictor):
            input_for_model = pred_features_scaled.reshape(pred_features_scaled.shape[0], 1, pred_features_scaled.shape[1])
            
        predictions = active_model.predict(input_for_model).flatten()
        
        results = []
        for index, row in pred_df.iterrows():
            results.append({
                "scats_id": row.get('scats_number', 'N/A'),
                "datetime": str(row.get('Date_Time', 'N/A')),
                "prediction": round(float(predictions[index]), 2),
                "lat": row.get('latitude', None),
                "lng": row.get('longitude', None),
                "name": row.get('location', 'N/A')
            })
            
        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)