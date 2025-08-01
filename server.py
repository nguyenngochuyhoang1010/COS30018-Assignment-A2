from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import traceback
import uuid
import heapq # For A* priority queue
import json

# Import your existing classes
from utils.dataloader import TrafficDataLoader
from model.saemodel import StackedAutoencoder
from model.lstm_model import LSTMTrafficPredictor
from model.gru_model import GRUTrafficPredictor
from model.xgboost_model import XGBoostTrafficPredictor


# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json'} # Add json for graph file
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global State ---
data_loader = None
active_model = None
scaler = None
X_train_columns = None
scats_graph = None
graph_connections = None # To store connections from JSON

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Graph and A* Implementation ---

def load_default_graph():
    """Loads the default graph connections from the JSON file."""
    global graph_connections
    try:
        with open('graph_data.json', 'r') as f:
            data = json.load(f)
            # Convert string keys to integers
            graph_connections = {int(k): v for k, v in data['connections'].items()}
        print("Default graph data loaded successfully.")
    except Exception as e:
        print(f"Could not load default graph_data.json: {e}")
        graph_connections = {} # Start with an empty graph if file is missing

def build_scats_graph(df, connections):
    """
    Builds a graph from the SCATS site data using the provided connections.
    """
    graph = {}
    if not connections: return graph

    locations = df[['scats_number', 'latitude', 'longitude']].drop_duplicates('scats_number').set_index('scats_number').to_dict('index')

    for site, neighbors in connections.items():
        if site in locations:
            if site not in graph:
                graph[site] = {'pos': (locations[site]['latitude'], locations[site]['longitude']), 'neighbors': {}}
            for neighbor in neighbors:
                if neighbor in locations:
                    if neighbor not in graph:
                         graph[neighbor] = {'pos': (locations[neighbor]['latitude'], locations[neighbor]['longitude']), 'neighbors': {}}
                    dist = np.sqrt((locations[site]['latitude'] - locations[neighbor]['latitude'])**2 +
                                   (locations[site]['longitude'] - locations[neighbor]['longitude'])**2)
                    graph[site]['neighbors'][neighbor] = dist
                    graph[neighbor]['neighbors'][site] = dist
    return graph

def a_star_search(graph, start_node, end_node, model, scaler, feature_columns):
    """
    A* search algorithm to find the fastest path.
    """
    if not all([start_node in graph, end_node in graph]):
        return None

    def heuristic(node, goal):
        pos1 = graph[node]['pos']
        pos2 = graph[goal]['pos']
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    open_set = [(heuristic(start_node, end_node), 0, start_node, [start_node])]
    closed_set = set()
    g_costs = {node: float('inf') for node in graph}
    g_costs[start_node] = 0

    while open_set:
        _, g_cost, current_node, path = heapq.heappop(open_set)

        if current_node == end_node:
            return [graph[node]['pos'] for node in path]

        if current_node in closed_set:
            continue
        closed_set.add(current_node)

        for neighbor, distance in graph[current_node]['neighbors'].items():
            if neighbor in closed_set:
                continue

            dummy_features = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
            if 'latitude' in dummy_features.columns and 'longitude' in dummy_features.columns:
                dummy_features['latitude'] = graph[neighbor]['pos'][0]
                dummy_features['longitude'] = graph[neighbor]['pos'][1]

            scaled_features = scaler.transform(dummy_features)
            
            input_for_model = scaled_features
            if isinstance(active_model, (LSTMTrafficPredictor, GRUTrafficPredictor)):
                input_for_model = scaled_features.reshape(1, 1, scaled_features.shape[1])

            predicted_volume = model.predict(input_for_model)[0]
            
            traffic_delay_factor = 1 + (predicted_volume / 500) 
            travel_time_cost = distance * traffic_delay_factor
            new_g_cost = g_cost + travel_time_cost

            if new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor, end_node)
                heapq.heappush(open_set, (f_cost, new_g_cost, neighbor, path + [neighbor]))

    return None


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    try:
        with open("index.html") as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Error: index.html not found.", 404

@app.route('/get-headers', methods=['POST'])
def get_headers():
    """Safely reads the header row of an uploaded CSV."""
    try:
        if 'file' not in request.files: return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({"error": "No file was selected."}), 400
        if not allowed_file(file.filename): return jsonify({"error": "Invalid file type. Please upload a CSV."}), 400

        temp_filename = f"temp_{uuid.uuid4().hex}.csv"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        try:
            file.save(temp_filepath)
            header_row = int(request.form.get('header_row', 0))
            headers = pd.read_csv(temp_filepath, header=header_row, nrows=0).columns.str.strip().tolist()
            return jsonify({"headers": headers})
        finally:
            if os.path.exists(temp_filepath): os.remove(temp_filepath)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"A server error occurred: {str(e)}"}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Provides summary statistics for the dashboard."""
    if data_loader is None or data_loader.df_final is None:
        return jsonify({"error": "Data not processed yet."}), 400
    try:
        df = data_loader.df_final
        stats = {
            "totalRecords": f"{len(df):,}",
            "uniqueSites": df['scats_number'].nunique(),
            "dateRange": f"{df['Date_Time'].min().strftime('%Y-%m-%d')} to {df['Date_Time'].max().strftime('%Y-%m-%d')}",
            "trafficByHour": df.groupby(df['Date_Time'].dt.hour)['Traffic_Volume'].mean().round(2).to_dict()
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/scats-locations', methods=['GET'])
def get_scats_locations():
    """Returns a list of all unique SCATS sites with their locations."""
    if data_loader is None or data_loader.df_final is None:
        return jsonify({"error": "Data not processed yet."}), 400
    try:
        locations = data_loader.get_scats_locations()
        return jsonify(locations)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Handles data uploading, mapping, and preprocessing."""
    global data_loader, scaler, X_train_columns, scats_graph
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    try:
        column_mapping = json.loads(request.form.get('column_mapping'))
        header_row = int(request.form.get('header_row', 0))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)
        file.save(filepath)

        data_loader = TrafficDataLoader(filepath, column_mapping, header_row)
        data_loader.load_and_initial_clean()
        
        # FIX: Ensure scats_number is an integer to match the graph keys
        if 'scats_number' in data_loader.df_raw.columns:
            data_loader.df_raw['scats_number'] = pd.to_numeric(data_loader.df_raw['scats_number'], errors='coerce').astype('Int64')
            data_loader.df_raw.dropna(subset=['scats_number'], inplace=True)

        data_loader.process_data()
        X_scaled, y, fitted_scaler = data_loader.prepare_for_model()
        
        if X_scaled is None: raise ValueError("Data preprocessing failed.")
        
        scaler = fitted_scaler
        X_train_columns = X_scaled.columns.tolist()
        
        scats_graph = build_scats_graph(data_loader.df_final, graph_connections)

        return jsonify({"message": "Data preprocessed successfully!", "rows_processed": len(data_loader.df_final)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during preprocessing: {e}"}), 500

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """Trains the selected model."""
    global active_model, data_loader
    if not data_loader or data_loader.df_final is None: return jsonify({"error": "Please preprocess data first."}), 400
    try:
        data = request.get_json()
        model_name = data.get('model', 'lstm')
        epochs = int(data.get('epochs', 50))
        
        X_scaled, y, _ = data_loader.prepare_for_model()
        X_train, X_test, y_train, y_test = data_loader.chronological_train_test_split(X_scaled, y)
        
        X_train_data, y_train_data = X_train.values, y_train.values
        X_test_data, y_test_data = X_test.values, y_test.values

        if model_name == 'lstm':
            model_instance = LSTMTrafficPredictor(input_dim=X_train_data.shape[1])
            X_train_reshaped = X_train_data.reshape(X_train_data.shape[0], 1, X_train_data.shape[1])
            X_test_reshaped = X_test_data.reshape(X_test_data.shape[0], 1, X_test_data.shape[1])
            model_instance.train(X_train_reshaped, y_train_data, epochs=epochs, validation_data=(X_test_reshaped, y_test_data))
        elif model_name == 'gru':
            model_instance = GRUTrafficPredictor(input_dim=X_train_data.shape[1])
            X_train_reshaped = X_train_data.reshape(X_train_data.shape[0], 1, X_train_data.shape[1])
            X_test_reshaped = X_test_data.reshape(X_test_data.shape[0], 1, X_test_data.shape[1])
            model_instance.train(X_train_reshaped, y_train_data, epochs=epochs, validation_data=(X_test_reshaped, y_test_data))
        elif model_name == 'sae':
            model_instance = StackedAutoencoder(input_dim=X_train_data.shape[1], encoding_dims=[128, 64, 32])
            model_instance.train_full_model(X_train_data, y_train_data, epochs=epochs, validation_data=(X_test_data, y_test_data))
        elif model_name == 'xgboost':
            model_instance = XGBoostTrafficPredictor()
            model_instance.train(X_train_data, y_train_data, eval_set=[(X_test_data, y_test_data)])
        else:
            return jsonify({"error": "Invalid model type specified"}), 400
            
        active_model = model_instance
        history = active_model.history.history if hasattr(active_model.history, 'history') else active_model.history
        return jsonify({"message": f"{model_name.upper()} model trained.", "history": history})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during training: {e}"}), 500

@app.route('/api/find-route', methods=['POST'])
def find_route():
    """Finds the fastest route between two SCATS sites using A*."""
    if not all([scats_graph, active_model, scaler, X_train_columns]):
        return jsonify({"error": "Graph not built or model not trained. Please process data and train a model."}), 400
    
    data = request.get_json()
    start_node = int(data.get('start_node'))
    end_node = int(data.get('end_node'))

    if start_node == end_node:
        return jsonify({"error": "Start and end points cannot be the same."}), 400

    try:
        path_coords = a_star_search(scats_graph, start_node, end_node, active_model, scaler, X_train_columns)
        if path_coords:
            return jsonify({"path": path_coords})
        else:
            return jsonify({"error": "No path found between the selected points. The graph may be disconnected."}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during route finding: {str(e)}"}), 500

@app.route('/predict-csv', methods=['POST'])
def predict_batch():
    """Makes predictions on a batch of data from an uploaded CSV."""
    if not all([active_model, scaler, X_train_columns, data_loader]):
        return jsonify({"error": "Please preprocess data and train a model first."}), 400
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    try:
        pred_df = pd.read_csv(file, header=data_loader.header_row)
        pred_df.columns = pred_df.columns.str.strip()
        
        rename_dict = {v: k for k, v in data_loader.mapping.items() if k != 'volume_columns'}
        pred_df.rename(columns=rename_dict, inplace=True)
        if len(data_loader.mapping['volume_columns']) == 1:
            pred_df.rename(columns={data_loader.mapping['volume_columns'][0]: 'Traffic_Volume'}, inplace=True)
        
        pred_df['Date_Time'] = pd.to_datetime(pred_df['date'], errors='coerce')
        df_features = data_loader.engineer_features(pred_df.copy(), is_prediction=True)

        for col in X_train_columns:
            if col not in df_features.columns: df_features[col] = 0
        
        pred_features_scaled = scaler.transform(df_features[X_train_columns])
        
        input_for_model = pred_features_scaled
        if isinstance(active_model, (LSTMTrafficPredictor, GRUTrafficPredictor)):
            input_for_model = pred_features_scaled.reshape(pred_features_scaled.shape[0], 1, pred_features_scaled.shape[1])
            
        predictions = active_model.predict(input_for_model).flatten()
        
        df_features['prediction'] = predictions
        
        results = []
        for index, row in df_features.iterrows():
            results.append({
                "scats_id": row.get('scats_number', 'N/A'),
                "datetime": str(row.get('Date_Time', 'N/A')),
                "prediction": round(float(row['prediction']), 2),
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
    load_default_graph() # Load the graph when the server starts
    app.run(debug=True, port=5000)