import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import threading # For running training in background
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import webbrowser # NEW: For opening HTML map in browser
import folium     # NEW: For creating interactive maps

# Adjust these imports based on your file structure
from utils.dataloader import TrafficDataLoader
from model.saemodel import StackedAutoencoder
from model.lstm_model import LSTMTrafficPredictor

class TFPSApp:
    """
    Traffic Flow Prediction System Application.
    This class orchestrates the GUI and interacts with data loading,
    model training, and prediction functionalities.
    """
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System (TFPS)")
        master.geometry("800x700") # Set initial window size

        # Initialize data_loader with the base filename.
        self.data_loader = TrafficDataLoader('Scats Data October 2006.csv')
        
        # Initialize both model instances. We will select which one to use later.
        # input_dim is 18 based on your dataloader.py feature_columns
        self.sae_model = StackedAutoencoder(input_dim=18, encoding_dims=[128, 64, 32])
        self.lstm_model = LSTMTrafficPredictor(input_dim=18, units=50) # Default LSTM units
        
        self.active_model = None # This will hold the currently selected and trained model
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None # To store the scaler from data_loader
        
        self.create_widgets()

        # NEW: Handle window closing gracefully
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Tab 1: Data Loading & Preprocessing
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="1. Data Loading & Preprocessing")
        self.create_data_widgets(self.data_tab)

        # Tab 2: Model Training
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="2. Model Training")
        self.create_model_widgets(self.model_tab)

        # Tab 3: Traffic Prediction & Route Guidance
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="3. Traffic Prediction & Route Guidance")
        self.create_prediction_widgets(self.prediction_tab)

        # Tab 4: Route Guidance (Future - Placeholder for now)
        self.future_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.future_tab, text="4. Route Guidance (Future)")
        self.create_future_widgets(self.future_tab)
        # NEW: Add a "Visualize Map" button to the Future tab
        self.visualize_map_button = ttk.Button(self.future_tab, text="Visualize SCATS Sites on Map", command=self.create_and_show_map)
        self.visualize_map_button.pack(padx=20, pady=20)


    def create_data_widgets(self, tab):
        # Frame for CSV File selection
        csv_frame = ttk.LabelFrame(tab, text="Data Loading & Preprocessing")
        csv_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(csv_frame, text="CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.csv_file_entry = ttk.Entry(csv_frame, width=50)
        self.csv_file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # Display the expected path, which is constructed by the dataloader
        self.csv_file_entry.insert(0, self.data_loader.csv_file_path)
        self.csv_file_entry.config(state='readonly') # Make it readonly as it's hardcoded to 'data/filename'

        # Load & Process Data button
        self.load_button = ttk.Button(csv_frame, text="Load & Process Data", command=self.load_and_process_data)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        self.data_status_label = ttk.Label(csv_frame, text="Status: Ready to load data")
        self.data_status_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # Configuration for column weights
        csv_frame.grid_columnconfigure(1, weight=1)

    def update_status(self, message, label_type="data"):
        """Helper to update status labels and ensure GUI updates."""
        if label_type == "data":
            self.data_status_label.config(text=message)
        elif label_type == "model":
            self.model_status_label.config(text=message)
        elif label_type == "prediction":
            # Add prediction status label if needed
            pass
        self.master.update_idletasks() # Force GUI update

    def load_and_process_data(self):
        self.update_status("Status: Loading and preprocessing data...", "data")
        
        try:
            # Step 1: Load and initial clean
            self.data_loader.load_and_initial_clean()

            # Step 2: Transform to long format
            self.data_loader.transform_to_long_format()

            # Step 3: Engineer features
            self.data_loader.engineer_features()

            # Step 4: Prepare for model (scaling and train-test split)
            X_scaled, y, scaler = self.data_loader.prepare_for_model()
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.chronological_train_test_split(X_scaled, y)
            self.scaler = scaler # Store the scaler for later inverse transformation

            self.update_status("Status: Data loaded and preprocessed successfully!", "data")
            messagebox.showinfo("Data Load Success", "Traffic data loaded and preprocessed successfully!")

            # After data is loaded, populate SCATS site combo for prediction
            if self.data_loader.df_final is not None:
                scats_numbers = sorted(self.data_loader.df_final['SCATS Number'].unique().tolist())
                self.scats_site_combo['values'] = scats_numbers
                if scats_numbers:
                    self.scats_site_combo.set(scats_numbers[0]) # Set default to first SCATS number

        except FileNotFoundError:
            messagebox.showerror("File Error", f"CSV file not found at: '{self.data_loader.csv_file_path}'. Please ensure it's in the correct 'data/' subdirectory relative to where the application is launched.")
            self.update_status("Status: Error loading data (File not found)", "data")
        except pd.errors.EmptyDataError:
            messagebox.showerror("Data Error", "CSV file is empty or malformed.")
            self.update_status("Status: Error loading data (Empty CSV)", "data")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during data loading: {e}")
            self.update_status("Status: Error loading data", "data")
            print(f"Error during data loading: {e}") # Print to console for debugging

    def create_model_widgets(self, tab):
        model_frame = ttk.LabelFrame(tab, text="Model Training")
        model_frame.pack(padx=10, pady=10, fill="x")

        # Model Selection
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_selection = ttk.Combobox(model_frame, values=["Stacked Autoencoder", "LSTM"], state="readonly")
        self.model_selection.set("Stacked Autoencoder") # Default selection
        self.model_selection.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(model_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(model_frame, width=10)
        self.epochs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.epochs_entry.insert(0, "50") # Default epochs

        self.train_button = ttk.Button(model_frame, text="Train Model", command=self.start_model_training)
        self.train_button.grid(row=1, column=2, padx=5, pady=5) # Adjusted row

        self.model_status_label = ttk.Label(model_frame, text="Status: Model not trained.")
        self.model_status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w") # Adjusted row

        # Placeholder for training progress bar (optional)
        self.progress_bar = ttk.Progressbar(model_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew") # Adjusted row

        # Placeholder for training plot (e.g., loss curve)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss (MSE)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=model_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew") # Adjusted row

        model_frame.grid_columnconfigure(1, weight=1)
        model_frame.grid_rowconfigure(4, weight=1) # Allow plot to expand

    def start_model_training(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Training Error", "Please load and preprocess data first.")
            return

        try:
            epochs = int(self.epochs_entry.get())
            if epochs <= 0:
                raise ValueError("Epochs must be a positive integer.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid epochs value: {e}")
            return

        selected_model_name = self.model_selection.get()
        self.update_status(f"Status: Training {selected_model_name} (This may take a while)...", "model")
        self.train_button.config(state=tk.DISABLED) # Disable button during training
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()

        # NEW: Make the thread a daemon thread
        training_thread = threading.Thread(target=self.train_model_thread, args=(epochs, selected_model_name), daemon=True)
        training_thread.start()

    def train_model_thread(self, epochs, model_name):
        # Define a common batch size for both models, adjust as needed for performance/memory
        current_batch_size = 64 # Changed from 32 to 64 for potential speedup

        try:
            X_train_data = self.X_train.values # Convert DataFrame to NumPy array
            y_train_data = self.y_train.values # Convert Series to NumPy array

            if model_name == "Stacked Autoencoder":
                self.active_model = self.sae_model # Set SAE as active model
                # Pre-train autoencoders
                self.sae_model.pretrain_autoencoders(X_train_data, epochs=epochs // 2, batch_size=current_batch_size)
                # Train the full model
                history = self.sae_model.train_full_model(X_train_data, y_train_data, epochs=epochs, batch_size=current_batch_size)
            elif model_name == "LSTM":
                self.active_model = self.lstm_model # Set LSTM as active model
                # Reshape data for LSTM: (samples, timesteps, features)
                # Current X_train_data is (samples, 18), so reshape to (samples, 1, 18)
                X_train_reshaped = X_train_data.reshape(X_train_data.shape[0], self.lstm_model.timesteps, X_train_data.shape[1])
                history = self.lstm_model.train(X_train_reshaped, y_train_data, epochs=epochs, batch_size=current_batch_size)
            else:
                raise ValueError("Unknown model selected.")

            # Plot training history (assuming history object is returned by train/train_full_model)
            self.ax.clear()
            self.ax.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                self.ax.plot(history.history['val_loss'], label='Validation Loss')
            self.ax.set_title(f"{model_name} Training Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss (MSE)")
            self.ax.legend()
            self.canvas.draw_idle()

            self.master.after(0, self.update_model_training_status, f"Status: {model_name} trained successfully!", True)

        except Exception as e:
            self.master.after(0, self.update_model_training_status, f"Status: Error during training: {e}", False)
            print(f"Error during model training: {e}")

    def update_model_training_status(self, message, success):
        self.update_status(message, "model")
        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate", value=0)
        self.train_button.config(state=tk.NORMAL) # Re-enable button
        if success:
            messagebox.showinfo("Model Training", message)
        else:
            messagebox.showerror("Model Training Error", message)

    def on_closing(self):
        """
        Handles the window close event to ensure proper application termination.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy() # Destroys the main Tkinter window
            # Daemon threads will terminate automatically when main thread exits.

    def create_prediction_widgets(self, tab):
        prediction_frame = ttk.LabelFrame(tab, text="Traffic Prediction")
        prediction_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(prediction_frame, text="SCATS Site:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # Changed to Combobox for SCATS site selection (populated after data load)
        self.scats_site_combo = ttk.Combobox(prediction_frame, width=15, state="readonly")
        self.scats_site_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(prediction_frame, text="Date (MM/DD/YYYY):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.date_entry = ttk.Entry(prediction_frame, width=15)
        self.date_entry.insert(0, "10/01/2006") # Example date
        self.date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(prediction_frame, text="Time (HH:MM):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.time_entry = ttk.Entry(prediction_frame, width=15)
        self.time_entry.insert(0, "08:00") # Example time
        self.time_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.predict_button = ttk.Button(prediction_frame, text="Predict Traffic", command=self.predict_traffic)
        self.predict_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="ns")

        self.prediction_result_label = ttk.Label(prediction_frame, text="Predicted Traffic Volume: N/A")
        self.prediction_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        prediction_frame.grid_columnconfigure(1, weight=1)

    def predict_traffic(self):
        if self.active_model is None: # Use active_model
            messagebox.showwarning("Prediction Error", "Please train a model first.")
            return
        if self.X_train is None or self.y_train is None or self.scaler is None: # Use X_train for column names
            messagebox.showwarning("Prediction Error", "Data not loaded or processed correctly.")
            return

        scats_number = self.scats_site_combo.get()
        date_str = self.date_entry.get()
        time_str = self.time_entry.get()

        if not scats_number or not date_str or not time_str:
            messagebox.showerror("Input Error", "Please fill in all prediction fields.")
            return

        try:
            scats_number = int(scats_number)
            prediction_datetime = pd.to_datetime(f"{date_str} {time_str}", format="%m/%d/%Y %H:%M")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid date or time format: {e}. Use MM/DD/YYYY and HH:MM.")
            return

        try:
            # Create a dummy DataFrame for the single prediction point
            predict_df = pd.DataFrame([{
                'SCATS Number': scats_number,
                'Date_Time': prediction_datetime,
                'hour': prediction_datetime.hour,
                'minute': prediction_datetime.minute,
                'day_of_week': prediction_datetime.dayofweek,
                'day_of_year': prediction_datetime.dayofyear,
                'week_of_year': prediction_datetime.isocalendar().week.astype(int),
                'month': prediction_datetime.month,
                'year': prediction_datetime.year,
                'is_weekend': (prediction_datetime.dayofweek >= 5).astype(int)
            }])

            # Populate lag features by looking up historical data
            historical_site_data = self.data_loader.df_final[
                self.data_loader.df_final['SCATS Number'] == scats_number
            ].set_index('Date_Time').sort_index()

            def get_historical_volume(dt):
                if dt in historical_site_data.index:
                    return historical_site_data.loc[dt, 'Traffic_Volume']
                return np.nan

            predict_df['traffic_volume_lag_1'] = get_historical_volume(prediction_datetime - pd.Timedelta(minutes=15))
            predict_df['traffic_volume_lag_4'] = get_historical_volume(prediction_datetime - pd.Timedelta(hours=1))
            predict_df['traffic_volume_lag_96'] = get_historical_volume(prediction_datetime - pd.Timedelta(hours=24))

            last_known_rolling_mean = historical_site_data['traffic_volume_rolling_mean_4'].iloc[-1] if not historical_site_data.empty else 0
            predict_df['traffic_volume_rolling_mean_4'] = last_known_rolling_mean

            predict_features_df = predict_df.drop(columns=['SCATS Number', 'Date_Time'])

            # Ensure the order of columns matches the training data (X_train)
            model_feature_columns = self.X_train.columns.tolist()
            predict_input = predict_features_df[model_feature_columns]

            if predict_input.isnull().any().any():
                messagebox.showwarning("Prediction Warning", "Missing historical data for lag features. Prediction might be inaccurate.")
                predict_input = predict_input.fillna(0)

            # Scale the input features using the *fitted* scaler
            predict_input_scaled = self.scaler.transform(predict_input)

            # Reshape for LSTM if the active model is LSTM
            if isinstance(self.active_model, LSTMTrafficPredictor):
                predict_input_scaled = predict_input_scaled.reshape(predict_input_scaled.shape[0], self.active_model.timesteps, predict_input_scaled.shape[1])
            
            # Make prediction using the active model
            predicted_scaled_volume = self.active_model.predict(predict_input_scaled)

            predicted_volume = predicted_scaled_volume[0][0] # Assuming single output regression

            self.prediction_result_label.config(text=f"Predicted Traffic Volume: {predicted_volume:.2f} vehicles")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.prediction_result_label.config(text="Predicted Traffic Volume: Error")
            print(f"Error during prediction: {e}")


    def create_future_widgets(self, tab):
        ttk.Label(tab, text="This tab is for future route guidance functionalities.").pack(padx=20, pady=20)
        # Add widgets for future development here

    # NEW METHOD: For creating and showing the map
    def create_and_show_map(self):
        if self.data_loader.df_final is None:
            messagebox.showwarning("Map Error", "Please load and process data first to visualize SCATS sites.")
            return

        map_file_path = "traffic_map.html"
        
        try:
            # Get unique SCATS sites with their latest known coordinates
            # This extracts the latest latitude and longitude for each unique SCATS site
            scats_locations = self.data_loader.df_final.groupby('SCATS Number').agg({
                'NB_LATITUDE': 'last',
                'NB_LONGITUDE': 'last',
                'Location': 'first' # Get any location name for display
            }).reset_index()

            if scats_locations.empty:
                messagebox.showwarning("Map Error", "No SCATS site data available for mapping.")
                return

            # Determine map center (average of all latitudes/longitudes)
            map_center = [scats_locations['NB_LATITUDE'].mean(), scats_locations['NB_LONGITUDE'].mean()]
            
            # Create a Folium map object
            m = folium.Map(location=map_center, zoom_start=12)

            # Add markers for each SCATS site
            for idx, row in scats_locations.iterrows():
                folium.Marker(
                    location=[row['NB_LATITUDE'], row['NB_LONGITUDE']],
                    popup=f"SCATS ID: {row['SCATS Number']}<br>Location: {row['Location']}",
                    tooltip=f"SCATS ID: {row['SCATS Number']}"
                ).add_to(m)

            # Save the map to an HTML file
            m.save(map_file_path)
            
            # Open the HTML file in the default web browser
            webbrowser.open(map_file_path)
            messagebox.showinfo("Map Generated", f"Interactive map generated and opened in browser: {map_file_path}")

        except Exception as e:
            messagebox.showerror("Map Generation Error", f"An error occurred while generating the map: {e}")
            print(f"Map generation error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TFPSApp(root)
    root.mainloop()