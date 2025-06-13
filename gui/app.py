import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import webbrowser
import folium

# Adjust these imports based on your file structure
from utils.dataloader import TrafficDataLoader
from model.saemodel import StackedAutoencoder
from model.lstm_model import LSTMTrafficPredictor

class TFPSApp:
    """
    Traffic Flow Prediction System Application.
    """
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System (TFPS)")
        master.geometry("800x700")

        self.data_loader = TrafficDataLoader('Scats Data October 2006.csv')
        
        self.sae_model = StackedAutoencoder(input_dim=18, encoding_dims=[128, 64, 32])
        self.lstm_model = LSTMTrafficPredictor(input_dim=18)
        
        self.active_model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None
        
        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="1. Data Loading & Preprocessing")
        self.create_data_widgets(self.data_tab)

        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="2. Model Training")
        self.create_model_widgets(self.model_tab)

        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="3. Traffic Prediction")
        self.create_prediction_widgets(self.prediction_tab)

        self.future_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.future_tab, text="4. Route Guidance (Future)")
        self.create_future_widgets(self.future_tab)

    def create_data_widgets(self, tab):
        csv_frame = ttk.LabelFrame(tab, text="Data Loading")
        csv_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(csv_frame, text="CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.csv_file_entry = ttk.Entry(csv_frame, width=50)
        self.csv_file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.csv_file_entry.insert(0, self.data_loader.csv_file_path)
        self.csv_file_entry.config(state='readonly')

        self.load_button = ttk.Button(csv_frame, text="Load & Process Data", command=self.load_and_process_data)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        self.data_status_label = ttk.Label(csv_frame, text="Status: Ready to load data")
        self.data_status_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        csv_frame.grid_columnconfigure(1, weight=1)

    def update_status(self, message, label_widget):
        label_widget.config(text=f"Status: {message}")
        self.master.update_idletasks()

    def load_and_process_data(self):
        self.update_status("Loading and preprocessing data...", self.data_status_label)
        try:
            self.data_loader.load_and_initial_clean()
            self.data_loader.transform_to_long_format()
            self.data_loader.engineer_features()
            X_scaled, y, scaler = self.data_loader.prepare_for_model()
            
            if X_scaled is None:
                raise ValueError("Data preparation failed, returned None.")

            self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.chronological_train_test_split(X_scaled, y)
            self.scaler = scaler

            self.update_status("Data loaded and preprocessed successfully!", self.data_status_label)
            messagebox.showinfo("Success", "Traffic data loaded and preprocessed successfully!")

            if self.data_loader.df_final is not None:
                scats_numbers = sorted(self.data_loader.df_final['SCATS Number'].unique().tolist())
                self.scats_site_combo['values'] = scats_numbers
                if scats_numbers:
                    self.scats_site_combo.set(scats_numbers[0])

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during data loading: {e}")
            self.update_status(f"Error loading data: {e}", self.data_status_label)
            print(f"Error during data loading: {e}")

    def create_model_widgets(self, tab):
        model_frame = ttk.LabelFrame(tab, text="Model Training")
        model_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_selection = ttk.Combobox(model_frame, values=["Stacked Autoencoder", "LSTM"], state="readonly")
        self.model_selection.set("Stacked Autoencoder")
        self.model_selection.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(model_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(model_frame, width=10)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.train_button = ttk.Button(model_frame, text="Train Model", command=self.start_model_training)
        self.train_button.grid(row=1, column=2, padx=5, pady=5)

        self.model_status_label = ttk.Label(model_frame, text="Status: Model not trained.")
        self.model_status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=model_frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=3, sticky="nsew")
        model_frame.grid_columnconfigure(1, weight=1)
        model_frame.grid_rowconfigure(4, weight=1)

    def start_model_training(self):
        if self.X_train is None:
            messagebox.showwarning("Training Error", "Please load data first.")
            return

        try:
            epochs = int(self.epochs_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Epochs must be an integer.")
            return

        model_name = self.model_selection.get()
        self.update_status(f"Training {model_name}...", self.model_status_label)
        self.train_button.config(state=tk.DISABLED)

        threading.Thread(target=self.train_model_thread, args=(epochs, model_name), daemon=True).start()

    def train_model_thread(self, epochs, model_name):
        try:
            X_train_data = self.X_train.values
            y_train_data = self.y_train.values
            history = None

            if model_name == "Stacked Autoencoder":
                self.active_model = self.sae_model
                # --- FIX: Changed method call from .train() to .train_full_model() ---
                history = self.active_model.train_full_model(X_train_data, y_train_data, epochs=epochs, batch_size=64)
            elif model_name == "LSTM":
                self.active_model = self.lstm_model
                X_train_reshaped = X_train_data.reshape(X_train_data.shape[0], 1, X_train_data.shape[1])
                history = self.active_model.train(X_train_reshaped, y_train_data, epochs=epochs, batch_size=64)
            
            self.master.after(0, self.update_after_training, model_name, history, True)
        except Exception as e:
            self.master.after(0, self.update_after_training, model_name, None, False, str(e))

    def update_after_training(self, model_name, history, success, error_msg=None):
        self.train_button.config(state=tk.NORMAL)
        if success:
            self.update_status(f"{model_name} trained successfully!", self.model_status_label)
            messagebox.showinfo("Training Complete", f"{model_name} has been trained.")
            self.ax.clear()
            self.ax.plot(history.history['loss'], label='Training Loss')
            self.ax.set_title(f"{model_name} Training Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss (MSE)")
            self.ax.legend()
            self.canvas.draw()
        else:
            self.update_status(f"Error training {model_name}: {error_msg}", self.model_status_label)
            messagebox.showerror("Training Error", f"An error occurred: {error_msg}")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy()

    def create_prediction_widgets(self, tab):
        prediction_frame = ttk.LabelFrame(tab, text="Traffic Prediction")
        prediction_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(prediction_frame, text="SCATS Site:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scats_site_combo = ttk.Combobox(prediction_frame, width=15, state="readonly")
        self.scats_site_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(prediction_frame, text="Date (MM/DD/YYYY):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.date_entry = ttk.Entry(prediction_frame, width=15)
        self.date_entry.insert(0, "10/01/2006")
        self.date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(prediction_frame, text="Time (HH:MM):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.time_entry = ttk.Entry(prediction_frame, width=15)
        self.time_entry.insert(0, "08:00")
        self.time_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.predict_button = ttk.Button(prediction_frame, text="Predict Traffic", command=self.predict_traffic)
        self.predict_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="ns")

        self.prediction_result_label = ttk.Label(prediction_frame, text="Predicted Traffic Volume: N/A", font=("Helvetica", 12))
        self.prediction_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        prediction_frame.grid_columnconfigure(1, weight=1)

    def predict_traffic(self):
        if self.active_model is None or self.data_loader is None or self.X_train is None:
            messagebox.showerror("Error", "Model not trained or data not loaded.")
            return
        
        try:
            scats_site = int(self.scats_site_combo.get())
            date_str = self.date_entry.get()
            time_str = self.time_entry.get()
            
            # 1. Find the most recent data point for the selected site to use as a template
            site_df = self.data_loader.df_final[self.data_loader.df_final['SCATS Number'] == scats_site]
            if site_df.empty:
                raise ValueError(f"No historical data available for SCATS site {scats_site}")
            
            template_row = site_df.iloc[[-1]].copy()

            # 2. Update time-based features from user input
            prediction_datetime = pd.to_datetime(f"{date_str} {time_str}", format="%m/%d/%Y %H:%M")
            template_row['hour'] = prediction_datetime.hour
            template_row['minute'] = prediction_datetime.minute
            template_row['day_of_week'] = prediction_datetime.dayofweek
            template_row['day_of_year'] = prediction_datetime.dayofyear
            template_row['week_of_year'] = prediction_datetime.isocalendar().week
            template_row['month'] = prediction_datetime.month
            template_row['year'] = prediction_datetime.year
            template_row['is_weekend'] = (template_row['day_of_week'] >= 5).astype(int)
            
            # Note: Lagged features in the template_row are used as an approximation.
            # A more advanced implementation would re-calculate them based on the prediction time.

            # 3. Ensure the feature columns match the training columns
            feature_columns = self.X_train.columns
            prediction_input_df = template_row[feature_columns]

            # 4. Scale the data using the *fitted* scaler
            scaled_input = self.scaler.transform(prediction_input_df)
            
            # 5. Reshape for LSTM model if necessary
            if isinstance(self.active_model, LSTMTrafficPredictor):
                scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
            
            # 6. Predict
            predicted_volume = self.active_model.predict(scaled_input)
            
            # The output might be nested, so we extract the single float value
            final_prediction = predicted_volume[0][0] if isinstance(predicted_volume[0], list) or isinstance(predicted_volume[0], np.ndarray) else predicted_volume[0]

            self.prediction_result_label.config(text=f"Predicted Traffic Volume: {final_prediction:.2f}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f'An error occurred during prediction: {e}')
            self.prediction_result_label.config(text="Predicted Traffic Volume: Error")
            print(f"Prediction Error Trace: {e}")

    def create_future_widgets(self, tab):
        ttk.Label(tab, text="Route guidance functionality will be implemented in a future version.").pack(pady=20, padx=20)
        self.visualize_map_button = ttk.Button(tab, text="Visualize SCATS Sites on Map", command=self.create_and_show_map)
        self.visualize_map_button.pack(padx=20, pady=10)

    def create_and_show_map(self):
        if self.data_loader.df_final is None:
            messagebox.showwarning("Map Error", "Please load data first.")
            return

        map_file_path = "traffic_map.html"
        try:
            scats_locations = self.data_loader.df_final.groupby('SCATS Number').agg({
                'NB_LATITUDE': 'last', 'NB_LONGITUDE': 'last', 'Location': 'first'
            }).dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE']).reset_index()

            if scats_locations.empty:
                messagebox.showwarning("Map Error", "No location data available.")
                return

            map_center = [scats_locations['NB_LATITUDE'].mean(), scats_locations['NB_LONGITUDE'].mean()]
            m = folium.Map(location=map_center, zoom_start=12)

            for _, row in scats_locations.iterrows():
                folium.Marker(
                    location=[row['NB_LATITUDE'], row['NB_LONGITUDE']],
                    popup=f"SCATS ID: {row['SCATS Number']}<br>Location: {row['Location']}",
                    tooltip=f"SCATS ID: {row['SCATS Number']}"
                ).add_to(m)

            m.save(map_file_path)
            webbrowser.open(f"file://{os.path.realpath(map_file_path)}")
            messagebox.showinfo("Map Generated", f"Map opened in browser: {map_file_path}")
        except Exception as e:
            messagebox.showerror("Map Generation Error", f"Failed to generate map: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TFPSApp(root)
    root.mainloop()