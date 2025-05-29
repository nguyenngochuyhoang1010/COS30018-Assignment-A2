import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import threading # For running training in background
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Adjust these imports based on your file structure
from utils.dataloader import TrafficDataLoader
from model.saemodel import StackedAutoencoder


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
        # The data_loader itself will prepend 'data/' to this filename.
        self.data_loader = TrafficDataLoader('Scats Data October 2006.csv')
        
        # Initialize sae_model with the correct input_dim and encoding_dims
        # input_dim is 18 based on your dataloader.py feature_columns
        self.sae_model = StackedAutoencoder(input_dim=18, encoding_dims=[128, 64, 32])
        
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None # To store the scaler from data_loader
        self.trained_model = None # To store the trained Keras model

        self.create_widgets()

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

        ttk.Label(model_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(model_frame, width=10)
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.epochs_entry.insert(0, "50") # Default epochs

        self.train_button = ttk.Button(model_frame, text="Train Model", command=self.start_model_training)
        self.train_button.grid(row=0, column=2, padx=5, pady=5)

        self.model_status_label = ttk.Label(model_frame, text="Status: Model not trained.")
        self.model_status_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # Placeholder for training progress bar (optional)
        self.progress_bar = ttk.Progressbar(model_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Placeholder for training plot (e.g., loss curve)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss (MSE)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=model_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        model_frame.grid_columnconfigure(1, weight=1)
        model_frame.grid_rowconfigure(3, weight=1) # Allow plot to expand

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

        self.update_status("Status: Training model (This may take a while)...", "model")
        self.train_button.config(state=tk.DISABLED) # Disable button during training
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()

        # Run training in a separate thread to keep GUI responsive
        training_thread = threading.Thread(target=self.train_model_thread, args=(epochs,))
        training_thread.start()

    def train_model_thread(self, epochs):
        try:
            # Step 1: Pre-train autoencoders
            # The X_train here is the scaled features from the DataLoader
            self.sae_model.pretrain_autoencoders(self.X_train.values, epochs=epochs, batch_size=32)

            # Step 2: Train the full model
            # Use X_train.values and y_train.values for numpy array input
            history = self.sae_model.train_full_model(self.X_train.values, self.y_train.values, epochs=epochs, batch_size=32)
            self.trained_model = self.sae_model.full_model # Store the trained Keras model instance

            # Plot training history (assuming history object is returned by train_full_model)
            self.ax.clear()
            self.ax.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                self.ax.plot(history.history['val_loss'], label='Validation Loss')
            self.ax.set_title("Training Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss (MSE)")
            self.ax.legend()
            self.canvas.draw_idle()


            self.master.after(0, self.update_model_training_status, "Status: Model trained successfully!", True)

        except Exception as e:
            self.master.after(0, self.update_model_training_status, f"Status: Error during training: {e}", False)
            print(f"Error during model training: {e}")

    def update_model_training_status(self, message, success):
        self.update_status(message, "model")
        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate", value=0)
        self.train_button.config(state=tk.NORMAL) # Re-enable button
        if success:
            messagebox.showinfo("Model Training", "Model trained successfully!")
        else:
            messagebox.showerror("Model Training Error", message)

    def create_prediction_widgets(self, tab):
        prediction_frame = ttk.LabelFrame(tab, text="Traffic Prediction")
        prediction_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(prediction_frame, text="SCATS Site:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scats_site_entry = ttk.Entry(prediction_frame, width=20)
        self.scats_site_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # Example default value if available, or fetch from loaded data
        # self.scats_site_entry.insert(0, "SCATS Number (e.g., 249)")

        ttk.Label(prediction_frame, text="Date (MM/DD/YYYY):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.date_entry = ttk.Entry(prediction_frame, width=20)
        self.date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.date_entry.insert(0, "10/01/2006") # Example default date

        ttk.Label(prediction_frame, text="Time (HH:MM):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.time_entry = ttk.Entry(prediction_frame, width=20)
        self.time_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.time_entry.insert(0, "08:00") # Example default time

        self.predict_button = ttk.Button(prediction_frame, text="Predict Traffic", command=self.predict_traffic)
        self.predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        ttk.Label(prediction_frame, text="Predicted Traffic Volume:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.predicted_volume_label = ttk.Label(prediction_frame, text="N/A")
        self.predicted_volume_label.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        prediction_frame.grid_columnconfigure(1, weight=1)

    def predict_traffic(self):
        if self.trained_model is None:
            messagebox.showwarning("Prediction Error", "Please train the model first.")
            return
        if self.X_test is None or self.y_test is None:
            messagebox.showwarning("Prediction Error", "Data not loaded/processed. Cannot make predictions.")
            return

        scats_site_input = self.scats_site_entry.get()
        date_input = self.date_entry.get()
        time_input = self.time_entry.get()

        try:
            # Combine date and time
            prediction_datetime_str = f"{date_input} {time_input}"
            prediction_dt = pd.to_datetime(prediction_datetime_str, format="%m/%d/%Y %H:%M", errors='raise')

            # Filter relevant data for the specific SCATS site to get its latest features
            # This is a simplified approach. A more robust system would interpolate or find closest known data.
            target_scats_data = self.data_loader.df_final[
                (self.data_loader.df_final['SCATS Number'] == int(scats_site_input)) &
                (self.data_loader.df_final['Date_Time'] < prediction_dt)
            ].sort_values(by='Date_Time', ascending=False)

            if target_scats_data.empty:
                messagebox.showerror("Prediction Error", f"No historical data found for SCATS site {scats_site_input} before the specified time.")
                return

            # Take the most recent entry for the site to get its base features
            # This assumes that other features (like lat/long, etc.) are static for a site
            base_features_row = target_scats_data[self.X_train.columns.tolist()].iloc[0].copy()

            # Update time-based features to match the prediction_dt
            base_features_row['hour'] = prediction_dt.hour
            base_features_row['minute'] = prediction_dt.minute
            base_features_row['day_of_week'] = prediction_dt.dayofweek
            base_features_row['day_of_year'] = prediction_dt.dayofyear
            base_features_row['week_of_year'] = prediction_dt.isocalendar().week.astype(int)
            base_features_row['month'] = prediction_dt.month
            base_features_row['year'] = prediction_dt.year
            base_features_row['is_weekend'] = (prediction_dt.dayofweek >= 5).astype(int)

            # For lagged features and rolling mean, this simple approach cannot accurately calculate them
            # for a future arbitrary timestamp without a more sophisticated time-series imputation/forecasting
            # of those features themselves. For this example, we'll use the lagged values from the
            # most recent known point, which is a simplification.
            # A truly robust solution would require forecasting these lagged features or
            # having enough sequential data leading up to the prediction_dt.

            # Rescale the single prediction input using the *same scaler* used for training
            prediction_input_scaled = self.scaler.transform(pd.DataFrame([base_features_row]))

            # Make prediction
            predicted_scaled_volume = self.trained_model.predict(prediction_input_scaled)

            predicted_volume = predicted_scaled_volume[0][0] # Assuming single output regression

            self.predicted_volume_label.config(text=f"{predicted_volume:.2f} vehicles")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input format: {e}. Please check SCATS Site (integer), Date (MM/DD/YYYY), and Time (HH:MM).")
        except KeyError as e:
            messagebox.showerror("Data Error", f"Missing expected column for prediction: {e}. Ensure all features used for training are available.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred during prediction: {e}")
            print(f"Prediction Error: {e}") # Print to console for debugging


    def create_future_widgets(self, tab):
        ttk.Label(tab, text="This tab is for future route guidance functionalities.").pack(padx=20, pady=20)
        # Add widgets for future development here

if __name__ == "__main__":
    root = tk.Tk()
    app = TFPSApp(root)
    root.mainloop()