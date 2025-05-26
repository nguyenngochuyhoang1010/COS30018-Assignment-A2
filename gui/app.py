import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os # For checking file existence

# Assuming these modules will be in the parent directory or correctly in PYTHONPATH
# from utils.dataloader import TrafficDataLoader
# from models.saemodel import StackedAutoencoder

class TFPSApp:
    """
    Traffic Flow Prediction System Graphical User Interface (GUI).
    Allows users to input parameters, trigger predictions, and potentially visualize results.
    """
    def __init__(self, master):
        """
        Initializes the TFPS GUI application.

        Args:
            master (tk.Tk): The root Tkinter window.
        """
        self.master = master
        master.title("Traffic Flow Prediction System (TFPS)")
        master.geometry("800x600") # Set initial window size
        master.resizable(True, True) # Allow window resizing

        # --- Styling ---
        style = ttk.Style()
        style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Inter', 10))
        style.configure('TButton', font=('Inter', 10, 'bold'), padding=6, relief="raised", borderwidth=2, foreground='white', background='#4CAF50')
        style.map('TButton', background=[('active', '#45a049')]) # Darker green on hover
        style.configure('TEntry', font=('Inter', 10), padding=5)
        style.configure('TCombobox', font=('Inter', 10), padding=5)

        # --- Data and Model Placeholders ---
        # In a real application, you would load these once the app starts or via a button
        self.data_loader = None
        self.sae_model = None
        self.X_scaled = None
        self.y = None
        self.scaler = None
        self.scats_sites = [] # To populate SCATS site dropdowns

        # --- Main Frame ---
        self.main_frame = ttk.Frame(master, padding="20 20 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Title ---
        self.title_label = ttk.Label(self.main_frame, text="Traffic Flow Prediction System", font=('Inter', 16, 'bold'))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # --- Data Loading Section ---
        self.data_frame = ttk.LabelFrame(self.main_frame, text="1. Data Loading & Preprocessing", padding="15 15 15 15")
        self.data_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        ttk.Label(self.data_frame, text="CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.csv_path_entry = ttk.Entry(self.data_frame, width=50)
        self.csv_path_entry.insert(0, 'Scats Data October 2006.csv') # Default path
        self.csv_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.load_data_button = ttk.Button(self.data_frame, text="Load & Process Data", command=self._load_and_process_data)
        self.load_data_button.grid(row=0, column=2, padx=5, pady=5)

        self.data_status_label = ttk.Label(self.data_frame, text="Status: Ready to load data.")
        self.data_status_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # --- Model Training Section ---
        self.model_frame = ttk.LabelFrame(self.main_frame, text="2. Model Training", padding="15 15 15 15")
        self.model_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        ttk.Label(self.model_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.epochs_entry = ttk.Entry(self.model_frame, width=10)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.train_model_button = ttk.Button(self.model_frame, text="Train Model", command=self._train_model)
        self.train_model_button.grid(row=0, column=2, padx=5, pady=5)

        self.model_status_label = ttk.Label(self.model_frame, text="Status: Model not trained.")
        self.model_status_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # --- Prediction Section ---
        self.prediction_frame = ttk.LabelFrame(self.main_frame, text="3. Traffic Prediction & Route Guidance", padding="15 15 15 15")
        self.prediction_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        ttk.Label(self.prediction_frame, text="SCATS Site:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scats_site_combo = ttk.Combobox(self.prediction_frame, width=15, state="readonly")
        self.scats_site_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(self.prediction_frame, text="Date (MM/DD/YYYY):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.date_entry = ttk.Entry(self.prediction_frame, width=15)
        self.date_entry.insert(0, "10/01/2006") # Example date
        self.date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(self.prediction_frame, text="Time (HH:MM):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.time_entry = ttk.Entry(self.prediction_frame, width=15)
        self.time_entry.insert(0, "08:00") # Example time
        self.time_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.predict_button = ttk.Button(self.prediction_frame, text="Predict Traffic", command=self._predict_traffic)
        self.predict_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="ns")

        self.prediction_result_label = ttk.Label(self.prediction_frame, text="Predicted Traffic Volume: N/A")
        self.prediction_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # --- Route Guidance (Placeholder for future development) ---
        self.route_guidance_frame = ttk.LabelFrame(self.main_frame, text="4. Route Guidance (Future)", padding="15 15 15 15")
        self.route_guidance_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        ttk.Label(self.route_guidance_frame, text="This section will be developed for route guidance.").pack(padx=5, pady=5)


        # Configure grid weights for responsiveness
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_rowconfigure(4, weight=1)

        self.data_frame.grid_columnconfigure(1, weight=1)
        self.model_frame.grid_columnconfigure(1, weight=1)
        self.prediction_frame.grid_columnconfigure(1, weight=1)


    def _load_and_process_data(self):
        """
        Handles loading and preprocessing of the traffic data.
        """
        csv_path = self.csv_path_entry.get()
        if not os.path.exists(csv_path):
            messagebox.showerror("File Error", f"CSV file not found at: {csv_path}")
            self.data_status_label.config(text="Status: Error - File not found.")
            return

        try:
            # Import DataLoader here to avoid circular dependencies if app.py is imported by main.py
            from utils.dataloader import TrafficDataLoader

            self.data_loader = TrafficDataLoader(csv_path)
            self.data_status_label.config(text="Status: Loading raw data...")
            self.master.update_idletasks() # Update GUI immediately

            self.data_loader.load_and_initial_clean()
            self.data_status_label.config(text="Status: Transforming data...")
            self.master.update_idletasks()

            self.data_loader.transform_to_long_format()
            self.data_status_label.config(text="Status: Engineering features...")
            self.master.update_idletasks()

            self.data_loader.engineer_features()
            self.data_status_label.config(text="Status: Preparing data for model...")
            self.master.update_idletasks()

            self.X_scaled, self.y, self.scaler = self.data_loader.prepare_for_model()

            if self.X_scaled is not None and self.y is not None:
                self.data_status_label.config(text="Status: Data loaded and processed successfully!")
                # Populate SCATS site dropdown
                self.scats_sites = sorted(self.data_loader.df_final['SCATS Number'].unique().tolist())
                self.scats_site_combo['values'] = self.scats_sites
                if self.scats_sites:
                    self.scats_site_combo.set(self.scats_sites[0]) # Set default to first site
            else:
                self.data_status_label.config(text="Status: Error during data processing.")

        except Exception as e:
            messagebox.showerror("Data Processing Error", f"An error occurred during data processing: {e}")
            self.data_status_label.config(text="Status: Error during data processing.")
            print(f"Error: {e}") # Print to console for debugging


    def _train_model(self):
        """
        Handles the training of the Stacked Autoencoder model.
        """
        if self.X_scaled is None or self.y is None:
            messagebox.showwarning("Training Error", "Please load and process data first.")
            return

        try:
            epochs = int(self.epochs_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for epochs.")
            return

        try:
            # Import StackedAutoencoder here
            from models.saemodel import StackedAutoencoder

            self.model_status_label.config(text="Status: Splitting data for training...")
            self.master.update_idletasks()

            X_train, X_test, y_train, y_test = self.data_loader.chronological_train_test_split(self.X_scaled, self.y)

            if X_train is None:
                self.model_status_label.config(text="Status: Error splitting data.")
                return

            input_dim = X_train.shape[1]
            encoding_dims = [input_dim // 2, input_dim // 4] # Example encoding dimensions

            self.sae_model = StackedAutoencoder(input_dim=input_dim, encoding_dims=encoding_dims)

            self.model_status_label.config(text="Status: Pre-training autoencoders...")
            self.master.update_idletasks()
            self.sae_model.pretrain_autoencoders(X_train.to_numpy(), epochs=epochs // 2) # Use half epochs for pre-training

            self.model_status_label.config(text="Status: Training full model...")
            self.master.update_idletasks()
            self.sae_model.train_full_model(X_train.to_numpy(), y_train.to_numpy(), epochs=epochs)

            self.model_status_label.config(text="Status: Model trained successfully!")
            messagebox.showinfo("Training Complete", "Model training finished successfully!")

            # Optionally, evaluate the model on test data
            if X_test is not None and y_test is not None:
                predictions = self.sae_model.predict(X_test.to_numpy())
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                print(f"Model Evaluation on Test Set - MSE: {mse:.2f}, R2: {r2:.2f}")
                self.model_status_label.config(text=f"Status: Model trained. Test MSE: {mse:.2f}")


        except Exception as e:
            messagebox.showerror("Model Training Error", f"An error occurred during model training: {e}")
            self.model_status_label.config(text="Status: Error during model training.")
            print(f"Error: {e}") # Print to console for debugging

    def _predict_traffic(self):
        """
        Handles traffic prediction for a specified SCATS site, date, and time.
        """
        if self.sae_model is None:
            messagebox.showwarning("Prediction Error", "Please train the model first.")
            return
        if self.X_scaled is None or self.y is None or self.scaler is None:
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
            # Combine date and time
            prediction_datetime = pd.to_datetime(f"{date_str} {time_str}", format="%m/%d/%Y %H:%M")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid date or time format: {e}. Use MM/DD/YYYY and HH:MM.")
            return

        # Filter data for the specific SCATS site
        site_data = self.data_loader.df_final[self.data_loader.df_final['SCATS Number'] == scats_number].copy()
        if site_data.empty:
            messagebox.showwarning("Prediction Error", f"No historical data found for SCATS site {scats_number}.")
            return

        # Sort by datetime to ensure correct lookup
        site_data = site_data.sort_values(by='Date_Time')

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
            # This requires careful handling of edge cases (e.g., predicting for the very first timestamp)
            # For simplicity, we'll assume historical data exists for the lags.
            # In a production system, you'd fetch real-time data or use a more sophisticated lookup.

            # Get the relevant historical data for the SCATS site
            historical_site_data = self.data_loader.df_final[
                self.data_loader.df_final['SCATS Number'] == scats_number
            ].set_index('Date_Time').sort_index()

            # Function to get historical volume for a given timestamp
            def get_historical_volume(dt):
                if dt in historical_site_data.index:
                    return historical_site_data.loc[dt, 'Traffic_Volume']
                return np.nan # Return NaN if data not found

            # Calculate lag features for the prediction point
            predict_df['traffic_volume_lag_1'] = get_historical_volume(prediction_datetime - pd.Timedelta(minutes=15))
            predict_df['traffic_volume_lag_4'] = get_historical_volume(prediction_datetime - pd.Timedelta(hours=1))
            predict_df['traffic_volume_lag_96'] = get_historical_volume(prediction_datetime - pd.Timedelta(hours=24))

            # Calculate rolling mean (simplified for a single point, might need more context)
            # For a single prediction point, rolling mean is tricky.
            # A common approach is to use the last known rolling mean or a simplified average.
            # For this example, we'll just use the last known rolling mean from historical data if available.
            last_known_rolling_mean = historical_site_data['traffic_volume_rolling_mean_4'].iloc[-1] if not historical_site_data.empty else 0
            predict_df['traffic_volume_rolling_mean_4'] = last_known_rolling_mean

            # Drop 'SCATS Number' and 'Date_Time' from the prediction input as they are not features for the model
            predict_features_df = predict_df.drop(columns=['SCATS Number', 'Date_Time'])

            # Ensure the order of columns matches the training data (X_scaled)
            # This is crucial for consistent input to the model.
            # Get the feature columns from the trained model's input
            model_feature_columns = self.X_scaled.columns.tolist()
            predict_input = predict_features_df[model_feature_columns]

            # Handle any NaNs that might have resulted from missing historical lag data
            if predict_input.isnull().any().any():
                messagebox.showwarning("Prediction Warning", "Missing historical data for lag features. Prediction might be inaccurate.")
                # You might choose to impute these NaNs (e.g., with 0 or mean) or prevent prediction
                predict_input = predict_input.fillna(0) # Simple imputation for demonstration

            # Scale the input features using the *fitted* scaler
            predict_input_scaled = self.scaler.transform(predict_input)

            # Make prediction
            predicted_volume = self.sae_model.predict(predict_input_scaled)[0][0]

            self.prediction_result_label.config(text=f"Predicted Traffic Volume: {predicted_volume:.2f}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.prediction_result_label.config(text="Predicted Traffic Volume: Error")
            print(f"Error during prediction: {e}")


# Main entry point for the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = TFPSApp(root)
    root.mainloop()