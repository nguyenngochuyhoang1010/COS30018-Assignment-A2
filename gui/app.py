import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel, Listbox, MULTIPLE
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

class ColumnMappingWindow(Toplevel):
    """A Toplevel window to allow users to map their CSV columns to required columns."""
    def __init__(self, master, columns, required_cols):
        super().__init__(master)
        self.title("Map CSV Columns")
        self.geometry("600x500")
        self.transient(master)
        self.grab_set()

        self.column_mapping = None
        self.csv_columns = columns
        self.required_cols = required_cols

        self.create_widgets()

    def create_widgets(self):
        self.mapping_vars = {}
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=10, pady=10, expand=True, fill="both")

        ttk.Label(main_frame, text="Please map your CSV columns to the required data fields.",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        for i, (req_col, desc) in enumerate(self.required_cols.items()):
            if req_col != 'volume_columns':
                ttk.Label(main_frame, text=f"{desc}:").grid(row=i + 1, column=0, padx=5, pady=5, sticky="w")
                var = tk.StringVar()
                default_selection = next((c for c in self.csv_columns if req_col.lower().replace("_", " ") in c.lower()), self.csv_columns[0])
                var.set(default_selection)
                ttk.Combobox(main_frame, textvariable=var, values=self.csv_columns, state="readonly").grid(row=i + 1, column=1, padx=5, pady=5, sticky="ew")
                self.mapping_vars[req_col] = var
        
        vol_label = ttk.Label(main_frame, text="Traffic Volume Columns:")
        vol_label.grid(row=len(self.required_cols), column=0, padx=5, pady=5, sticky="nw")
        
        self.volume_listbox = Listbox(main_frame, selectmode=MULTIPLE, exportselection=False, height=10)
        for col in self.csv_columns:
            self.volume_listbox.insert(tk.END, col)
            if col.strip().startswith('V') and len(col.strip()) > 1 and col.strip()[1:].isdigit():
                 self.volume_listbox.selection_set(tk.END)
        self.volume_listbox.grid(row=len(self.required_cols), column=1, padx=5, pady=5, sticky="ew")

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=len(self.required_cols) + 1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Confirm Mapping", command=self.confirm_mapping).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side="left", padx=10)
        
        main_frame.grid_columnconfigure(1, weight=1)

    def confirm_mapping(self):
        self.column_mapping = {req: var.get() for req, var in self.mapping_vars.items()}
        
        selected_indices = self.volume_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Validation Error", "Please select at least one traffic volume column.", parent=self)
            return
            
        self.column_mapping['volume_columns'] = [self.volume_listbox.get(i) for i in selected_indices]
        self.destroy()

class TFPSApp:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Flow Prediction System (TFPS)")
        master.geometry("800x700")

        self.data_loader = None
        self.column_mapping = None
        
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
        
        self.browse_button = ttk.Button(csv_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        self.map_button = ttk.Button(csv_frame, text="Map Columns", command=self.open_column_mapper, state=tk.DISABLED)
        self.map_button.grid(row=1, column=2, padx=5, pady=5)

        self.load_button = ttk.Button(csv_frame, text="Load & Process Data", command=self.load_and_process_data, state=tk.DISABLED)
        self.load_button.grid(row=1, column=3, padx=5, pady=5)
        
        # --- NEW: Added entry for header row index ---
        ttk.Label(csv_frame, text="Header Row Index:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.header_row_var = tk.StringVar(value="0")
        self.header_row_entry = ttk.Entry(csv_frame, textvariable=self.header_row_var, width=10)
        self.header_row_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.data_status_label = ttk.Label(csv_frame, text="Status: Please select a CSV file.")
        self.data_status_label.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="w")
        csv_frame.grid_columnconfigure(1, weight=1)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filepath:
            self.csv_file_entry.delete(0, tk.END)
            self.csv_file_entry.insert(0, filepath)
            self.update_status("File selected. Please set Header Row Index and map columns.", self.data_status_label)
            self.map_button.config(state=tk.NORMAL)
            self.load_button.config(state=tk.DISABLED)

    def open_column_mapper(self):
        filepath = self.csv_file_entry.get()
        if not filepath: return messagebox.showerror("Error", "Please select a file first.")
        
        try:
            header_row = int(self.header_row_var.get())
            # --- FIX: Use the user-provided header_row value ---
            headers = pd.read_csv(filepath, nrows=0, header=header_row).columns.str.strip().tolist()
            
            required_cols = {
                'scats_number': 'SCATS Site ID / Junction',
                'location': 'Location Name (Optional)',
                'latitude': 'Latitude (Optional)',
                'longitude': 'Longitude (Optional)',
                'date': 'Date or DateTime',
                'volume_columns': 'Traffic Volume Columns'
            }

            mapper_window = ColumnMappingWindow(self.master, headers, required_cols)
            self.master.wait_window(mapper_window)
            
            if mapper_window.column_mapping:
                self.column_mapping = mapper_window.column_mapping
                self.update_status("Columns mapped successfully. Ready to load data.", self.data_status_label)
                self.load_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("File Read Error", f"Could not read the CSV file header on row {header_row}. Please check the file and Header Row Index. Error: {e}")

    def update_status(self, message, label_widget):
        label_widget.config(text=f"Status: {message}")
        self.master.update_idletasks()

    def load_and_process_data(self):
        file_path = self.csv_file_entry.get()
        if not file_path or not self.column_mapping: return messagebox.showerror("Error", "Please select a file and map columns.")

        self.update_status("Loading and preprocessing data...", self.data_status_label)
        try:
            header_row = int(self.header_row_var.get())
            # --- FIX: Pass header_row to DataLoader ---
            self.data_loader = TrafficDataLoader(file_path, self.column_mapping, header_row)
            
            self.data_loader.load_and_initial_clean()
            # Determine data format and process accordingly
            if self.data_loader.is_long_format:
                self.data_loader.process_long_format()
            else:
                self.data_loader.transform_to_long_format()

            self.data_loader.engineer_features()
            X_scaled, y, scaler = self.data_loader.prepare_for_model()
            
            if X_scaled is None: raise ValueError("Data preparation returned None.")

            self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.chronological_train_test_split(X_scaled, y)
            self.scaler = scaler

            self.update_status("Data loaded and preprocessed successfully!", self.data_status_label)
            messagebox.showinfo("Success", "Data loaded and preprocessed successfully!")

            if self.data_loader.df_final is not None:
                scats_numbers = sorted(self.data_loader.df_final['scats_number'].unique().tolist())
                self.scats_site_combo['values'] = scats_numbers
                if scats_numbers: self.scats_site_combo.set(scats_numbers[0])
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
        if self.X_train is None: return messagebox.showwarning("Training Error", "Please load data first.")
        try: epochs = int(self.epochs_entry.get())
        except ValueError: return messagebox.showerror("Input Error", "Epochs must be an integer.")

        model_name = self.model_selection.get()
        self.update_status(f"Training {model_name}...", self.model_status_label)
        self.train_button.config(state=tk.DISABLED)
        threading.Thread(target=self.train_model_thread, args=(epochs, model_name), daemon=True).start()

    def train_model_thread(self, epochs, model_name):
        try:
            X_train_data, y_train_data = self.X_train.values, self.y_train.values
            history = None
            if model_name == "Stacked Autoencoder":
                self.active_model = self.sae_model
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
            self.ax.set_title(f"{model_name} Training Loss"); self.ax.set_xlabel("Epoch"); self.ax.set_ylabel("Loss (MSE)"); self.ax.legend()
            self.canvas.draw()
        else:
            self.update_status(f"Error training {model_name}: {error_msg}", self.model_status_label)
            messagebox.showerror("Training Error", f"An error occurred: {error_msg}")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"): self.master.destroy()

    def create_prediction_widgets(self, tab):
        prediction_frame = ttk.LabelFrame(tab, text="Traffic Prediction")
        prediction_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(prediction_frame, text="SCATS Site:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.scats_site_combo = ttk.Combobox(prediction_frame, width=15, state="readonly")
        self.scats_site_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(prediction_frame, text="Date (MM/DD/YYYY):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.date_entry = ttk.Entry(prediction_frame, width=15); self.date_entry.insert(0, "10/01/2006")
        self.date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(prediction_frame, text="Time (HH:MM):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.time_entry = ttk.Entry(prediction_frame, width=15); self.time_entry.insert(0, "08:00")
        self.time_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.predict_button = ttk.Button(prediction_frame, text="Predict Traffic", command=self.predict_traffic)
        self.predict_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="ns")

        self.prediction_result_label = ttk.Label(prediction_frame, text="Predicted Traffic Volume: N/A", font=("Helvetica", 12))
        self.prediction_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        prediction_frame.grid_columnconfigure(1, weight=1)

    def predict_traffic(self):
        if self.active_model is None: return messagebox.showerror("Error", "Model not trained or data not loaded.")
        
        try:
            scats_site = int(self.scats_site_combo.get())
            date_str = self.date_entry.get(); time_str = self.time_entry.get()
            
            site_df = self.data_loader.df_final[self.data_loader.df_final['scats_number'] == scats_site]
            if site_df.empty: raise ValueError(f"No historical data for SCATS site {scats_site}")
            
            template_row = site_df.iloc[[-1]].copy()
            prediction_datetime = pd.to_datetime(f"{date_str} {time_str}", format="%m/%d/%Y %H:%M")
            template_row['hour'] = prediction_datetime.hour; template_row['minute'] = prediction_datetime.minute
            template_row['day_of_week'] = prediction_datetime.dayofweek; template_row['day_of_year'] = prediction_datetime.dayofyear
            template_row['week_of_year'] = prediction_datetime.isocalendar().week; template_row['month'] = prediction_datetime.month
            template_row['year'] = prediction_datetime.year; template_row['is_weekend'] = (template_row['day_of_week'] >= 5).astype(int)
            
            feature_columns = self.X_train.columns
            for col in feature_columns:
                if col not in template_row: template_row[col] = 0
            prediction_input_df = template_row[feature_columns]

            scaled_input = self.scaler.transform(prediction_input_df)
            if isinstance(self.active_model, LSTMTrafficPredictor):
                scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
            
            predicted_volume = self.active_model.predict(scaled_input)
            final_prediction = predicted_volume[0][0] if isinstance(predicted_volume[0], (list, np.ndarray)) else predicted_volume[0]
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
        if self.data_loader is None: return messagebox.showwarning("Map Error", "Please load data first.")

        map_file_path = "traffic_map.html"
        try:
            lat_col, lon_col, scats_col, loc_col = 'latitude', 'longitude', 'scats_number', 'location'
            required_map_cols = [lat_col, lon_col, scats_col, loc_col]
            if not all(col in self.data_loader.df_final.columns for col in required_map_cols):
                 raise KeyError(f"One of the required map columns is missing: {required_map_cols}")

            scats_locations = self.data_loader.df_final.groupby(scats_col).agg({lat_col: 'last', lon_col: 'last', loc_col: 'first'}).dropna(subset=[lat_col, lon_col]).reset_index()
            if scats_locations.empty: return messagebox.showwarning("Map Error", "No valid location data found.")

            map_center = [scats_locations[lat_col].mean(), scats_locations[lon_col].mean()]
            m = folium.Map(location=map_center, zoom_start=12)
            for _, row in scats_locations.iterrows():
                folium.Marker(location=[row[lat_col], row[lon_col]], popup=f"ID: {row[scats_col]}<br>Location: {row[loc_col]}", tooltip=f"ID: {row[scats_col]}").add_to(m)

            m.save(map_file_path)
            webbrowser.open(f"file://{os.path.realpath(map_file_path)}")
        except KeyError as e: messagebox.showerror("Map Generation Error", f"A required column is missing from mapping: {e}")
        except Exception as e: messagebox.showerror("Map Generation Error", f"Failed to generate map: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TFPSApp(root)
    root.mainloop()