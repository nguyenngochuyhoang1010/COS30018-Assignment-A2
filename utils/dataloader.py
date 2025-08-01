import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os # Import os module to handle file paths

class TrafficDataLoader:
    def __init__(self, csv_file_path, column_mapping, header_row=0):
        self.csv_file_path = csv_file_path
        self.mapping = column_mapping
        self.header_row = header_row
        self.df_raw = None
        self.df_processed = None
        self.df_final = None
        self.scaler = None
        self.is_long_format = False # Flag to check data format

    def load_and_initial_clean(self):
        print(f"Loading data from: {self.csv_file_path} with header on row {self.header_row}")
        try:
            self.df_raw = pd.read_csv(self.csv_file_path, header=self.header_row)
            self.df_raw.columns = self.df_raw.columns.str.strip()
        except Exception as e:
            raise ValueError(f"Could not read CSV. Check file and header row index. Error: {e}")

        rename_dict = {v: k for k, v in self.mapping.items() if k != 'volume_columns'}
        self.df_raw.rename(columns=rename_dict, inplace=True)
        
        if len(self.mapping['volume_columns']) == 1:
            self.is_long_format = True
            self.df_raw.rename(columns={self.mapping['volume_columns'][0]: 'Traffic_Volume'}, inplace=True)
        
        print("Columns renamed. Detected format:", "Long" if self.is_long_format else "Wide")

    def process_long_format(self):
        """Processes data that is already in a long format."""
        print("Processing long-format data.")
        self.df_processed = self.df_raw.copy()
        self.df_processed['Date_Time'] = pd.to_datetime(self.df_processed['date'], errors='coerce')
        self.df_processed.dropna(subset=['Date_Time'], inplace=True)
        if 'Traffic_Volume' in self.df_processed.columns:
            self.df_processed['Traffic_Volume'] = pd.to_numeric(self.df_processed['Traffic_Volume'], errors='coerce')
            self.df_processed.dropna(subset=['Traffic_Volume'], inplace=True)

    def transform_to_long_format(self):
        """Melts wide-format data (like the original SCATS file)."""
        if self.df_raw is None: raise ValueError("Raw data not loaded.")
        volume_columns = [col.strip() for col in self.mapping['volume_columns']]
        id_vars = [k for k in self.mapping.keys() if k != 'volume_columns' and k in self.df_raw.columns]
        df_melted = self.df_raw.melt(id_vars=id_vars, value_vars=volume_columns, var_name='Time_Interval_Code', value_name='Traffic_Volume')
        times = pd.to_datetime([f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]], format='%H:%M').strftime('%H:%M').tolist()
        time_code_map = {vol_col: times[i % len(times)] for i, vol_col in enumerate(volume_columns)}
        df_melted['Time_Interval'] = df_melted['Time_Interval_Code'].map(time_code_map)
        df_melted['date'] = pd.to_datetime(df_melted['date'], errors='coerce')
        df_melted['Date_Time'] = pd.to_datetime(df_melted['date'].dt.strftime('%Y-%m-%d') + ' ' + df_melted['Time_Interval'], errors='coerce')
        df_melted['Traffic_Volume'] = pd.to_numeric(df_melted['Traffic_Volume'], errors='coerce')
        df_melted.dropna(subset=['Date_Time', 'Traffic_Volume'], inplace=True)
        self.df_processed = df_melted.drop(columns=['Time_Interval_Code'])

    def engineer_features(self, df=None, is_prediction=False):
        """
        Adds time-based and lagged features. Can operate on a provided dataframe for prediction.
        """
        if is_prediction:
            if df is None:
                raise ValueError("A DataFrame must be provided when is_prediction is True.")
            input_df = df.copy() # Work on a copy
        else:
            if self.df_processed is None:
                raise ValueError("Data not processed yet. Run process_data() first.")
            input_df = self.df_processed.copy()

        scats_col = 'scats_number'
        if scats_col not in input_df.columns:
            raise ValueError(f"Required column '{scats_col}' not found in the dataframe.")

        df_engineered = input_df.sort_values(by=[scats_col, 'Date_Time']).reset_index(drop=True)

        # Time-based features
        df_engineered['hour'] = df_engineered['Date_Time'].dt.hour.astype('Int64')
        df_engineered['minute'] = df_engineered['Date_Time'].dt.minute.astype('Int64')
        df_engineered['day_of_week'] = df_engineered['Date_Time'].dt.dayofweek.astype('Int64')
        df_engineered['day_of_year'] = df_engineered['Date_Time'].dt.dayofyear.astype('Int64')
        df_engineered['week_of_year'] = df_engineered['Date_Time'].dt.isocalendar().week.astype('Int64')
        df_engineered['month'] = df_engineered['Date_Time'].dt.month.astype('Int64')
        df_engineered['year'] = df_engineered['Date_Time'].dt.year.astype('Int64')
        df_engineered['is_weekend'] = (df_engineered['day_of_week'] >= 5).astype('Int64')

        # For prediction, we assume the input might not have the target column to create lags from.
        # A more robust solution would involve joining historical data to create these lags.
        # For now, we'll create them if possible, otherwise they will be filled with 0 later.
        if 'Traffic_Volume' in df_engineered.columns:
            df_engineered['traffic_volume_lag_1'] = df_engineered.groupby(scats_col)['Traffic_Volume'].shift(1)
            df_engineered['traffic_volume_lag_4'] = df_engineered.groupby(scats_col)['Traffic_Volume'].shift(4)
            df_engineered['traffic_volume_lag_96'] = df_engineered.groupby(scats_col)['Traffic_Volume'].shift(96)
            df_engineered['traffic_volume_rolling_mean_4'] = df_engineered.groupby(scats_col)['Traffic_Volume'].transform(lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
            df_engineered.dropna(subset=['traffic_volume_lag_96'], inplace=True)
        
        df_engineered.drop_duplicates(subset=[scats_col, 'Date_Time'], inplace=True)

        if is_prediction:
            return df_engineered
        else:
            self.df_final = df_engineered

    def prepare_for_model(self, target_column='Traffic_Volume'):
        """Prepares the final data for the model, ensuring consistent feature sets."""
        if self.df_final is None: raise ValueError("Features not engineered.")
        for col in ['latitude', 'longitude', 'location']:
            if col not in self.df_final.columns:
                self.df_final[col] = 0 if col != 'location' else 'N/A'
        
        feature_columns = [col for col in ['latitude', 'longitude', 'hour', 'minute', 'day_of_week', 'day_of_year',
                                             'week_of_year', 'month', 'year', 'is_weekend', 'traffic_volume_lag_1',
                                             'traffic_volume_lag_4', 'traffic_volume_lag_96', 'traffic_volume_rolling_mean_4']
                                           if col in self.df_final.columns]
        
        X = self.df_final[feature_columns].copy()
        y = self.df_final[target_column]
        
        features_to_scale = [col for col in X.columns if X[col].dtype in ['int64', 'float64', 'Int64']]
        self.scaler = MinMaxScaler()
        X_scaled = X.copy()
        X_scaled[features_to_scale] = self.scaler.fit_transform(X_scaled[features_to_scale])
        return X_scaled, y, self.scaler

    def chronological_train_test_split(self, X, y, test_size=0.2):
        if X is None: return None, None, None, None
        split_index = int(len(X) * (1 - test_size))
        return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]

    def get_scats_locations(self):
        """Returns a list of unique SCATS sites with names and coordinates."""
        if self.df_final is None: return []
        
        # Ensure optional columns exist
        for col in ['location', 'latitude', 'longitude']:
            if col not in self.df_final.columns:
                return [] # Not enough data for locations

        locations_df = self.df_final[['scats_number', 'location', 'latitude', 'longitude']].drop_duplicates(subset=['scats_number'])
        
        return [{'id': row['scats_number'], 'name': row['location'], 'lat': row['latitude'], 'lng': row['longitude']} for index, row in locations_df.iterrows()]

    def process_data(self):
        """A single method to run the main processing steps."""
        if self.is_long_format:
            self.process_long_format()
        else:
            self.transform_to_long_format()
        self.engineer_features()
