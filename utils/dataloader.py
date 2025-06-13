import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os # Import os module to handle file paths

class TrafficDataLoader:
    def __init__(self, csv_file_path):
        self.csv_file_path = os.path.join('data', csv_file_path)
        self.df_raw = None
        self.df_processed = None
        self.df_final = None
        self.scaler = None

    def load_and_initial_clean(self):
        print(f"Loading data from: {self.csv_file_path}")
        try:
            self.df_raw = pd.read_csv(self.csv_file_path, header=1)
            self.df_raw.columns = self.df_raw.columns.str.strip()
            self.df_raw = self.df_raw.rename(columns={'Unnamed: 9': 'Date'})

            metadata_path = os.path.join('data', 'SCATS_Site_Metadata.csv')
            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
                metadata_df.columns = metadata_df.columns.str.strip()
                self.df_raw = self.df_raw.merge(metadata_df, on='SCATS Number', how='left')
            else:
                print(f"\nWARNING: Metadata file not found at {metadata_path}. Skipping merge.")
            
            return self.df_raw
        except FileNotFoundError:
            print(f"Error: The file '{self.csv_file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            raise # Re-raise the exception to be caught by the GUI

    def transform_to_long_format(self):
        if self.df_raw is None:
            print("Raw data not loaded.")
            return None

        volume_columns = [col for col in self.df_raw.columns if col.startswith('V') and col[1:].isdigit()]
        
        # --- FIX 1: Add missing columns to id_vars to prevent them from being dropped ---
        id_vars = [var for var in [
            'SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date',
            'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY'
            ] if var in self.df_raw.columns]

        df_melted = self.df_raw.melt(
            id_vars=id_vars,
            value_vars=volume_columns,
            var_name='Time_Interval_Code',
            value_name='Traffic_Volume'
        )

        times = pd.to_datetime([f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]], format='%H:%M').strftime('%H:%M').tolist()
        time_code_map = {f'V{i:02d}': t for i, t in enumerate(times)}
        df_melted['Time_Interval'] = df_melted['Time_Interval_Code'].map(time_code_map)
        
        df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%d/%m/%Y', errors='coerce')
        df_melted['Date_Time'] = pd.to_datetime(df_melted['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_melted['Time_Interval'], errors='coerce')
        df_melted['Traffic_Volume'] = pd.to_numeric(df_melted['Traffic_Volume'], errors='coerce')
        
        initial_rows = len(df_melted)
        df_melted.dropna(subset=['Date_Time', 'Traffic_Volume'], inplace=True)
        print(f"Dropped {initial_rows - len(df_melted)} rows with missing Date_Time or Traffic_Volume.")

        self.df_processed = df_melted.drop(columns=['Time_Interval_Code'])
        return self.df_processed

    def engineer_features(self):
        if self.df_processed is None:
            print("Processed data not available.")
            return None

        print("\nEngineering time-based and lagged features...")
        df = self.df_processed.sort_values(by=['SCATS Number', 'Date_Time']).reset_index(drop=True)

        df['hour'] = df['Date_Time'].dt.hour.astype('Int64')
        df['minute'] = df['Date_Time'].dt.minute.astype('Int64')
        df['day_of_week'] = df['Date_Time'].dt.dayofweek.astype('Int64')
        df['day_of_year'] = df['Date_Time'].dt.dayofyear.astype('Int64')
        df['week_of_year'] = df['Date_Time'].dt.isocalendar().week.astype('Int64')
        df['month'] = df['Date_Time'].dt.month.astype('Int64')
        df['year'] = df['Date_Time'].dt.year.astype('Int64')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('Int64')

        # Lagged and rolling features
        df['traffic_volume_lag_1'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(1)
        df['traffic_volume_lag_4'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(4)
        df['traffic_volume_lag_96'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(96)
        df['traffic_volume_rolling_mean_4'] = df.groupby('SCATS Number')['Traffic_Volume'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
        )

        initial_rows = len(df)
        df.dropna(subset=['traffic_volume_lag_96'], inplace=True)
        print(f"Dropped {initial_rows - len(df)} rows due to NaN values from lagged features.")

        df.drop_duplicates(subset=['SCATS Number', 'Date_Time'], inplace=True)
        self.df_final = df
        return self.df_final

    def prepare_for_model(self, target_column='Traffic_Volume'):
        if self.df_final is None:
            print("Features not engineered.")
            return None, None, None

        # --- FIX 2: Add the columns back to the final feature set ---
        feature_columns = [col for col in [
            'NB_LATITUDE', 'NB_LONGITUDE', 'HF VicRoads Internal',
            'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY',
            'hour', 'minute', 'day_of_week', 'day_of_year', 'week_of_year',
            'month', 'year', 'is_weekend',
            'traffic_volume_lag_1', 'traffic_volume_lag_4', 'traffic_volume_lag_96',
            'traffic_volume_rolling_mean_4'
        ] if col in self.df_final.columns]

        X = self.df_final[feature_columns]
        y = self.df_final[target_column]

        features_to_scale = [col for col in feature_columns if X[col].dtype in ['int64', 'float64', 'Int64']]
        self.scaler = MinMaxScaler()
        X_scaled = X.copy()
        
        # Handle potential all-NaN columns before scaling
        for col in features_to_scale:
            if X_scaled[col].isnull().all():
                X_scaled[col].fillna(0, inplace=True)

        X_scaled[features_to_scale] = self.scaler.fit_transform(X_scaled[features_to_scale])
        
        return X_scaled, y, self.scaler

    def chronological_train_test_split(self, X, y, test_size=0.2):
        if X is None or y is None:
            return None, None, None, None
            
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        data_loader = TrafficDataLoader('Scats Data October 2006.csv')
        data_loader.load_and_initial_clean()
        data_loader.transform_to_long_format()
        data_loader.engineer_features()
        X, y, scaler = data_loader.prepare_for_model()
        if X is not None:
            print(f"Final number of features for the model: {X.shape[1]}")
            X_train, X_test, y_train, y_test = data_loader.chronological_train_test_split(X, y)
            print("\nData loading and preprocessing complete.")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")