import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os # Import os module to handle file paths

class TrafficDataLoader:
    def __init__(self, csv_file_path):
        # Using os.path.join for better cross-platform compatibility
        self.csv_file_path = os.path.join('data', csv_file_path)
        self.df_raw = None
        self.df_processed = None
        self.df_final = None
        self.scaler = None

    def load_and_initial_clean(self):
        print(f"Loading data from: {self.csv_file_path}")
        try:
            self.df_raw = pd.read_csv(self.csv_file_path, header=1)
            
            # --- FIX ---
            # Strip leading/trailing whitespace from all column names.
            # This is the crucial fix for the KeyError.
            self.df_raw.columns = self.df_raw.columns.str.strip()

            self.df_raw = self.df_raw.rename(columns={'Unnamed: 9': 'Date'})

            # Merge with metadata if available
            metadata_path = os.path.join('data', 'SCATS_Site_Metadata.csv')
            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
                # Also strip whitespace from metadata columns for a robust merge
                metadata_df.columns = metadata_df.columns.str.strip()
                print(f"\nMerging with metadata from: {metadata_path}")
                
                # Ensure the merge key also has stripped whitespace if necessary
                # Assuming 'SCATS Number' is the merge key and it's clean.
                self.df_raw = self.df_raw.merge(metadata_df, on='SCATS Number', how='left')
                print("\nDataFrame head after merge:")
                print(self.df_raw.head())
            else:
                print(f"\nWARNING: Metadata file not found at {metadata_path}. Skipping metadata merge.")

            print("Initial DataFrame head after loading and renaming:")
            print(self.df_raw.head())
            print("\nInitial DataFrame info:")
            self.df_raw.info()
            return self.df_raw
        except FileNotFoundError:
            print(f"Error: The file '{self.csv_file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred during loading: {e}")
            return None


    def transform_to_long_format(self):
        """
        Melts the DataFrame to transform 15-minute interval volume columns into rows.
        Creates a combined datetime column.
        """
        if self.df_raw is None:
            print("Raw data not loaded. Please call load_and_initial_clean() first.")
            return None

        # Identify columns containing traffic volume data.
        volume_columns = [col for col in self.df_raw.columns if col.startswith('V') and len(col) == 3 and col[1:].isdigit()]

        # Define identifier variables that should remain as columns after melting.
        id_vars = [
            'SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 'NB_LONGITUDE',
            'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc',
            'NB_TYPE_SURVEY', 'Date'
        ]
        
        # Ensure all id_vars exist in the dataframe before melting
        id_vars = [var for var in id_vars if var in self.df_raw.columns]

        print(f"\nMelting {len(volume_columns)} volume columns into rows...")
        df_melted = self.df_raw.melt(
            id_vars=id_vars,
            value_vars=volume_columns,
            var_name='Time_Interval_Code',
            value_name='Traffic_Volume'
        )

        # Create a mapping for 'Time_Interval_Code' (V00-V95) to actual 15-minute time strings.
        times = pd.to_datetime([f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]], format='%H:%M').strftime('%H:%M').tolist()
        time_code_map = {f'V{i:02d}': t for i, t in enumerate(times)}

        df_melted['Time_Interval'] = df_melted['Time_Interval_Code'].map(time_code_map)
        df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%d/%m/%Y', errors='coerce')

        # Combine 'Date' and 'Time_Interval' into a single 'Date_Time' column.
        df_melted['Date_Time'] = pd.to_datetime(df_melted['Date'].dt.strftime('%Y-%m-%d') + ' ' + df_melted['Time_Interval'])
        
        df_melted['Traffic_Volume'] = pd.to_numeric(df_melted['Traffic_Volume'], errors='coerce')

        self.df_processed = df_melted.drop(columns=['Time_Interval_Code'])

        print("\nDataFrame head after melting and datetime conversion:")
        print(self.df_processed.head())
        print("\nDataFrame info after melting and datetime conversion:")
        self.df_processed.info()
        return self.df_processed

    def engineer_features(self):
        """
        Adds time-based and lagged features to the DataFrame.
        """
        if self.df_processed is None:
            print("Processed data not available. Please call transform_to_long_format() first.")
            return None

        print("\nEngineering time-based and lagged features...")
        df = self.df_processed.sort_values(by=['SCATS Number', 'Date_Time']).reset_index(drop=True)

        # Time-based features
        df['hour'] = df['Date_Time'].dt.hour
        df['minute'] = df['Date_Time'].dt.minute
        df['day_of_week'] = df['Date_Time'].dt.dayofweek
        df['day_of_year'] = df['Date_Time'].dt.dayofyear
        df['week_of_year'] = df['Date_Time'].dt.isocalendar().week.astype(int)
        df['month'] = df['Date_Time'].dt.month
        df['year'] = df['Date_Time'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Lagged and rolling features
        df['traffic_volume_lag_1'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(1)
        df['traffic_volume_lag_4'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(4)
        df['traffic_volume_lag_96'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(96)
        df['traffic_volume_rolling_mean_4'] = df.groupby('SCATS Number')['Traffic_Volume'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
        )

        initial_rows = len(df)
        df_features = df.dropna(subset=['traffic_volume_lag_96'])
        print(f"Dropped {initial_rows - len(df_features)} rows due to NaN values from lagged features.")

        original_rows_before_drop = len(df_features)
        df_features = df_features.drop_duplicates(subset=['SCATS Number', 'Date_Time'])
        if len(df_features) < original_rows_before_drop:
            print(f"Dropped {original_rows_before_drop - len(df_features)} duplicate (SCATS Number, Date_Time) entries.")

        self.df_final = df_features

        print("\nDataFrame head after feature engineering:")
        print(self.df_final.head())

        return self.df_final

    def prepare_for_model(self, target_column='Traffic_Volume', features_to_scale=None):
        """
        Prepares data for model training: splits into X/y and scales features.
        """
        if self.df_final is None:
            print("Features not engineered. Please call engineer_features() first.")
            return None, None, None

        feature_columns = [
            'NB_LATITUDE', 'NB_LONGITUDE',
            'hour', 'minute', 'day_of_week', 'day_of_year', 'week_of_year',
            'month', 'year', 'is_weekend',
            'traffic_volume_lag_1', 'traffic_volume_lag_4', 'traffic_volume_lag_96',
            'traffic_volume_rolling_mean_4'
        ]
        
        # Filter out any columns that might not exist in the dataframe
        feature_columns = [col for col in feature_columns if col in self.df_final.columns]

        X = self.df_final[feature_columns]
        y = self.df_final[target_column]

        if features_to_scale is None:
            features_to_scale = [col for col in feature_columns if X[col].dtype in ['int64', 'float64', 'int32']]

        self.scaler = MinMaxScaler()
        X_scaled = X.copy()
        X_scaled[features_to_scale] = self.scaler.fit_transform(X[features_to_scale])

        print("\nFeatures (X) head after scaling:")
        print(X_scaled.head())
        
        return X_scaled, y, self.scaler

    def chronological_train_test_split(self, X, y, test_size=0.2):
        """
        Performs a chronological train-test split for time-series data.
        """
        if X is None or y is None:
            print("Data not prepared. Call prepare_for_model() first.")
            return None, None, None, None

        X = X.loc[y.index]
        split_index = int(len(X) * (1 - test_size))

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        print(f"\nChronological Train-Test Split:")
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test

# Example usage (for testing this module directly)
if __name__ == "__main__":
    data_loader = TrafficDataLoader('Scats Data October 2006.csv')
    data_loader.load_and_initial_clean()
    data_loader.transform_to_long_format()
    data_loader.engineer_features()
    X, y, scaler = data_loader.prepare_for_model()
    if X is not None:
        X_train, X_test, y_train, y_test = data_loader.chronological_train_test_split(X, y)
        print("\nData loading and preprocessing complete. Ready for model training.")