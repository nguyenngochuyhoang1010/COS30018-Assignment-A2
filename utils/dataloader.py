import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os # Import os module to handle file paths

class TrafficDataLoader:
    """
    A class to load, preprocess, and prepare traffic data for machine learning models.
    It includes functionalities for data cleaning, feature engineering, and splitting.
    """
    def __init__(self, csv_file_path):
        """
        Initializes the DataLoader with the path to the raw CSV data.

        Args:
            csv_file_path (str): The path to the 'Scats Data October 2006.csv' file.
        """
        # Adjust the path to look inside the 'data' directory
        self.csv_file_path = os.path.join('data', csv_file_path)
        self.df_raw = None
        self.df_processed = None
        self.df_final = None
        self.scaler = None # Scaler for numerical features

    def load_and_initial_clean(self):
        """
        Loads the CSV file, skipping the first header row and renaming the date column.
        """
        print(f"Loading data from: {self.csv_file_path}")
        # The actual header is on the second row (index 1), so we use header=1
        # The first row contains general titles which we can skip.
        self.df_raw = pd.read_csv(self.csv_file_path, header=1)

        # Rename the 'Unnamed: 9' column to 'Date' as it contains the date information.
        # This column was 'Start Time' in the first header row, but shifted.
        self.df_raw = self.df_raw.rename(columns={'Unnamed: 9': 'Date'})

        print("Initial DataFrame head after loading and renaming:")
        print(self.df_raw.head())
        print("\nInitial DataFrame info:")
        self.df_raw.info()
        return self.df_raw

    def transform_to_long_format(self):
        """
        Melts the DataFrame to transform 15-minute interval volume columns into rows.
        Creates a combined datetime column.
        """
        if self.df_raw is None:
            print("Raw data not loaded. Please call load_and_initial_clean() first.")
            return None

        # Identify columns containing traffic volume data. These are columns starting
        # with 'V' followed by two digits (e.g., V00, V01, ..., V95).
        volume_columns = [col for col in self.df_raw.columns if col.startswith('V') and len(col) == 3]

        # Define identifier variables that should remain as columns after melting.
        # These columns provide context for each traffic volume reading.
        id_vars = [
            'SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE', 'NB_LONGITUDE',
            'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc',
            'NB_TYPE_SURVEY', 'Date'
        ]

        print(f"\nMelting {len(volume_columns)} volume columns into rows...")
        df_melted = self.df_raw.melt(
            id_vars=id_vars,
            value_vars=volume_columns,
            var_name='Time_Interval_Code', # New column for the original 'Vxx' code
            value_name='Traffic_Volume'    # New column for the actual traffic volume
        )

        # Create a mapping for 'Time_Interval_Code' (V00-V95) to actual 15-minute time strings.
        # There are 96 intervals in a day (24 hours * 4 intervals/hour).
        times = pd.to_datetime([f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]).strftime('%H:%M').tolist()
        time_code_map = {f'V{i:02d}': t for i, t in enumerate(times)}

        # Apply the mapping to create a readable 'Time_Interval' column.
        df_melted['Time_Interval'] = df_melted['Time_Interval_Code'].map(time_code_map)

        # Convert 'Date' to datetime objects, handling potential mixed formats by inferring.
        df_melted['Date'] = pd.to_datetime(df_melted['Date'], infer_datetime_format=True, errors='coerce')

        # Combine 'Date' and 'Time_Interval' into a single 'Date_Time' column.
        # This is critical for time-series analysis.
        df_melted['Date_Time'] = df_melted.apply(
            lambda row: row['Date'].replace(
                hour=pd.to_datetime(row['Time_Interval']).hour,
                minute=pd.to_datetime(row['Time_Interval']).minute
            ), axis=1
        )

        # Convert 'Traffic_Volume' to numeric. 'coerce' will turn non-numeric values into NaN.
        df_melted['Traffic_Volume'] = pd.to_numeric(df_melted['Traffic_Volume'], errors='coerce')

        # Drop the original 'Time_Interval_Code' as 'Time_Interval' and 'Date_Time' are more useful.
        self.df_processed = df_melted.drop(columns=['Time_Interval_Code'])

        print("\nDataFrame head after melting and datetime conversion:")
        print(self.df_processed.head())
        print("\nDataFrame info after melting and datetime conversion:")
        self.df_processed.info()
        return self.df_processed

    def engineer_features(self):
        """
        Adds time-based and lagged features to the DataFrame.
        Sorts data by SCATS Number and Date_Time for correct lag calculation.
        """
        if self.df_processed is None:
            print("Processed data not available. Please call transform_to_long_format() first.")
            return None

        print("\nEngineering time-based and lagged features...")
        # Ensure data is sorted for correct lag feature calculation.
        df = self.df_processed.sort_values(by=['SCATS Number', 'Date_Time']).reset_index(drop=True)

        # Time-based features extracted from 'Date_Time'
        df['hour'] = df['Date_Time'].dt.hour
        df['minute'] = df['Date_Time'].dt.minute # Could be useful for 15-min intervals
        df['day_of_week'] = df['Date_Time'].dt.dayofweek # Monday=0, Sunday=6
        df['day_of_year'] = df['Date_Time'].dt.dayofyear
        df['week_of_year'] = df['Date_Time'].dt.isocalendar().week.astype(int)
        df['month'] = df['Date_Time'].dt.month
        df['year'] = df['Date_Time'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) # 1 for Sat/Sun, 0 otherwise

        # Lagged features for Traffic_Volume
        # These are crucial for time-series prediction: predicting future based on past.
        # We use `groupby('SCATS Number')` to ensure lags are calculated independently
        # for each traffic sensor site, preventing data leakage across sites.

        # Lag 1: Traffic volume from the previous 15-minute interval
        df['traffic_volume_lag_1'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(1)

        # Lag 4: Traffic volume from 1 hour ago (4 * 15-minute intervals)
        df['traffic_volume_lag_4'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(4)

        # Lag 96: Traffic volume from 24 hours ago (96 * 15-minute intervals)
        # This captures daily periodicity in traffic patterns.
        df['traffic_volume_lag_96'] = df.groupby('SCATS Number')['Traffic_Volume'].shift(96)

        # Rolling Mean: Average traffic volume over the last 4 intervals (1 hour)
        df['traffic_volume_rolling_mean_4'] = df.groupby('SCATS Number')['Traffic_Volume'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
        )

        # Handle missing values created by shifting (the first few rows for each SCATS number)
        # For now, we'll drop them. In a real scenario, you might impute them or handle differently.
        initial_rows = len(df)
        df_features = df.dropna(subset=['traffic_volume_lag_96']) # Drop rows where 24-hour lag is NaN
        print(f"Dropped {initial_rows - len(df_features)} rows due to NaN values from lagged features.")

        self.df_final = df_features

        print("\nDataFrame head after feature engineering:")
        print(self.df_final.head())
        print("\nDataFrame info after feature engineering:")
        print(self.df_final.info())
        print("\nMissing values after feature engineering:")
        print(self.df_final.isnull().sum())

        return self.df_final

    def prepare_for_model(self, target_column='Traffic_Volume', features_to_scale=None):
        """
        Prepares the data for model training by splitting into features (X) and target (y),
        and scaling numerical features.
        """
        if self.df_final is None:
            print("Features not engineered. Please call engineer_features() first.")
            return None, None, None

        # Define features and target
        # Exclude identifier columns and the target itself from features
        # Include engineered features and relevant original numerical columns
        feature_columns = [
            'NB_LATITUDE', 'NB_LONGITUDE', 'HF VicRoads Internal',
            'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY',
            'hour', 'minute', 'day_of_week', 'day_of_year', 'week_of_year',
            'month', 'year', 'is_weekend',
            'traffic_volume_lag_1', 'traffic_volume_lag_4', 'traffic_volume_lag_96',
            'traffic_volume_rolling_mean_4'
        ]

        # Filter out any columns that might not exist in the dataframe
        feature_columns = [col for col in feature_columns if col in self.df_final.columns]

        X = self.df_final[feature_columns]
        y = self.df_final[target_column]

        # Scale numerical features
        if features_to_scale is None:
            # Default to scaling all numerical features identified
            features_to_scale = [col for col in feature_columns if X[col].dtype in ['int64', 'float64']]

        self.scaler = MinMaxScaler()
        X_scaled = X.copy()
        X_scaled[features_to_scale] = self.scaler.fit_transform(X[features_to_scale])

        print("\nFeatures (X) head after scaling:")
        print(X_scaled.head())
        print("\nTarget (y) head:")
        print(y.head())

        return X_scaled, y, self.scaler

    def chronological_train_test_split(self, X, y, test_size=0.2):
        """
        Performs a chronological train-test split for time-series data.

        Args:
            X (pd.DataFrame): Features DataFrame.
            y (pd.Series): Target Series.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if X is None or y is None:
            print("Data not prepared. Call prepare_for_model() first.")
            return None, None, None, None

        # Ensure X and y are aligned by index
        X = X.loc[y.index]

        # Calculate the split point based on the test_size
        split_index = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        print(f"\nChronological Train-Test Split:")
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
        print(f"Training period: {self.df_final['Date_Time'].iloc[0]} to {self.df_final['Date_Time'].iloc[split_index-1]}")
        print(f"Testing period: {self.df_final['Date_Time'].iloc[split_index]} to {self.df_final['Date_Time'].iloc[-1]}")

        return X_train, X_test, y_train, y_test

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Assuming 'Scats Data October 2006.csv' is in a 'data' directory relative to this script
    data_loader = TrafficDataLoader('Scats Data October 2006.csv')

    # Load and clean data
    data_loader.load_and_initial_clean()

    # Transform to long format
    data_loader.transform_to_long_format()

    # Engineer features
    data_loader.engineer_features()

    # Prepare data for model (scaling)
    X, y, scaler = data_loader.prepare_for_model()

    # Perform chronological train-test split
    X_train, X_test, y_train, y_test = data_loader.chronological_train_test_split(X, y)

    print("\nData loading and preprocessing complete. Ready for model training.")