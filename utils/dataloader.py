import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_traffic_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    return df

def scale_data(df, column='volume'):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler

def create_sequences(series, window_size=4):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return X, y