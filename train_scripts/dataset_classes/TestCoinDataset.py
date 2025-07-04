import pandas as pd

import torch
from torch.utils.data import Dataset

class TestCoinDataset(Dataset):
    def __init__(self, csv_path, input_coins, input_features, output_coins, output_features, input_window, output_window):
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.input_cols = [f'{c}_{f}' for c in input_coins for f in input_features]
        self.output_cols = [f'{c}_{f}' for c in output_coins for f in output_features]

        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.df) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        analysis_rows = self.df.iloc[idx:idx + self.input_window]
        prediction_rows = self.df.iloc[idx + self.input_window:idx + self.input_window + self.output_window]

        # first 4 columns are BTC_open/close/low_high, and then same 4 for each ETH, BNB, XRP. Each column is a timestamp
        analysis_matrix = analysis_rows[self.input_cols].to_numpy()
        prediction_target = prediction_rows[self.output_cols].to_numpy()

        x, y = analysis_matrix.T, prediction_target.T

        return torch.tensor(x, dtype=torch.float32) / 100, torch.tensor(y, dtype=torch.float32) / 100