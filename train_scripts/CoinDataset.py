import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class CoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p = 0.2, augmentation_noise_std=0.01, augment_constant_c=1, augment_scale_s=0.25, z_norm_means_csv_path="", z_norm_stds_csv_path=""):
        self.df = pd.read_csv(csv_path)
        
        # first column is open_time, so skip it
        start, end  = {'BTC': (1, 5), 'ETH': (5, 9), 'BNB': (9, 13), 'XRP': (13, 17)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        self.input_window = input_window
        self.output_window = output_window

        self.augmentation_p = augmentation_p
        self.augmentation_noise_std = augmentation_noise_std
        self.augment_constant_c = augment_constant_c
        self.augment_scale_s = augment_scale_s

        self.z_norm_means_df = pd.read_csv(z_norm_means_csv_path)
        self.z_norm_stds_df = pd.read_csv(z_norm_stds_csv_path)

    def __len__(self):
        return len(self.df) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        analysis_rows = self.df.iloc[idx : idx + self.input_window]
        prediction_rows = self.df.iloc[idx + self.input_window : idx + self.input_window + self.output_window]

        # first 4 columns are BTC_open/close/low_high, and then same 4 for each ETH, BNB, XRP. Each column is a timestamp
        analysis_matrix = analysis_rows[analysis_rows.columns[1:]].to_numpy()
        prediction_target = prediction_rows[self.coin_cols].to_numpy()

        x,y = analysis_matrix.T, prediction_target.T

        if np.random.rand() < self.augmentation_p:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def rescale_to_real_price(self, price):
        means = self.z_norm_means_df[self.coin_cols].to_numpy()
        stds = self.z_norm_stds_df[self.coin_cols].to_numpy()
        
        real_price = np.power(10, price.T * stds + means)        
        
        return real_price.T
    
    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x_aug = x + np.random.normal(scale=self.augmentation_noise_std, size=x.shape)

            # this dict explains the low <= close & open <= high logic for each coin
            clip_rules = {(0,1): (2, 3), (4,5): (6, 7), (8,9): (10, 11), (12,13): (14, 15)}

            for ((open_row, close_row), (low_row, high_row)) in clip_rules.items():
                x_aug[open_row] = np.clip(x_aug[open_row], x_aug[low_row], x_aug[high_row])
                x_aug[close_row] = np.clip(x_aug[close_row], x_aug[low_row], x_aug[high_row])
        if torch.rand(1) < self.augmentation_p:
            x_aug = x + np.random.uniform(-self.augment_constant_c, self.augment_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x_aug = x * (1 + np.random.uniform(-self.augment_scale_s, self.augment_scale_s))

        return x_aug