import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class LogZNormCoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p, augmentation_noise_std, augment_constant_c, augment_scale_s, train_session_dir, training_dataset):
        print("THIS DATASET IS NOT RECOMMENDED, IT CREATES A DISTRIBUTION SHIFT BETWEEN TRAINING AND VALIDATION PARTITIONS SINCE COIN PRICES ARE SO HIGH TOWARDS THE END")
        raise ZeroDivisionError("You should not use this dataset")
        self.df = pd.read_csv(csv_path, index_col="open_time")
        
        if training_dataset:
            self.df = np.log(self.df + 0.00000001)
            train_col_means, train_col_stds = self.df.mean(), self.df.std()
            self.df = (self.df - train_col_means) / (train_col_stds + 0.00000001)
            
            self.train_col_means_df = pd.DataFrame(train_col_means).T
            self.train_col_stds_df = pd.DataFrame(train_col_stds).T
            self.train_col_means_df.to_csv(os.path.join(train_session_dir, "z_norm_means.csv"), index=False)
            self.train_col_stds_df.to_csv(os.path.join(train_session_dir, "z_norm_stds.csv"), index=False)
        else:
            self.train_col_means_df = pd.read_csv(os.path.join(train_session_dir, "z_norm_means.csv"))
            self.train_col_stds_df = pd.read_csv(os.path.join(train_session_dir, "z_norm_stds.csv"))

            self.df = np.log(self.df + 0.00000001)
            self.df.sub(self.train_col_means_df.iloc[0, :]).div(self.train_col_stds_df + 0.00000001)

        # first column is open_time, so skip it
        start, end  = {'BTC': (0, 4), 'ETH': (4, 8), 'BNB': (8, 12), 'XRP': (12, 16)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        self.input_window = input_window
        self.output_window = output_window

        self.augmentation_p = augmentation_p
        self.augmentation_noise_std = augmentation_noise_std
        self.augment_constant_c = augment_constant_c
        self.augment_scale_s = augment_scale_s

    def __len__(self):
        return len(self.df) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        analysis_rows = self.df.iloc[idx : idx + self.input_window]
        prediction_rows = self.df.iloc[idx + self.input_window : idx + self.input_window + self.output_window]

        # first 4 columns are BTC_open/close/low_high, and then same 4 for each ETH, BNB, XRP. Each column is a timestamp
        analysis_matrix = analysis_rows[analysis_rows.columns].to_numpy()
        prediction_target = prediction_rows[self.coin_cols].to_numpy()

        x,y = analysis_matrix.T, prediction_target.T

        if np.random.rand() < self.augmentation_p:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def rescale_to_real_price(self, price):
        means = self.train_col_means_df[self.coin_cols].to_numpy()
        stds = self.train_col_stds_df[self.coin_cols].to_numpy()
        
        real_price = np.exp(price * stds + means)

        return real_price

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(scale=self.augmentation_noise_std, size=x.shape)

            # this dict explains the low <= close & open <= high logic for each coin
            clip_rules = {(0,1): (2, 3), (4,5): (6, 7), (8,9): (10, 11), (12,13): (14, 15)}

            for ((open_row, close_row), (low_row, high_row)) in clip_rules.items():
                x[open_row] = np.clip(x[open_row], x[low_row], x[high_row])
                x[close_row] = np.clip(x[close_row], x[low_row], x[high_row])
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augment_constant_c, self.augment_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1 + np.random.uniform(-self.augment_scale_s, self.augment_scale_s))

        return x