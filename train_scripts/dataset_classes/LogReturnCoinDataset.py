import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class LogReturnCoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p, augmentation_noise_std, augment_constant_c, augment_scale_s, distribution_scale, distribution_clip):
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.df = np.log(self.df / self.df.shift(1))
        self.df.dropna(inplace=True)

        self.df = self.df * distribution_scale
        self.df = self.df.clip(-distribution_clip, distribution_clip)

        # first column is open_time, so skip it
        start, end  = {'BTC': (0, 4), 'ETH': (4, 8), 'BNB': (8, 12), 'XRP': (12, 16)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        self.input_window = input_window
        self.output_window = output_window

        self.augmentation_p = augmentation_p
        self.augmentation_noise_std = augmentation_noise_std
        self.augment_constant_c = augment_constant_c
        self.augment_scale_s = augment_scale_s
        self.distribution_scale = distribution_scale

    def __len__(self):
        return len(self.df) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        analysis_rows = self.df.iloc[idx:idx + self.input_window]
        prediction_rows = self.df.iloc[idx + self.input_window:idx + self.input_window + self.output_window]

        # first 4 columns are BTC_open/close/low_high, and then same 4 for each ETH, BNB, XRP. Each column is a timestamp
        analysis_matrix = analysis_rows[analysis_rows.columns].to_numpy()
        prediction_target = prediction_rows[self.coin_cols].to_numpy()

        x, y = analysis_matrix.T, prediction_target.T

        if np.random.rand() < self.augmentation_p:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def rescale_to_real_price(self, price, initial_prices):
        price = torch.tensor(price / self.distribution_scale)
        
        real_price = torch.zeros((price.shape[0] + 1, price.shape[1]))
        real_price[0] = initial_prices

        for t in range(price.shape[0]):
            real_price[t + 1] = real_price[t] * torch.exp(price[t])
        real_price = real_price[1:]

        return real_price

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augment_constant_c, self.augment_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augment_scale_s, self.augment_scale_s))

        return x