import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class LogReturnTransformCoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p, augmentation_noise_std, augment_constant_c, augment_scale_s, transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset):
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.df = np.log(self.df / self.df.shift(1))
        self.df.dropna(inplace=True)

        # first column is open_time, so skip it
        start, end  = {'BTC': (0, 4), 'ETH': (4, 8), 'BNB': (8, 12), 'XRP': (12, 16)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        if training_dataset:
            if transform_name == "QuantileTransformer":
                self.transform = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=42)
            elif transform_name == "PowerTransformer":
                self.transform = PowerTransformer(method="yeo-johnson")

            self.df = pd.DataFrame(self.transform.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
            joblib.dump(self.transform, os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
        else:
            self.transform = joblib.load( os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
            self.df = pd.DataFrame(self.transform.transform(self.df), columns=self.df.columns, index=self.df.index)

        self.input_window = input_window
        self.output_window = output_window

        self.augmentation_p = augmentation_p
        self.augmentation_noise_std = augmentation_noise_std
        self.augment_constant_c = augment_constant_c
        self.augment_scale_s = augment_scale_s

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
        y_pred_full = np.zeros((price.shape[0], 16))
        y_pred_full[:, :4] = price
        y_pred_inversed = self.transform.inverse_transform(y_pred_full)
        y_pred_rescaled = torch.tensor(y_pred_inversed[:, :4])

        price_full = np.zeros((price.shape[0], 16))
        price_full[:, :4] = price
        
        price_full_inverted = self.transform.inverse_transform(price_full)
        price_inverted = torch.tensor(price_full_inverted[:, :4])

        real_price = torch.zeros((price_inverted.shape[0] + 1, price_inverted.shape[1]))
        real_price[0] = initial_prices

        for t in range(y_pred_rescaled.shape[0]):
            real_price[t + 1] = real_price[t] * torch.exp(y_pred_rescaled[t])
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