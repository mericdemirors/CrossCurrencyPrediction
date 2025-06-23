import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer

class LogReturnTransformCoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p, augmentation_noise_std, augment_constant_c, augment_scale_s, output_distribution, n_quantiles, transform, train_session_dir):
        self.df = pd.read_csv(csv_path)

        # first column is open_time, so skip it
        start, end  = {'BTC': (1, 5), 'ETH': (5, 9), 'BNB': (9, 13), 'XRP': (13, 17)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        if transform == 0:
            self.transform = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=42)
            self.df[self.df.columns[1:]] = pd.DataFrame(self.transform.fit_transform(self.df[self.df.columns[1:]]), columns=self.df[self.df.columns[1:]].columns, index=self.df[self.df.columns[1:]].index)
            joblib.dump(self.transform, os.path.join(train_session_dir,"dataset_transformer.pkl"))
        else:
            self.transform = joblib.load( os.path.join(train_session_dir,"dataset_transformer.pkl"))
            self.df[self.df.columns[1:]] = pd.DataFrame(self.transform.transform(self.df[self.df.columns[1:]]), columns=self.df[self.df.columns[1:]].columns, index=self.df[self.df.columns[1:]].index)

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
        analysis_matrix = analysis_rows[analysis_rows.columns[1:]].to_numpy()
        prediction_target = prediction_rows[self.coin_cols].to_numpy()

        x, y = analysis_matrix.T, prediction_target.T

        if np.random.rand() < self.augmentation_p:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def rescale_to_real_price(self, price):           
        y_pred_full = np.zeros((16, price.shape[1]))
        y_pred_full[:4, :] = price
        y_pred_inversed = self.transform.inverse_transform(y_pred_full.T)
        y_pred_rescaled = y_pred_inversed[:, :4]

        return y_pred_rescaled

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augment_constant_c, self.augment_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augment_scale_s, self.augment_scale_s))

        return x