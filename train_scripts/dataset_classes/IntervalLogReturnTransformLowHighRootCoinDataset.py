import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class IntervalLogReturnTransformLowHighRootCoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p, augmentation_noise_std, augmentation_constant_c, augmentation_scale_s, transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset):
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.df.loc[:, "BTC_low"] = np.log(self.df["BTC_low"] / self.df["BTC_open"])
        self.df.loc[:, "BTC_high"] = np.log(self.df["BTC_high"] / self.df["BTC_open"])
        self.df.loc[:, "BTC_close"] = np.log(self.df["BTC_close"] / self.df["BTC_open"])

        self.df.loc[:, "ETH_low"] = np.log(self.df["ETH_low"] / self.df["ETH_open"])
        self.df.loc[:, "ETH_high"] = np.log(self.df["ETH_high"] / self.df["ETH_open"])
        self.df.loc[:, "ETH_close"] = np.log(self.df["ETH_close"] / self.df["ETH_open"])

        self.df.loc[:, "BNB_low"] = np.log(self.df["BNB_low"] / self.df["BNB_open"])
        self.df.loc[:, "BNB_high"] = np.log(self.df["BNB_high"] / self.df["BNB_open"])
        self.df.loc[:, "BNB_close"] = np.log(self.df["BNB_close"] / self.df["BNB_open"])

        self.df.loc[:, "XRP_low"] = np.log(self.df["XRP_low"] / self.df["XRP_open"])
        self.df.loc[:, "XRP_high"] = np.log(self.df["XRP_high"] / self.df["XRP_open"])
        self.df.loc[:, "XRP_close"] = np.log(self.df["XRP_close"] / self.df["XRP_open"])

        btc_open = self.df["BTC_open"].values
        self.df.iloc[1:, self.df.columns.get_loc("BTC_open")] = np.log(btc_open[1:] / (btc_open[:-1]))
        eth_open = self.df["ETH_open"].values
        self.df.iloc[1:, self.df.columns.get_loc("ETH_open")] = np.log(eth_open[1:] / (eth_open[:-1]))
        bnb_open = self.df["BNB_open"].values
        self.df.iloc[1:, self.df.columns.get_loc("BNB_open")] = np.log(bnb_open[1:] / (bnb_open[:-1]))
        xrp_open = self.df["XRP_open"].values
        self.df.iloc[1:, self.df.columns.get_loc("XRP_open")] = np.log(xrp_open[1:] / (xrp_open[:-1]))

        self.df = self.df.iloc[1:]

        start, end  = {'BTC': (0, 4), 'ETH': (4, 8), 'BNB': (8, 12), 'XRP': (12, 16)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        cols_to_normalize = []
        for col in self.df.columns:
            if "low" in col or "high" in col:
                self.df[col] = np.sqrt(np.abs(self.df[col])) * np.sign(self.df[col])
            else:
                cols_to_normalize.append(col)

        if training_dataset:
            if transform_name == "QuantileTransformer":
                self.transform = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=42)
            elif transform_name == "PowerTransformer":
                self.transform = PowerTransformer(method="yeo-johnson")

            self.df_norm_cols = pd.DataFrame(self.transform.fit_transform(self.df[cols_to_normalize]), columns=cols_to_normalize, index=self.df.index)
            self.df[cols_to_normalize] = self.df_norm_cols

            joblib.dump(self.transform, os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
        else:
            self.transform = joblib.load( os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))

            self.df_norm_cols = pd.DataFrame(self.transform.transform(self.df[cols_to_normalize]), columns=cols_to_normalize, index=self.df.index)
            self.df[cols_to_normalize] = self.df_norm_cols

        self.input_window = input_window
        self.output_window = output_window

        self.augmentation_p = augmentation_p
        self.augmentation_noise_std = augmentation_noise_std
        self.augmentation_constant_c = augmentation_constant_c
        self.augmentation_scale_s = augmentation_scale_s

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
        # reverse the low high square rooting
        price_low = torch.tensor(np.square(price[:, 2]) * np.sign(price[:, 2]))
        price_high = torch.tensor(np.square(price[:, 3]) * np.sign(price[:, 3]))

        price_with_zero_cols_no_low_high = np.zeros((price.shape[0], 8))
        price_with_zero_cols_no_low_high[:, :2] = price[:, :2]
        price_with_zero_cols_inverted_no_low_high = self.transform.inverse_transform(price_with_zero_cols_no_low_high)
        price_with_zero_cols_inverted_only_coin = torch.tensor(price_with_zero_cols_inverted_no_low_high[:, :4])
        
        # till here, we didn't do anything with the low high. "torch.tensor(price_with_zero_cols_inverted_no_low_high[:, :4])" line gives extra 2 columns
        # fill this extra 2 columns with low and high squared
        price_with_zero_cols_inverted_only_coin[:, 2] = price_low
        price_with_zero_cols_inverted_only_coin[:, 3] = price_high

        real_price = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]))
        real_price[0] = initial_prices

        for t in range(price_with_zero_cols_inverted_only_coin.shape[0]):
            real_price[t + 1, 0] = real_price[t, 0] * torch.exp(price_with_zero_cols_inverted_only_coin[t, 0])
            real_price[t + 1, 1:] = real_price[t + 1, 0] * torch.exp(price_with_zero_cols_inverted_only_coin[t, 1:])
        real_price = real_price[1:]

        return real_price

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augmentation_constant_c, self.augmentation_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augmentation_scale_s, self.augmentation_scale_s))

        # low is always negative, high is always positive
        low_cols = [2, 6, 10 ,14]
        high_cols = [3, 7, 11 ,15]

        for low_col in low_cols:
            x[low_col] = np.clip(x[low_col], a_min=min(x[low_col]), a_max=0)
        for high_col in high_cols:
            x[high_col] = np.clip(x[high_col], a_min=0, a_max=max(x[high_col]))

        # cloe is always in between low and high
        close_cols = [1, 5, 9, 13]
        for close_col in close_cols:
            x[close_col] = np.clip(x[close_col], a_min=x[close_col+1], a_max=x[close_col+2])

        return x