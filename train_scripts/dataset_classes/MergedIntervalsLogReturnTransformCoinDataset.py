import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class MergedIntervalsLogReturnTransformCoinDataset(Dataset):
    def __init__(self, csv_path, coin_symbol, input_window, output_window, augmentation_p, augmentation_noise_std, augment_constant_c, augment_scale_s, transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset):
        self.df = pd.read_csv(csv_path, index_col="open_time")

        coins = ['BTC', 'ETH', 'BNB', 'XRP']

        for coin in coins:
            lows = self.df[f'{coin}_low'].values
            highs = self.df[f'{coin}_high'].values

            trajectory = []
            stability = []

            for i in range(len(self.df)):
                row_i = len(self.df) -1 - i
                if row_i == 0:
                    break
                if i % 2 == 1:
                    trajectory.append(None)
                    stability.append(None)
                    continue
                
                low1, high1 = lows[row_i-1], highs[row_i-1]
                low2, high2 = lows[row_i], highs[row_i]

                lowest_low = min(low1, low2)
                highest_high = max(high1, high2)

                if lowest_low == low1 and highest_high == high2:
                    traj = 1 # going up
                elif lowest_low == low2 and highest_high == high1:
                    traj = -1 # going down
                else:
                    traj = 0 # no info

                if high1 - low1 > high2 - low2:
                    stab = -1 # less stabil
                else:
                    stab = 1 # more stabil

                trajectory.append(traj)
                stability.append(stab)

            # None to match original length
            while len(trajectory) < len(self.df):
                trajectory.append(None)
            while len(stability) < len(self.df):
                stability.append(None)

            self.df[f'{coin}_trajectory'] = list(reversed(trajectory))
            self.df[f'{coin}_stability'] = list(reversed(stability))

        coin_cols = [[f"{coin}_{feature}" for feature in ["open", "close", "low", "high", "trajectory", "stability"]] for coin in coins]
        coin_cols = [x for xs in coin_cols for x in xs]
        self.df = self.df[coin_cols].dropna(axis=0, how='any')

        price_cols = [col for col in self.df.columns if not ("trajectory" in col or "stability" in col)]

        self.df[price_cols] = np.log(self.df[price_cols] / self.df[price_cols].shift(1))
        self.df.dropna(inplace=True)

        start, end  = {'BTC': (0, 4), 'ETH': (6, 10), 'BNB': (12, 16), 'XRP': (18, 22)}[coin_symbol]
        self.coin_cols = self.df.columns[start: end]

        if training_dataset:
            if transform_name == "QuantileTransformer":
                self.transform = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=42)
            elif transform_name == "PowerTransformer":
                self.transform = PowerTransformer(method="yeo-johnson")

            self.df[price_cols] = pd.DataFrame(self.transform.fit_transform(self.df[price_cols]), columns=price_cols, index=self.df.index)
            joblib.dump(self.transform, os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
        else:
            self.transform = joblib.load( os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
            self.df[price_cols] = pd.DataFrame(self.transform.transform(self.df[price_cols]), columns=price_cols, index=self.df.index)

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

        analysis_matrix = analysis_rows[analysis_rows.columns].to_numpy()
        prediction_target = prediction_rows[self.coin_cols].to_numpy()

        x, y = analysis_matrix.T, prediction_target.T

        if np.random.rand() < self.augmentation_p:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def rescale_to_real_price(self, price, initial_prices):
        price_with_zero_cols = np.zeros((price.shape[0], 16))
        price_with_zero_cols[:, :4] = price
        price_with_zero_cols_inverted = self.transform.inverse_transform(price_with_zero_cols)
        price_with_zero_cols_inverted_only_coin = torch.tensor(price_with_zero_cols_inverted[:, :4])

        real_price = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]))
        real_price[0] = initial_prices

        for t in range(price_with_zero_cols_inverted_only_coin.shape[0]):
            real_price[t + 1] = real_price[t] * torch.exp(price_with_zero_cols_inverted_only_coin[t])
        real_price = real_price[1:]

        return real_price

    def augment(self, x):
        x_orig = x.copy()
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augment_constant_c, self.augment_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augment_scale_s, self.augment_scale_s))
        
        # no augmentation to the trajectory, stability columns
        x[4] = x_orig[4]
        x[5] = x_orig[5]
        x[10] = x_orig[10]
        x[11] = x_orig[11]
        x[16] = x_orig[16]
        x[17] = x_orig[17]
        x[22] = x_orig[22]
        x[23] = x_orig[23]
        
        return x