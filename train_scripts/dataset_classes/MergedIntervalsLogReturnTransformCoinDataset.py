import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class MergedIntervalsLogReturnTransformCoinDataset(Dataset):
    def __init__(self, csv_path, input_coins, input_features, output_coins, output_features, input_window, output_window,
                 transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset,
                 augmentation_p, augmentation_noise_std, augmentation_constant_c, augmentation_scale_s):
                
        self.df = pd.read_csv(csv_path, index_col="open_time")

        coins = list(set(input_coins).union(set(output_coins)))
        features = list(set(input_features).union(set(output_features)))
        
        for coin in coins:
            open_data = self.df[f'{coin}_open'].values
            close_data = self.df[f'{coin}_close'].values
            low_data = self.df[f'{coin}_low'].values
            high_col = self.df[f'{coin}_high'].values

            opens = []
            mids = []
            closes = []
            lows = []
            secondary_lows = []
            highs = []
            secondary_highs = []
            trajectory = []
            stability = []

            for i in range(len(self.df)):
                row_i = len(self.df) -1 - i
                if row_i == 0:
                    break
                if i % 2 == 1:
                    opens.append(None)
                    mids.append(None)
                    closes.append(None)
                    lows.append(None)
                    highs.append(None)
                    secondary_lows.append(None)
                    secondary_highs.append(None)
                    trajectory.append(None)
                    stability.append(None)
                    continue
                
                low1, high1 = low_data[row_i-1], high_col[row_i-1]
                low2, high2 = low_data[row_i], high_col[row_i]

                lowest_low = min(low1, low2)
                highest_high = max(high1, high2)

                if lowest_low == low1 and highest_high == high2:
                    traj = highest_high - lowest_low # going up
                elif lowest_low == low2 and highest_high == high1:
                    traj = lowest_low - highest_high # going down
                else:
                    traj = 0 # no info

                if high1 - low1 > high2 - low2:
                    stab = (high1 - low1) - (high2 - low2) # more stabil
                else:
                    stab = (high1 - low1) - (high2 - low2) # less stabil
                
                opens.append(open_data[row_i-1])
                mids.append((close_data[row_i-1] + open_data[row_i])/2)
                closes.append(close_data[row_i])
                lows.append(lowest_low)
                highs.append(highest_high)
                secondary_lows.append(low1 + low2 - lowest_low)
                secondary_highs.append(high1 + high2 - highest_high)
                trajectory.append(traj)
                stability.append(stab)

            # None to match original length
            while len(opens) < len(self.df):
                opens.append(None)
            while len(mids) < len(self.df):
                mids.append(None)
            while len(closes) < len(self.df):
                closes.append(None)
            while len(lows) < len(self.df):
                lows.append(None)
            while len(highs) < len(self.df):
                highs.append(None)
            while len(secondary_lows) < len(self.df):
                secondary_lows.append(None)
            while len(secondary_highs) < len(self.df):
                secondary_highs.append(None)
            while len(trajectory) < len(self.df):
                trajectory.append(None)
            while len(stability) < len(self.df):
                stability.append(None)

            self.df[f'{coin}_open'] = list(reversed(opens))
            self.df[f'{coin}_mid'] = list(reversed(mids))
            self.df[f'{coin}_close'] = list(reversed(closes))
            self.df[f'{coin}_low1'] = list(reversed(lows))
            self.df[f'{coin}_high1'] = list(reversed(highs))
            self.df[f'{coin}_low2'] = list(reversed(secondary_lows))
            self.df[f'{coin}_high2'] = list(reversed(secondary_highs))
            self.df[f'{coin}_trajectory'] = list(reversed(trajectory))
            self.df[f'{coin}_stability'] = list(reversed(stability))

        coin_cols = [[f"{coin}_{feature}" for feature in features] for coin in coins] # = [[BTC cols...], [ETH cols...], ...]
        coin_cols = [x for xs in coin_cols for x in xs] # = [BTC cols..., ETH cols..., ...] puts everything in one big list
        self.df = self.df[coin_cols].dropna(axis=0, how='any')

        self.input_cols = [f'{c}_{f}' for c in input_coins for f in input_features]
        self.output_cols = [f'{c}_{f}' for c in output_coins for f in output_features]
        self.input_col_indices = [list(self.df.columns).index(col) for col in self.input_cols]
        self.output_col_indices = [list(self.df.columns).index(col) for col in self.output_cols]

        traj_stab_cols = [col for col in self.df.columns if "trajectory" in col or "stability" in col]
        price_cols = [col for col in self.df.columns if col not in traj_stab_cols]

        self.df[price_cols] = np.log(self.df[price_cols] / self.df[price_cols].shift(1))
        self.df.dropna(inplace=True)
        self.df_copy_for_traj_stab = self.df.copy()

        if training_dataset:
            if transform_name == "QuantileTransformer":
                self.transform = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=42)
            elif transform_name == "PowerTransformer":
                self.transform = PowerTransformer(method="yeo-johnson")

            self.df = pd.DataFrame(self.transform.fit_transform(self.df.values), columns=self.df.columns, index=self.df.index)
            joblib.dump(self.transform, os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
        else:
            self.transform = joblib.load( os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
            self.df = pd.DataFrame(self.transform.transform(self.df.values), columns=self.df.columns, index=self.df.index)

        self.df[traj_stab_cols] = self.df_copy_for_traj_stab[traj_stab_cols]

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

        analysis_matrix = analysis_rows[self.input_cols].to_numpy()
        prediction_target = prediction_rows[self.output_cols].to_numpy()

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
            x = x + np.random.uniform(-self.augmentation_constant_c, self.augmentation_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augmentation_scale_s, self.augmentation_scale_s))
        
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