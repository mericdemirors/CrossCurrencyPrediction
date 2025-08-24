import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class LogReturnTransformCoinDataset(Dataset):
    def __init__(self, csv_path, input_coins, input_features, output_coins, output_features, input_window, output_window,
                 transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset,
                 augmentation_p, augmentation_noise_std, augmentation_constant_c, augmentation_scale_s):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.df = np.log(self.df / self.df.shift(1))
        self.df.dropna(inplace=True)

        self.input_cols = [f'{c}_{f}' for c in input_coins for f in input_features]
        self.output_cols = [f'{c}_{f}' for c in output_coins for f in output_features]
        self.input_col_indices = [list(self.df.columns).index(col) for col in self.input_cols]
        self.output_col_indices = [list(self.df.columns).index(col) for col in self.output_cols]

        if training_dataset:
            if transform_name == "QuantileTransformer":
                self.transform = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=42)
            elif transform_name == "PowerTransformer":
                self.transform = PowerTransformer(method="yeo-johnson")

            self.df = pd.DataFrame(self.transform.fit_transform(self.df.values), columns=self.df.columns, index=self.df.index)
            joblib.dump(self.transform, os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
        else:
            self.transform = joblib.load(os.path.join(train_session_dir,f'dataset_{transform_name}.pkl'))
            self.df = pd.DataFrame(self.transform.transform(self.df.values), columns=self.df.columns, index=self.df.index)

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
        analysis_matrix = analysis_rows[self.input_cols].to_numpy()
        prediction_target = prediction_rows[self.output_cols].to_numpy()

        x, y = analysis_matrix.T, prediction_target.T

        if np.random.rand() < self.augmentation_p:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def rescale_to_real_price(self, price, initial_prices, profit_inference=False):
        price_with_zero_cols = np.zeros((price.shape[0], len(self.df.columns)))
        price_with_zero_cols[:, self.output_col_indices] = price
        price_with_zero_cols_inverted = self.transform.inverse_transform(price_with_zero_cols)
        price_with_zero_cols_inverted_only_coin = torch.tensor(price_with_zero_cols_inverted[:, self.output_col_indices])

        real_price_based_on_only_predictions = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]))
        real_price_based_on_real_data = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]))
        real_price_based_on_only_predictions[0] = initial_prices.float()
        real_price_based_on_real_data[0] = initial_prices.float()

        df = pd.read_csv(self.csv_path, index_col="open_time")
        matches = np.all(np.isclose(df[self.output_cols].values, initial_prices.numpy(), atol=1e-4), axis=1)
        initial_prices_index = np.where(matches)[0]

        # if we are doing profit inference we need some predictions in the future that's data not in the df
        # so we can't relate them to the real data we can only base them on the predictions
        if profit_inference:
            real_price_based_on_only_predictions = initial_prices * np.exp(np.cumsum(price_with_zero_cols_inverted_only_coin, axis=0))
            return real_price_based_on_only_predictions
    
        real_data_to_relate = torch.tensor(df[self.output_cols].iloc[initial_prices_index.item(): (initial_prices_index+price_with_zero_cols_inverted_only_coin.shape[0]).item()].values).squeeze().float()
        real_price_based_on_real_data = real_data_to_relate * torch.exp(price_with_zero_cols_inverted_only_coin)

        real_price_based_on_only_predictions = initial_prices * np.exp(np.cumsum(price_with_zero_cols_inverted_only_coin, axis=0))
        # for t in range(price_with_zero_cols_inverted_only_coin.shape[0]):
        #     real_price_based_on_only_predictions[t + 1] = real_price_based_on_only_predictions[t] * torch.exp(price_with_zero_cols_inverted_only_coin[t])
            
        #     real_data_to_relate = torch.tensor(df[self.output_cols].iloc[initial_prices_index + t].values).squeeze().float()
        #     real_price_based_on_real_data[t + 1] = real_data_to_relate * torch.exp(price_with_zero_cols_inverted_only_coin[t])

        return real_price_based_on_real_data, real_price_based_on_only_predictions

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augmentation_constant_c, self.augmentation_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augmentation_scale_s, self.augmentation_scale_s))

        return x