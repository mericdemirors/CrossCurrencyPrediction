import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class MergedIntervalsLogReturnTransformCoinDataset(Dataset):
    def __init__(self, csv_path, input_coins, input_features, output_coins, output_features, input_window, output_window,
                 merge_count, transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset,
                 augmentation_p, augmentation_noise_std, augmentation_constant_c, augmentation_scale_s):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.merge_count = merge_count
        merges = []
        for i in range(merge_count):
            merge_df = self.df.iloc[range(0, len(self.df), merge_count)]
            merge_df = merge_df.reset_index().add_suffix("_" + str(i))
            merges.append(merge_df)

        self.df = pd.concat(merges, axis=1)
        self.df.dropna(inplace=True)
        self.df = self.df.rename(columns={"open_time_0": "open_time"})
        self.df = self.df.set_index("open_time")
        self.df = self.df.drop(columns=[col for col in self.df.columns if "open_time" in col])

        self.input_coins = input_coins
        self.output_coins = output_coins
        for i in range(1, merge_count):
            for c in list(set(input_coins + output_coins)):
                self.df[f"{c}_open_{i}"] = (self.df[f"{c}_close_{i-1}"] + self.df[f"{c}_open_{i}"]) / 2
                self.df = self.df.drop(columns=[f"{c}_close_{i-1}"])

        new_order = sorted(self.df.columns, key= lambda x: x.split("_")[0])
        self.df = self.df[new_order]
        
        self.df = np.log(self.df / self.df.shift(1))
        self.df.dropna(inplace=True)

        # input_features =[f for f in input_features for i in range(merge_count)]
        self.input_cols = [f'{c}_{f}' for c in input_coins for f in input_features]
        self.input_cols = [col for col in self.df.columns for input_col in self.input_cols if input_col in col]
        self.output_cols = [f'{c}_{f}' for c in output_coins for f in output_features]
        self.output_cols = [col for col in self.df.columns for output_col in self.output_cols if output_col in col]
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

    def rescale_to_real_price(self, price, initial_prices):
        price_with_zero_cols = np.zeros((price.shape[0], len(self.df.columns)))
        price_with_zero_cols[:, self.output_col_indices] = price
        price_with_zero_cols_inverted = self.transform.inverse_transform(price_with_zero_cols)
        price_with_zero_cols_inverted_only_coin = torch.tensor(price_with_zero_cols_inverted[:, self.output_col_indices])

        real_price_based_on_only_predictions = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]))
        real_price_based_on_real_data = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]))
        real_price_based_on_only_predictions[0] = initial_prices.float()
        real_price_based_on_real_data[0] = initial_prices.float()

        df = pd.read_csv(self.csv_path, index_col="open_time")
        df = self.set_raw_dataset_for_evaluation(df)
        matches = np.all(np.isclose(df[self.output_cols].values, initial_prices.numpy(), atol=1e-4), axis=1)
        initial_prices_index = np.where(matches)[0]

        real_data_to_relate = torch.tensor(df[self.output_cols].iloc[initial_prices_index.item(): (initial_prices_index+price_with_zero_cols_inverted_only_coin.shape[0]).item()].values).squeeze().float()
        real_price_based_on_real_data = real_data_to_relate * torch.exp(price_with_zero_cols_inverted_only_coin)

        real_price_based_on_only_predictions = initial_prices * np.exp(np.cumsum(price_with_zero_cols_inverted_only_coin, axis=0))
        # for t in range(price_with_zero_cols_inverted_only_coin.shape[0]):
        #     real_price_based_on_only_predictions[t + 1] = real_price_based_on_only_predictions[t] * torch.exp(price_with_zero_cols_inverted_only_coin[t])
            
        #     real_data_to_relate = torch.tensor(df[self.output_cols].iloc[initial_prices_index + t].values).squeeze().float()
        #     real_price_based_on_real_data[t + 1] = real_data_to_relate * torch.exp(price_with_zero_cols_inverted_only_coin[t])

        return real_price_based_on_real_data, real_price_based_on_only_predictions

    def set_raw_dataset_for_evaluation(self, raw_df):
        merges = []
        for i in range(self.merge_count):
            merge_df = raw_df.iloc[range(0, len(raw_df), self.merge_count)]
            merge_df = merge_df.reset_index().add_suffix("_" + str(i))
            merges.append(merge_df)

        raw_df = pd.concat(merges, axis=1)
        raw_df.dropna(inplace=True)
        raw_df = raw_df.rename(columns={"open_time_0": "open_time"})
        raw_df = raw_df.set_index("open_time")
        raw_df = raw_df.drop(columns=[col for col in raw_df.columns if "open_time" in col])

        for i in range(1, self.merge_count):
            for c in list(set(self.input_coins + self.output_coins)):
                raw_df[f"{c}_open_{i}"] = (raw_df[f"{c}_close_{i-1}"] + raw_df[f"{c}_open_{i}"]) / 2
                raw_df = raw_df.drop(columns=[f"{c}_close_{i-1}"])

        new_order = sorted(raw_df.columns, key= lambda x: x.split("_")[0])
        raw_df = raw_df[new_order]
        return raw_df

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augmentation_constant_c, self.augmentation_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augmentation_scale_s, self.augmentation_scale_s))

        return x