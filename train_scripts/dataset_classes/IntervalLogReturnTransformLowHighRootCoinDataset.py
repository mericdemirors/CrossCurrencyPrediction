import os
import joblib
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

class IntervalLogReturnTransformLowHighRootCoinDataset(Dataset):
    def __init__(self, csv_path, input_coins, input_features, output_coins, output_features, input_window, output_window,
                 transform_name, output_distribution, n_quantiles, train_session_dir, training_dataset,
                 augmentation_p, augmentation_noise_std, augmentation_constant_c, augmentation_scale_s):
        self.df = pd.read_csv(csv_path, index_col="open_time")

        self.csv_path = csv_path
        self.input_features = input_features
        self.output_coins = output_coins
        self.input_cols = [f'{c}_{f}' for c in input_coins for f in input_features]
        self.output_features = output_features
        self.output_cols = [f'{c}_{f}' for c in output_coins for f in output_features]
        self.input_col_indices = [list(self.df.columns).index(col) for col in self.input_cols]
        self.output_col_indices = [list(self.df.columns).index(col) for col in self.output_cols]

        cols_to_process = set(self.input_cols).union(set(self.output_cols))
        for col in cols_to_process:
            [c,f] = col.split("_")
            if f == "open":
                continue
            else:
                self.df.loc[:, f'{c}_{f}'] = np.log(self.df[f'{c}_{f}'] / self.df[f'{c}_open'])
        for col in cols_to_process:
            [c,f] = col.split("_")
            if f == "open":
                c_open = self.df[f'{c}_open'].values
                self.df.iloc[1:, self.df.columns.get_loc(f'{c}_open')] = np.log(c_open[1:] / (c_open[:-1]))
            else:
                continue

        self.low_high_cols_to_root = [col for col in cols_to_process if "low" in col or "high" in col]
        for col in self.low_high_cols_to_root:
            self.df[col] = np.sqrt(np.abs(self.df[col])) * np.sign(self.df[col])

        self.df = self.df.iloc[1:]
        self.df_copy_for_low_high_roots = self.df.copy()

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

        self.df[self.low_high_cols_to_root] = self.df_copy_for_low_high_roots[self.low_high_cols_to_root]

        self.low_high_cols_in_output = [col for col in self.output_cols if "low" in col or "high" in col]
        self.low_high_col_indices_in_output = [self.output_cols.index(col) for col in self.low_high_cols_in_output]

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
        # reverse the low high square rooting
        low_high_prices = price[:, self.low_high_col_indices_in_output]
        low_high_prices = torch.square(low_high_prices) * torch.sign(low_high_prices)

        price_with_zero_cols = np.zeros((price.shape[0], len(self.df.columns)))
        price_with_zero_cols[:, self.output_col_indices] = price
        price_with_zero_cols_inverted = self.transform.inverse_transform(price_with_zero_cols)
        price_with_zero_cols_inverted_only_coin = torch.tensor(price_with_zero_cols_inverted[:, self.output_col_indices]).float()

        # till here, we didn't do anything with the low high
        price_with_zero_cols_inverted_only_coin[:, self.low_high_col_indices_in_output] = low_high_prices.float()

        real_price_based_on_only_predictions = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]), dtype=torch.float)
        real_price_based_on_real_data = torch.zeros((price_with_zero_cols_inverted_only_coin.shape[0] + 1, price_with_zero_cols_inverted_only_coin.shape[1]), dtype=torch.float)
        real_price_based_on_only_predictions[0] = initial_prices.float()
        real_price_based_on_real_data[0] = initial_prices.float()

        df = pd.read_csv(self.csv_path, index_col="open_time")

        # get the output columns' indices without open feature
        nonopen_column_indices_in_output_columns = [self.output_cols.index(col) for col in self.output_cols if "open" not in col]

        # if there is no open feature, we need to fine the open prices of the coins from the dataset
        matches = np.all(np.isclose(df[self.df.columns[self.output_col_indices]].values, initial_prices.numpy(), atol=1e-4), axis=1)
        matching_row_index = np.where(matches)[0]

        for t in range(price_with_zero_cols_inverted_only_coin.shape[0]):
            if "open" in self.output_features: # if we have some open features in the output, we will relate the other features to them
                # get the output columns' indices with open feature
                open_column_indices_in_output_columns = [self.output_cols.index(col) for col in self.output_cols if "open" in col]
            else:
                # get the interval t's open values for the coins
                opens_row_index = matching_row_index + t + 1
                # create a list open prices to relate
                opens_to_relate = torch.tensor(df[[f'{c}_open' for c in self.output_coins]].iloc[opens_row_index].values).squeeze()

            # now to compute the features with the open features have
            if "open" in self.output_features:
                # calculate the open feature related to previous interval's open price
                real_price_based_on_only_predictions[t + 1, open_column_indices_in_output_columns] = real_price_based_on_only_predictions[t, open_column_indices_in_output_columns] * torch.exp(price_with_zero_cols_inverted_only_coin[t, open_column_indices_in_output_columns]).float()
                
                # for other columns we relate them to the open feature of the same coin
                for i in range(len(self.output_coins)):
                    real_price_based_on_only_predictions[t + 1, nonopen_column_indices_in_output_columns[i*(len(self.output_features)-1):(i+1)*(len(self.output_features)-1)]] = real_price_based_on_only_predictions[t + 1, open_column_indices_in_output_columns[i]] * torch.exp(price_with_zero_cols_inverted_only_coin[t, nonopen_column_indices_in_output_columns[i*(len(self.output_features)-1):(i+1)*(len(self.output_features)-1)]]).float()
            else:
                # if we don't have open in the output columns that means we have the opens_to_relate vector
                for i in range(len(self.output_coins)):
                    # for other columns we relate them to the opens_to_relate of the same coin
                    opens_to_relate_i = opens_to_relate.item() if opens_to_relate.dim() == 0 else opens_to_relate[i]
                    real_price_based_on_only_predictions[t + 1, nonopen_column_indices_in_output_columns[i*(len(self.output_features)):(i+1)*(len(self.output_features))]] = opens_to_relate_i * torch.exp(price_with_zero_cols_inverted_only_coin[t, nonopen_column_indices_in_output_columns[i*(len(self.output_features)):(i+1)*(len(self.output_features))]]).float()

        for t in range(price_with_zero_cols_inverted_only_coin.shape[0]):
            if "open" in self.output_features: # if we have some open features in the output, we will relate the other features to them
                # get the output columns' indices with open feature
                open_column_indices_in_output_columns = [self.output_cols.index(col) for col in self.output_cols if "open" in col]
            else:
                # get the interval t's open values for the coins
                opens_row_index = matching_row_index + t + 1
                # create a list open prices to relate
                opens_to_relate = torch.tensor(df[[f'{c}_open' for c in self.output_coins]].iloc[opens_row_index].values).squeeze()

            # now to compute the features with the open features have
            if "open" in self.output_features:
                # even if we have the open columns in our output features, we will still relate them to the real open price of the previous day
                # we will not relate them to the open price that is calculated over the predictions, so take the open of the previous day
                opens_row_index_previous_interval = matching_row_index + t
                
                opens_row_index_previous_interval = torch.tensor(df[[f'{c}_open' for c in self.output_coins]].iloc[opens_row_index_previous_interval].values).squeeze()

                # calculate the open feature related to previous interval's open price
                real_price_based_on_real_data[t + 1, open_column_indices_in_output_columns] = (opens_row_index_previous_interval * torch.exp(price_with_zero_cols_inverted_only_coin[t, open_column_indices_in_output_columns])).float()
                
                # for other columns we relate them to the open feature of the same coin
                for i in range(len(self.output_coins)):
                    real_price_based_on_real_data[t + 1, nonopen_column_indices_in_output_columns[i*(len(self.output_features)-1):(i+1)*(len(self.output_features)-1)]] = real_price_based_on_real_data[t + 1, open_column_indices_in_output_columns[i]] * torch.exp(price_with_zero_cols_inverted_only_coin[t, nonopen_column_indices_in_output_columns[i*(len(self.output_features)-1):(i+1)*(len(self.output_features)-1)]]).float()
            else:
                # if we don't have open in the output columns that means we have the opens_to_relate vector
                for i in range(len(self.output_coins)):
                    # for other columns we relate them to the opens_to_relate of the same coin
                    opens_to_relate_i = opens_to_relate.item() if opens_to_relate.dim() == 0 else opens_to_relate[i]
                    real_price_based_on_real_data[t + 1, nonopen_column_indices_in_output_columns[i*(len(self.output_features)):(i+1)*(len(self.output_features))]] = opens_to_relate_i * torch.exp(price_with_zero_cols_inverted_only_coin[t, nonopen_column_indices_in_output_columns[i*(len(self.output_features)):(i+1)*(len(self.output_features))]]).float()
    
        real_price_based_on_real_data = real_price_based_on_real_data[1:]
        real_price_based_on_only_predictions = real_price_based_on_only_predictions[1:]
        
        return real_price_based_on_real_data, real_price_based_on_only_predictions

    def augment(self, x):
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.normal(loc=0.0, scale=self.augmentation_noise_std, size=x.shape)
        if torch.rand(1) < self.augmentation_p:
            x = x + np.random.uniform(-self.augmentation_constant_c, self.augmentation_constant_c)
        if torch.rand(1) < self.augmentation_p:
            x = x * (1.0 + np.random.uniform(-self.augmentation_scale_s, self.augmentation_scale_s))

        self.output_cols = [f'{c}_{f}' for c in self.output_coins for f in self.output_features]
        self.output_col_indices = [list(self.df.columns).index(col) for col in self.output_cols]

        # low is always negative, high is always positive
        low_cols = [self.output_cols.index(col) for col in self.output_cols if "low" in col]
        for low_col in low_cols:
            x[low_col] = np.clip(x[low_col], a_min=None, a_max=0)

        high_cols = [self.output_cols.index(col) for col in self.output_cols if "high" in col]
        for high_col in high_cols:
            x[high_col] = np.clip(x[high_col], a_min=0, a_max=None)

        close_cols = [self.output_cols.index(col) for col in self.output_cols if "close" in col]
        if len(low_cols) == len(high_cols) and len(low_cols) == len(close_cols):
            for e, close_col in enumerate(close_cols):
                x[close_col] = np.clip(x[close_col], a_min=x[low_cols[e]], a_max=x[high_cols[e]])

        return x