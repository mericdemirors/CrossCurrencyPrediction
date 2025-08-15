import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from matplotlib.collections import LineCollection

import torch

from import_model import import_model
from import_dataset import import_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_evaluation_graphs(train_session_dir):
    # read the training session data and recreate the training environment for inference
    json_to_inference = os.path.join(train_session_dir,"args.json")
    with open(json_to_inference, 'r') as f:
        data = json.load(f)

    model_name = data["model_name"]
    train_pred_name = f'{model_name}_train_pred_series.pt'
    val_pred_name = f'{model_name}_val_pred_series.pt'

    target_and_pred_names = [train_pred_name, val_pred_name]
    model_pt_name = [x for x in os.listdir(train_session_dir) if x.endswith(".pt") and x not in target_and_pred_names][0]
    model_pt_path = os.path.join(train_session_dir, model_pt_name)

    args = argparse.Namespace(**data)

    input_cols = [f'{c}_{f}' for c in args.input_coins for f in args.input_features]
    output_cols = [f'{c}_{f}' for c in args.output_coins for f in args.output_features]
    output_col_indices_in_input_cols = [input_cols.index(col) for col in output_cols]
    target_coin_indices = [args.input_coins.index(c) for c in args.output_coins]

    model_kwargs = {"input_features": len(args.input_coins)*len(args.input_features), "output_features": len(args.output_coins)*len(args.output_features),
    "input_window": args.input_window, "output_window": args.output_window,
    "dropout": args.dropout, "num_layers": args.num_layers, "hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
    "teacher_forcing_ratio": args.teacher_forcing_ratio, "input_cols":input_cols, "output_cols":output_cols,
    "target_coin_indices": target_coin_indices, "output_col_indices_in_input_cols": output_col_indices_in_input_cols,
    "device": device}

    model = import_model(args.model_name, **model_kwargs)
    model.load_state_dict(torch.load(model_pt_path, weights_only=True))
    model = model.eval()
    if hasattr(model, "set_teacher_forcing_ratio"):
        model.set_teacher_forcing_ratio(0)

    base_dataset_kwargs = {"input_coins": args.input_coins, "input_features": args.input_features, "output_coins": args.output_coins,
    "output_features": args.output_features, "input_window": args.input_window, "output_window": args.output_window,
    "augmentation_noise_std": args.augmentation_noise_std, "augmentation_constant_c": args.augmentation_constant_c, "augmentation_scale_s": args.augmentation_scale_s,
    "transform_name":args.transform_name, "output_distribution": args.output_distribution, "n_quantiles": args.n_quantiles, "train_session_dir": train_session_dir}

    val_inference_dataset_kwargs = {**base_dataset_kwargs, "csv_path": args.val_csv_path, "augmentation_p": 0, "training_dataset":0}
    val_inference_dataset = import_dataset(args.dataset_name, **val_inference_dataset_kwargs)

    train_inference_dataset_kwargs = {**base_dataset_kwargs, "csv_path": args.train_csv_path, "augmentation_p": 0, "training_dataset":1}
    train_inference_dataset = import_dataset(args.dataset_name, **train_inference_dataset_kwargs)

    # MODEL'S PREDICTIONS
    train_pred_path = os.path.join(train_session_dir, train_pred_name)
    val_pred_path = os.path.join(train_session_dir, val_pred_name)

    train_pred_series = torch.load(train_pred_path)
    val_pred_series = torch.load(val_pred_path)

    # DATA THE MODEL SAW DURING THE TRAINING
    train_learned_dataframe_crop = train_inference_dataset.df.iloc[data["input_window"]:]
    val_learned_dataframe_crop = val_inference_dataset.df.iloc[data["input_window"]:]

    # REAL PRICE DATA
    train_df = pd.read_csv(data["train_csv_path"], index_col="open_time")
    train_df_preds = train_df.loc[train_learned_dataframe_crop.index]
    # we take the data["input_window"]th data as the initial price instead of data["input_window"]-1
    # it's because we are shifting the dates 1 interval back during the preprocessing, so the first day is kinda wasted for the lof return computations
    # and so the initial price is also shifted one interval further than it would be at an unshifted dataset
    t1 = datetime.strptime(train_learned_dataframe_crop.index[0], "%Y-%m-%d %H:%M:%S")
    t2 = datetime.strptime(train_learned_dataframe_crop.index[1], "%Y-%m-%d %H:%M:%S")
    delta = t2 - t1
    t0 = t1 - delta
    train_initial_prices = torch.tensor(train_df.loc[str(t0)][output_cols].values).unsqueeze(0)

    val_df = pd.read_csv(data["val_csv_path"], index_col="open_time")
    val_df_preds = val_df.loc[val_learned_dataframe_crop.index]
    t1 = datetime.strptime(val_learned_dataframe_crop.index[0], "%Y-%m-%d %H:%M:%S")
    t2 = datetime.strptime(val_learned_dataframe_crop.index[1], "%Y-%m-%d %H:%M:%S")
    delta = t2 - t1
    t0 = t1 - delta
    val_initial_prices = torch.tensor(val_df.loc[str(t0)][output_cols].values).unsqueeze(0)

    def plot_the_dataset_distributoins(pred_series, learned_dataframe_crop, inference_dataset, dataset_portion):
        # prepend zeros for different trust values
        pred_series_with_different_trusts = []
        for trust in range(pred_series.shape[2]):
            pred_series_with_different_trusts.append(torch.cat((torch.zeros(pred_series.shape[0], trust), pred_series[:,:-1,trust], pred_series[:,-1,trust:]), dim=1))

        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Distributions of predictions on the {dataset_portion} dataset.\n Blue: price_t+1 from price_t, Greens: price_t+i from price_t where i>1\nAll other plots check the temporal fit to the dataset, this plot checks overall ability to match real world p(x)', fontsize=12)

        for i in tqdm(range(len(output_cols)), desc=f'plotting {dataset_portion} dataset distributions', leave=False):
            row_count = math.ceil(len(output_cols)**0.5)
            plt.subplot(row_count, math.ceil(len(output_cols)//row_count), i + 1)
            
            # this is the data from dataset
            ground_truth = learned_dataframe_crop[inference_dataset.output_cols].values.T[i]
            sns.kdeplot(data=ground_truth, fill=False, color="orange", label="Dataset")
            
            # this is the predictions of the model
            for trust, pred_series_to_plot in reversed(list(enumerate(pred_series_with_different_trusts))):
                if trust != 0:
                    sns.kdeplot(data=pred_series_to_plot[i], fill=False, color="green", label="Other Predictions", alpha=1/(trust+1))
                else:
                    sns.kdeplot(data=pred_series_to_plot[i], fill=False, color="blue", label="Predictions", alpha=1/(trust+1))

            plt.title(f'{output_cols[i]}')
            plt.xlabel("Time")
            plt.xlim(ground_truth.min(), ground_truth.max())
            plt.ylabel("Value")
            plt.legend()

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(reversed(labels), reversed(handles)))
            plt.legend(by_label.values(), by_label.keys())    

        plt.savefig(os.path.join(train_session_dir, f'evaluation_graphs/distributions_on_the_dataset_{dataset_portion}.png'))
    
    def plot_the_future_dataset_predictions(pred_series, learned_dataframe_crop, inference_dataset, dataset_portion):
        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Predictions on the {dataset_portion} dataset.\n Blue: price_t+i from price_t where i>0, fading away for further predictions', fontsize=12)
        
        pred_legend = Line2D([0], [0], color=(0.2, 0.4, 0.8, 1), label="Future Predictions")

        for i in tqdm(range(len(output_cols)), desc=f'plotting {dataset_portion} dataset predictions', leave=False):
            row_count = math.ceil(len(output_cols)**0.5)
            plt.subplot(row_count, math.ceil(len(output_cols)//row_count), i + 1)
            
            # this is the data from dataset
            ground_truth = learned_dataframe_crop[inference_dataset.output_cols].values.T[i]
            plt.plot(ground_truth, label="Dataset", color="orange", zorder=1)

            tensor = pred_series[i]
            num_of_intervals, output_window_len = tensor.shape

            # X axis indices, all predictions shifted one tick right
            x = torch.arange(output_window_len).unsqueeze(0) + torch.arange(num_of_intervals).unsqueeze(1)

            # if we have 8 output window that means we have 7 segments in our line.
            # So we create the start and end coordinated for each output_window-1 segment in one prediction
            x_start = x[:, :-1]
            x_end = x[:, 1:]
            y_start = tensor[:, :-1]
            y_end = tensor[:, 1:]

            # this is the whole segments we will plot
            segments = np.stack([ np.stack([x_start, y_start], axis=2),
                                np.stack([x_end, y_end], axis=2)], axis=2).reshape(-1, 2, 2)

            # set up the fading away trust
            alpha_fade = np.linspace(1.0, 0.1, output_window_len - 1) ** 4
            # and repeat them so we get the same fading away effect for all predictionss' 7 segments
            alphas = np.tile(alpha_fade, num_of_intervals)

            # set up the colors
            colors = np.ones((segments.shape[0], 4))
            colors[:, 0] = 0.2  # R
            colors[:, 1] = 0.4  # G
            colors[:, 2] = 0.8  # B
            colors[:, 3] = alphas  # Alpha fade

            ax = plt.gca()
            lc = LineCollection(segments, colors=colors, linewidths=1)
            ax.add_collection(lc)

            plt.title(f'{output_cols[i]}')
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.ylim(ground_truth.min()*0.9, ground_truth.max()*1.1)
            plt.legend()

            handles, labels = ax.get_legend_handles_labels()
            handles.append(pred_legend)
            labels.append("Future Predictions")
            by_label = OrderedDict(zip(reversed(labels), reversed(handles)))
            plt.legend(by_label.values(), by_label.keys())
            
        plt.savefig(os.path.join(train_session_dir, f'evaluation_graphs/future_predictions_on_the_dataset_{dataset_portion}.png'))

    def plot_the_autoregressive_dataset_predictions(learned_dataframe_crop, inference_dataset, dataset_portion):
        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Autoregressive predictions on the {dataset_portion} dataset.\n Blue: filling the missing (non-predicted) input features from ground truth, Green: filling the missing (non-predicted) input features with zeros', fontsize=12)

        autoregressive_tensor = torch.tensor(inference_dataset.df[inference_dataset.input_cols].values).float()
        for e, interval in enumerate(range(len(inference_dataset))):
            x = autoregressive_tensor[interval:interval+args.input_window].T.unsqueeze(0).float().to(model_kwargs["device"])
            with torch.no_grad():
                next_interval = model.call(x, None)
            
            autoregressive_tensor[interval+args.input_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T[0]

            if e == len(inference_dataset) - 1:
                autoregressive_tensor[interval+args.input_window:interval+args.input_window+args.output_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T

        cols_to_zero_out = [False if x in inference_dataset.output_col_indices else True for x in range(autoregressive_tensor.shape[1])]
        autoregressive_tensor_with_zeros = torch.tensor(inference_dataset.df[inference_dataset.input_cols].values).float()
        for e, interval in enumerate(range(len(inference_dataset))):
            x = autoregressive_tensor_with_zeros[interval:interval+args.input_window].T.unsqueeze(0).float().to(model_kwargs["device"])
            with torch.no_grad():
                next_interval = model.call(x, None)
            
            autoregressive_tensor_with_zeros[interval+args.input_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T[0]
            autoregressive_tensor_with_zeros[interval+args.input_window, cols_to_zero_out] = 0

            if e == len(inference_dataset) - 1:
                autoregressive_tensor_with_zeros[interval+args.input_window:interval+args.input_window+args.output_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T

        for i in tqdm(range(len(output_cols)), desc=f'plotting autoregressive {dataset_portion} dataset predictions', leave=False):
            row_count = math.ceil(len(output_cols)**0.5)
            plt.subplot(row_count, math.ceil(len(output_cols)//row_count), i + 1)
            
            # this is the data from dataset
            ground_truth = learned_dataframe_crop[inference_dataset.output_cols].values.T[i]
            plt.plot(ground_truth, label="Dataset", color="orange", zorder=1)
            
            autoregressive_col = autoregressive_tensor[args.input_window:, inference_dataset.output_col_indices].T[i]
            plt.plot(autoregressive_col, label="Autoregressive Predictions", color="Blue", zorder=3)
            
            autoregressive_col_with_zeros = autoregressive_tensor_with_zeros[args.input_window:, inference_dataset.output_col_indices].T[i]
            plt.plot(autoregressive_col_with_zeros, label="Autoregressive Predictions With Zeros", color="Green", zorder=2)

            plt.title(f'{output_cols[i]}')
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.ylim(ground_truth.min()*0.9, ground_truth.max()*1.1)
            plt.legend()

            plt.savefig(os.path.join(train_session_dir, f'evaluation_graphs/autoregressive_predictions_on_the_dataset_{dataset_portion}.png'))

    def plot_the_future_price_predictions(inference_dataset, learned_dataframe_crop, initial_prices, pred_series, df_preds, dataset_portion):
        # rescale the dataset values into real prices
        rescaled_target_series_to_plot, _ = inference_dataset.rescale_to_real_price(torch.from_numpy(learned_dataframe_crop[inference_dataset.output_cols].values), initial_prices)
        rescaled_target_series_to_plot = rescaled_target_series_to_plot.T

        # prepend zeros for different trust values
        pred_series_with_different_trusts = []
        for trust in range(pred_series.shape[2]):
            pred_series_with_different_trusts.append(torch.cat((torch.zeros(pred_series.shape[0], trust), pred_series[:,:-1,trust], pred_series[:,-1,trust:]), dim=1))

        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Predictions on the {dataset_portion} dataset.\n Red: real prices from API, Orange: re-scaled prices from dataset\'s normalization (should overlap with red)\nBlue: price_t+i from price_t where price_t is taken from previous prediction and i>0 , fading away for further predictions\nBlack: price_t+i from price_t where price_t is the real price of previous interval and i>0 , fading away for further predictions', fontsize=12)

        daily_pred_legend = Line2D([0], [0], color=(0.0, 0.0, 0.0, 1), label="Daily Future Predictions")
        pred_legend = Line2D([0], [0], color=(0.2, 0.4, 0.8, 1), label="Future Predictions")

        for i in tqdm(range(len(output_cols)), desc=f'plotting {dataset_portion} price predictions', leave=False):
            row_count = math.ceil(len(output_cols)**0.5)
            plt.subplot(row_count, math.ceil(len(output_cols)//row_count), i + 1)
            
            # this is the real price data directly from the source
            real_prices = df_preds[inference_dataset.output_cols].values.T[i]
            plt.plot(real_prices, label="Prices", color="red", zorder=1)

            # this is the data from dataset, first scaled and then rescaled (used to see the divergence preprocess caused)
            ground_truth = rescaled_target_series_to_plot[i]
            plt.plot(ground_truth, label="Dataset", color="orange", zorder=1)
            
            rescaled_pred_series_on_real_data, rescaled_pred_series_on_predictions = [], []
            # this is the predictions of the model
            
            for trust, pred_series_to_plot in reversed(list(enumerate(pred_series_with_different_trusts))):
                # first one's predictions are based on the real data, second one's predictions are based on the previous day's predictions
                rescaled_pred_series_to_plot_based_on_real_data, rescaled_pred_series_to_plot_based_on_predictions = inference_dataset.rescale_to_real_price(pred_series_to_plot.T, initial_prices)
                rescaled_pred_series_on_real_data.append(rescaled_pred_series_to_plot_based_on_real_data)
                rescaled_pred_series_on_predictions.append(rescaled_pred_series_to_plot_based_on_predictions)

            stacked_rescaled_pred_series_on_real_data = torch.stack(rescaled_pred_series_on_real_data)
            stacked_rescaled_pred_series_on_real_data = stacked_rescaled_pred_series_on_real_data.permute((2,1,0))

            stacked_rescaled_pred_series_on_predictions = torch.stack(rescaled_pred_series_on_predictions)
            stacked_rescaled_pred_series_on_predictions = stacked_rescaled_pred_series_on_predictions.permute((2,1,0))

            tensor = stacked_rescaled_pred_series_on_predictions[i]
            num_of_intervals, output_window_len = tensor.shape

            # X axis indices, all predictions shifted one tick right
            x = torch.arange(output_window_len).unsqueeze(0) + torch.arange(num_of_intervals).unsqueeze(1)

            # if we have 8 output window that means we have 7 segments in our line.
            # So we create the start and end coordinated for each output_window-1 segment in one prediction
            x_start = x[:, :-1]
            x_end = x[:, 1:]
            y_start = tensor[:, :-1]
            y_end = tensor[:, 1:]

            # this is the whole segments we will plot
            segments = np.stack([ np.stack([x_start, y_start], axis=2),
                                np.stack([x_end, y_end], axis=2)], axis=2).reshape(-1, 2, 2)

            # set up the fading away trust
            alpha_fade = np.linspace(1.0, 0.1, output_window_len - 1) ** 4
            # and repeat them so we get the same fading away effect for all predictionss' 7 segments
            alphas = np.tile(alpha_fade, num_of_intervals)

            # set up the colors
            colors = np.ones((segments.shape[0], 4))
            colors[:, 0] = 0.2  # R
            colors[:, 1] = 0.4  # G
            colors[:, 2] = 0.8  # B
            colors[:, 3] = alphas  # Alpha fade

            ax = plt.gca()
            lc = LineCollection(segments, colors=colors, linewidths=1)
            ax.add_collection(lc)

            tensor = stacked_rescaled_pred_series_on_real_data[i]
            num_of_intervals, output_window_len = tensor.shape

            # X axis indices, all predictions shifted one tick right
            x = torch.arange(output_window_len).unsqueeze(0) + torch.arange(num_of_intervals).unsqueeze(1)

            # if we have 8 output window that means we have 7 segments in our line.
            # So we create the start and end coordinated for each output_window-1 segment in one prediction
            x_start = x[:, :-1]
            x_end = x[:, 1:]
            y_start = tensor[:, :-1]
            y_end = tensor[:, 1:]

            # this is the whole segments we will plot
            segments = np.stack([ np.stack([x_start, y_start], axis=2),
                                np.stack([x_end, y_end], axis=2)], axis=2).reshape(-1, 2, 2)

            # set up the fading away trust
            alpha_fade = np.linspace(1.0, 0.1, output_window_len - 1) ** 4
            # and repeat them so we get the same fading away effect for all predictionss' 7 segments
            alphas = np.tile(alpha_fade, num_of_intervals)

            # set up the colors
            colors = np.ones((segments.shape[0], 4))
            colors[:, 0] = 0.0  # R
            colors[:, 1] = 0.0  # G
            colors[:, 2] = 0.0  # B
            colors[:, 3] = alphas  # Alpha fade

            ax = plt.gca()
            lc = LineCollection(segments, colors=colors, linewidths=1)
            ax.add_collection(lc)

            plt.title(f'{output_cols[i]}')
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.ylim(ground_truth.min()*0.9, ground_truth.max()*1.1)
            plt.legend()

            handles, labels = ax.get_legend_handles_labels()
            handles.append(daily_pred_legend)
            labels.append("Daily Future Predictions")
            handles.append(pred_legend)
            labels.append("Future Predictions")
            by_label = OrderedDict(zip(reversed(labels), reversed(handles)))
            plt.legend(by_label.values(), by_label.keys())

        plt.savefig(os.path.join(train_session_dir, f'evaluation_graphs/future_predictions_on_the_prices_{dataset_portion}.png'))

    def plot_the_autoregressive_price_predictions(learned_dataframe_crop, inference_dataset, initial_prices, dataset_portion):
        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Autoregressive predictions on the {dataset_portion} dataset.\n Blue: filling the missing (non-predicted) input features from ground truth, Green: filling the missing (non-predicted) input features with zeros', fontsize=12)

        autoregressive_tensor = torch.tensor(inference_dataset.df[inference_dataset.input_cols].values).float()
        for e, interval in enumerate(range(len(inference_dataset))):
            x = autoregressive_tensor[interval:interval+args.input_window].T.unsqueeze(0).float().to(model_kwargs["device"])
            with torch.no_grad():
                next_interval = model.call(x, None)
            
            autoregressive_tensor[interval+args.input_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T[0]

            if e == len(inference_dataset) - 1:
                autoregressive_tensor[interval+args.input_window:interval+args.input_window+args.output_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T

        cols_to_zero_out = [False if x in inference_dataset.output_col_indices else True for x in range(autoregressive_tensor.shape[1])]
        autoregressive_tensor_with_zeros = torch.tensor(inference_dataset.df[inference_dataset.input_cols].values).float()
        for e, interval in enumerate(range(len(inference_dataset))):
            x = autoregressive_tensor_with_zeros[interval:interval+args.input_window].T.unsqueeze(0).float().to(model_kwargs["device"])
            with torch.no_grad():
                next_interval = model.call(x, None)
            
            autoregressive_tensor_with_zeros[interval+args.input_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T[0]
            autoregressive_tensor_with_zeros[interval+args.input_window, cols_to_zero_out] = 0

            if e == len(inference_dataset) - 1:
                autoregressive_tensor_with_zeros[interval+args.input_window:interval+args.input_window+args.output_window, inference_dataset.output_col_indices] = next_interval.detach().cpu().float().squeeze().T

        for i in tqdm(range(len(output_cols)), desc=f'plotting autoregressive {dataset_portion} dataset predictions', leave=False):
            row_count = math.ceil(len(output_cols)**0.5)
            plt.subplot(row_count, math.ceil(len(output_cols)//row_count), i + 1)
            
            _, rescaled_target_series_to_plot = inference_dataset.rescale_to_real_price(torch.from_numpy(learned_dataframe_crop[inference_dataset.output_cols].values), initial_prices)
            ground_truth = rescaled_target_series_to_plot.T[i]
            plt.plot(ground_truth, label="Dataset", color="orange", zorder=1)
            
            _, rescaled_autoregressive_series_to_plot = inference_dataset.rescale_to_real_price(autoregressive_tensor[args.input_window:, inference_dataset.output_col_indices], initial_prices)
            rescaled_autoregressive_col = rescaled_autoregressive_series_to_plot.T[i]
            plt.plot(rescaled_autoregressive_col, label="Autoregressive Predictions", color="blue", zorder=3)
            
            _, rescaled_autoregressive_zero_series_to_plot = inference_dataset.rescale_to_real_price(autoregressive_tensor_with_zeros[args.input_window:, inference_dataset.output_col_indices], initial_prices)
            rescaled_autoregressive_col_with_zeros = rescaled_autoregressive_zero_series_to_plot.T[i]
            plt.plot(rescaled_autoregressive_col_with_zeros, label="Autoregressive Predictions", color="green", zorder=2)

            plt.title(f'{output_cols[i]}')
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.ylim(ground_truth.min()*0.9, ground_truth.max()*1.1)
            plt.legend()

            plt.savefig(os.path.join(train_session_dir, f'evaluation_graphs/autoregressive_predictions_on_the_dataset_{dataset_portion}.png'))

    os.makedirs(os.path.join(train_session_dir, "evaluation_graphs"))
    plot_the_dataset_distributoins(train_pred_series, train_learned_dataframe_crop, train_inference_dataset, "train")
    plot_the_dataset_distributoins(val_pred_series, val_learned_dataframe_crop, val_inference_dataset, "val")
    plot_the_future_dataset_predictions(train_pred_series, train_learned_dataframe_crop, train_inference_dataset, "train")
    plot_the_future_dataset_predictions(val_pred_series, val_learned_dataframe_crop, val_inference_dataset, "val")
    # plot_the_autoregressive_dataset_predictions(train_learned_dataframe_crop, train_inference_dataset, "train")
    plot_the_autoregressive_dataset_predictions(val_learned_dataframe_crop, val_inference_dataset, "val")
    plot_the_future_price_predictions(train_inference_dataset, train_learned_dataframe_crop, train_initial_prices, train_pred_series, train_df_preds, "train")
    plot_the_future_price_predictions(val_inference_dataset, val_learned_dataframe_crop, val_initial_prices, val_pred_series, val_df_preds, "val")
    # plot_the_autoregressive_price_predictions(train_learned_dataframe_crop, train_inference_dataset, train_initial_prices, "train")
    plot_the_autoregressive_price_predictions(val_learned_dataframe_crop, val_inference_dataset, val_initial_prices, "val")