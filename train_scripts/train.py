import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from import_model import import_model
from import_dataset import import_dataset

def loss(model, prediction, target, loss_name, zero_weight, variance_weight, logic_loss_weight, l1_loss_weight, l2_loss_weight):
    # --- Base loss ---
    if loss_name == "MSE":
        base_loss_fn = nn.MSELoss()
    elif loss_name == "MAE":
        base_loss_fn = nn.L1Loss()
    elif loss_name == "SmoothL1":
        base_loss_fn = nn.SmoothL1Loss()

    base_loss = base_loss_fn(prediction, target)

    # feature-wise losses
    base_loss_open = base_loss_fn(prediction[:, 0, :], target[:, 0, :])
    base_loss_close = base_loss_fn(prediction[:, 1, :], target[:, 1, :])
    base_loss_low = base_loss_fn(prediction[:, 2, :], target[:, 2, :])
    base_loss_high = base_loss_fn(prediction[:, 3, :], target[:, 3, :])

    # --- Logical constraints ---
    open_ = prediction[:, 0, :]
    close = prediction[:, 1, :]
    low = prediction[:, 2, :]
    high = prediction[:, 3, :]

    # high should be >= low
    high_low = torch.relu(low - high)

    # open and close should be in [low, high]
    open_low = torch.relu(low - open_)
    open_high = torch.relu(open_ - high)
    close_low = torch.relu(low - close)
    close_high = torch.relu(close - high)

    logic_loss = (high_low + open_low + open_high + close_low + close_high).mean()

    # regularization
    l1_reg = torch.tensor(0., device=prediction.device)
    l2_reg = torch.tensor(0., device=prediction.device)
    for param in model.parameters():
        if param.requires_grad:
            l1_reg += param.abs().sum()
            l2_reg += (param ** 2).sum()

    # punishes the model for predicting zero, trying to push predictions away from the mean
    zero_penalty = torch.abs(prediction - target) * torch.abs(target).mean()
    zero_penalty = torch.tensor(0., device=prediction.device)

    # --- Var reward ---
    var_open = prediction[:, 0, :].var(unbiased=False)
    var_close = prediction[:, 1, :].var(unbiased=False)
    var_low = prediction[:, 2, :].var(unbiased=False)
    var_high = prediction[:, 3, :].var(unbiased=False)

    variance_reward = -(var_open + var_close + var_low + var_high)

    # --- Final loss ---
    total_loss = base_loss + zero_weight * zero_penalty + variance_reward * variance_weight + logic_loss_weight * logic_loss + l1_loss_weight * l1_reg + l2_loss_weight * l2_reg
    return total_loss, [base_loss.item(), base_loss_open.item(), base_loss_close.item(),  base_loss_low.item(), base_loss_high.item(), (zero_weight*zero_penalty).item(),
                        (variance_weight*variance_reward).item(), (logic_loss_weight*logic_loss).item(), (l1_loss_weight*l1_reg).item(), (l2_loss_weight*l2_reg).item()]

def plot_losses(train_losses, val_losses, lr_steps, train_session_dir, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot vertical lines for LR drops
    ax = plt.gca()
    for epoch_idx in lr_steps:
        ax.axvline(x=epoch_idx, color="grey", linestyle="--", label="LR drop")

    ax.set_xticks(np.linspace(0, len(train_losses) - 1, 4, dtype=int))

    # To avoid repeating the label multiple times in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())    
    
    plt.savefig(os.path.join(train_session_dir, f'{model_name}_train_val_losses_plot.png'))
    plt.close()

def plot_weight_grad_norms(weight_norms_epoch, grad_norms_epoch, train_session_dir, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(weight_norms_epoch, label="Weight", color="blue")
    plt.plot(grad_norms_epoch, label="Grad", color="orange")
    plt.xlabel("Epoch Steps")
    plt.ylabel("L2 Norm")
    plt.title(f'{model_name} - Weight and Grad Norms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_session_dir, f'{model_name}_weight_grad_norms_plot.png'))
    plt.close()

def plot_weight_grad_dists(weights, grads, train_session_dir, model_name, epoch_idx):
    plt.hist(weights, bins=50, label="Weights", alpha=0.7)
    plt.hist(grads, bins=50, label="Grads", alpha=0.7)
    plt.legend()
    plt.grid(True)
    plt.title(f'{model_name} - Weight and Grad Distributions')
    plt.xlabel("Values")
    plt.xlim((-1, 1))
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(train_session_dir, f'{model_name}_epoch{epoch_idx:03d}_weight_grad_dists_plot.png'))
    plt.close()

def run_inference_and_plot(model, loader, train_session_dir, model_name, epoch_idx, save=False):
    model = model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model.call(x_batch, y_batch)
            all_predictions.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    # Concatenate all batches [batch, 4, 8] -> [total_seq, 4, 8]
    all_targets = torch.cat(all_targets, dim=0)  # [n, 4, 8]
    all_predictions = torch.cat(all_predictions, dim=0)  # [n, 4, 8]

    # Reshape into full time series: [n, 4, 8] -> [4, n, 8]
    target_series = all_targets.permute(1, 0, 2)
    pred_series = all_predictions.permute(1, 0, 2)
    
    if save:
        torch.save(target_series, os.path.join(train_session_dir, f'{model_name}_target_series.pt'))
        torch.save(pred_series, os.path.join(train_session_dir, f'{model_name}_pred_series.pt'))

    target_series_to_plot = torch.cat((target_series[:,:,0], target_series[:,-1]), dim=1)
    pred_series_with_different_trusts = []
    for trust in range(8):
        pred_series_with_different_trusts.append(torch.cat((torch.zeros(pred_series.shape[0], trust), pred_series[:,:,trust], pred_series[:,-1,trust:]), dim=1))

    feature_names = ["Open", "Close", "Low", "High"]
    plt.figure(figsize=(20, 10))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        
        plt.plot(target_series_to_plot[i], label="Ground Truth", color="orange")
        for trust, pred_series_to_plot in enumerate(pred_series_with_different_trusts):
            plt.plot(pred_series_to_plot[i], label="Prediction", color="green", alpha=1/(trust+1))

        plt.title(f'{feature_names[i]}')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(reversed(labels), reversed(handles)))
        plt.legend(by_label.values(), by_label.keys())    

    plt.tight_layout()
    plt.savefig(os.path.join(train_session_dir, f'{model_name}_prediction_plot_{epoch_idx:03d}.png'))
    plt.close()
    model = model.train()

def train_with_args(args):
    model_kwargs = {"input_features": args.input_features, "output_features": args.output_features,
    "input_window": args.input_window, "output_window": args.output_window,
    "dropout": args.dropout, "num_layers": args.num_layers,
    "hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
    "teacher_forcing_ratio": args.teacher_forcing_ratio,
    "target_coin_index": args.target_coin_index,
    "num_coins": args.num_coins, "device": device}

    model = import_model(args.model_name, **model_kwargs)

    if args.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=args.lr_decrease)

    train_session_dir = os.path.join("train_sessions", f'{args.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(train_session_dir, exist_ok=True)

    with open(os.path.join(train_session_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    base_dataset_kwargs = { "coin_symbol": args.coin_symbol, "input_window": args.input_window, "output_window": args.output_window,
    "augmentation_noise_std": args.augmentation_noise_std, "augment_constant_c": args.augment_constant_c, "augment_scale_s": args.augment_scale_s,
    "z_norm_means_csv_path": args.z_norm_means_csv_path, "z_norm_stds_csv_path": args.z_norm_stds_csv_path,
    "distribution_scale": args.distribution_scale, "distribution_clip": args.distribution_clip, "transform_name":args.transform_name,
    "output_distribution": args.output_distribution, "n_quantiles": args.n_quantiles, "train_session_dir": train_session_dir}
    
    train_dataset_kwargs = {**base_dataset_kwargs, "csv_path": args.train_csv_path, "augmentation_p": args.augmentation_p, "train_dataset":1}
    val_dataset_kwargs = {**base_dataset_kwargs, "csv_path": args.val_csv_path, "augmentation_p": args.augmentation_p, "train_dataset":0}
    
    inference_train_dataset_kwargs = {**base_dataset_kwargs, "csv_path": args.train_csv_path, "augmentation_p": 0, "train_dataset":1}
    inference_val_dataset_kwargs = {**base_dataset_kwargs, "csv_path": args.val_csv_path, "augmentation_p": 0, "train_dataset":0}
    
    train_dataset = import_dataset(args.dataset_name, **train_dataset_kwargs)
    val_dataset = import_dataset(args.dataset_name, **val_dataset_kwargs)
    
    inference_train_dataset = import_dataset(args.dataset_name, **inference_train_dataset_kwargs)
    inference_val_dataset = import_dataset(args.dataset_name, **inference_val_dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    inference_train_loader = DataLoader(inference_train_dataset, batch_size=args.batch_size, shuffle=False)
    inference_val_loader = DataLoader(inference_val_dataset, batch_size=args.batch_size, shuffle=False)

    train_model(model=model, model_name=args.model_name, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, epoch_plot_step=args.epoch_plot_step,
                early_stop_patience=args.early_stop_patience, optimizer=optimizer, scheduler=scheduler,
                teacher_forcing_ratio_decrease=args.teacher_forcing_ratio_decrease, zero_weight=args.zero_weight,
                variance_weight=args.variance_weight, logical_loss_weight=args.logical_loss_weight, l1_loss_weight=args.l1_loss_weight,
                l2_loss_weight=args.l2_loss_weight, loss_name=args.loss_name, loss_fn=loss,
                train_session_dir=train_session_dir, inference_dataloaders=(inference_train_loader,inference_val_loader),
                plot_weight_grad=args.plot_weight_grad)

def train_model(model, model_name, train_loader, val_loader, epochs, epoch_plot_step, early_stop_patience, optimizer, scheduler, teacher_forcing_ratio_decrease,
                zero_weight, variance_weight, logical_loss_weight, l1_loss_weight, l2_loss_weight, loss_name, loss_fn, train_session_dir,
                inference_dataloaders, plot_weight_grad):
    best_val_loss = float("inf")
    early_stop_step = 0

    train_losses, val_losses = [], []
    weight_norms_epoch, grad_norms_epoch = [], []
    lr_steps, current_lr = [], optimizer.param_groups[0]["lr"]

    for epoch_idx in range(epochs):
        # --- Training ---
        model = model.train()
        train_loss, individual_train_losses = 0.0, None

        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader), desc="train_loader", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model.call(batch_x, batch_y)

            loss, individual_losses = loss_fn(model, output, batch_y, loss_name, zero_weight, variance_weight, logical_loss_weight, l1_loss_weight, l2_loss_weight)
            loss.backward()
            optimizer.step()

            if plot_weight_grad:
                with torch.no_grad():
                    weight_norms, grad_norms = [], []
                    weights, grads = [], []

                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            weight_norms.append(param.detach().norm(2).item())
                            weights.append(param.detach().cpu().numpy().flatten()) 
                            if param.grad is not None:
                                grad_norms.append(param.grad.detach().norm(2).item())
                                grads.append(param.grad.detach().cpu().numpy().flatten()) 

                    weight_norms_epoch.append(sum(weight_norms) / len(weight_norms))
                    grad_norms_epoch.append(sum(grad_norms) / len(grad_norms))

                    weights = np.concatenate(weights)
                    grads = np.concatenate(grads)

                    plot_weight_grad_dists(weights, grads, train_session_dir, model_name, epoch_idx)

                    if epoch_plot_step != -1 and batch_idx > 0 and batch_idx % epoch_plot_step == 0:
                        run_inference_and_plot(model, inference_dataloaders[0], train_session_dir, f'{model_name}_train_mid_step_{batch_idx}', epoch_idx, epochs)
                        run_inference_and_plot(model, inference_dataloaders[1], train_session_dir, f'{model_name}_val_mid_step_{batch_idx}', epoch_idx, epochs)

            train_loss += loss.item()
            if individual_train_losses is not None:
                individual_train_losses += np.array(individual_losses)
            else:
                individual_train_losses = np.array(individual_losses)

        train_loss /= len(train_loader)
        individual_train_losses /= len(train_loader)
        train_losses.append(train_loss)

        # --- val ---
        model = model.eval()
        val_loss, individual_val_losses = 0.0, None

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="val_loader", leave=False):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model.call(batch_x, batch_y)

                loss, individual_losses = loss_fn(model, output, batch_y, loss_name, zero_weight, variance_weight, logical_loss_weight, l1_loss_weight, l2_loss_weight)
                
                val_loss += loss.item()
                if individual_val_losses is not None:
                    individual_val_losses += np.array(individual_losses)
                else:
                    individual_val_losses = np.array(individual_losses)
        
        val_loss /= len(val_loader)
        individual_val_losses /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch_idx:03d} | Train Loss: {train_loss:.4f} ({individual_train_losses}) \n         | Val Loss: {val_loss:.4f} ({individual_val_losses})')

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_step = 0

            [os.remove(os.path.join(train_session_dir,x)) for x in os.listdir(train_session_dir) if x.endswith(".pt")]
            
            torch.save(model.state_dict(), os.path.join(train_session_dir, f'{model_name}_epoch{epoch_idx:03d}_val{val_loss:.4f}.pt'))
            print("New best model.")
        else:
            early_stop_step += 1
            if early_stop_step >= early_stop_patience:
                print("Early stopping.")
                break

        scheduler.step(val_loss)
        
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            lr_steps.append(epoch_idx)
            current_lr = new_lr

        if hasattr(model, "set_teacher_forcing_ratio"):
            model.set_teacher_forcing_ratio(model.teacher_forcing_ratio-teacher_forcing_ratio_decrease)

        run_inference_and_plot(model, inference_dataloaders[0], train_session_dir, f'{model_name}_train', epoch_idx, save=(early_stop_step==0))
        run_inference_and_plot(model, inference_dataloaders[1], train_session_dir, f'{model_name}_val', epoch_idx, save=(early_stop_step==0))

    plot_losses(train_losses, val_losses, lr_steps, train_session_dir, model_name)
    
    if plot_weight_grad:
        plot_weight_grad_norms(weight_norms_epoch, grad_norms_epoch, train_session_dir, model_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="") # name of the model to use
    parser.add_argument("--input_features", type=int, default=16) # = (number of coins * features per coin) in the input
    parser.add_argument("--output_features", type=int, default=4) # = (number of coins * features per coin) in the output
    parser.add_argument("--input_window", type=int, default=28) # = number of time-series data in the input
    parser.add_argument("--output_window", type=int, default=8) # = number of time-series data in the output
    parser.add_argument("--dropout", type=float, default=0.0) # dropout p for the models
    parser.add_argument("--num_layers", type=int, default=1) # number of layers in the LSTM/GRU/Transformer models
    parser.add_argument("--hidden_dim", type=int, default=64) # dimension size for the LSTM/GRU/Transformer models
    parser.add_argument("--num_heads", type=int, default=4) # number of heads for the Multiheadattention/Transformer
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0) # teacher forcing start ratio (what percentage of the training samples will be predicted utilizing the real data for later values in the output_window)
    parser.add_argument("--teacher_forcing_ratio_decrease", type=float, default=0) # decrease in the teacher forcing ratio after each epoch
    parser.add_argument("--target_coin_index", type=int, default=0) # index of the coin models will predict
    parser.add_argument("--num_coins", type=int, default=4) # number of coins in the dataset
    parser.add_argument("--coin_symbol", type=str, default="BTC") # symbol of the coin for the dataset
    parser.add_argument("--dataset_name", type=str, default="") # name of the dataset to use
    parser.add_argument("--train_csv_path", type=str, default="") # csv path to load for training dataset
    parser.add_argument("--val_csv_path", type=str, default="") # csv path to load for validation dataset
    parser.add_argument("--z_norm_means_csv_path", type=str, default="") # mean values to reverse the z_normalization (only for the LogZNormCoinDataset dataset)
    parser.add_argument("--z_norm_stds_csv_path", type=str, default="") # std values to reverse the z_normalization (only for the LogZNormCoinDataset dataset)
    parser.add_argument("--augmentation_p", type=float, default=0) # probability of a sample being augmented (also the probability of each augmentation being applied to that sample)
    parser.add_argument("--augmentation_noise_std", type=float, default=0) # std for the gaussian noise augmentation
    parser.add_argument("--augment_constant_c", type=float, default=0) # min max limit for the constant to be added to all samples
    parser.add_argument("--augment_scale_s", type=float, default=0) # min max limit for the scale to be multiplied with all samples
    parser.add_argument("--distribution_scale", type=float, default=100) # scale to multiply the log returns with (only for the LogReturnCoinDataset dataset)
    parser.add_argument("--distribution_clip", type=float, default=10) # scale to clip the log returns' both sides with (only for the LogReturnCoinDataset dataset)
    parser.add_argument("--transform_name", type=str, default="QuantileTransformer") # sklearn.preprocessing QuantileTransformer or PowerTransformer to use
    parser.add_argument("--output_distribution", type=str, default="normal") # sklearn.preprocessing QuantileTransformer distribution type (only for the LogReturnTransformCoinDataset dataset)
    parser.add_argument("--n_quantiles", type=int, default=1000) # sklearn.preprocessing QuantileTransformer number of quantiles (only for the LogReturnTransformCoinDataset dataset)
    parser.add_argument("--plot_weight_grad", type=int, default=0) # whether to plot weight and gradient plots at each epoch
    parser.add_argument("--loss_name", type=str, default="") # name of the loss to use
    parser.add_argument("--zero_weight", type=float, default=0) # weight of the loss penalizing the wrong zero predictions
    parser.add_argument("--variance_weight", type=float, default=0) # weight of the reward for high variance
    parser.add_argument("--logical_loss_weight", type=float, default=0) # weight of the custom logical loss (only for the LogZNormCoinDataset dataset)
    parser.add_argument("--l1_loss_weight", type=float, default=0) # weight for the l1 regularization
    parser.add_argument("--l2_loss_weight", type=float, default=0) # weight for the l2 regularization
    parser.add_argument("--batch_size", type=int, default=32) # batch size
    parser.add_argument("--epochs", type=int, default=10) # number of epochs
    parser.add_argument("--epoch_plot_step", type=int, default=100) # whether to plo mid-epoch inference plots, if so at every what batch to do it
    parser.add_argument("--optimizer_name", type=str, default="Adam") # name of the optimizer to use
    parser.add_argument("--lr", type=float, default=1e-3) # learning rate
    parser.add_argument("--lr_patience", type=int, default=5) # patience for learning rate scheduler
    parser.add_argument("--lr_decrease", type=float, default=0.95) # multiplicative decrease for the learning rate scheduler
    parser.add_argument("--early_stop_patience", type=int, default=5) # early stop after no improvement in validation score
    args = parser.parse_args()
    train_with_args(args)

if __name__ == "__main__":
    main()