import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from import_model import import_model
from LogZNormCoinDataset import LogZNormCoinDataset
from LogReturnCoinDataset import LogReturnCoinDataset

def loss(model, prediction, target, logic_loss_weight, l1_loss_weight, l2_loss_weight):
    # --- Base loss ---
    base_loss = nn.MSELoss()(prediction, target)

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

    # --- Final loss ---
    total_loss = base_loss + logic_loss_weight * logic_loss + l1_loss_weight * l1_reg + l2_loss_weight * l2_reg
    return total_loss, [base_loss.item(), (logic_loss_weight*logic_loss).item(), (l1_loss_weight*l1_reg).item(), (l2_loss_weight*l2_reg).item()]

def plot_losses(train_losses, val_losses, train_session_dir, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_session_dir, f"{model_name}_train_val_loss_plot.png"))
    plt.close()

def run_inference_and_plot(model, loader, train_session_dir, model_name, epoch):
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

    target_series = torch.cat((target_series[:,:,0], target_series[:,-1]), dim=1)
    pred_series = torch.cat((pred_series[:,:,0], pred_series[:,-1]), dim=1)

    feature_names = ['Open', 'Close', 'Low', 'High']
    plt.figure(figsize=(20, 10))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(target_series[i], label='Ground Truth', color='orange')
        plt.plot(pred_series[i], label='Prediction', color='green')
        plt.title(f'{feature_names[i]}')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(train_session_dir, f"{model_name}_prediction_plot_{epoch}.png"))
    plt.close()

def train_with_args(args):
    if args.z_norm_means_csv_path == "" and args.z_norm_stds_csv_path == "":
        train_dataset = LogReturnCoinDataset(args.train_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=args.augmentation_p, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, domain_scale=args.domain_scale)
        val_dataset = LogReturnCoinDataset(args.val_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=0, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, domain_scale=args.domain_scale)

        inference_train_dataset = LogReturnCoinDataset(args.train_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=0, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, domain_scale=args.domain_scale)
        inference_val_dataset = LogReturnCoinDataset(args.val_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=0, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, domain_scale=args.domain_scale)
    
        args.logical_loss_weight = 0
    else:
        train_dataset = LogZNormCoinDataset(args.train_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=args.augmentation_p, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, z_norm_means_csv_path=args.z_norm_means_csv_path, z_norm_stds_csv_path=args.z_norm_stds_csv_path)
        val_dataset = LogZNormCoinDataset(args.val_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=0, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, z_norm_means_csv_path=args.z_norm_means_csv_path, z_norm_stds_csv_path=args.z_norm_stds_csv_path)

        inference_train_dataset = LogZNormCoinDataset(args.train_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=0, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, z_norm_means_csv_path=args.z_norm_means_csv_path, z_norm_stds_csv_path=args.z_norm_stds_csv_path)
        inference_val_dataset = LogZNormCoinDataset(args.val_csv_path, coin_symbol=args.coin_symbol, input_window=args.input_window, output_window=args.output_window, augmentation_p=0, augmentation_noise_std=args.augmentation_std, augment_constant_c=args.augment_constant_c, augment_scale_s=args.augment_scale_s, z_norm_means_csv_path=args.z_norm_means_csv_path, z_norm_stds_csv_path=args.z_norm_stds_csv_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    inference_train_loader = DataLoader(inference_train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    inference_val_loader = DataLoader(inference_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model_kwargs = {
        "input_features": args.input_features,
        "output_features": args.output_features,
        "input_window": args.input_window,
        "output_window": args.output_window,
        "dropout": args.dropout,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "target_coin_index": args.target_coin_index,
        "num_coins": args.num_coins,
        "device": device
    }

    model = import_model(args.model_name, **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_session_dir = os.path.join("train_sessions", f"{args.model_name}_{start_time}")
    os.makedirs(train_session_dir, exist_ok=True)

    with open(os.path.join(train_session_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, early_stop_patience=args.early_stop_patience, optimizer=optimizer, scheduler=scheduler, logical_loss_weight=args.logical_loss_weight, l1_loss_weight=args.l1_loss_weight, l2_loss_weight=args.l2_loss_weight, loss_fn=loss, train_session_dir=train_session_dir, inference_dataloaders=(inference_train_loader,inference_val_loader))

def train_model(model, train_loader, val_loader, epochs, early_stop_patience, optimizer, scheduler,  logical_loss_weight, l1_loss_weight, l2_loss_weight, loss_fn=loss, train_session_dir="", inference_dataloaders=(None, None)):
    best_val_loss = float("inf")
    early_stop_step = 0
    model_name = model.__class__.__name__

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Training ---
        model = model.train()
        train_loss = 0.0
        individual_train_losses = None

        for x_batch, y_batch in tqdm(train_loader, desc="train_loader", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model.call(x_batch, y_batch)
            loss, individual_losses = loss_fn(model, output, y_batch, logical_loss_weight, l1_loss_weight, l2_loss_weight)
            loss.backward()
            optimizer.step()

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
        val_loss = 0.0
        individual_val_losses = None

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc="val_loader", leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model.call(x_batch, y_batch)
                loss, individual_losses = loss_fn(model, output, y_batch, logical_loss_weight, l1_loss_weight, l2_loss_weight)
                
                val_loss += loss.item()
                if individual_val_losses is not None:
                    individual_val_losses += np.array(individual_losses)
                else:
                    individual_val_losses = np.array(individual_losses)
        
        val_loss /= len(val_loader)
        individual_val_losses /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} ({individual_train_losses}) \n         | Val Loss: {val_loss:.4f} ({individual_val_losses})")

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_step = 0

            [os.remove(os.path.join(train_session_dir,x)) for x in os.listdir(train_session_dir) if x.endswith(".pt")]
            

            torch.save(model.state_dict(), os.path.join(train_session_dir, f"{model_name}_epoch{epoch+1:02d}_val{val_loss:.4f}.pt"))
            print(f"New best model.")
        else:
            early_stop_step += 1
            if early_stop_step >= early_stop_patience:
                print(f"Early stopping.")
                break

        scheduler.step(val_loss)

        if hasattr(model, "set_teacher_forcing_ratio"):
            model.set_teacher_forcing_ratio(model.teacher_forcing_ratio-0.1)

        run_inference_and_plot(model, inference_dataloaders[0], train_session_dir, model_name + "_train", epoch)
        run_inference_and_plot(model, inference_dataloaders[1], train_session_dir, model_name + "_val", epoch)

    plot_losses(train_losses, val_losses, train_session_dir, model_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--input_features", type=int, default=16)
    parser.add_argument("--output_features", type=int, default=4)
    parser.add_argument("--input_window", type=int, default=28)
    parser.add_argument("--output_window", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    parser.add_argument("--target_coin_index", type=int, default=0)
    parser.add_argument("--num_coins", type=int, default=4)
    parser.add_argument("--coin_symbol", type=str, default="BTC")
    parser.add_argument("--train_csv_path", type=str, default="")
    parser.add_argument("--val_csv_path", type=str, default="")
    parser.add_argument("--z_norm_means_csv_path", type=str, default="")
    parser.add_argument("--z_norm_stds_csv_path", type=str, default="")
    parser.add_argument("--augmentation_p", type=float, default=0.75)
    parser.add_argument("--augmentation_std", type=float, default=0.05)
    parser.add_argument("--augment_constant_c", type=float, default=1)
    parser.add_argument("--augment_scale_s", type=float, default=0.25)
    parser.add_argument("--domain_scale", type=float, default=100)
    parser.add_argument("--logical_loss_weight", type=float, default=1e-5)
    parser.add_argument("--l1_loss_weight", type=float, default=0.0)
    parser.add_argument("--l2_loss_weight", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early_stop_patience", type=int, default=7)
    args = parser.parse_args()
    train_with_args(args)

if __name__ == "__main__":
    main()