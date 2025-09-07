import warnings
warnings.filterwarnings("ignore")

import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch

from import_model import import_model
from import_dataset import import_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_orders(oclh):
    # calculate each (buy at low, sell at an upcoming high) profit
    profits = []
    for low_interval in range(oclh.shape[1]):
        for high_interval in range(low_interval + 1, oclh.shape[1]):
            profit = oclh[3][high_interval] - oclh[2][low_interval]
            if profit > 0:
                profits.append(((low_interval, high_interval), (oclh[2][low_interval], oclh[3][high_interval]), profit, (2, 3)))

    # calculate sandwiches
    sandwiches = []
    while len(profits) > 0:
        max_b_s_p_l_h = max(profits, key=lambda item: item[2])
        if max_b_s_p_l_h[2] > 0:
            sandwiches.append(max_b_s_p_l_h)
            profits = [x for x in profits if x[0][1] < max_b_s_p_l_h[0][0] or x[0][0] > max_b_s_p_l_h[0][1]]

    # collect remaining slices
    slices = [x for x in range(oclh.shape[1]) if x not in sum(([list((range(x[0][0], x[0][1]+1))) for x in sandwiches]), [])]

    orders = sandwiches

    # create buy sell orders from slices and collect all orders in a list
    for s in slices:
        slice_ = oclh[:, s]
        if slice_[3] - slice_[0] > 0 or slice_[1] - slice_[2] > 0:
            # if high-open is bigger than close-low
            if slice_[3] - slice_[0] > slice_[1] - slice_[2]:
                orders.append(((s, s), (slice_[0], slice_[3]), slice_[3]-slice_[0], (0, 3))) # buy at open, sell at high
            else:
                orders.append(((s, s), (slice_[2], slice_[1]), slice_[1]-slice_[2], (2, 1))) # buy at low, sell at close

    # sorted orders in the format of [(buy_interval_index, sell_interval_index), (buy_price, sell_price), profit]
    orders = sorted(orders, key=lambda x:x[0][0])
    orders = [{"buy_interval":x[0][0], "sell_interval":x[0][1], "buy_price":x[1][0], "sell_price":x[1][1], "profit":x[2], "buy_sell_features":x[3]} for x in orders]

    return orders

def toast_bread_agent(inference_dataset, df, all_predictions, bank_start, error_margin):
    """
    Toast Bread strategy is a strategy that I came up with, which utilizes the limited future insight that keeps flowing when time progresses. It consists of 2 stages: stage A where the agent creates a plan and stage B/C where the agent acts upon the newly came future insight. Before explaining the underlying logic, first lets talk about a scenario where our trained models make predictions for the 8 upcoming time-series data with 6 hours intervals. It predicts 4 features for each interval: open, close, low and high. So our future-insight is in this format: [d1.0-6, d1.6-12, d1.12-18, d1.18-24, d2.0-6, d2.6-12, d2.12-18, d2.18-24] where di.s-e means ith interval's [open, close, low, end] information for the interval between start hour s and end hour h.
    And these are the terms for having common ground on the explanations.
    'slice' is a 6 hour interval, it contains an open and a close price that is at the start and end of the 6 hour interval. And it contains low and high prices which are the lowest and highest prices the coin has achieved in this interval. We don't have any time information about these two. 'di.s-e' is a slice.
    'sandwich' is consecutive slices back to back. A single slice is not a sandwich.

    And we set all of our buy and sell orders with an error margin. Lets say the coin comes 100$ close to our prediction price, then we take the action, and if it doesn't we don't do anything. Below is how the strategy works step by step:
    A. 1- find the best profiting sandwich, that has the profit margin of (highest-lowest), where 'lowest' is the low price of the slice that is at the beginning of the sandwich; and 'highest' is the high price of the slice that is at the end of the sandwich.
    A. 2- create a buy-sell order to buy the coin at sandwich's lowest and sell at sandwich's highest
    A. 3- remove this sandwich from the data
    A. 4- continue doing A.1-2-3 steps at remaining data (but we approach them as seperate time-series data since removed sandwich split them into two unrelated parts) till there is no profitable sandwich left.
    A. 5- for all remaining slices in our data, do one of these:
    A. 5.1- create a buy order at open price if we can sell at close price or high price with profit
    A. 5.2- create a buy order at low price if we can sell at close price with profit

    And now we have a plan with a list of set buy-sell orders for the sandwichs (also could be empty), and also buy-sell orders for the slices (also could be empty). If we have some orders for the current interval that we are in right now, we do them. Now we go to next time interval. Since we enter the new interval, we now got next predictions from the models. It covers again upcoming 8 intervals so there is a 7 interval overlap but we select to trust newly came predictions rather than old ones. After entering the next interval, now we can be in two stages, first easier stage B:
    B. 1- we dont have any ongoing plan (meaning we are not inbetween some buy-sell order of a sandwich from stage A. We either did a slice buy-sell order, or didn't take any actions at all)
    B. 2- if so, then we go back to stage A where we had no plan, and do A.1-2-3-4-5 buy-sell order setting steps again with the newly came predictions.

    or harder stage C:
    C. 1- we have some ongoing plan (meaning we had a set buy-sell order which we completed the buy part and now waiting for the sell part)
    C. 2- we trust the newly came slices more than our past data so we discard all plans other than the ongoing one because.
    C. 3- and we check the state of our ongoing plan, which can be updated in three different way as below:
    C. 4.1- if we have a better selling point at a newly predicted slice's high price on the timeline, we change our current ongoing plan's selling point to that point, and discard any prediction that comes before that. do the steps A-1-2-3-4-5 for the remaining data
    C. 4.2- if we don't have a better selling point at a newly came slice further on the timeline and our previous selling slice's high price is still valid, continue with our plan and do the A-1-2-3-4-5 with remaining data
    C. 4.3, if we don't have a better selling point at further on the timeline and our previous selling slice's high price is now invalid (which means that the newly came predictions contradict with the past ones), find the highest selling point that we can find, set our ongoing plan's selling point to that slice's high price to get away with lowest money loss. and do the A-1-2-3-4-5 with remaining ones

    This method is named after a toast bread since it's easier to visualize this 6 hour intervals with an image that has a width rather than a timeline of the data (it's confusing to think about our open, close, low, high features when first 2 has a timestamp and second two's occurance time is not knoww which means we can't set a buy-sell order like buying at di.12-18.low and selling at di.12-18.high since high might have occured before the low). Also it helps with the easy-to-remember non-terminologic words by naming the intervals as slices and back to back intervals as sandwiches.
    This way we can show the effects of 'knowing predicted future' on the agent profits compared to 'unknown future' and also 'ground-truth future'.
    """
    bank = bank_start
    wallet = 0
    ongoing_order = None
    sold_at_first_order_open_to_buy_later_for_first_order = False
    buys, sells = [], []

    cols_and_indices = {col:e for e,col in enumerate(df.columns)}
    permute_order = [cols_and_indices[col] for col in ["open", "close", "low", "high"]]

    intended_orders = []
    values = []
    all_predictions = all_predictions.numpy()
    for i in tqdm(range(len(df)), desc="toast bread", leave=False):
        if i % 100 == 0:
            gc.collect()
        oclh_initial = df.iloc[i].values

        oclh_predictions = all_predictions[i]
        oclh_predictions = inference_dataset.rescale_to_real_price(torch.from_numpy(oclh_predictions.T), torch.from_numpy(oclh_initial.T), profit_inference=True)

        # now transform the oclh_predictions back into the open,close,low,high ordered columns
        oclh_predictions = oclh_predictions[:, permute_order].numpy()

        # if we are at the last iteration that means we already lived through the csv intervals
        # and only thing to do is to calculate the upcoming orders and plan the future strategy
        # so return both the scaled predictions and also the planned orders
        if i == len(df)-1:
            # return oclh_predictions
            upcoming_orders = get_orders(oclh_predictions.T)
            return values, buys, sells, intended_orders, oclh_predictions, upcoming_orders

        oclh_to_buy_sell_from = df.iloc[i+1].values
        # now transform the oclh back into the open,close,low,high ordered columns
        oclh_to_buy_sell_from = oclh_to_buy_sell_from[permute_order]

        if ongoing_order is None:
            orders = get_orders(oclh_predictions.T)

            if len(orders) > 0 and orders[0]["buy_interval"] == 0:
                ongoing_order = orders[0]
                if bank > 0:
                    real_price_for_buying = oclh_to_buy_sell_from[ongoing_order["buy_sell_features"][0]]
                    
                    if ongoing_order["buy_price"] + error_margin * real_price_for_buying > real_price_for_buying:
                        ongoing_order["buy_price"] = ongoing_order["buy_price"] + error_margin * real_price_for_buying
                        intended_order = ongoing_order.copy()
                        intended_order["buy_interval"] = intended_order["buy_interval"] + i + 1
                        intended_order["sell_interval"] = intended_order["sell_interval"] + i + 1
                        intended_orders.append(intended_order)
                        wallet += bank / ongoing_order["buy_price"]
                        bank = 0
                        buys.append((i+1, ongoing_order["buy_price"], ongoing_order["buy_sell_features"][0]))
                        # print("buying at", i, "to open the ongoing order", ongoing_order)
                    else:
                        ongoing_order = None
                        values.append(bank + wallet * oclh_to_buy_sell_from[1])
                        continue
            else:
                values.append(bank + wallet * oclh_to_buy_sell_from[1])
                continue
            
            if ongoing_order["sell_interval"] == 0:
                if wallet > 0:
                    real_price_for_selling = oclh_to_buy_sell_from[ongoing_order["buy_sell_features"][1]]

                    if ongoing_order["sell_price"] - error_margin * real_price_for_selling < real_price_for_selling:
                        ongoing_order["sell_price"] = ongoing_order["sell_price"] - error_margin * real_price_for_selling
                        bank += wallet * ongoing_order["sell_price"]
                        wallet = 0
                        sells.append((i+1, ongoing_order["sell_price"], ongoing_order["buy_sell_features"][1]))
                    else: # if we couldn't profit, sell at today's close
                        ongoing_order["sell_price"] = oclh_to_buy_sell_from[1]
                        bank += wallet * ongoing_order["sell_price"]
                        wallet = 0
                        sells.append((i+1, ongoing_order["sell_price"], ongoing_order["buy_sell_features"][1]))

                    # print("selling at", i, "to open the ongoing order", ongoing_order)
                    ongoing_order = None
            
            if ongoing_order:
                ongoing_order["buy_interval"] -= 1
                ongoing_order["sell_interval"] -= 1

        else:
            try:
                first_order = get_orders(oclh_predictions.T)[0]
            except IndexError:
                first_order = None

            if first_order is not None:
                # ongoing order eats the first order if it's profit will get bigger
                if first_order["buy_price"] > ongoing_order["sell_price"]:
                    ongoing_order = {"buy_interval": ongoing_order["buy_interval"], "sell_interval": first_order["sell_interval"],
                                    "buy_price": ongoing_order["buy_price"],"sell_price": first_order["sell_price"],
                                    "profit": first_order["sell_price"] - ongoing_order["buy_price"],
                                    "buy_sell_features": (ongoing_order["buy_sell_features"][0], first_order["buy_sell_features"][1])}
                else: # there is more profit chance at selling ongoing and then doing the first order do that
                    if first_order["buy_interval"] == 0 and ongoing_order["sell_interval"] != 0:
                        # set the ongoing_order sell date as the open price of the first_order's buy interval
                        ongoing_order = {"buy_interval": ongoing_order["buy_interval"], "sell_interval": 0,
                                    "buy_price": ongoing_order["buy_price"],"sell_price": oclh_predictions[0, 0],
                                    "profit": oclh_predictions[0, 0] - ongoing_order["buy_price"],
                                    "buy_sell_features": (ongoing_order["buy_sell_features"][0], 0)}
                        
                        sold_at_first_order_open_to_buy_later_for_first_order = True
                    elif first_order["buy_interval"] > 0:
                        # set the ongoing_order sell date as the highest that comes before first order
                        highest_interval_before_first_order, highest_price_before_first_order = np.argmax(oclh_predictions[3, :first_order["buy_interval"]]), max(oclh_predictions[3, :first_order["buy_interval"]])
                        ongoing_order = {"buy_interval": ongoing_order["buy_interval"], "sell_interval": highest_interval_before_first_order,
                                    "buy_price": ongoing_order["buy_price"],"sell_price": highest_price_before_first_order,
                                    "profit": highest_price_before_first_order - ongoing_order["buy_price"],
                                    "buy_sell_features": (ongoing_order["buy_sell_features"][0], 3)}

            # if we set the ongoing_order sell date as today, sell it
            if ongoing_order["sell_interval"] == 0:
                if wallet > 0:
                    real_price_for_selling = oclh_to_buy_sell_from[ongoing_order["buy_sell_features"][1]]

                    if ongoing_order["sell_price"] - error_margin * real_price_for_selling < real_price_for_selling:
                        ongoing_order["sell_price"] = ongoing_order["sell_price"] - error_margin * real_price_for_selling
                        bank += wallet * ongoing_order["sell_price"]
                        wallet = 0
                        sells.append((i+1, ongoing_order["sell_price"], ongoing_order["buy_sell_features"][1]))
                    else: # if we couldn't profit, sell at today's close
                        ongoing_order["sell_price"] = oclh_to_buy_sell_from[1]
                        bank += wallet * ongoing_order["sell_price"]
                        wallet = 0
                        sells.append((i+1, ongoing_order["sell_price"], ongoing_order["buy_sell_features"][1]))

                    # print("selling at", i, "to open the ongoing order", ongoing_order)
                    ongoing_order = None

            if sold_at_first_order_open_to_buy_later_for_first_order:
                sold_at_first_order_open_to_buy_later_for_first_order = False
                ongoing_order = first_order
                if bank > 0:
                    real_price_for_buying = oclh_to_buy_sell_from[ongoing_order["buy_sell_features"][0]]
                    
                    if ongoing_order["buy_price"] + error_margin * real_price_for_buying> real_price_for_buying:
                        ongoing_order["buy_price"] = ongoing_order["buy_price"] + error_margin * real_price_for_buying
                        intended_order = ongoing_order.copy()
                        intended_order["buy_interval"] = intended_order["buy_interval"] + i + 1
                        intended_order["sell_interval"] = intended_order["sell_interval"] + i + 1
                        intended_orders.append(intended_order)
                        wallet += bank / ongoing_order["buy_price"]
                        bank = 0
                        buys.append((i+1, ongoing_order["buy_price"], ongoing_order["buy_sell_features"][0]))
                        # print("buying at", i, "sold_at_first_order_open_to_buy_later_for_first_order", ongoing_order)
                    else:
                        ongoing_order = None
                        values.append(bank + wallet * oclh_to_buy_sell_from[1])
                        continue
            
            if ongoing_order:
                ongoing_order["buy_interval"] -= 1
                ongoing_order["sell_interval"] -= 1

        values.append(bank + wallet * oclh_to_buy_sell_from[1])

def plot_profit_inference(df, buys, sells, intended_orders, upcoming_oclh_predictions, upcoming_orders, delta):
    df = df.reset_index()

    # Compute interval end as next row's open_time
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = df["open_time"].shift(-1)
    df.loc[df.index[-1], "close_time"] = df.loc[df.index[-1], "open_time"] + pd.Timedelta(delta)

    fig = go.Figure()

    # --- Open dots (at start of interval) ---
    fig.add_trace(go.Scatter(x=df["open_time"], y=df["open"],mode="markers", name="Open",marker=dict(color="purple", size=4, symbol="circle")))

    # --- Close dots (at end of interval = next start) ---
    fig.add_trace(go.Scatter(x=df["close_time"], y=df["close"],mode="markers", name="Close",marker=dict(color="blue", size=4, symbol="circle")))

    # --- High lines (across interval) ---
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(x=[row["open_time"], row["close_time"]],y=[row["high"], row["high"]],mode="lines", line=dict(color="green"),showlegend=(i == 0), name="High"))

    # --- Low lines (across interval) ---
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(x=[row["open_time"], row["close_time"]],y=[row["low"], row["low"]],mode="lines", line=dict(color="red"),showlegend=(i == 0), name="Low"))

    # --- Vertical low-high lines to seperate the intervals ---
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(x=[row["open_time"], row["open_time"]],y=[row["low"], row["high"]],mode="lines",line=dict(color="gray", width=1, dash="dot"),showlegend=False))
        fig.add_trace(go.Scatter(x=[row["close_time"], row["close_time"]],y=[row["low"], row["high"]],mode="lines",line=dict(color="gray", width=1, dash="dot"),showlegend=False))

    # calculates the time of the buy sell to plot nicely
    def get_trade_time(row, feature):
        open_t = row["open_time"]
        end_t = row["close_time"]

        if feature == 0:
            return open_t
        elif feature == 1:
            return end_t
        else:
            return open_t + (end_t - open_t) / 2 # midpoint

    # Draw trades
    for i, ((buy_idx, buy_price, buy_feature), (sell_idx, sell_price, sell_feature)) in enumerate(zip(buys, sells)):
        buy_t = get_trade_time(df.iloc[buy_idx], buy_feature)
        sell_t = get_trade_time(df.iloc[sell_idx], sell_feature)

        fig.add_trace(go.Scatter(x=[buy_t, sell_t],y=[buy_price, sell_price],mode="lines+markers", line=dict(color="black" if sell_price > buy_price else "brown",width=3),showlegend=(i == 0),marker=dict(size=6), name="Trade"))

    # puts a red shade on the predictions lines
    cutoff_time = df["close_time"].iloc[-1]
    fig.add_vrect(x0=cutoff_time,x1=cutoff_time + delta * upcoming_oclh_predictions.shape[0],fillcolor="lightcoral",
                opacity=0.2,layer="below",line_width=0,annotation_text="Prediction",annotation_position="top left")

    # calculating time indices for the predictions
    t_minus1 = df["open_time"].iloc[-1]
    t_minus2 = df["open_time"].iloc[-2]

    delta = t_minus1 - t_minus2
    predicted_time = pd.date_range(start=t_minus1+delta, periods=upcoming_oclh_predictions.shape[0], freq=delta)

    # --- Predicted Open dots (at start of interval) ---
    fig.add_trace(go.Scatter(x=predicted_time,y=upcoming_oclh_predictions.T[0],mode="markers",name="Pred Open",marker=dict(color="purple", size=6, symbol="circle-open")))

    # --- Predicted Close dots (at end of interval = next start) ---
    pred_end_time = list(predicted_time[1:]) + [predicted_time[-1] + (predicted_time[1] - predicted_time[0])]
    fig.add_trace(go.Scatter(x=pred_end_time,y=upcoming_oclh_predictions.T[1],mode="markers",name="Pred Close",marker=dict(color="blue", size=6, symbol="circle-open")))

    # --- Predicted High lines (across interval) ---
    for i in range(len(predicted_time)):
        fig.add_trace(go.Scatter(x=[predicted_time[i], pred_end_time[i]],y=[upcoming_oclh_predictions[i, 3], upcoming_oclh_predictions[i, 3]],mode="lines",line=dict(color="green", dash="dot"),showlegend=(i == 0),name="Pred High"))

    # --- Predicted Low lines (across interval) ---
    for i in range(len(predicted_time)):
        fig.add_trace(go.Scatter(x=[predicted_time[i], pred_end_time[i]],y=[upcoming_oclh_predictions[i, 2], upcoming_oclh_predictions[i, 2]],mode="lines",line=dict(color="red", dash="dot"),showlegend=(i == 0),name="Pred Low"))

    # --- Predicted vertical Low-High lines ---
    for i in range(len(predicted_time)):
        start_t = predicted_time[i]
        low_val = upcoming_oclh_predictions[i, 2]
        high_val = upcoming_oclh_predictions[i, 3]
        fig.add_trace(go.Scatter(x=[start_t, start_t],y=[low_val, high_val],mode="lines",line=dict(color="gray", width=1, dash="dot"),showlegend=(i == 0),name="Pred Low-High"))

    # calculates the buy sell time for the planned orders
    def get_predicted_trade_time(interval_idx, feature_idx, time_slots):
        try:
            start = time_slots[interval_idx]
        except KeyError: # in case our sell date was out of the df timeline at the last intended order
            start = time_slots[len(time_slots)-1] + delta * (interval_idx - (len(time_slots) - 1))
        end = start + delta

        # Position according to feature
        if feature_idx in [0, 1]:  # open or close -> start or end
            if feature_idx == 0:
                return start
            else:
                return end
        else:  # low or high -> midpoint of interval
            return start + delta / 2

    # Plot predicted upcoming_orders
    for i, order in enumerate(intended_orders):
        buy_time = get_predicted_trade_time(order['buy_interval'], order['buy_sell_features'][0], df["open_time"])
        sell_time = get_predicted_trade_time(order['sell_interval'], order['buy_sell_features'][1], df["open_time"])
        buy_price = order['buy_price']
        sell_price = order['sell_price']
        profit_ratio = sell_price / buy_price

        # Draw future planned orders
        fig.add_trace(go.Scatter(x=[buy_time, sell_time],y=[buy_price, sell_price],mode='lines+markers+text',line=dict(color='gray', width=1, dash="dash"),showlegend=(i == 0),marker=dict(size=4),textposition="top center",name="Intended Trade"))

    # Plot predicted upcoming_orders
    for i, order in enumerate(upcoming_orders):
        buy_time = get_predicted_trade_time(order['buy_interval'], order['buy_sell_features'][0], predicted_time)
        sell_time = get_predicted_trade_time(order['sell_interval'], order['buy_sell_features'][1], predicted_time)
        buy_price = order['buy_price']
        sell_price = order['sell_price']
        profit_ratio = sell_price / buy_price

        # Draw future planned orders
        fig.add_trace(go.Scatter(x=[buy_time, sell_time],y=[buy_price, sell_price],mode='lines+markers+text',line=dict(color='black', width=2, dash="dot"),marker=dict(size=6),showlegend=(i == 0),text=[None, f"{profit_ratio:.4f}x"],textposition="top center",name="Intended Future Trade"))

    fig.update_layout(height=600)
    fig.show()

def profit_inference(train_session_dir, csv_to_infer, bank_start, plot_interactive_plot=False):
    # read the training session data and recreate the training environment for inference
    json_to_inference = os.path.join(train_session_dir,"args.json")
    with open(json_to_inference, 'r') as f:
        data = json.load(f)

    model_name = data["model_name"]

    model_pt_name = [x for x in os.listdir(train_session_dir) if x.endswith(".pt") and "series" not in x][0]
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
    "transform_name":args.transform_name, "output_distribution": args.output_distribution, "n_quantiles": args.n_quantiles, 
    "merge_count": args.merge_count, "price_loss_with_real": args.price_loss_with_real, "price_loss_weight": args.price_loss_weight,
    "train_session_dir": train_session_dir}
    
    inference_dataset_kwargs = {**base_dataset_kwargs, "csv_path": csv_to_infer, "augmentation_p": 0, "training_dataset":0}
    inference_dataset = import_dataset(args.dataset_name, **inference_dataset_kwargs)

    all_predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(inference_dataset) + args.output_window), desc="profit inference", leave=False):
            x_batch, y_batch = inference_dataset[i]
            x_batch = x_batch.unsqueeze(0)
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(0)
            y_batch = y_batch.to(device)

            preds = model.call(x_batch, y_batch)
            all_predictions.append(preds.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)  # [n, features, timeline]
    torch.save(all_predictions, os.path.join(train_session_dir, f'{model_name}_inference_series.pt'))

    # read the data, get the columns of the coin
    df = pd.read_csv(csv_to_infer, index_col="open_time")
    df = df[inference_dataset.output_cols]
    df.columns = [col.split("_")[1] for col in df.columns]

    # calculate where will the predictions start from
    learned_dataframe_crop = inference_dataset.df.iloc[data["input_window"]:]
    t1 = datetime.strptime(learned_dataframe_crop.index[0], "%Y-%m-%d %H:%M:%S")
    t2 = datetime.strptime(learned_dataframe_crop.index[1], "%Y-%m-%d %H:%M:%S")
    delta = t2 - t1
    t0 = t1 - delta

    # get that portion of the data
    # df dataset will only be used to retrieve the real prices of the current interval
    # and we will use the all_predictions tensor for the upcoming interval planning
    df = df.loc[str(t0):]

    # df[i] -> should give the real price of the interval i
    # all_predictions[i] -> should give the upcoming args.output_window intervals' predictions after the interval i
    # so if we are at the end of the interval0, df[i] will give the data from interval0, and all_predictions will give the data from interval1-8
    # we will use the df[i] as an initial price for the all_predictions[i]

    values, buys, sells, intended_orders, upcoming_oclh_predictions, upcoming_orders = toast_bread_agent(inference_dataset, df, all_predictions, bank_start=bank_start, error_margin=0)
    if plot_interactive_plot:
        plot_profit_inference(df, buys, sells, intended_orders, upcoming_oclh_predictions, upcoming_orders, delta)
    
    return values

def compare_profits(train_session_dir, trading_agents_and_values, toast_bread_values, toast_bread_wait, bank_start):
    start_filled_toast_bread = [bank_start for _ in range(toast_bread_wait + 1)] + toast_bread_values

    plt.figure(figsize=(30, 15))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trading_agents_and_values)))

    max_value = bank_start
    for (k, v), c in zip(trading_agents_and_values.items(), colors):
        max_value = max(max(v), max_value)
        if k in ["volatility_agent", "low_high_slope_agent", "yesterday_trend_breaking_agent", "candlestick_pattern_agent", "buy_yesterdays_low_sell_yesterdays_high_agent"]:
            plt.plot(v, color=c, label=k, linestyle="dashed")
        else:
            plt.plot(v, color=c, label=k, linestyle="dotted")
        plt.text(len(v) - 1, v[-1], f" {k}", color=c, va="center", fontsize=15)
    
    for y in range(bank_start, int(max_value), 1000):
        plt.axhline(y=y, color="gray", linestyle=":", linewidth=0.8)

    plt.plot(start_filled_toast_bread, color="black", linestyle="solid", linewidth=2, label="toast_bread")
    plt.text(len(start_filled_toast_bread) - 1, start_filled_toast_bread[-1], " toast_bread", color="black", va="center", fontsize=15, fontweight="bold")
    plt.legend()

    plt.title(f'Trading Strategies Profit Comparison')
    plt.xlabel("Time")
    plt.ylabel("$ Value")
    plt.savefig(os.path.join(train_session_dir, f'evaluation_graphs/profit_comparison_with_equal_start.png'))
    plt.show()