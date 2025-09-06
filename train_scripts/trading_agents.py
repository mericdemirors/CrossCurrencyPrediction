import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

def random_all_in_agent(df, bank_start, buy_probability=0.3):
    bank = bank_start
    wallet = 0

    action = "buy"

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        price_to_buy_or_sell = (today_data["open"] + today_data["close"] + today_data["low"] + today_data["high"]) / 4

        if random.random() < buy_probability:
            if action == "buy" and bank > 0:
                wallet = bank / price_to_buy_or_sell
                bank = 0
                action = "sell"
            elif action == "sell" and wallet > 0:
                bank = wallet * price_to_buy_or_sell
                wallet = 0
                action = "buy"

        values.append(bank + wallet * today_data["close"])

    return values

def random_portional_agent(df, bank_start, buy_probability=0.15):
    bank = bank_start
    wallet = 0

    action = "buy"

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        price_to_buy_or_sell = (today_data["open"] + today_data["close"] + today_data["low"] + today_data["high"]) / 4

        if random.random() < buy_probability:
            portion_to_use = random.random()

            if action == "buy" and bank > 0:
                wallet = wallet + (bank*portion_to_use) / price_to_buy_or_sell
                bank = bank - bank*portion_to_use
                action = "sell"
            elif action == "sell" and wallet > 0:
                bank = bank + wallet * portion_to_use * price_to_buy_or_sell
                wallet = wallet - wallet*portion_to_use
                action = "buy"

        values.append(bank + wallet * today_data["close"])

    return values

def holder_buyer_seller_agent(df, bank_start, wait_period=1):
    bank = bank_start
    wallet = 0
    prev_price_to_buy_or_sell = np.inf
    buying_period = wait_period
    action = "buy"

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        price_to_buy_or_sell = today_data["open"]

        # if it's time to buy, and we have money
        if action == "buy" and bank > 0:
            wallet = bank / today_data["open"]
            bank = 0
            action = "sell"
            prev_price_to_buy_or_sell = price_to_buy_or_sell
            buying_period = wait_period

        # either we bought and now waiting for hold times to pass
        # or we are waiting for
        if buying_period != 0:
            buying_period -= 1
            values.append(bank + wallet * today_data["close"])
            continue

        # if it's time to sell, and we have coin, and we can profit from selling
        if action == "sell" and wallet > 0 and price_to_buy_or_sell > prev_price_to_buy_or_sell:
            bank = wallet * today_data["close"]
            wallet = 0
            action = "buy"
            buying_period = wait_period
        
        values.append(bank + wallet * today_data["close"])

    return values

def holder_agent(df, bank_start):
    bank = bank_start
    wallet = 0

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        if i == 0:
            wallet = bank / today_data["open"]
            bank = 0
        values.append(bank + wallet * today_data["close"])

    return values

def martingale_agent(df, bank_start, base_bet=1, set_holding_period=2):
    bank = bank_start
    wallet = 0   # Amount of coin
    current_bet = base_bet
    prev_price_to_buy = np.inf
    holding_period = 0

    action = "buy"

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        price_to_buy_or_sell = (today_data["open"] + today_data["close"] + today_data["low"] + today_data["high"]) / 4

        # Handle buy logic
        if action == "buy" and bank >= current_bet:
            coins_bought = current_bet / price_to_buy_or_sell
            wallet += coins_bought
            bank -= current_bet
            prev_price_to_buy = price_to_buy_or_sell
            holding_period = set_holding_period
            action = "sell"
        
        # Wait for the holding period to pass
        elif action == "sell":
            holding_period -= 1
            if holding_period > 0:
                values.append(bank + wallet * today_data["close"])
                continue
            else:
                # Holding period is over; decide whether to sell
                coin_value_now = wallet * price_to_buy_or_sell
                cost_basis = coins_bought * prev_price_to_buy

                profit = coin_value_now - cost_basis

                if profit >= 0:
                    pass  # Reset to base bet
                else:
                    # Double the previous bet or bet enough to cover the loss
                    current_bet = current_bet * 2
                    if current_bet > bank:
                        bank = bank + wallet * price_to_buy_or_sell
                        wallet = 0
                        if current_bet > bank:
                            current_bet = base_bet
                action = "buy"
        
        values.append(bank + wallet * today_data["close"])

    return values

def dollar_cost_averaging_agent(df, bank_start, period=32):
    bank = bank_start
    wallet = 0
    periodic_investment_budget = bank_start / (len(df) / period)

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        price_to_buy_or_sell = (today_data["open"] + today_data["close"] + today_data["low"] + today_data["high"]) / 4

        if i % period == 0 and bank >= periodic_investment_budget:
            # Invest the fixed amount
            quantity_bought = periodic_investment_budget / price_to_buy_or_sell
            wallet += quantity_bought
            bank -= periodic_investment_budget

        values.append(bank + wallet * today_data["close"])

    return values

def sma_crossover_agent(df, bank_start, short_window=1, long_window=2):
    bank = bank_start
    wallet = 0
    action = "buy"  # Initial action can be to buy if there's bank

    # Calculate Simple Moving Averages
    df['SMA_short'] =((df["open"] + df["close"] + df["low"] + df["high"]) / 4).rolling(window=short_window, min_periods=1).mean()
    df['SMA_long'] = ((df["open"] + df["close"] + df["low"] + df["high"]) / 4).rolling(window=long_window, min_periods=1).mean()

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]

        # Check for crossover signals (avoiding the first few periods where SMAs are still calculating)
        if i > long_window:
            previous_row = df.iloc[i-1]
            short_above_long = previous_row['SMA_short'] > previous_row['SMA_long']
            short_below_long = previous_row['SMA_short'] < previous_row['SMA_long']

            if short_above_long and action == "buy" and bank > 0:
                # Buy with all available bank
                wallet = bank / today_data["open"]
                bank = 0
                action = "sell"

            elif short_below_long and action == "sell" and wallet > 0:
                # Sell all available wallet
                bank = wallet * today_data["open"]
                wallet = 0
                action = "buy"

        values.append(bank + wallet * today_data["close"])

    return values

def volatility_agent(df, bank_start, low_volatility_threshold_coeff=3, high_volatility_threshold_coeff=2.0):
    bank = bank_start
    wallet = 0
    action = "buy"  # Initial action

    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=16, min_periods=1).mean()

    values = []
    for i in range(len(df)):
        today_data = df.iloc[i]
        price_to_buy_or_sell = today_data["open"]
        atr = today_data["ATR"]

        # Define dynamic volatility thresholds based on the mean ATR
        mean_atr = df['ATR'].iloc[:i].mean() if i > 0 else atr  # Use historical mean

        if mean_atr > 0:
            low_volatility_threshold = mean_atr * low_volatility_threshold_coeff
            high_volatility_threshold = mean_atr * high_volatility_threshold_coeff
        else:
            low_volatility_threshold = 0
            high_volatility_threshold = float('inf')

        # Buy logic: After a period of low volatility, expecting a breakout
        if atr < low_volatility_threshold and action == "buy" and bank > 0:
            # Buy with all available bank
            wallet = bank / price_to_buy_or_sell
            bank = 0
            action = "sell"

        # Sell logic: After a period of high volatility, expecting consolidation
        elif atr > high_volatility_threshold and action == "sell" and wallet > 0:
            # Sell all available wallet
            bank = wallet * price_to_buy_or_sell
            wallet = 0
            action = "buy"

        values.append(bank + wallet * today_data["close"])

    return values

def percentage_change_following_agent(df, bank_start, buying_start=8, selling_start=16):
    bank = bank_start
    wallet = 0
    action = "buy"  # Initial action

    values = []
    for i in range(len(df)):
        price_to_buy_or_sell = df['open'].iloc[i]
        last_close_price = df['close'].iloc[i-1]
        # Check for buy signal
        if action == "buy" and bank > 0 and i >= buying_start:
            start_close_price = df['close'].iloc[i - buying_start]
            percentage_change = (last_close_price - start_close_price) / start_close_price
            if percentage_change >= 0.05:
                # Buy with all available bank
                wallet = bank / price_to_buy_or_sell
                bank = 0
                action = "sell"

        # Check for sell signal
        elif action == "sell" and wallet > 0 and i >= selling_start:
            start_close_price = df['close'].iloc[i - selling_start]
            percentage_change = (last_close_price - start_close_price) / start_close_price
            if percentage_change <= -0.05:
                # Sell all available wallet
                bank = wallet * price_to_buy_or_sell
                wallet = 0
                action = "buy"

        values.append(bank + wallet * df['close'].iloc[i])

    return values

def low_high_slope_agent(df, bank_start, window=16, error_margin=0):
    bank = bank_start
    wallet = 0
    values = []
    action = "buy"

    for i in range(len(df)):
        if i < window:
            # Not enough history for prediction
            values.append(bank + wallet * df.iloc[i]["close"])
            continue

        history = df.iloc[i-window:i]  # last window days
        today_data = df.iloc[i]

        # calculate slope of the low and high
        slope_low = (history["low"].iloc[-1] - history["low"].iloc[0]) / window
        slope_high = (history["high"].iloc[-1] - history["high"].iloc[0]) / window

        # predict the today's low, today's high and tomorrow's high
        predicted_low = history["low"].iloc[-1] + slope_low
        predicted_high = history["high"].iloc[-1] + slope_high
        predicted_next_high = history["high"].iloc[-1] + slope_high * 2
        
        # buy at open and sell at today's high
        open_buy_condition = ((predicted_high - today_data["open"]) > error_margin) and ((today_data["high"] - predicted_high) > error_margin)
        # buy at today's low and sell at tomorrow's high
        buy_condition = ((predicted_next_high - predicted_low) > 2 * error_margin) and ((predicted_low - today_data["low"]) > error_margin)

        if action == "sell":
            # sell the moment you go negative, or if today's predicted high is bigger than buying price
            sell_condition = ((today_data["open"] - buyed_at < 0) or (predicted_high - buyed_at) > error_margin) and ((today_data["high"] - predicted_high) > error_margin)

        # if it's better to buy today's opening and sell at high rather than today's low to tomorrows high, do that
        if action == "buy" and open_buy_condition and predicted_high - today_data["open"] > predicted_next_high - predicted_low:
            wallet = bank / today_data["open"]
            bank = 0
            action = "sell"
            buyed_at = today_data["open"]

            bank = wallet * (predicted_high - error_margin)
            wallet = 0
            action = "buy"
        else:
            if action == "buy" and buy_condition and bank > 0:
                wallet = bank / (predicted_low + error_margin)
                bank = 0
                action = "sell"
                buyed_at = (predicted_low + error_margin)
            elif action == "sell" and sell_condition:
                bank = wallet * (predicted_high - error_margin)
                wallet = 0
                action = "buy"

        # Track portfolio value
        values.append(bank + wallet * today_data["close"])

    return values

def yesterday_trend_breaking_agent(df, bank_start, lookback=4):
    bank = bank_start
    wallet = 0
    values = []
    action = "buy"

    for i in range(len(df)):
        if i < lookback:
            values.append(bank + wallet * df.iloc[i]["close"])
            continue

        yesterday = df.iloc[i-1]
        today_data = df.iloc[i]

        # check the previous day's min max limits, except yesterday
        trend_high = df["high"].iloc[i-lookback:i-1].max()
        trend_low = df["low"].iloc[i-lookback:i-1].min()

        # check if yesterdays high or low break these limits
        # buy if yesterday peaked a new high (price will go up) or sell if yesterday dipped a new low (price will go down)
        sell_condition = yesterday["close"] > trend_high
        buy_condition = yesterday["close"] < trend_low

        # use execution price within today's range
        price_to_buy = today_data["open"]
        price_to_sell = today_data["close"]

        if action == "buy" and buy_condition and bank > 0:
            wallet = bank / price_to_buy
            bank = 0
            action = "sell"
        elif action == "sell" and sell_condition and wallet > 0:
            bank = wallet * price_to_sell
            wallet = 0
            action = "buy"

        values.append(bank + wallet * today_data["close"])
    return values

def mean_reversion_agent(df, bank_start, window=4, threshold=0.2):
    bank = bank_start
    wallet = 0
    values = []
    action = "buy"

    df["ma"] = df["close"].rolling(window).mean()

    for i in range(len(df)):
        if i == 0 or pd.isna(df.iloc[i-1]["ma"]):
            values.append(bank + wallet * df.iloc[i]["close"])
            continue

        yesterday = df.iloc[i-1]
        today_data = df.iloc[i]

        deviation = (yesterday["close"] - yesterday["ma"]) / yesterday["ma"]
        buy_condition = deviation < -threshold
        sell_condition = deviation > threshold

        price_to_buy = today_data["open"]
        price_to_sell = today_data["open"]

        if action == "buy" and buy_condition and bank > 0:
            wallet = bank / price_to_buy
            bank = 0
            action = "sell"
        elif action == "sell" and sell_condition and wallet > 0:
            bank = wallet * price_to_sell
            wallet = 0
            action = "buy"

        values.append(bank + wallet * today_data["close"])
    return values

def candlestick_pattern_agent(df, bank_start, body_ratio=0.2):
    bank = bank_start
    wallet = 0
    values = []
    action = "buy"

    for i in range(len(df)):
        if i == 0:
            values.append(bank + wallet * df.iloc[i]["close"])
            continue

        yesterday = df.iloc[i-1]
        today_data = df.iloc[i]

        body = abs(yesterday["close"] - yesterday["open"])
        candle_range = yesterday["high"] - yesterday["low"]

        # Example: Hammer pattern (bullish reversal)
        buy_condition = (body < (candle_range * body_ratio)) and ((yesterday["close"] > yesterday["open"]))
        # Example: Shooting star pattern (bearish reversal)
        sell_condition = (body < (candle_range * body_ratio)) and ((yesterday["close"] < yesterday["open"]))

        price_to_buy = today_data["open"]
        price_to_sell = today_data["open"]

        if action == "buy" and buy_condition and bank > 0:
            wallet = bank / price_to_buy
            bank = 0
            action = "sell"
        elif action == "sell" and sell_condition and wallet > 0:
            bank = wallet * price_to_sell
            wallet = 0
            action = "buy"

        values.append(bank + wallet * today_data["close"])
    return values

def buy_yesterdays_low_sell_yesterdays_high_agent(df, bank_start, error_margin=0):
    bank = bank_start
    wallet = 0
    values = []
    action = "buy"

    for i in range(len(df)):
        today_data = df.iloc[i]
        if i == 0:
            values.append(bank + wallet * today_data["close"])
            continue

        yesterday = df.iloc[i-1]
        # look at yesterday's low, and buy at that price + some margin if it happens today
        if action == "buy" and yesterday["low"] - today_data["low"] > error_margin and bank > 0 and today_data["low"] <= yesterday["low"] <= today_data["high"]:
            wallet = bank / (yesterday["low"] + error_margin)
            bank = 0
            action = "sell"
            buyed_at = today_data["low"]
        # if we are selling, look at yesterday's high, and sell at that price - some margin if it happens today
        elif action == "sell" and today_data["high"] - yesterday["high"] > error_margin and wallet > 0 and today_data["low"] <= yesterday["high"] <= today_data["high"]:
            bank = wallet * (yesterday["high"] - error_margin)
            wallet = 0
            action = "buy"
        # if we are selling and in the negative at the end of the day, sell today from closing price
        elif action == "sell" and buyed_at > today_data["close"]:
            bank = wallet * today_data["close"]
            wallet = 0
            action = "buy"

        values.append(bank + wallet * today_data["close"])
    return values

# ----------------------------------------------------------------------------------------------------------

strategies = {
        "random_all_in_agent": random_all_in_agent,
        "random_portional_agent": random_portional_agent,
        "holder_buyer_seller_agent": holder_buyer_seller_agent,
        "holder_agent": holder_agent,
        "martingale_agent": martingale_agent,
        "dollar_cost_averaging_agent": dollar_cost_averaging_agent,
        "sma_crossover_agent": sma_crossover_agent,
        "volatility_agent": volatility_agent,
        "percentage_change_following_agent": percentage_change_following_agent,
        "low_high_slope_agent": low_high_slope_agent,
        "yesterday_trend_breaking_agent": yesterday_trend_breaking_agent,
        "mean_reversion_agent": mean_reversion_agent,
        "candlestick_pattern_agent": candlestick_pattern_agent,
        "buy_yesterdays_low_sell_yesterdays_high_agent":buy_yesterdays_low_sell_yesterdays_high_agent,
    }

def grid_search_strategies(csv_path, coin_name, bank_start):
    df = pd.read_csv(csv_path, index_col="open_time")
    df = df[[col for col in df.columns if coin_name in col]]
    df.columns = [col.split("_")[1] for col in df.columns]

    hyperparameters = {
        'random_all_in_agent': {'buy_probability': [x/20 for x in range(20)]},
        'random_portional_agent': {'buy_probability': [x/20 for x in range(20)]},
        'holder_buyer_seller_agent': {'wait_period': [1, 2, 4, 8, 16, 32, 48, 64]},
        'holder_agent': {},
        'martingale_agent': {'base_bet': [1, 2, 5, 10, 20, 50, 100],'set_holding_period': [1, 2, 4, 8, 16, 32]},
        'dollar_cost_averaging_agent': {'period': [1, 2, 4, 8, 16, 32, 64]},
        'sma_crossover_agent': {'short_window': [1, 2, 4, 8, 16],'long_window': [2, 4, 8, 16, 32, 64]},
        'volatility_agent': {'low_volatility_threshold_coeff': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],'high_volatility_threshold_coeff': [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]},
        'percentage_change_following_agent': {'buying_start': [1, 2, 4, 8, 16, 32],'selling_start': [1, 2, 4, 8, 16, 32]},
        'low_high_slope_agent': {'window': [1, 2, 4, 8, 16, 32, 48],'error_margin': [1, 2, 4, 8, 16, 32, 50, 100, 250, 500]},
        'yesterday_trend_breaking_agent': {'lookback': [1, 2, 4, 8, 16, 32, 48]},
        'mean_reversion_agent': {'window': [1, 2, 4, 8, 16, 32, 48],'threshold': [0.01, 0.025, 0.05, 0.1, 0.2]},
        'candlestick_pattern_agent': {'body_ratio': [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]},
        "buy_yesterdays_low_sell_yesterdays_high_agent": {"error_margin":[1, 2, 4, 8, 16, 32, 50, 100, 250, 500]}}

    best_results = {}
    for name, agent in strategies.items():
        print(f"\n--- Finding Best Result for {name} ---")
        params_grid = hyperparameters[name]
        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())

        best_last_value = -1
        best_params = None

        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            try:
                values = agent(df.copy(), bank_start, **params)
                last_value = values[-1] if values else bank_start
                if last_value > best_last_value:
                    best_last_value = last_value
                    best_params = params
            except Exception as e:
                print(f"Error with {name} and params {params}: {e}")
        
        best_results[name] = {'params': best_params, 'last_value': best_last_value}

    print("\n--- Best Results for Each Strategy ---")
    for name, result in best_results.items():
        print(f"Strategy: {name:<30} | Best Parameters: {str(result['params']):<60} | Best Final Value: {result['last_value']:.2f}")

def get_trading_agent_values(csv_to_infer, coin_to_infer, bank_start):
    df = pd.read_csv(csv_to_infer, index_col="open_time")
    df = df[[col for col in df.columns if coin_to_infer in col]]
    df.columns = [col.split("_")[1] for col in df.columns]

    agents_and_values = {}
    for agent_name, agent_func in tqdm(strategies.items(), desc="trading agents", leave=False):
        agents_and_values[agent_name] = agent_func(df.copy(), bank_start)

    return agents_and_values