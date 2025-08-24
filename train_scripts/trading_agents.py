import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

def random_all_in_agent(df, bank_start, buy_probability=0.8):
    bank = bank_start
    wallet = 0

    action = "buy"

    values = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4

        if random.random() < buy_probability:
            if action == "buy" and bank > 0:
                wallet = bank / price_to_buy_or_sell
                bank = 0
                action = "sell"
            elif action == "sell" and wallet > 0:
                bank = wallet * price_to_buy_or_sell
                wallet = 0
                action = "buy"

        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def random_portional_agent(df, bank_start, buy_probability=0.8):
    bank = bank_start
    wallet = 0

    action = "buy"

    values = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4

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

        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def holder_buyer_seller_agent(df, bank_start, wait_period=1):
    bank = bank_start
    wallet = 0
    prev_price_to_buy_or_sell = np.inf
    buying_period = wait_period
    action = "buy"

    values = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4

        # if it's time to buy, and we have money
        if action == "buy" and bank > 0:
            wallet = bank / row_data["low"]
            bank = 0
            action = "sell"
            prev_price_to_buy_or_sell = price_to_buy_or_sell
            buying_period = wait_period

        # either we bought and now waiting for hold times to pass
        # or we are waiting for
        if buying_period != 0:
            buying_period -= 1
            values.append(bank + wallet * price_to_buy_or_sell)
            continue

        # if it's time to sell, and we have coin, and we can profit from selling
        if action == "sell" and wallet > 0 and price_to_buy_or_sell > prev_price_to_buy_or_sell:
            bank = wallet * row_data["high"]
            wallet = 0
            action = "buy"
            buying_period = wait_period
        
        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def martingale_agent(df, bank_start, base_bet=1, set_holding_period=2):
    bank = bank_start
    wallet = 0   # Amount of coin
    current_bet = base_bet
    prev_price_to_buy = np.inf
    holding_period = 0

    action = "buy"

    values = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4

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
                values.append(bank + wallet * price_to_buy_or_sell)
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
        
        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def dollar_cost_averaging_agent(df, bank_start, period=32):
    bank = bank_start
    wallet = 0
    periodic_investment_budget = bank_start / (len(df) / period)

    values = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4

        if i % period == 0 and bank >= periodic_investment_budget:
            # Invest the fixed amount
            quantity_bought = periodic_investment_budget / price_to_buy_or_sell
            wallet += quantity_bought
            bank -= periodic_investment_budget

        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def sma_crossover_agent(df, bank_start, short_window=1, long_window=8):
    bank = bank_start
    wallet = 0
    action = "buy"  # Initial action can be to buy if there's bank

    # Calculate Simple Moving Averages
    df['SMA_short'] =((df["open"] + df["close"] + df["low"] + df["high"]) / 4).rolling(window=short_window, min_periods=1).mean()
    df['SMA_long'] = ((df["open"] + df["close"] + df["low"] + df["high"]) / 4).rolling(window=long_window, min_periods=1).mean()

    values = []
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4

        # Check for crossover signals (avoiding the first few periods where SMAs are still calculating)
        if i > long_window:
            previous_row = df.iloc[i-1]
            short_above_long = previous_row['SMA_short'] < previous_row['SMA_long'] and row_data['SMA_short'] > row_data['SMA_long']
            short_below_long = previous_row['SMA_short'] > previous_row['SMA_long'] and row_data['SMA_short'] < row_data['SMA_long']

            if short_above_long and action == "buy" and bank > 0:
                # Buy with all available bank
                wallet = bank / price_to_buy_or_sell
                bank = 0
                action = "sell"

            elif short_below_long and action == "sell" and wallet > 0:
                # Sell all available wallet
                bank = wallet * price_to_buy_or_sell
                wallet = 0
                action = "buy"

        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def volatility_agent(df, bank_start, low_volatility_threshold_coeff=2, high_volatility_threshold_coeff=1.75):
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
    for i in tqdm(range(len(df))):
        row_data = df.iloc[i]
        price_to_buy_or_sell = (row_data["open"] + row_data["close"] + row_data["low"] + row_data["high"]) / 4
        atr = row_data["ATR"]

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

        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def trend_following_agent(df, bank_start, buying_start=8, selling_start=16):
    bank = bank_start
    wallet = 0
    action = "buy"  # Initial action

    values = []
    for i in tqdm(range(len(df))):
        price_to_buy_or_sell = df['close'].iloc[i]

        # Check for buy signal
        if action == "buy" and bank > 0 and i >= buying_start:
            previous_price = df['close'].iloc[i - buying_start]
            percentage_change = (price_to_buy_or_sell - previous_price) / previous_price
            if percentage_change >= 0.05:
                # Buy with all available bank
                wallet = bank / price_to_buy_or_sell
                bank = 0
                action = "sell"

        # Check for sell signal
        elif action == "sell" and wallet > 0 and i >= selling_start:
            previous_price = df['close'].iloc[i - selling_start]
            percentage_change = (price_to_buy_or_sell - previous_price) / previous_price
            if percentage_change <= -0.05:
                # Sell all available wallet
                bank = wallet * price_to_buy_or_sell
                wallet = 0
                action = "buy"

        values.append(bank + wallet * price_to_buy_or_sell)

    return values

def grid_search_strategies(csv_path, coin_name, bank_start):
    df = pd.read_csv(csv_path, index_col="open_time")
    df = df[[col for col in df.columns if coin_name in col]]
    df.columns = [col.split("_")[1] for col in df.columns]

    strategies = {
        'random_all_in_agent': random_all_in_agent,
        'random_portional_agent': random_portional_agent,
        'holder_buyer_seller_agent': holder_buyer_seller_agent,
        'martingale_agent': martingale_agent,
        'dollar_cost_averaging_agent': dollar_cost_averaging_agent,
        'sma_crossover_agent': sma_crossover_agent,
        'volatility_agent': volatility_agent,
        'trend_following_agent': trend_following_agent
    }

    hyperparameters = {
        'random_all_in_agent': {'buy_probability': [x/10 for x in range(10)]},
        'random_portional_agent': {'buy_probability': [x/10 for x in range(10)]},
        'holder_buyer_seller_agent': {'wait_period': [1, 4, 8, 16, 32, 48]},
        'martingale_agent': {'base_bet': [1, 2, 5, 10, 20, 50, 100, 200, 500], 'set_holding_period': [1,2,4,8,16,32]},
        'dollar_cost_averaging_agent': {'period': [1, 4, 8, 16, 32, 48]},
        'sma_crossover_agent': {'short_window': [1, 4, 8, 16], 'long_window': [8, 16, 32, 48]},
        'volatility_agent': {'low_volatility_threshold_coeff': [1.0, 1.25, 1.5, 1.75, 2], 'high_volatility_threshold_coeff': [1.5, 1.75, 2, 2.25, 2.5, 3.0]},
        'trend_following_agent': {'buying_start': [1, 4, 8, 16], 'selling_start': [1, 4, 8, 16]}
    }


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

def get_trading_agent_values(csv_to_infer, coin_to_infer, bank_start=1000):
    df = pd.read_csv(csv_to_infer, index_col="open_time")
    df = df[[col for col in df.columns if coin_to_infer in col]]
    df.columns = [col.split("_")[1] for col in df.columns]

    random_all_in_values = random_all_in_agent(df.copy(), bank_start)
    random_portional_values = random_portional_agent(df.copy(), bank_start)
    holder_buyer_seller_values = holder_buyer_seller_agent(df.copy(), bank_start)
    martingale_values = martingale_agent(df.copy(), bank_start)
    dollar_cost_averaging_values = dollar_cost_averaging_agent(df.copy(), bank_start)
    sma_crossover_values = sma_crossover_agent(df.copy(), bank_start)
    volatility_values = volatility_agent(df.copy(), bank_start)
    trend_following_values = trend_following_agent(df.copy(), bank_start)

    agents = ["random_all_in_values", "random_portional_values", "holder_buyer_seller_values", "martingale_values", "dollar_cost_averaging_values", "sma_crossover_values", "volatility_values", "trend_following_values"]
    values = [random_all_in_values, random_portional_values, holder_buyer_seller_values, martingale_values, dollar_cost_averaging_values, sma_crossover_values, volatility_values, trend_following_values]

    return {agent:value for (agent,value) in zip(agents, values)}

