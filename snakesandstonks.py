import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

def generate_mock_data():
    np.random.seed(42)
    days = config.SIMULATION_DAYS
    start_price = config.START_PRICE
    
    if config.MARKET_TYPE == "TRENDING":
        trend = np.linspace(0, 20, days)
        noise = np.random.normal(0, config.VOLATILITY, days)
        prices = start_price + trend + np.cumsum(noise)
        
    elif config.MARKET_TYPE == "MEAN_REVERTING":
        prices = [start_price]
        current_price = start_price
        for _ in range(days - 1):
            pull = (start_price - current_price) * config.REVERSION_SPEED
            noise = np.random.normal(0, config.VOLATILITY)
            current_price += pull + noise
            prices.append(current_price)
            
    else: # "RANDOM_WALK"
        noise = np.random.normal(0, config.VOLATILITY, days)
        prices = start_price + np.cumsum(noise)

    dates = pd.date_range(start="2023-01-01", periods=days)
    return pd.DataFrame({'Date': dates, 'Close': prices})

def run_simulation(strategy_function):
    print("Starting Simulation...\n")
    
    df = generate_mock_data()
    
    # --- Settings ---
    initial_cash = config.INITIAL_CASH
    commission_fee = config.COMMISSION_FEE  # Cost per trade
    trade_size = config.TRADE_SIZE        # Shares per buy/sell
    # ----------------
    
    cash = initial_cash
    shares = 0
    portfolio_value_history = []
    buy_signals = []  
    sell_signals = []

    # Start loop (give them 20 days of data to start)
    for i in range(20, len(df)):
        historical_slice = df.iloc[:i]
        current_price = historical_slice['Close'].iloc[-1]
        current_date = historical_slice['Date'].iloc[-1]
        
        signal = strategy_function(historical_slice, cash, shares)
        
        action_taken = "HOLD"
        
        if signal == "BUY" and cash >= (current_price * trade_size) + commission_fee:
            shares += trade_size
            cash -= (current_price * trade_size)
            cash -= commission_fee
            action_taken = "BUY"
            buy_signals.append((current_date, current_price))
            
        elif signal == "SELL" and shares >= trade_size:
            shares -= trade_size
            cash += (current_price * trade_size)
            cash -= commission_fee 
            action_taken = "SELL"
            sell_signals.append((current_date, current_price))
        
        # Track portfolio value daily
        total_value = cash + (shares * current_price)
        portfolio_value_history.append(total_value)

    # calc final results
    final_price = df['Close'].iloc[-1]
    final_value = cash + (shares * final_price)
    total_profit = final_value - initial_cash
    percent_return = (total_profit / initial_cash) * 100
    
    # calc benchmark
    start_price = df['Close'].iloc[0]
    max_shares_possible = (initial_cash - commission_fee) // start_price
    benchmark_value = (max_shares_possible * final_price) + (initial_cash - (max_shares_possible * start_price) - commission_fee)
    benchmark_return = ((benchmark_value - initial_cash) / initial_cash) * 100

    print("-" * 40)
    print(f"FINAL RESULTS")
    print("-" * 40)
    print(f"Initial Cash:      ${initial_cash:.2f}")
    print(f"Final Value:       ${final_value:.2f}")
    print(f"Total Profit:      ${total_profit:.2f}")
    print(f"Return on Inv:     {percent_return:.2f}%")
    print("-" * 40)
    print(f"📊 BENCHMARK (Buy & Hold)")
    print(f"Benchmark Return:  {benchmark_return:.2f}%")
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Stock Price', alpha=0.5)
    
    # Plot Buy Markers
    if buy_signals:
        b_dates, b_prices = zip(*buy_signals)
        plt.scatter(b_dates, b_prices, marker='^', color='green', s=100, label='Buy')
        
    # Plot Sell Markers
    if sell_signals:
        s_dates, s_prices = zip(*sell_signals)
        plt.scatter(s_dates, s_prices, marker='v', color='red', s=100, label='Sell')

    plt.title(f"Trading Strategy Results (Profit: ${total_profit:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
