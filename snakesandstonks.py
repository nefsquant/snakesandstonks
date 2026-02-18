import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

def generate_mock_data(seed=42):
    if seed is not None:
        np.random.seed(seed)
        
    days = config.SIMULATION_DAYS
    start_price = config.START_PRICE
    volatility = config.VOLATILITY
    market_type = getattr(config, 'MARKET_TYPE', 'COMPLEX')
    
    # SIMPLE MODES
    if market_type == "TRENDING":
        trend = np.linspace(0, 20, days)
        noise = np.random.normal(0, volatility, days)
        prices = start_price + trend + np.cumsum(noise)
        day_numbers = np.arange(1, days + 1)
        return pd.DataFrame({'Day': day_numbers, 'Close': prices})
        
    elif market_type == "MEAN_REVERTING":
        prices = [start_price]
        current_price = start_price
        for _ in range(days - 1):
            # Pull back to starting price
            reversion_speed = getattr(config, 'REVERSION_SPEED', 0.1)
            pull = (start_price - current_price) * reversion_speed
            noise = np.random.normal(0, volatility)
            current_price += pull + noise
            prices.append(current_price)
        day_numbers = np.arange(1, days + 1)
        return pd.DataFrame({'Day': day_numbers, 'Close': prices})
        
    elif market_type == "RANDOM_WALK":
        noise = np.random.normal(0, volatility, days)
        prices = start_price + np.cumsum(noise)
        day_numbers = np.arange(1, days + 1)
        return pd.DataFrame({'Day': day_numbers, 'Close': prices})
    
    prices = np.zeros(days)
    prices[0] = start_price
    returns = np.zeros(days)
    
    regime_length = 100
    num_regimes = (days // regime_length) + 1
    
    regimes = np.random.choice([0, 1, 2], size=num_regimes, p=[0.4, 0.25, 0.35])
    
    for i in range(1, days):
        current_regime = regimes[i // regime_length]
        
        if current_regime == 0: 
            trend = 0.0008 
        elif current_regime == 1:
            trend = -0.0005 
        else: 
            deviation = (prices[i-1] - start_price) / start_price
            trend = -deviation * 0.03 
        
        cycle = 0.002 * np.sin(2 * np.pi * i / 20)
        
        if i >= 5:
            recent_return = np.mean(returns[i-5:i])
            momentum = recent_return * 0.15  
        else:
            momentum = 0
        
        pct_volatility = volatility / start_price
        noise = np.random.normal(0, pct_volatility)
        
        daily_return = trend + cycle + momentum + noise
        returns[i] = daily_return
        prices[i] = prices[i-1] * (1 + daily_return)
    
    day_numbers = np.arange(1, days + 1)
    return pd.DataFrame({'Day': day_numbers, 'Close': prices})

def export_data_to_csv(seed=42, filename="market_data.csv"):
    market_type = getattr(config, 'MARKET_TYPE', 'COMPLEX')
    print(f"📊 Generating market data (Seed: {seed}, Type: {market_type})...")
    df = generate_mock_data(seed=seed)
    
    # Add some useful calculated columns for analysis
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100  # In percentage
    df['Volatility_10d'] = df['Close'].pct_change().rolling(window=10).std() * 100
    
    df.to_csv(filename, index=False)
    print(f"✅ Data exported to: {filename}")
    print(f"   - Total days: {len(df)}")
    print(f"   - Start price: ${df['Close'].iloc[0]:.2f}")
    print(f"   - End price: ${df['Close'].iloc[-1]:.2f}")
    print(f"   - Price change: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"\n💡 Tip: Open {filename} in Excel or use pandas.read_csv() to analyze!")

def _run_core_simulation(strategy_function, df):
    # --- Load Settings ---
    initial_cash = config.INITIAL_CASH
    commission_fee = config.COMMISSION_FEE
    trade_size = config.TRADE_SIZE
    
    position_limit = getattr(config, 'POSITION_LIMIT', 100)
    
    cash = initial_cash
    shares = 0
    
    # History Tracking
    buy_signals = []
    sell_signals = []
    portfolio_value_history = [] 
    position_history = []
    history_dates = []

    for i in range(20, len(df)):
        historical_slice = df.iloc[:i]
        current_price = historical_slice['Close'].iloc[-1]
        current_day = historical_slice['Day'].iloc[-1]
        
        signal = strategy_function(historical_slice, shares)
        
        if signal == "BUY":
            new_share_count = shares + trade_size
            within_limit = abs(new_share_count) <= position_limit
            
            if within_limit:
                shares += trade_size
                cash -= (current_price * trade_size)
                cash -= commission_fee
                buy_signals.append((current_day, current_price))
            
        elif signal == "SELL":
            new_share_count = shares - trade_size
            within_limit = abs(new_share_count) <= position_limit
            
            if within_limit:
                shares -= trade_size
                cash += (current_price * trade_size)
                cash -= commission_fee
                sell_signals.append((current_day, current_price))

        total_value = cash + (shares * current_price)
        portfolio_value_history.append(total_value)
        position_history.append(shares)
        history_dates.append(current_day)

    final_price = df['Close'].iloc[-1]
    final_value = cash + (shares * final_price)
    total_profit = final_value - initial_cash
    
    start_price = df['Close'].iloc[0]
    max_shares_cash = (initial_cash - commission_fee) // start_price
    benchmark_shares = min(max_shares_cash, position_limit)
    benchmark_value = (benchmark_shares * final_price) + (initial_cash - (benchmark_shares * start_price) - commission_fee)
    
    return {
        "final_value": final_value,
        "total_profit": total_profit,
        "benchmark_value": benchmark_value,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "history_dates": history_dates,
        "portfolio_values": portfolio_value_history,
        "position_history": position_history
    }

def run_simulation(strategy_function, seed=42):
    market_type = getattr(config, 'MARKET_TYPE', 'COMPLEX')
    print(f"🚀 Starting Simulation [Market: {market_type}] (Seed: {seed})...\n")
    
    df = generate_mock_data(seed=seed)
    results = _run_core_simulation(strategy_function, df)
    
    # Unpack basic results
    total_profit = results['total_profit']
    final_value = results['final_value']
    benchmark_value = results['benchmark_value']
    initial_cash = config.INITIAL_CASH
    
    percent_return = (total_profit / initial_cash) * 100
    benchmark_return = ((benchmark_value - initial_cash) / initial_cash) * 100

    perf_df = pd.DataFrame({
        'Day': results['history_dates'],
        'Value': results['portfolio_values'],
        'Position': results['position_history']
    }).set_index('Day')
    

    perf_df['Daily_Return'] = perf_df['Value'].pct_change()

    rolling_mean = perf_df['Daily_Return'].expanding(min_periods=20).mean()
    rolling_std = perf_df['Daily_Return'].expanding(min_periods=20).std()
    
    perf_df['Sharpe'] = (rolling_mean / rolling_std * np.sqrt(252)).fillna(0)
    
    final_sharpe = perf_df['Sharpe'].iloc[-1]

    # --- Print Stats ---
    benchmark_pnl = benchmark_value - initial_cash
    
    print("-" * 40)
    print(f"FINAL RESULTS")
    print("-" * 40)
    print(f"Your PnL:          ${total_profit:.2f}")
    print(f"Your Return:       {percent_return:.2f}%")
    print(f"Sharpe Ratio:      {final_sharpe:.2f}")
    print("-" * 40)
    print(f"BENCHMARK (Buy & Hold)")
    print(f"Benchmark PnL:     ${benchmark_pnl:.2f}")
    print(f"Benchmark Return:  {benchmark_return:.2f}%")
    print("-" * 40)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 1. Price Chart with Signals
    ax1.plot(df['Day'], df['Close'], label='Stock Price', color='gray', alpha=0.5)
    if results['buy_signals']:
        b_dates, b_prices = zip(*results['buy_signals'])
        ax1.scatter(b_dates, b_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
    if results['sell_signals']:
        s_dates, s_prices = zip(*results['sell_signals'])
        ax1.scatter(s_dates, s_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
    ax1.set_title(f"Market Price & Trades (Profit: ${total_profit:.2f})")
    ax1.set_ylabel("Price ($)")
    ax1.set_xlabel("Day")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Position Size (Shares Held)
    ax2.plot(perf_df.index, perf_df['Position'], color='blue', drawstyle='steps-post')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_title("Position Size (Shares Held)")
    ax2.set_ylabel("Shares")
    ax2.set_xlabel("Day")
    ax2.grid(True, alpha=0.3)
    
    # 3. Total Profit Over Time
    profit_curve = perf_df['Value'] - initial_cash
    ax3.fill_between(perf_df.index, profit_curve, 0, where=(profit_curve >= 0), color='green', alpha=0.3)
    ax3.fill_between(perf_df.index, profit_curve, 0, where=(profit_curve < 0), color='red', alpha=0.3)
    ax3.plot(perf_df.index, profit_curve, color='black', linewidth=1)
    ax3.set_title("Total Profit/Loss Over Time")
    ax3.set_ylabel("Profit ($)")
    ax3.set_xlabel("Day")
    ax3.grid(True, alpha=0.3)

    # 4. Sharpe Ratio Over Time
    ax4.plot(perf_df.index, perf_df['Sharpe'], color='purple')
    ax4.set_title(f"Sharpe Ratio (Risk-Adjusted Return) - Final: {final_sharpe:.2f}")
    ax4.set_ylabel("Sharpe")
    ax4.set_xlabel("Day")
    ax4.axhline(1.0, color='green', linestyle='--', label='Good (>1)')
    ax4.axhline(0.0, color='red', linestyle='--', label='Bad (<0)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_stress_test(strategy_function, runs=50):
    market_type = getattr(config, 'MARKET_TYPE', 'COMPLEX')
    print(f"Starting Stress Test [Market: {market_type}] ({runs} runs)...")
    
    profits = []
    sharpes = []
    wins_vs_benchmark = 0
    
    initial_cash = config.INITIAL_CASH
    
    for i in range(runs):
        df = generate_mock_data(seed=i)
        res = _run_core_simulation(strategy_function, df)
        
        profits.append(res['total_profit'])
        if res['total_profit'] > (res['benchmark_value'] - initial_cash):
            wins_vs_benchmark += 1
            
        # Calculate Sharpe for this run
        values = pd.Series(res['portfolio_values'])
        returns = values.pct_change()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        sharpes.append(sharpe)

    avg_profit = sum(profits) / len(profits)
    avg_sharpe = sum(sharpes) / len(sharpes)
    
    # Calculate benchmark stats
    benchmark_profits = []
    for i in range(runs):
        df = generate_mock_data(seed=i)
        res = _run_core_simulation(lambda x, y: "HOLD", df)  # Dummy to get benchmark
        benchmark_profits.append(res['benchmark_value'] - initial_cash)
    avg_benchmark = sum(benchmark_profits) / len(benchmark_profits)
    
    print("\n" + "=" * 40)
    print(f"STRESS TEST REPORT")
    print("=" * 40)
    print(f"Simulations Run:    {runs}")
    print(f"Average PnL:        ${avg_profit:.2f}")
    print(f"Average Sharpe:     {avg_sharpe:.2f}")
    print("-" * 40)
    print(f"Best Run:           ${max(profits):.2f}")
    print(f"Worst Run:          ${min(profits):.2f}")
    print("-" * 40)
    print(f"Benchmark Avg PnL:  ${avg_benchmark:.2f}")
    print(f"Win Rate (Make $):  {len([p for p in profits if p > 0]) / runs * 100:.0f}%")
    print(f"Win Rate (Beat Mkt):{wins_vs_benchmark / runs * 100:.0f}%")
