from snakesandstonks import run_simulation

def my_trading_strategy(data, current_cash, current_shares):
    """
    data: DataFrame with price history up to today
    current_cash: Float, amount of money you have available
    current_shares: Integer, number of shares you currently own
    
    Returns:
    - "BUY": Buys 10 shares
    - "SELL": Sells 10 shares
    - "HOLD": Does nothing
    """
    
    today_price = data['Close'].iloc[-1]
    
    # 1. Calculate a Simple Moving Average (SMA) of the last 5 days
    sma_5 = data['Close'].tail(5).mean()
    
    # 2. Strategy Logic:
    # If the price is below average, it might be "cheap" -> BUY
    if today_price < sma_5 and current_cash > (today_price * 10):
        return "BUY"
    
    # If the price is above average, take profit -> SELL
    elif today_price > sma_5 and current_shares >= 10:
        return "SELL"
        
    return "HOLD"

# DO NOT CHANGE BELOW THIS LINE!
if __name__ == "__main__":
    run_simulation(my_trading_strategy)
