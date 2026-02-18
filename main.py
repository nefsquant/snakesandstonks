from snakesandstonks import run_simulation
import config

def my_trading_strategy(data, current_shares):    
    # Get today's price
    today_price = data['Close'].iloc[-1]
    
    # Calculate the 20-day Simple Moving Average (our "fair price")
    fair_price = data['Close'].tail(20).mean()
    
    # Calculate how far price is from fair value 
    price_deviation = (today_price - fair_price) / fair_price * 100
    
    # Trading Logic:
    # If price is more than 2% below fair value → BUY
    if price_deviation < -2 and current_shares < 50:
        return "BUY"
    
    # If price is more than 2% above fair value → SELL
    elif price_deviation > 2 and current_shares > -50:
        return "SELL"
    
    # Otherwise, do nothing
    return "HOLD"

# DO NOT CHANGE BELOW THIS LINE!
if __name__ == "__main__":
    run_simulation(my_trading_strategy, config.SEED)
