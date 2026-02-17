from snakesandstonks import run_simulation

def my_trading_strategy(data, current_cash, current_shares):    
    today_price = data['Close'].iloc[-1]
    
    sma_5 = data['Close'].tail(5).mean()
    
    if today_price < sma_5 and current_cash > (today_price * 10):
        return "BUY"
    
    elif today_price > sma_5 and current_shares >= 10:
        return "SELL"
        
    return "HOLD"

# DO NOT CHANGE BELOW THIS LINE!
if __name__ == "__main__":
    run_simulation(my_trading_strategy)
