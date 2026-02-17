# --- Game Settings ---
INITIAL_CASH = 1000.0      # Starting money
COMMISSION_FEE = 2.00      # Cost per trade (Buy or Sell)
TRADE_SIZE = 10            # Number of shares per trade

# --- Market Conditions ---
# Options: "TRENDING", "MEAN_REVERTING", "RANDOM_WALK"
MARKET_TYPE = "MEAN_REVERTING" 

# --- Data Generation Settings ---
SIMULATION_DAYS = 200
START_PRICE = 100.0
VOLATILITY = 2.0           # Higher = wilder price swings
REVERSION_SPEED = 0.1      # Only used for MEAN_REVERTING (0.01 - 0.5)
