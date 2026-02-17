# Snakes and Stonks

## Prerequisites

Ensure the following tools are installed on your system:

* [Git]()
* [uv]()

## Setup and Installation

1. **Clone the repository:**
```bash
git clone <REPOSITORY_URL>

```


2. **Navigate to the project directory:**
```bash
cd <PROJECT_DIRECTORY>

```


3. **Install dependencies:**
The project uses `uv` to manage the Python environment. Run the following command to create the virtual environment and install required packages (pandas, numpy, matplotlib):
```bash
uv sync

```



## Execution

To run your trading bot against the simulation:

```bash
uv run main.py

```

## File Overview

* **`main.py`**: The only file you need to modify. It contains the `my_trading_strategy` function where you define your logic.
* **`config.py`**: Contains simulation parameters such as starting cash, transaction fees, and market volatility.
* **`backtester.py`**: The simulation engine. It generates data, executes trades, and calculates performance. **Do not modify this file.**

## Usage

Your strategy must be implemented in the `my_trading_strategy` function within `main.py`.

### Inputs

The function receives the current state of the market and portfolio:

1. **`data`** *(pandas.DataFrame)*: Historical price data up to the current simulation step.
2. **`cash`** *(float)*: Current available cash balance.
3. **`shares`** *(int)*: Current number of shares held.

### Outputs

The function must return one of the following strings:

* `"BUY"`: Purchase shares.
* `"SELL"`: Sell shares.
* `"HOLD"`: Take no action.
