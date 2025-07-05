# Bitcoin Analysis Module

This repository contains a Python module for performing various analyses on Bitcoin price data. It includes calculations for volatility, power-law models, and other statistical metrics.

## Installation

To use this module, you need to install the required Python packages. You can do this using `pip` and the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Input Data Requirements

The `price_df` pandas DataFrame passed to the `BitcoinAnalysis` class must have the following structure:

1.  **Index**: The index of the DataFrame must be a `DatetimeIndex`. Each index entry represents a single timestamp for the price data (e.g., daily).
2.  **Columns**: The DataFrame must contain the following columns with market data:
    *   `Open`: The opening price for the period.
    *   `High`: The highest price for the period.
    *   `Low`: The lowest price for the period.
    *   `Close`: The closing price for the period.

Here is an example of what the input DataFrame should look like:

```
                   Open     High      Low    Close
Date
2023-01-01      16547.9  16630.4  16508.4  16604.2
2023-01-02      16604.2  16759.3  16570.6  16670.8
...                 ...      ...      ...      ...
```

## Usage

Here is an example of how to use the `BitcoinAnalysis` module:

```python
import pandas as pd
import json
from bitcoin_analysis import BitcoinAnalysis, get_quantile_prices, calculate_benchmark_prices

# Load your bitcoin price data (e.g., from a CSV file)
# The CSV should have columns: Date, Open, High, Low, Close
price_df = pd.read_csv('your_bitcoin_data.csv', index_col='Date', parse_dates=True)

# Initialize the analysis object
analysis = BitcoinAnalysis(price_df)

# --- Perform Analyses ---

# 1. Get prepared price history with PLRR value
price_history_df = analysis.prepare_price_history()

# 2. Calculate GARCH volatility
garch_volatility_df = analysis.calculate_garch_volatility()
historical_volatility = garch_volatility_df[garch_volatility_df['type'] == 'historical']
forecast_volatility = garch_volatility_df[garch_volatility_df['type'] == 'forecast']


# 3. Calculate statistics
statistics = analysis.calculate_statistics()

# 4. Get quantile index
quantile_index_df = analysis.calculate_quantile_index_df()

# 5. Get power-law quantile prices
quantile_prices_df = get_quantile_prices()

# 6. Get benchmark prices
benchmark_prices_df = calculate_benchmark_prices()


# --- Save outputs to JSON ---

# Save volatility data
volatility_data = {
    "historical": historical_volatility.to_dict(orient='records'),
    "forecast": forecast_volatility.to_dict(orient='records')
}
with open('volatility.json', 'w') as f:
    json.dump(volatility_data, f)

# Save other analysis data
bitcoin_data = {
    "price_history": price_history_df.to_dict(orient='records'),
    "quantile_prices": quantile_prices_df.to_dict(orient='records'),
    "quantile_index": quantile_index_df.to_dict(orient='records'),
    "benchmark_price": benchmark_prices_df.to_dict(orient='records'),
    "statistics": statistics
}
with open('bitcoin-data.json', 'w') as f:
    json.dump(bitcoin_data, f)

```

## Module Functions (`bitcoin_analysis.py`)

### `BitcoinAnalysis` Class

This is the main class for the analysis.

-   `__init__(self, price_df)`: Initializes the class with a pandas DataFrame of historical Bitcoin prices. The DataFrame index should be datetime objects.
-   `calculate_garch_volatility(self)`: Calculates historical and forecasted GARCH(1,1) volatility.
-   `prepare_price_history(self)`: Returns a DataFrame with OHLC prices, the PLRR (`value`) and a timestamp.
-   `calculate_statistics(self)`: Calculates various price change statistics and volatility metrics, comparing them with a Power Law model.
-   `calculate_quantile_index_df(self)`: Calculates an index representing where the current price is within the power-law quantile bands.

### Standalone Functions

-   `get_quantile_prices()`: Calculates and returns a DataFrame of Bitcoin price predictions based on a power-law model, including median, 2.5 percentile, and 97.5 percentile prices.
-   `calculate_benchmark_prices()`: Calculates and returns a DataFrame of benchmark prices based on an exponential model, including a fit, lower bound, and upper bound.

## Output Files Format

The module generates two main JSON files: `volatility.json` and `bitcoin-data.json`.

### `volatility.json`

This file contains historical and forecasted volatility data.

-   `historical`: An array of objects, where each object represents the historical volatility for a given day.
    -   `time`: Unix timestamp in milliseconds.
    -   `volatility`: The calculated annualized GARCH volatility.
    -   `type`: Always "historical".
-   `forecast`: An array of objects, representing the forecasted volatility for the next 30 days.
    -   `time`: Unix timestamp in milliseconds.
    -   `volatility`: The forecasted annualized volatility.
    -   `type`: Always "forecast".

**Example `volatility.json` entry:**

```json
{
  "historical": [
    {
      "time": 1718668800000,
      "volatility": 0.40550867749611946,
      "type": "historical"
    }
  ],
  "forecast": [
    {
      "time": 1751760000000,
      "volatility": 0.41175036837143475,
      "type": "forecast"
    }
  ]
}
```

### `bitcoin-data.json`

This file is a JSON object containing several datasets from the analysis.

-   `price_history`: An array of objects containing daily OHLC price data, along with the calculated PLRR value.
    -   `time`: Unix timestamp in milliseconds.
    -   `open`, `high`, `low`, `close`: Daily prices.
    -   `value`: The Power-Law-Residual-Ratio (PLRR) for that day.
-   `quantile_prices`: An array of objects representing the power-law model price bands.
    -   `time`: Unix timestamp in milliseconds.
    -   `priceMedian`, `price2p5`, `price97p5`: The median, 2.5th percentile, and 97.5th percentile prices from the model.
-   `quantile_index`: An array of objects representing the historical quantile of the price.
    -   `time`: Unix timestamp in milliseconds.
    -   `value`: The quantile index, a value from 0 to 1.
-   `benchmark_price`: An array of objects for the exponential benchmark model.
    -   `time`: Unix timestamp in milliseconds.
    -   `priceFit`, `priceLB`, `priceUB`: The fitted, lower-bound, and upper-bound prices.
-   `statistics`: A JSON object containing various calculated metrics.
    -   `change1d`, `change30d`, `change1yr`: Percentage price change over the last 1, 30, and 365 days.
    -   `change1d_PL`, `change30d_PL`, `change1yr_PL`: Price change predicted by the Power Law model for the same periods.
    -   `volatility30d`: Realized 30-day volatility.
    -   `volatility30d_PL`: 30-day volatility predicted by the Power Law model.
