import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from arch import arch_model

COEFFICIENTS = {
    'median': [1.2e-17, 5.7 ,0.532544,4.114783e-05],
    '2p5': [1.2e-17, 5.7, -0.0437205, 2.1281429e-05],
    '97p5': [1.2e-17, 5.7, 3.49136, -0.000345542]
}

POWER_LAW_EXPONENT = 5.7
TIME_SPANS = np.arange(180, 450, 15)
BITCOIN_GENESIS_DATE = '2009-01-03'

class BitcoinAnalysis:
    def __init__(self, price_df):
        self.price_df = self._preprocess_data(price_df)
        self.price_df = self.calculate_plrr()

    def _preprocess_data(self, price_df):
        t0_datetime = pd.to_datetime(BITCOIN_GENESIS_DATE)
        price_df['TimeDays'] = (price_df.index - t0_datetime).days
        price_df['LogReturn'] = np.log(0.5*(price_df['Low']+price_df['High'])).diff()
        price_df['LogTimeDiff'] = np.log(price_df['TimeDays']).diff()
        price_df['value'] = np.zeros(len(price_df))
        price_df['time'] = ((price_df.index - datetime(1970, 1, 1)) / timedelta(milliseconds=1)).astype(int)
        return price_df

    def calculate_garch_volatility(self):
        data = self.price_df.copy()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()
        returns = data['Log_Returns'] * 100
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
        garch_fit = model.fit(disp='off')
        historical_volatility = garch_fit.conditional_volatility / 100
        annualized_volatility = historical_volatility * np.sqrt(365)
        historical_df = pd.DataFrame({
            'time': ((data.index - datetime(1970, 1, 1)) / timedelta(milliseconds=1)).astype(int),
            'volatility': annualized_volatility,
            'type': 'historical'
        })
        forecast = garch_fit.forecast(horizon=30)
        forecast_vol = np.sqrt(forecast.variance.values[-1, :]) / 100 * np.sqrt(365)
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_df = pd.DataFrame({
            'time': ((forecast_dates - datetime(1970, 1, 1)) / timedelta(milliseconds=1)).astype(int),
            'volatility': forecast_vol,
            'type': 'forecast'
        })
        return pd.concat([historical_df, forecast_df])

    def calculate_plrr(self):
        price_df = self.price_df.copy()
        for span in TIME_SPANS:
            price_df['MeanLogReturn'] = price_df['LogReturn'].rolling(span).mean()
            price_df['MeanLogTimeDiff'] = price_df['LogTimeDiff'].rolling(span).mean()
            price_df['LogSDev'] = price_df['LogReturn'].rolling(span).std()
            price_df['value'] += (1/len(TIME_SPANS)) * np.sqrt(span) * \
                (price_df['MeanLogReturn'] - POWER_LAW_EXPONENT*price_df['MeanLogTimeDiff']) / price_df['LogSDev']
        return price_df

    def prepare_price_history(self):
        price_history_df = self.price_df[['Open', 'High', 'Low', 'Close', 'value', 'time']]
        price_history_df = price_history_df.dropna(subset=['value'])
        price_history_df = price_history_df.rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
        )
        return price_history_df

    def calculate_statistics(self):
        price_df = self.price_df
        end_date = datetime.now()
        t0 = pd.to_datetime('2009-01-03')
        days_end = (end_date - t0).days
        log_time = np.log(days_end)
        k = 5.7
        intercept = 0.21923
        slope = -0.02242
        close_prices = price_df['Close']
        changes = {
            'change1d': self._calculate_price_change(close_prices, 2),
            'change30d': self._calculate_price_change(close_prices, 30),
            'change1yr': self._calculate_price_change(close_prices, 365)
        }
        pl_changes = {
            'change1d_PL': self._calculate_pl_change(days_end, 1, k),
            'change30d_PL': self._calculate_pl_change(days_end, 30, k),
            'change1yr_PL': self._calculate_pl_change(days_end, 365, k)
        }
        volatility30d = (np.exp(np.log(close_prices).diff().rolling(60).std() * np.sqrt(30))-1)*100
        volatility30d_PL = (np.exp((intercept+slope*log_time)*np.sqrt(30))-1)*100
        return {**changes, **pl_changes,
                'volatility30d': volatility30d.iloc[-1],
                'volatility30d_PL': volatility30d_PL}

    def _calculate_price_change(self, prices, days):
        return (prices.iloc[-1] - prices.iloc[-days]) / prices.iloc[-days] * 100

    def _calculate_pl_change(self, days_end, days, k):
        return (np.exp(k*np.log(days_end/(days_end-days)))-1)*100

    def calculate_quantile_index_df(self):
        price_df = self.price_df
        priceLB = calculate_power_law_price(price_df['TimeDays'], COEFFICIENTS['2p5'])
        priceUB = calculate_power_law_price(price_df['TimeDays'], COEFFICIENTS['97p5'])
        priceMedian = calculate_power_law_price(price_df['TimeDays'], COEFFICIENTS['median'])
        priceLB.index = price_df.index
        priceUB.index = price_df.index
        priceMedian.index = price_df.index
        
        # Calculate quantile index using piecewise interpolation with extrapolation
        quantile_index = self._interpolate_quantile_index(
            price_df['Close'], priceLB, priceMedian, priceUB
        )
        
        quantile_index_df = pd.DataFrame({
            'time': ((price_df.index - datetime(1970, 1, 1)) / timedelta(milliseconds=1)).astype(int),
            'value': quantile_index
        })
        return quantile_index_df.dropna()
    
    def _interpolate_quantile_index(self, current_price, priceLB, priceMedian, priceUB):
        """
        Interpolate quantile index using piecewise linear interpolation between
        priceLB (2.5%), priceMedian (50%), and priceUB (97.5%) with extrapolation.
        
        Returns quantile index where:
        - 0.025 corresponds to priceLB
        - 0.5 corresponds to priceMedian  
        - 0.975 corresponds to priceUB
        - Values below priceLB are extrapolated (can go negative)
        - Values above priceUB are extrapolated (can exceed 1.0)
        """
        quantile_index = np.zeros_like(current_price, dtype=float)
        
        # Define quantile levels
        q_lb = 0.025
        q_median = 0.5
        q_ub = 0.975
        
        # Case 1: Price below priceLB (extrapolate downward)
        below_lb = current_price < priceLB
        if np.any(below_lb):
            # Linear extrapolation using slope between LB and median
            slope_lower = (q_median - q_lb) / (priceMedian[below_lb] - priceLB[below_lb])
            quantile_index[below_lb] = q_lb + slope_lower * (current_price[below_lb] - priceLB[below_lb])
        
        # Case 2: Price between priceLB and priceMedian
        between_lb_median = (current_price >= priceLB) & (current_price <= priceMedian)
        if np.any(between_lb_median):
            # Linear interpolation between LB and median
            weight = (current_price[between_lb_median] - priceLB[between_lb_median]) / \
                    (priceMedian[between_lb_median] - priceLB[between_lb_median])
            quantile_index[between_lb_median] = q_lb + weight * (q_median - q_lb)
        
        # Case 3: Price between priceMedian and priceUB
        between_median_ub = (current_price > priceMedian) & (current_price <= priceUB)
        if np.any(between_median_ub):
            # Linear interpolation between median and UB
            weight = (current_price[between_median_ub] - priceMedian[between_median_ub]) / \
                    (priceUB[between_median_ub] - priceMedian[between_median_ub])
            quantile_index[between_median_ub] = q_median + weight * (q_ub - q_median)
        
        # Case 4: Price above priceUB (extrapolate upward)
        above_ub = current_price > priceUB
        if np.any(above_ub):
            # Linear extrapolation using slope between median and UB
            slope_upper = (q_ub - q_median) / (priceUB[above_ub] - priceMedian[above_ub])
            quantile_index[above_ub] = q_ub + slope_upper * (current_price[above_ub] - priceUB[above_ub])
        
        return quantile_index

def calculate_quantile_prices(coefficients):
    t0 = pd.to_datetime('2009-01-03')
    tSq = pd.to_datetime('2015-06-01')
    tEq = pd.to_datetime('2025-12-31')
    date_array = pd.date_range(start=tSq, end=tEq)
    days_array = np.array((date_array - t0).days)
    priceMedian = calculate_power_law_price(days_array, coefficients['median'])
    price2p5 = calculate_power_law_price(days_array, coefficients['2p5'])
    price97p5 = calculate_power_law_price(days_array, coefficients['97p5'])
    return pd.DataFrame({
        'time': ((date_array - datetime(1970, 1, 1)) / timedelta(milliseconds=1)).astype(int),
        'priceMedian': priceMedian,
        'price2p5': price2p5,
        'price97p5': price97p5
    })

def calculate_benchmark_prices():
    t0 = pd.to_datetime('2023-01-01')
    tSq = pd.to_datetime('2023-01-01')
    tEq = pd.to_datetime('2025-12-31')
    date_array = pd.date_range(start=tSq, end=tEq)
    days_array = np.array((date_array - t0).days)
    coefficients = {
        'LB': [9.92-0.42, 0.0020],
        'UB': [9.92+0.42, 0.0020],
        'Fit': [9.92, 0.0020]
    }
    return pd.DataFrame({
        'time': ((date_array - datetime(1970, 1, 1)) / timedelta(milliseconds=1)).astype(int),
        'priceLB': calculate_exponential_price(days_array, coefficients['LB']),
        'priceUB': calculate_exponential_price(days_array, coefficients['UB']),
        'priceFit': calculate_exponential_price(days_array, coefficients['Fit'])
    })

def calculate_power_law_price(days_array, coefficients):
    return np.exp(coefficients[2] + coefficients[3]*days_array) * (days_array**coefficients[1]) * coefficients[0]

def calculate_exponential_price(days_array, coefficients):
    return np.exp(coefficients[0] + coefficients[1] * days_array)

def get_quantile_prices():
    return calculate_quantile_prices(COEFFICIENTS) 