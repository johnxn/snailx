# coding=utf-8
import numpy


def calculate_forecast(df_continuous, args):
    fast_window = args['fast_window']
    slow_window = args['slow_window']
    forecast_scalar = args['forecast_scalar']
    volatility_lookback_window = int(args['volatility_lookback_window'])
    forecast_max = args['forecast_max']

    daily_price_diff = df_continuous['AdjustPrice'].diff()
    daily_price_diff_vol = daily_price_diff.rolling(window=volatility_lookback_window).std()

    ewmac_fast = df_continuous['AdjustPrice'].ewm(span=fast_window, min_periods=fast_window,
                                                  adjust=False).mean()
    ewmac_slow = df_continuous['AdjustPrice'].ewm(span=slow_window, min_periods=slow_window,
                                                  adjust=False).mean()
    raw_forecast = ewmac_fast - ewmac_slow
    forecast = (raw_forecast / daily_price_diff_vol * forecast_scalar)
    forecast = forecast.clip(-forecast_max, forecast_max)
    forecast.iloc[0] = 0 if numpy.isnan(forecast.iloc[0]) else forecast.iloc[0]
    forecast = forecast.ffill()
    return forecast
