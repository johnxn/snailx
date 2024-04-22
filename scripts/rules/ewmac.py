# coding=utf-8
import numpy


def calculate_forecast(df_continuous, args):
    fast_window = args['fast_window']
    slow_window = args['slow_window']
    forecast_scalar = args['forecast_scalar']
    volatility_lookback_window = args['volatility_lookback_window']
    forecast_max = args['forecast_max']

    daily_price_diff = df_continuous['AdjustPrice'].diff()
    daily_price_diff_vol = daily_price_diff.rolling(window=volatility_lookback_window).std()
    daily_price_diff_vol[0] = 0 if numpy.isnan(daily_price_diff_vol[0]) else daily_price_diff_vol[0]
    daily_price_diff_vol = daily_price_diff_vol.ffill()

    ewmac_fast = df_continuous['AdjustPrice'].ewm(span=fast_window, min_periods=fast_window,
                                                  adjust=False).mean()
    ewmac_slow = df_continuous['AdjustPrice'].ewm(span=slow_window, min_periods=slow_window,
                                                  adjust=False).mean()
    raw_forecast = ewmac_fast - ewmac_slow
    raw_forecast[0] = 0 if numpy.isnan(raw_forecast[0]) else raw_forecast[0]
    raw_forecast = raw_forecast.ffill()
    forecast = (raw_forecast / daily_price_diff_vol * forecast_scalar)
    forecast = forecast.clip(-forecast_max, forecast_max)
    return forecast
