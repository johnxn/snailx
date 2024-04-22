# coding=utf-8
import numpy
import pandas as pd


def calculate_forecast(df_continuous, args):
    forecast_scalar = args['forecast_scalar']
    smooth_days = args['smooth_days']
    volatility_lookback_window = args['volatility_lookback_window']
    forecast_max = args['forecast_max']

    daily_price_diff = df_continuous['AdjustPrice'].diff()
    daily_price_diff_vol = daily_price_diff.rolling(window=volatility_lookback_window).std()
    daily_price_diff_vol[0] = 0 if numpy.isnan(daily_price_diff_vol[0]) else daily_price_diff_vol[0]
    daily_price_diff_vol = daily_price_diff_vol.ffill()

    carry_contract_date = pd.to_datetime(df_continuous['CarryContract'], format="%Y%m")
    price_contract_date = pd.to_datetime(df_continuous['CurrentContract'], format="%Y%m")
    delta_days = (carry_contract_date - price_contract_date).dt.days
    annual_return_price = (df_continuous['CurrentPrice'] - df_continuous['CarryPrice']) / (delta_days / 365.0)
    annual_price_diff_vol = daily_price_diff_vol * 16
    raw_carry = (annual_return_price / annual_price_diff_vol).rolling(window=smooth_days).mean()
    forecast = raw_carry * forecast_scalar
    forecast[0] = 0 if numpy.isnan(forecast[0]) else forecast[0]
    forecast = forecast.ffill()
    forecast = forecast.clip(-forecast_max, forecast_max)
    return forecast
