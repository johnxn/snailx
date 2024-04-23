# coding=utf-8
import numpy
import pandas as pd


def calculate_forecast(df_continuous, args):
    forecast_scalar = args['forecast_scalar']
    smooth_days = int(args['smooth_window'])
    volatility_lookback_window = int(args['volatility_lookback_window'])
    forecast_max = args['forecast_max']

    daily_price_diff = df_continuous['AdjustPrice'].diff()
    daily_price_diff_vol = daily_price_diff.rolling(window=volatility_lookback_window).std()

    carry_contract_date = pd.to_datetime(df_continuous['CarryContract'], format="%Y%m")
    price_contract_date = pd.to_datetime(df_continuous['CurrentContract'], format="%Y%m")
    delta_days = (carry_contract_date - price_contract_date).dt.days
    annual_return_price = (df_continuous['CurrentPrice'] - df_continuous['CarryPrice']) / (delta_days / 365.0)
    annual_price_diff_vol = daily_price_diff_vol * 16
    raw_carry = (annual_return_price / annual_price_diff_vol).rolling(window=smooth_days).mean()
    forecast = raw_carry * forecast_scalar
    forecast = forecast.clip(-forecast_max, forecast_max)
    forecast.iloc[0] = 0 if numpy.isnan(forecast.iloc[0]) else forecast.iloc[0]
    forecast = forecast.ffill()
    return forecast
