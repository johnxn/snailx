# coding=utf-8

import pandas as pd
from data_blob import DataBlob
from data_source_akshare import DataSourceAkshare
from data_source_norgate import DataSourceNorgate
from strategy_robert import StrategyRobert
import mailer
import util

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # 设置宽度不限制


def run_china_market():
    csv_config_dict = dict(
        futures_single_contracts_dir='data/china_futures/single_contracts',
        futures_continuous_dir='data/output/backtest2/continuous',
        futures_combined_dir='data/output/backtest2/combined',
        futures_roll_calendar_dir='data/output/backtest2/roll_calendar',
        daily_account_value_file_path='data/output/backtest2/daily_account_value.csv',
        portfolio_config_file_path='config/symbols_china_futures.xlsx',
        strategy_rules_config_file_path='config/strategy_rules.xlsx',
        strategy_parameters_config_file_path='config/strategy_parameters.xlsx',
    )
    data_source = DataSourceAkshare()
    data_blob = DataBlob(csv_config_dict, data_source)
    data_blob.update_single_contracts_in_portfolio()
    data_blob.update_roll_calendar_in_portfolio()
    data_blob.update_data_continuous_in_portfolio()
    data_blob.run_strategy(StrategyRobert)
    signal_date, df_signal = data_blob.get_latest_operation_signals()
    mailer.send_mail(subject=f"China Futures {util.datetime_to_str(signal_date)}", content=df_signal.to_html(index=True))


def run_us_market():
    csv_config_dict = dict(
        futures_single_contracts_dir='data/futures/single_contracts',
        futures_continuous_dir='data/output/us_ewmac/continuous',
        futures_combined_dir='data/output/us_ewmac/combined',
        futures_roll_calendar_dir='data/output/us_ewmac/roll_calendar',
        daily_account_value_file_path='data/output/us_ewmac/daily_account_value.csv',
        portfolio_config_file_path='config/symbols_us_futures.xlsx',
        strategy_rules_config_file_path='config/strategy_rules.xlsx',
        strategy_parameters_config_file_path='config/strategy_parameters.xlsx',
    )
    data_source = DataSourceNorgate()
    data_blob = DataBlob(csv_config_dict, data_source)
    data_blob.update_single_contracts_in_portfolio()
    data_blob.update_roll_calendar_in_portfolio()
    data_blob.update_data_continuous_in_portfolio()
    data_blob.run_strategy(StrategyRobert)
    signal_date, df_signal = data_blob.get_latest_operation_signals()
    mailer.send_mail(subject=f"US Futures {util.datetime_to_str(signal_date)}", content=df_signal.to_html(index=True))


if __name__ == "__main__":
    run_us_market()
    run_china_market()
