# coding=utf-8

import pandas as pd
from data_blob import DataBlob
from data_source_akshare import DataSourceAkshare
from data_source_norgate import DataSourceNorgate
from futu_api.futu_broker import FutuDataBroker
from strategy_robert import StrategyRobert
import mailer
import util

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # 设置宽度不限制


def run_china_market():
    csv_config_dict = dict(
        futures_single_contracts_dir='data/china_futures/single_contracts',
        futures_continuous_dir='data/china_futures/continuous',
        futures_combined_dir='data/china_futures/combined',
        futures_roll_calendar_dir='data/china_futures/roll_calendar',
        daily_account_value_file_path='data/china_futures/daily_account_value.csv',
        portfolio_config_file_path='config/symbols_china_futures.csv',
        strategy_rules_config_file_path='config/strategy_rules_china.csv',
        strategy_parameters_config_file_path='config/strategy_parameters_china.csv',
    )
    data_source = DataSourceAkshare()
    data_blob = DataBlob(csv_config_dict, data_source, None)
    data_blob.update_single_contracts_in_portfolio()
    data_blob.update_roll_calendar_in_portfolio()
    data_blob.update_data_continuous_in_portfolio()
    data_blob.run_strategy(StrategyRobert)
    signal_date, df_signal = data_blob.get_latest_operation_signals()
    mailer.send_mail(subject=f"China Futures {util.datetime_to_str(signal_date)}", content=df_signal.to_html(index=True))


def run_us_market():
    csv_config_dict = dict(
        futures_single_contracts_dir='data/futures/single_contracts',
        futures_continuous_dir='data/futures/continuous',
        futures_combined_dir='data/futures/combined',
        futures_roll_calendar_dir='data/futures/roll_calendar',
        symbol_config_file_path='data/futures/all_symbols.csv',
        daily_account_value_file_path='data/futures/daily_account_value.csv',
        daily_symbol_weight_file_path='data/futures/daily_symbol_weights.csv',
        daily_rule_weight_file_path='data/futures/daily_rule_weights.csv',
        strategy_rules_config_file_path='data/futures/strategy_rules.csv',
        strategy_parameters_config_file_path='data/futures/strategy_parameters.csv',
    )
    data_source = DataSourceNorgate()
    data_broker = FutuDataBroker()
    data_blob = DataBlob(csv_config_dict, data_source, data_broker)
    data_blob.update_daily_account_value()
    data_blob.update_single_contracts_in_portfolio()
    data_blob.generate_roll_calendar_in_portfolio(by_volume=True)
    data_blob.update_roll_calendar_in_portfolio(by_volume=False)
    data_blob.update_data_continuous_in_portfolio()
    data_blob.run_strategy(StrategyRobert, is_live=True)
    signal_date, df_signal = data_blob.get_latest_operation_signals()
    mailer.send_mail(subject=f"US Futures {util.datetime_to_str(signal_date)}", content=df_signal.to_html(index=True))
    data_broker.destroy()


if __name__ == "__main__":
    run_us_market()
    # run_china_market()
