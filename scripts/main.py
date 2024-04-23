# coding=utf-8

import pandas as pd
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy
import util
from data_blob import DataBlob
from data_source_akshare import DataSourceAkshare
from strategy_robert import StrategyRobert

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)  # 设置宽度不限制

csv_config_dict = dict(
    futures_single_contracts_dir='data/china_futures/single_contracts',
    futures_continuous_dir='data/china_futures/continuous',
    futures_combined_dir='data/china_futures/combined',
    futures_roll_calendar_dir='data/china_futures/roll_calendar',
    portfolio_config_file_path='config/symbols_china_futures.xlsx',
    strategy_rules_config_file_path='config/strategy_rules.xlsx',
    strategy_parameters_config_file_path='config/strategy_parameters.xlsx',
)

if __name__ == "__main__":
    data_source = DataSourceAkshare()
    data_blob = DataBlob(csv_config_dict, data_source)
    # data_blob.populate_single_contracts_in_portfolio()
    # data_blob.update_data()
    data_blob.run_strategy(StrategyRobert)
