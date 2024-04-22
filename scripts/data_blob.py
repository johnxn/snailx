# coding=utf-8
import pandas as pd
import os
import numpy
import util
from data_source import DataSource


class DataBlob(object):
    def __init__(self, csv_config_dict, data_source: DataSource):
        project_dir = util.get_project_dir()
        self.futures_single_contracts_dir = os.path.join(project_dir, csv_config_dict['futures_single_contracts_dir'])
        self.futures_continuous_dir = os.path.join(project_dir, csv_config_dict['futures_continuous_dir'])
        self.futures_forecast_dir = os.path.join(project_dir, csv_config_dict['futures_forecast_dir'])
        self.futures_combined_dir = os.path.join(project_dir, csv_config_dict['futures_combined_dir'])
        self.futures_roll_calendar_dir = os.path.join(project_dir, csv_config_dict['futures_roll_calendar_dir'])
        self.portfolio_config_file_path = os.path.join(project_dir, csv_config_dict['portfolio_config_file_path'])
        self.strategy_rules_config_file_path = os.path.join(project_dir, csv_config_dict['strategy_rules_config_file_path'])
        self.strategy_parameters_config_file_path = os.path.join(project_dir, csv_config_dict['strategy_parameters_config_file_path'])

        self.data_source = data_source

        self.df_portfolio_config = None
        self.df_roll_calendar_cache_dict = {}
        self.df_contract_cache_dict = {}
        self.df_continuous_cache_dict = {}
        self.df_forecast_cache_dict = {}
        self.df_combined_cache_dict = {}

        self.strategy_rules = None
        self.strategy_parameters = None

        self.make_data_dirs()

    def make_data_dirs(self):
        for d in [self.futures_single_contracts_dir,
                  self.futures_continuous_dir,
                  self.futures_forecast_dir,
                  self.futures_combined_dir,
                  self.futures_roll_calendar_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        for d in [self.futures_single_contracts_dir]:
            for symbol in self.get_portfolio_symbol_list():
                sub_dir = os.path.join(d, symbol)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

    def get_portfolio_config(self) -> pd.DataFrame:
        if self.df_portfolio_config is None:
            self.df_portfolio_config = pd.read_excel(self.portfolio_config_file_path)
        return self.df_portfolio_config

    def get_portfolio_symbol_list(self):
        return list(self.get_portfolio_config()['Symbol'])
    
    def get_single_contract_file_path(self, symbol, contract_date):
        return os.path.join(self.futures_single_contracts_dir, f"{symbol}/{contract_date}.csv")

    def get_single_contract(self, symbol, contract_date) -> pd.DataFrame:
        if (symbol, contract_date) in self.df_contract_cache_dict:
            return self.df_contract_cache_dict[(symbol, contract_date)]
        single_contract_filepath = self.get_single_contract_file_path(symbol, contract_date)
        df = pd.read_csv(single_contract_filepath, index_col='Date', parse_dates=['Date'])
        self.df_contract_cache_dict[(symbol, contract_date)] = df
        return df

    def update_single_contract(self, symbol, contract_date):
        contracts_data_dir = os.path.join(self.futures_single_contracts_dir, symbol)
        if not os.path.exists(contracts_data_dir):
            os.mkdir(contracts_data_dir)
        single_contract_filepath = self.get_single_contract_file_path(symbol, contract_date)
        df = self.data_source.download_single_contract(symbol, contract_date)
        if df is None:
            print(f"update single contract {symbol}{contract_date} failed")
            return
        if os.path.exists(single_contract_filepath):
            df_old = self.get_single_contract(symbol, contract_date)
            df_added = df[df.index > df_old.index[-1]]
            if len(df_added) > 0:
                print(f'add {len(df_added)} rows data for single contract {symbol}{contract_date}')
                df = pd.concat([df_old, df_added])
                df.to_csv(single_contract_filepath, index=False)
        else:
            print(f"download contract data for single contract {symbol}{contract_date}")
            df.to_csv(single_contract_filepath, index=False)

    def update_single_contracts_in_portfolio(self):
        symbol_list = self.get_portfolio_symbol_list()
        for symbol in symbol_list:
            contract_date_list = self.data_source.get_active_contract_dates(symbol)
            for contract_date in contract_date_list:
                self.update_single_contract(symbol, contract_date)
        self.df_contract_cache_dict = {}

    def populate_single_contracts_in_portfolio(self):
        symbol_list = self.get_portfolio_symbol_list()
        contracts_dict = self.data_source.download_all_single_contracts(symbol_list)
        for (symbol, contract_date), df_contract in contracts_dict.items():
            single_contract_filepath = self.get_single_contract_file_path(symbol, contract_date)
            if os.path.exists(single_contract_filepath):
                print(f"overriding exists single contract, {symbol}{contract_date}")
            df_contract.to_csv(single_contract_filepath, index=False)
        self.df_contract_cache_dict = {}

    def get_roll_calendar(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_roll_calendar_cache_dict:
            roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
            self.df_roll_calendar_cache_dict[symbol] = pd.read_csv(roll_calendar_file_path, index_col='Date',
                                                                   parse_dates=['Date'])
        return self.df_roll_calendar_cache_dict[symbol]

    def build_roll_calendar_by_volume(self, symbol, last_date=None):
        contract_date_list = []
        for filename in os.listdir(os.path.join(self.futures_single_contracts_dir, symbol)):
            if filename.endswith('.csv'):
                contract_date_list.append(filename.split('.')[0])
        df_volume = None
        for contract_date in contract_date_list:
            df = self.get_single_contract(symbol, contract_date)
            if last_date is not None:
                df = df[df.index > last_date]
            df = df['volume']
            if df_volume is None:
                df_volume = df
            else:
                df_volume = pd.merge(df_volume, df, left_index=True, right_index=True, how='outer')
            # 先随便给个名字, 避免merge的时候报错
            df_volume.columns = list(range(len(df_volume.columns)))
        df_volume.columns = contract_date_list

        df_roll_calendar = df_volume.apply(lambda row: row.nlargest(2).index.tolist(), axis=1, result_type='expand')
        df_roll_calendar.columns = ['FirstContract', 'SecondContract']
        df_roll_calendar['CarryContract'] = df_roll_calendar.apply(
            lambda row: min(row['FirstContract'], row['SecondContract']), axis=1)
        df_roll_calendar['CurrentContract'] = df_roll_calendar.apply(
            lambda row: max(row['FirstContract'], row['SecondContract']), axis=1)

        last_carry_date, last_current_date = None, None
        for date, row in df_roll_calendar.iterrows():
            carry_date, current_date = row['CarryContract'], row['CurrentContract']
            if last_current_date is not None and current_date < last_current_date:
                df_roll_calendar.loc[date, 'CarryContract'] = last_carry_date
                df_roll_calendar.loc[date, 'CurrentContract'] = last_current_date
            last_carry_date, last_current_date = carry_date, current_date

        return df_roll_calendar, df_volume

    def update_roll_calendar_in_portfolio(self, auto_roll=True):
        df_symbol = self.get_portfolio_config()
        for _, row in df_symbol.iterrows():
            symbol = row['Symbol']
            if auto_roll:
                self.build_roll_calendar(symbol)
            else:
                carry_date = row['CarryContract']
                current_date = row['CurrentContract']
                self.build_roll_calendar(symbol, carry_date, current_date)
        self.df_roll_calendar_cache_dict = {}

    def build_roll_calendar(self, symbol, carry_date=None, current_date=None):
        roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
        if not os.path.exists(roll_calendar_file_path):
            df_roll_calendar = self.build_roll_calendar_by_volume(symbol)
        else:
            df_roll_calendar = self.get_roll_calendar(symbol)
        last_date = df_roll_calendar.index[-1]
        if carry_date is not None and current_date is not None:
            # 用户指定了carry和current的contract
            df_current = self.get_single_contract(symbol, current_date)
            df_roll_calendar_added = pd.DataFrame({'CarryContract': carry_date, 'CurrentContract': current_date},
                                                  index=df_current.index[df_current.index > last_date])
        else:
            df_roll_calendar_added = self.build_roll_calendar_by_volume(symbol, last_date)
        if len(df_roll_calendar_added) > 0:
            print(f"update {len(df_roll_calendar_added)} rows of roll calendar for {symbol}")
            df_roll_calendar = pd.concat([df_roll_calendar, df_roll_calendar_added])
            df_roll_calendar.to_csv(roll_calendar_file_path)

    def get_data_continuous(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_continuous_cache_dict:
            continuous_file_path = os.path.join(self.futures_continuous_dir, f"{symbol}.csv")
            self.df_continuous_cache_dict[symbol] = pd.read_csv(continuous_file_path, index_col='Date',
                                                                   parse_dates=['Date'])
        return self.df_continuous_cache_dict[symbol]

    def build_data_continuous(self, symbol) -> pd.DataFrame:
        df_roll_calendar = self.get_roll_calendar(symbol)
        df_continuous = df_roll_calendar
        new_columns = ['CarryPrice', 'CarryVolume', 'CurrentPrice', 'CurrentVolume']
        for column in new_columns:
            df_continuous[column] = None
        last_current_date = None
        for date, row in df_continuous.iterrows():
            carry_date, current_date = row['CarryContract'], row['CurrentContract']
            df_carry = self.get_single_contract(symbol, carry_date)
            df_current = self.get_single_contract(symbol, current_date)
            if last_current_date is not None and current_date != last_current_date:
                df_current_last = self.get_single_contract(symbol, last_current_date)
                spread = df_current.loc[date]['close'] - df_current_last.loc[date]['close']
                df_continuous['AdjustPrice'] = df_continuous['AdjustPrice'] + spread
            df_continuous.loc[date, 'CarryPrice'] = df_carry.loc[date, 'close']
            df_continuous.loc[date, 'CarryVolume'] = df_carry.loc[date, 'volume']
            df_continuous.loc[date, 'CurrentPrice'] = df_current.loc[date, 'close']
            df_continuous.loc[date, 'CurrentVolume'] = df_current.loc[date, 'volume']
            df_continuous.loc[date, 'AdjustPrice'] = df_continuous.loc[date, 'CurrentPrice']
            last_current_date = current_date

        continuous_file_path = os.path.join(self.futures_continuous_dir, f"{symbol}.csv")
        df_continuous.to_csv(continuous_file_path)
        return df_continuous

    def update_data_continuous_in_portfolio(self):
        df_symbol = self.get_portfolio_config()
        for _, row in df_symbol.iterrows():
            symbol = row['Symbol']
            continuous_file_path = os.path.join(self.futures_continuous_dir, f"{symbol}.csv")
            if not os.path.exists(continuous_file_path):
                df_continuous = self.build_data_continuous(symbol)
            else:
                df_continuous = self.get_data_continuous(symbol)
                df_roll_calendar = self.get_roll_calendar(symbol)
                if len(df_roll_calendar) > len(df_continuous):
                    df_continuous = self.build_data_continuous(symbol)
                else:
                    print(f'no new data for symbol: {symbol}')
                    continue
            df_continuous.to_csv(continuous_file_path, index=True)
        self.df_continuous_cache_dict = {}

    def get_forecast(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_forecast_cache_dict:
            self.df_forecast_cache_dict[symbol] = pd.read_csv(os.path.join(self.futures_forecast_dir, f"{symbol}.csv"),
                                                              index_col='Date', parse_dates=['Date'])
        return self.df_forecast_cache_dict[symbol]

    def get_combined_data(self, name) -> pd.DataFrame:
        if name not in self.df_combined_cache_dict:
            self.df_combined_cache_dict[name] = pd.read_csv(os.path.join(self.futures_combined_dir, f"{name}.csv"),
                                                            index_col='Date', parse_dates=['Date'])
        return self.df_combined_cache_dict[name]

    def get_strategy_rules(self) -> list:
        if self.strategy_rules is None:
            df_strategy_rules = pd.read_csv(self.strategy_rules_config_file_path)
            dict_list = df_strategy_rules.to_dict(orient='records')
            self.strategy_rules = [{k: v for k, v in d.items()} for d in dict_list]
        return self.strategy_rules

    def get_strategy_parameters(self) -> dict:
        if self.strategy_parameters is None:
            df_strategy_parameters = pd.read_csv(self.strategy_parameters_config_file_path)
            dict_list = df_strategy_parameters.to_dict(orient='records')
            self.strategy_parameters = {d['Name']: d['Value'] for d in dict_list}
        return self.strategy_parameters

    def run_strategy(self, strategy_class):
        s = strategy_class(self)
        if s.simulate():
            df_forecast_dict = s.get_df_forecast_dict()
            df_combined_data_dict = s.get_df_combined_data_dict()
            for symbol, df_forecast in df_forecast_dict.items():
                df_forecast.to_csv(os.path.join(self.futures_forecast_dir, f"{symbol}.csv"))
            for name, df_combined in df_combined_data_dict.items():
                df_combined.to_csv(os.path.join(self.futures_combined_dir, f"{name}.csv"))
        self.df_forecast_cache_dict = {}
        self.df_combined_cache_dict = {}

    def update_data(self):
        # self.update_single_contracts_in_portfolio()
        self.update_roll_calendar_in_portfolio()
        # self.update_data_continuous_in_portfolio()
