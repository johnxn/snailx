# coding=utf-8
import pandas as pd
from datetime import timedelta
import os
import numpy
import util
from data_source import DataSource


class DataBlob(object):
    def __init__(self, csv_config_dict, data_source: DataSource):
        project_dir = util.get_project_dir()
        self.futures_single_contracts_dir = os.path.join(project_dir, csv_config_dict['futures_single_contracts_dir'])
        self.futures_continuous_dir = os.path.join(project_dir, csv_config_dict['futures_continuous_dir'])
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
        self.df_combined_cache_dict = {}

        self.strategy_rules = None
        self.strategy_parameters = None

        self.make_data_dirs()

    def make_data_dirs(self):
        for d in [self.futures_single_contracts_dir,
                  self.futures_continuous_dir,
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
            df_added = df_added.drop_duplicates()
            if len(df_added) > 0:
                print(f'add {len(df_added)} rows data for single contract {symbol}{contract_date}')
                df = pd.concat([df_old, df_added])
                df.to_csv(single_contract_filepath, index=True)
        else:
            print(f"download contract data for single contract {symbol}{contract_date}")
            df.to_csv(single_contract_filepath, index=True)

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
            df_contract.to_csv(single_contract_filepath, index=True)
        self.df_contract_cache_dict = {}
    
    def get_contract_date_list(self, symbol):
        contract_date_list = []
        for filename in os.listdir(os.path.join(self.futures_single_contracts_dir, symbol)):
            if filename.endswith('.csv'):
                contract_date_list.append(filename.split('.')[0])
        return contract_date_list
    
    def get_start_and_end_date(self, symbol):
        start_date = None
        end_date = None
        for contract_date in self.get_contract_date_list(symbol):
            df = self.get_single_contract(symbol, contract_date)
            if start_date is None or df.index[0] < start_date:
                start_date = df.index[0]
            if end_date is None or df.index[-1] > end_date:
                end_date = df.index[-1]
        return start_date, end_date

    def get_roll_calendar(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_roll_calendar_cache_dict:
            roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
            self.df_roll_calendar_cache_dict[symbol] = pd.read_csv(roll_calendar_file_path, index_col='Date',
                                                                   parse_dates=['Date'])
        return self.df_roll_calendar_cache_dict[symbol]
    
    def build_roll_calendar_by_config(self, symbol, last_date=None):
        pass

    def build_roll_calendar_by_volume(self, symbol, start_date=None, end_date=None):
        if start_date and end_date:
            print(f"generate new roll calendar for {symbol},  start_date {util.datetime_to_str(start_date)}, end_date {util.datetime_to_str(end_date)}")
        contract_date_list = self.get_contract_date_list(symbol)
        df_volume = None
        for contract_date in contract_date_list:
            df = self.get_single_contract(symbol, contract_date)
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index < end_date]
            if len(df) > 0:
                df = df[['Volume']]
                df.rename({'Volume':contract_date}, axis=1, inplace=True)
                if df_volume is None:
                    df_volume = df
                else:
                    df_volume = pd.merge(df_volume, df, left_index=True, right_index=True, how='outer')
        if df_volume is None or len(df_volume.columns) < 2:
            # print(f"no roll_calendar generated for {symbol}, start_date {util.datetime_to_str(start_date)}, end_date {util.datetime_to_str(end_date)}")
            return None

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

        return df_roll_calendar

    def update_roll_calendar_in_portfolio(self):
        for symbol in self.get_portfolio_symbol_list():
            roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
            if not os.path.exists(roll_calendar_file_path):
                start_date, end_date = self.get_start_and_end_date(symbol)
                end_date += timedelta(days=1)

                delta_days = timedelta(days=200)
                begin_date = start_date
                df_roll_calendar_section_list = []
                while True:
                    next_date = begin_date + delta_days
                    if next_date >= end_date:
                        next_date = end_date
                    df_section = self.build_roll_calendar_by_volume(symbol, begin_date, next_date)
                    if df_section is not None:
                        df_roll_calendar_section_list.append(df_section)
                    if next_date >= end_date:
                        break
                    begin_date = next_date
                df_roll_calendar = pd.concat(df_roll_calendar_section_list)
                df_roll_calendar.to_csv(roll_calendar_file_path)
            else:
                df_roll_calendar = self.get_roll_calendar(symbol)
                start_date = df_roll_calendar.index[-1]
                start_date += timedelta(days=1)
                df_roll_calendar_added = self.build_roll_calendar_by_volume(symbol, start_date)
                if df_roll_calendar_added and len(df_roll_calendar_added) > 0:
                    print(f"update {len(df_roll_calendar_added)} rows of roll calendar for {symbol}")
                    df_roll_calendar = pd.concat([df_roll_calendar, df_roll_calendar_added])
                    df_roll_calendar.to_csv(roll_calendar_file_path)
        self.df_roll_calendar_cache_dict = {}
    
    def generate_roll_config_in_portfolio(self):
        pass

    def get_data_continuous(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_continuous_cache_dict:
            continuous_file_path = os.path.join(self.futures_continuous_dir, f"{symbol}.csv")
            self.df_continuous_cache_dict[symbol] = pd.read_csv(continuous_file_path, index_col='Date',
                                                                   parse_dates=['Date'])
        return self.df_continuous_cache_dict[symbol]

    def build_data_continuous(self, symbol) -> pd.DataFrame:
        df_roll_calendar = self.get_roll_calendar(symbol)
        df_continuous = df_roll_calendar
        df_continuous.drop(columns=['FirstContract', 'SecondContract'], inplace=True)
        new_columns = ['CarryPrice', 'CarryVolume', 'CurrentPrice', 'CurrentVolume']
        for column in new_columns:
            df_continuous[column] = None
        last_current_contract_date = None
        for date, row in df_continuous.iterrows():
            carry_contract_date, current_contract_date = row['CarryContract'], row['CurrentContract']
            df_carry = self.get_single_contract(symbol, carry_contract_date)
            df_current = self.get_single_contract(symbol, current_contract_date)
            if last_current_contract_date is not None and current_contract_date != last_current_contract_date:
                df_current_last = self.get_single_contract(symbol, last_current_contract_date)
                if date in df_current_last.index:
                    spread = df_current.loc[date]['Close'] - df_current_last.loc[date]['Close']
                else:
                    print(f"missing {date} data for {symbol}{last_current_contract_date} while backjusting data!!!, current contract: {symbol}{current_contract_date}")
                    spread = 0
                df_continuous['AdjustPrice'] = df_continuous['AdjustPrice'] + spread
            if date in df_carry.index:
                df_continuous.loc[date, 'CarryPrice'] = df_carry.loc[date, 'Close']
                df_continuous.loc[date, 'CarryVolume'] = df_carry.loc[date, 'Volume']
            else:
                print(f"missing {date} data for carry contract {symbol}{carry_contract_date}")
                df_continuous.loc[date, 'CarryPrice'] = numpy.nan
                df_continuous.loc[date, 'CarryVolume'] = numpy.nan
            if date in df_current.index:
                df_continuous.loc[date, 'CurrentPrice'] = df_current.loc[date, 'Close']
                df_continuous.loc[date, 'CurrentVolume'] = df_current.loc[date, 'Volume']
            else:
                print(f"missing {date} data for current contract {symbol}{current_contract_date}")
                df_continuous.loc[date, 'CurrentPrice'] = numpy.nan
                df_continuous.loc[date, 'CurrentVolume'] = numpy.nan

            df_continuous.loc[date, 'AdjustPrice'] = df_continuous.loc[date, 'CurrentPrice']
            last_current_contract_date = current_contract_date

        return df_continuous

    def update_data_continuous_in_portfolio(self):
        for symbol in self.get_portfolio_symbol_list():
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

    def get_combined_data(self, name) -> pd.DataFrame:
        if name not in self.df_combined_cache_dict:
            self.df_combined_cache_dict[name] = pd.read_csv(os.path.join(self.futures_combined_dir, f"{name}.csv"),
                                                            index_col='Date', parse_dates=['Date'])
        return self.df_combined_cache_dict[name]

    def get_strategy_rules(self) -> list:
        if self.strategy_rules is None:
            df_strategy_rules = pd.read_excel(self.strategy_rules_config_file_path)
            self.strategy_rules = df_strategy_rules.to_dict(orient='records')
        return self.strategy_rules

    def get_strategy_parameters(self) -> dict:
        if self.strategy_parameters is None:
            df_strategy_parameters = pd.read_excel(self.strategy_parameters_config_file_path)
            dict_list = df_strategy_parameters.to_dict(orient='records')
            self.strategy_parameters = {d['Name']: d['Value'] for d in dict_list}
        return self.strategy_parameters

    def run_strategy(self, strategy_class):
        s = strategy_class(self)
        if s.simulate():
            df_combined_data_dict = s.get_df_combined_data_dict()
            for name, df_combined in df_combined_data_dict.items():
                df_combined.to_csv(os.path.join(self.futures_combined_dir, f"{name}.csv"), index=True)
        self.df_combined_cache_dict = {}

    def update_data(self):
        self.update_single_contracts_in_portfolio()
        self.update_roll_calendar_in_portfolio()
        self.update_data_continuous_in_portfolio()
