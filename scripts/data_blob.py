# coding=utf-8
import pandas as pd
from datetime import timedelta, datetime
import os
import numpy
import matplotlib.pyplot as plt
import util
from data_source import DataSource
from data_broker import DataBroker


class DataBlob(object):
    def __init__(self, csv_config_dict, data_source: DataSource,data_broker: DataBroker):
        project_dir = util.get_project_dir()
        self.futures_single_contracts_dir = os.path.join(project_dir, csv_config_dict['futures_single_contracts_dir'])
        self.futures_continuous_dir = os.path.join(project_dir, csv_config_dict['futures_continuous_dir'])
        self.futures_combined_dir = os.path.join(project_dir, csv_config_dict['futures_combined_dir'])
        self.futures_roll_calendar_dir = os.path.join(project_dir, csv_config_dict['futures_roll_calendar_dir'])
        self.symbol_config_file_path = os.path.join(project_dir, csv_config_dict['symbol_config_file_path'])
        self.strategy_rules_config_file_path = os.path.join(project_dir, csv_config_dict['strategy_rules_config_file_path'])
        self.strategy_parameters_config_file_path = os.path.join(project_dir, csv_config_dict['strategy_parameters_config_file_path'])
        self.daily_account_value_file_path = os.path.join(project_dir, csv_config_dict['daily_account_value_file_path'])
        self.daily_symbol_weight_file_path = os.path.join(project_dir, csv_config_dict['daily_symbol_weight_file_path'])
        self.daily_rule_weight_file_path = os.path.join(project_dir, csv_config_dict['daily_rule_weight_file_path'])

        self.data_source = data_source
        self.data_broker = data_broker

        self.df_roll_calendar_cache_dict = {}
        self.df_contract_cache_dict = {}
        self.df_continuous_cache_dict = {}
        self.df_combined_cache_dict = {}
        self.df_trade_calendar_cahce_dict = {}
        self.df_daily_account_value = None
        self.df_daily_symbol_weights = None
        self.df_daily_rule_weights = None

        self.all_symbol_info_list = None
        self.all_symbol_info_dict = None
        self.portfolio = None

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

    def get_all_symbol_info_list(self) -> list:
        if self.all_symbol_info_list is None:
            df_symbol_config = pd.read_csv(self.symbol_config_file_path)
            self.all_symbol_info_list = df_symbol_config.to_dict(orient='records')
        return self.all_symbol_info_list

    def get_all_symbol_info_dict(self) -> dict:
        if self.all_symbol_info_dict is None:
            all_symbol_info_list = self.get_all_symbol_info_list()
            self.all_symbol_info_dict = {symbol_info['symbol']: symbol_info for symbol_info in all_symbol_info_list}
        return self.all_symbol_info_dict

    def get_portfolio(self) -> list:
        df_symbol_weights = self.get_daily_symbol_weights()
        symbol_list = df_symbol_weights.columns
        if self.portfolio is None:
            self.portfolio = []
            all_symbol_info_dict = self.get_all_symbol_info_dict()
            for symbol in symbol_list:
                self.portfolio.append(all_symbol_info_dict[symbol])
        return self.portfolio

    def get_portfolio_symbol_list(self) -> list:
        return [v['symbol'] for v in self.get_portfolio()]
    
    def get_portfolio_symbol_info(self, symbol) -> dict:
        for info in self.get_portfolio():
            if info['symbol'] == symbol:
                return info
        return None

    def get_single_contract_file_path(self, symbol, contract_date):
        return os.path.join(self.futures_single_contracts_dir, f"{symbol}/{contract_date}.csv")

    def get_single_contract(self, symbol, contract_date) -> pd.DataFrame:
        if (symbol, contract_date) in self.df_contract_cache_dict:
            return self.df_contract_cache_dict[(symbol, contract_date)]
        single_contract_filepath = self.get_single_contract_file_path(symbol, contract_date)
        if not os.path.exists(single_contract_filepath):
            print(f"no single contarct for {symbol}{contract_date}")
            return None
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
            raise Exception(f"update single contract {symbol}{contract_date} failed")
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
        self.df_trade_calendar_cahce_dict = {}

    def populate_single_contracts_in_portfolio(self):
        symbol_list = self.get_portfolio_symbol_list()
        contracts_dict = self.data_source.download_all_single_contracts(symbol_list)
        for (symbol, contract_date), df_contract in contracts_dict.items():
            single_contract_filepath = self.get_single_contract_file_path(symbol, contract_date)
            if os.path.exists(single_contract_filepath):
                print(f"single contract alreday exists, skipped..., {symbol}{contract_date}")
                df_contract.to_csv(single_contract_filepath, index=True)
        self.df_contract_cache_dict = {}
    
    def get_contract_date_list(self, symbol):
        contract_date_list = []
        for filename in os.listdir(os.path.join(self.futures_single_contracts_dir, symbol)):
            if filename.endswith('.csv'):
                contract_date_list.append(filename.split('.')[0])
        return contract_date_list
    
    def get_start_and_end_date(self, symbol):
        df_trade_calendar = self.get_trade_calendar(symbol)
        return df_trade_calendar.index[0], df_trade_calendar.index[-1]
    
    def get_trade_calendar(self, symbol):
        if symbol not in self.df_trade_calendar_cahce_dict:
            df_trade_calendar = None
            df_list = []
            for contract_date in self.get_contract_date_list(symbol):
                df = self.get_single_contract(symbol, contract_date)
                df = df.drop(df.columns, axis=1)
                df_list.append(df)
            df_trade_calendar = pd.concat(df_list)
            # fucking insane...
            df_trade_calendar = df_trade_calendar.reset_index().drop_duplicates(subset='Date', keep='first').set_index('Date')
            self.df_trade_calendar_cahce_dict[symbol] = df_trade_calendar
        return self.df_trade_calendar_cahce_dict[symbol]

    def get_roll_calendar(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_roll_calendar_cache_dict:
            roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
            self.df_roll_calendar_cache_dict[symbol] = pd.read_csv(roll_calendar_file_path, index_col='Date',
                                                                   parse_dates=['Date'],
                                                                   dtype={'CarryContract': str, 'CurrentContract': str})
        return self.df_roll_calendar_cache_dict[symbol]
    
    def build_roll_calendar_by_config(self, symbol, start_date=None, end_date=None):
        symbol_info = self.get_portfolio_symbol_info(symbol)
        roll_offset_days = int(symbol_info['roll_offset_days'])
        expiry_offset_days = int(symbol_info['expiry_offset_days'])
        roll_offset_days += expiry_offset_days
        roll_cycle = [int(v) for v in symbol_info['roll_cycle'].split(',')]
        contract_cycle = [int(v) for v in symbol_info['contract_cycle'].split(',')]
        carry_offset = int(symbol_info['carry_offset'])
        df_roll_calendar = self.get_trade_calendar(symbol)
        if start_date is not None:
            df_roll_calendar = df_roll_calendar[df_roll_calendar.index >= start_date]
        if end_date is not None:
            df_roll_calendar = df_roll_calendar[df_roll_calendar.index < end_date]
        new_columns = ['CurrentContract', 'CarryContract']
        for column in new_columns:
            df_roll_calendar[column] = None
        for date, _ in df_roll_calendar.iterrows():
            current_year = None
            current_month = None
            for add_year in range(-1, 4):
                is_found = False
                for candidate_month in roll_cycle:
                    candidate_year = date.year + add_year
                    roll_date = datetime(candidate_year, candidate_month, 1) + timedelta(days=roll_offset_days)
                    if date < roll_date:
                        current_year = candidate_year
                        current_month = candidate_month
                        is_found = True
                        break
                if is_found:
                    break
            if current_year is None or current_month is None:
                raise Exception(f"can't find CurrentContract for {symbol} on {date}")

            current_month_index = contract_cycle.index(current_month)
            carry_month_index = current_month_index + carry_offset
            carry_year = current_year + carry_month_index // len(contract_cycle)
            carry_month = contract_cycle[carry_month_index % len(contract_cycle)]
            carry_contract_date = f"{carry_year}{carry_month:02d}"
            current_contract_date = f"{current_year}{current_month:02d}"
            df_roll_calendar.loc[date, 'CarryContract'] = carry_contract_date
            df_roll_calendar.loc[date, 'CurrentContract'] = current_contract_date
        return df_roll_calendar
    
    def merge_contract_volume(self, symbol, start_date=None, end_date=None):
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
        return df_volume

    def build_roll_calendar_by_volume(self, symbol, start_date=None, end_date=None):
        start_date_str = util.datetime_to_str(start_date) if start_date is not None else "None"
        end_date_str = util.datetime_to_str(end_date) if end_date is not None else "None"
        df_volume = self.merge_contract_volume(symbol, start_date, end_date)
        if df_volume is None or len(df_volume.columns) < 2:
            print(f"no roll_calendar generated for {symbol}, start_date {start_date_str}, end_date {end_date_str}")
            return None

        df_roll_calendar = df_volume.apply(lambda row: row.nlargest(2).index.tolist(), axis=1, result_type='expand')
        df_roll_calendar.columns = ['CurrentContract', 'CarryContract']
        return df_roll_calendar
    
    def generate_roll_calendar_in_portfolio(self, by_volume=True):
        for symbol in self.get_portfolio_symbol_list():
            roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
            if os.path.exists(roll_calendar_file_path):
                print(f"try to override exists roll calendar for {symbol}, please delete it manually")
                continue
            start_date, end_date = self.get_start_and_end_date(symbol)
            end_date += timedelta(days=1)

            delta_days = timedelta(days=200)
            begin_date = start_date
            df_roll_calendar_section_list = []
            while True:
                next_date = begin_date + delta_days
                if next_date >= end_date:
                    next_date = end_date
                if by_volume:
                    df_section = self.build_roll_calendar_by_volume(symbol, begin_date, next_date)
                else:
                    df_section = self.build_roll_calendar_by_config(symbol, begin_date, next_date)
                if df_section is not None:
                    df_roll_calendar_section_list.append(df_section)
                if next_date >= end_date:
                    break
                begin_date = next_date
            df_roll_calendar = pd.concat(df_roll_calendar_section_list)
            # 不往旧的合约roll
            last_carry_date, last_current_date = None, None
            for date, row in df_roll_calendar.iterrows():
                current_date = row['CurrentContract']
                if last_current_date is not None and current_date < last_current_date:
                    df_roll_calendar.loc[date, 'CarryContract'] = last_carry_date
                    df_roll_calendar.loc[date, 'CurrentContract'] = last_current_date
                last_carry_date, last_current_date = df_roll_calendar.loc[date, 'CarryContract'], df_roll_calendar.loc[
                    date, 'CurrentContract']

            df_roll_calendar = df_roll_calendar.drop_duplicates(subset='CurrentContract', keep='last')
            df_roll_calendar.to_csv(os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv"), index=True)

    def update_roll_calendar_in_portfolio(self, by_volume=True):
        for symbol in self.get_portfolio_symbol_list():
            roll_calendar_file_path = os.path.join(self.futures_roll_calendar_dir, f"{symbol}.csv")
            if not os.path.exists(roll_calendar_file_path):
                print(f"no roll calendar for {symbol}, please generate and modify it first!")
            else:
                df_roll_calendar = self.get_roll_calendar(symbol)
                start_date = df_roll_calendar.index[-1]
                start_date += timedelta(days=1)
                if by_volume is True:
                    df_roll_calendar_added = self.build_roll_calendar_by_volume(symbol, start_date)
                else:
                    df_roll_calendar_added = self.build_roll_calendar_by_config(symbol, start_date)
                if df_roll_calendar_added is not None and len(df_roll_calendar_added) > 0:
                    df_roll_calendar = pd.concat([df_roll_calendar, df_roll_calendar_added])
                    df_roll_calendar = df_roll_calendar.drop_duplicates(subset='CurrentContract', keep='last')
                    df_roll_calendar.to_csv(roll_calendar_file_path, index=True)

        self.df_roll_calendar_cache_dict = {}
    
    def get_data_continuous(self, symbol) -> pd.DataFrame:
        if symbol not in self.df_continuous_cache_dict:
            continuous_file_path = os.path.join(self.futures_continuous_dir, f"{symbol}.csv")
            self.df_continuous_cache_dict[symbol] = pd.read_csv(continuous_file_path, index_col='Date',
                                                                   parse_dates=['Date'],
                                                                    dtype={'CarryContract': str, 'CurrentContract': str})
        return self.df_continuous_cache_dict[symbol]

    def build_data_continuous(self, symbol) -> pd.DataFrame:
        df_roll_calendar = self.get_roll_calendar(symbol)
        df_trade_calendar = self.get_trade_calendar(symbol)
        df_continuous = df_roll_calendar.reindex(df_trade_calendar.index).bfill()
        new_columns = ['CarryPrice', 'CarryVolume', 'CurrentPrice', 'CurrentVolume']
        for column in new_columns:
            df_continuous[column] = None
        last_current_contract_date = None
        for date, row in df_continuous.iterrows():
            carry_contract_date, current_contract_date = row['CarryContract'], row['CurrentContract']
            df_carry = self.get_single_contract(symbol, carry_contract_date)
            df_current = self.get_single_contract(symbol, current_contract_date)
            if df_carry is None or df_current is None:
                print(f"missing contract data for {symbol}{carry_contract_date}, {symbol}{current_contract_date}, skipping {date}")
                continue
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

        if numpy.isnan(df_continuous.iloc[0]['AdjustPrice']):
            df_continuous = df_continuous.bfill()

        return df_continuous

    def update_data_continuous_in_portfolio(self):
        symbol_list = self.get_portfolio_symbol_list()
        for symbol in symbol_list:
            continuous_file_path = os.path.join(self.futures_continuous_dir, f"{symbol}.csv")
            df_continuous = self.build_data_continuous(symbol)
            df_continuous.to_csv(continuous_file_path, index=True)
        self.df_continuous_cache_dict = {}

    def get_combined_data(self, name) -> pd.DataFrame:
        dtype_dict = {
            'CarryContract' : str,
            'CurrentContract' : str,
            'DailyContracts' : int,
        }
        if name not in self.df_combined_cache_dict:
            self.df_combined_cache_dict[name] = pd.read_csv(os.path.join(self.futures_combined_dir, f"{name}.csv"),
                                                            index_col='Date', parse_dates=['Date'],
                                                            dtype=dtype_dict.get(name, float))
        return self.df_combined_cache_dict[name]

    def get_strategy_rules(self) -> list:
        if self.strategy_rules is None:
            df_strategy_rules = pd.read_csv(self.strategy_rules_config_file_path)
            self.strategy_rules = df_strategy_rules.to_dict(orient='records')
        return self.strategy_rules

    def get_strategy_parameters(self) -> dict:
        if self.strategy_parameters is None:
            df_strategy_parameters = pd.read_csv(self.strategy_parameters_config_file_path)
            dict_list = df_strategy_parameters.to_dict(orient='records')
            self.strategy_parameters = {d['Name']: d['Value'] for d in dict_list}
        return self.strategy_parameters
    
    def update_daily_account_value(self):
        today_dt = util.str_to_datetime(util.datetime_to_str(datetime.today()))
        daily_net_df = pd.read_csv(self.daily_account_value_file_path, index_col='Date', parse_dates=['Date'])
        total_value = self.data_broker.get_account_value()
        daily_net_df.loc[today_dt,'NetValue'] = total_value
        daily_net_df.to_csv(self.daily_account_value_file_path, index=True)

    def get_daily_account_value(self):
        if os.path.exists(self.daily_account_value_file_path):
            self.df_daily_account_value = pd.read_csv(self.daily_account_value_file_path, index_col='Date', parse_dates=['Date'])
            return self.df_daily_account_value
        return None

    def get_daily_symbol_weights(self):
        if os.path.exists(self.daily_symbol_weight_file_path):
            self.df_daily_symbol_weights = pd.read_csv(self.daily_symbol_weight_file_path, index_col='Date', parse_dates=['Date'])
            return self.df_daily_symbol_weights
        return None

    def get_daily_rule_weights(self):
        if os.path.exists(self.daily_rule_weight_file_path):
            self.df_daily_rule_weights = pd.read_csv(self.daily_rule_weight_file_path, index_col='Date', parse_dates=['Date'])
            return self.df_daily_rule_weights
        return None

    def plot_daily_account_value(self):
        df_daily_value = self.get_daily_account_value()
        df_daily_value = df_daily_value.ffill().fillna(0)
        df_daily_value['Total'] = df_daily_value['NetValue'] + df_daily_value['Cash']
        if df_daily_value is not None:
            df_daily_value['Total'].plot()
            plt.show()

    def run_strategy(self, strategy_class, is_live=False):
        s = strategy_class(self)
        s.simulate(is_live)
        df_combined_data_dict = s.get_df_combined_data_dict()
        for name, df_combined in df_combined_data_dict.items():
            df_combined.to_csv(os.path.join(self.futures_combined_dir, f"{name}.csv"), index=True)
        self.df_combined_cache_dict = {}

    def get_latest_operation_signals(self):
        df_list = []
        names = []
        df_daily_contracts = self.get_combined_data('DailyContracts')
        if len(df_daily_contracts) >= 2:
            df_diff_contracts = pd.DataFrame([df_daily_contracts.iloc[-1] - df_daily_contracts.iloc[-2]], columns=df_daily_contracts.columns)
            df_list.append(df_diff_contracts)
            names.append('DiffContracts')

        df_combined_list = []
        combined_names = ['DailyContracts', 'DailyWeights', 'CurrentContract', 'CarryContract', 'AdjustPrice',  'Forecast']
        signal_date = None
        for name in combined_names:
            df = self.get_combined_data(name)
            df = df[df.index == df.index[-1]]
            signal_date = df.index[0]
            df_combined_list.append(df)
        
        df_list += df_combined_list
        names += combined_names

        df_signal = pd.concat(df_list)
        df_signal.index = names
        return signal_date, df_signal

    def plot_simulated_daily_net_value(self):
        df_daily_net_value = self.get_combined_data('DailyNetValue')
        df_account_value = self.get_daily_account_value()
        df_daily_net_value = df_daily_net_value[df_daily_net_value.index >= df_account_value.index[0]]
        # df_daily_net_value = df_daily_net_value.copy()

        df_daily_net_value.plot()
        plt.show()

        df_daily_net_value_withou_combined = df_daily_net_value.drop('combined', axis=1)
        print("return correlation matrix")
        print(df_daily_net_value_withou_combined.corr())

        df_combined_adjust_price = self.get_combined_data('AdjustPrice')
        print("price correlation matrix")
        print(df_combined_adjust_price.corr())

        merged_index = df_account_value.index.union(df_daily_net_value.index)
        df_account_value = df_account_value.reindex(merged_index).ffill().fillna(0)
        df_daily_net_value['NetValue'] = df_daily_net_value['combined'] + df_account_value['NetValue'] + df_account_value['Cash']
        df_daily_net_value['DailyReturn'] = df_daily_net_value['NetValue'].pct_change()
        rolling_annual_return = df_daily_net_value['DailyReturn'].rolling(window=252).apply(lambda x: ((1 + x.mean()) ** 252) - 1, raw=True)
        rolling_annual_volatility = df_daily_net_value['DailyReturn'].rolling(window=252).std() * numpy.sqrt(252)
        rolling_sharpe_ratio = rolling_annual_return  / rolling_annual_volatility

        df_daily_net_value['MaxValue'] = df_daily_net_value['NetValue'].rolling(window=252).max()
        df_daily_net_value['Drawdown'] = (df_daily_net_value['NetValue'] - df_daily_net_value['MaxValue']) / df_daily_net_value['MaxValue']
        rolling_max_drawdown = df_daily_net_value['Drawdown'].rolling(window=252).min()

        # 计算净值的累积收益率
        cumulative_return = df_daily_net_value['NetValue'][-1] / df_daily_net_value['NetValue'][0]
        # 计算年化收益率
        years = (df_daily_net_value.index[-1] - df_daily_net_value.index[0]).days / 365.0
        annual_return = (cumulative_return ** (1 / years)) - 1

        # 计算年化波动率
        annual_volatility = df_daily_net_value['DailyReturn'].std() * numpy.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        print(f"total annual return: {annual_return}, sharpe ratio: {sharpe_ratio}")

        # 绘制图表
        plt.figure(figsize=(10, 6))

        # 绘制滚动年化收益率
        plt.subplot(2, 2, 1)
        rolling_annual_return.plot(color='blue')
        plt.title('Rolling Annual Return')
        plt.xlabel('Date')
        plt.ylabel('Return')

        # 绘制滚动年化波动率
        plt.subplot(2, 2, 2)
        rolling_annual_volatility.plot(color='green')
        plt.title('Rolling Annual Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')

        # 绘制滚动夏普比率
        plt.subplot(2, 2, 3)
        rolling_sharpe_ratio.plot(color='red')
        plt.title('Rolling Sharpe Ratio')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')

        # 绘制滚动最大回撤
        plt.subplot(2, 2, 4)
        rolling_max_drawdown.plot(color='purple')
        plt.title('Rolling Max Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')

        plt.tight_layout()  # 自动调整子图布局，防止重叠
        plt.suptitle('Performance Metrics', fontsize=16)  # 添加总标题
        plt.show()



