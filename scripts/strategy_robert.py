# coding=utf-8

import pandas as pd
from data_blob import DataBlob
import importlib


class Position(object):
    def __init__(self, symbol):
        self.symbol = symbol
        self.multiplier = 1
        self.commission = 0
        self.slippage = 0
        self.contract = 0

        self.cash = 0
        self.asset_value = 0


class StrategyRobert(object):
    def __init__(self, data_blob: DataBlob):
        self.data_blob = data_blob

        self.portfolio = []
        self.symbol_list = self.data_blob.get_portfolio_symbol_list()
        for symbol in self.symbol_list:
            self.portfolio.append(Position(symbol))

        # key: combined data type, Forecast, AdjustPrice...
        self.df_combined_data_dict = {}
        # key: symbol, ES, CL...
        self.df_forecast_dict = {}

    def get_df_combined_data_dict(self):
        return self.df_combined_data_dict

    def get_df_forecast_dict(self):
        return self.df_forecast_dict

    def get_buy_costs_ex(self, buy_price, contract, position):
        return contract * (buy_price + position.slippage) * position.multiplier + contract * position.commission

    def get_sell_reward_ex(self, sell_price, contract, position):
        return contract * (sell_price - position.slippage) * position.multiplier - contract * position.commission

    def simulate(self):
        self.calculate_forecast()
        self.generate_combined_data()

        strategy_parameters = self.data_blob.get_strategy_parameters()
        capital = strategy_parameters.get('capital', 0)
        annual_risk_target = strategy_parameters.get('annual_risk_target', 0)
        forecast_base = strategy_parameters.get('forecast_base', 10)
        idm = strategy_parameters.get('instrument_diversification_multiplier', 1)
        fdm = strategy_parameters.get('forecast_diversification_multiplier', 1)

        df_combined_adjust_close = self.df_combined_data_dict['AdjustClose']
        df_combined_forecast = self.df_combined_data_dict['Forecast']
        df_combined_daily_contract = pd.DataFrame(0, index=df_combined_adjust_close.index,
                                                  columns=df_combined_adjust_close.columns)
        df_combined_daily_net_value = pd.DataFrame(0, index=df_combined_adjust_close.index,
                                                   columns=df_combined_adjust_close.columns)
        # weird, but we don't want look ahead bias,
        df_combined_adjust_close = df_combined_adjust_close.shift(-1)
        df_combined_adjust_close = df_combined_adjust_close.ffill()
        for date in df_combined_forecast.index:
            if df_combined_forecast.loc[date].isna().any():
                continue
            mark_to_market = capital
            for position in self.portfolio:
                symbol = position.symbol
                position_weight = 1.0 / len(self.symbol_list)
                daily_risk_target = annual_risk_target / 16.0
                daily_position_capital = mark_to_market * daily_risk_target * position_weight
                price = df_combined_adjust_close.loc[date][symbol]
                daily_vol_value = df_combined_daily_contract.loc[date][symbol]
                forecast = df_combined_forecast.loc[date][symbol]
                if daily_vol_value == 0:
                    need_contract = 0
                else:
                    need_contract = (daily_position_capital / daily_vol_value) * idm * (forecast / forecast_base)
                need_contract = round(need_contract)
                if need_contract < position.contract:
                    sell_contract = position.contract - need_contract
                    position.cash += self.get_sell_reward_ex(price, sell_contract, position)
                    position.contract -= sell_contract
                elif need_contract > position.contract:
                    buy_contract = need_contract - position.contract
                    position.cash -= self.get_buy_costs_ex(price, buy_contract, position)
                    position.contract += buy_contract
                position.asset_value = position.contract * price * position.multiplier
                net_value = position.asset_value + position.cash
                df_combined_daily_contract.loc[date, symbol] = need_contract
                df_combined_daily_net_value.loc[date, symbol] = net_value
                mark_to_market += net_value
        df_combined_daily_net_value['combined'] = df_combined_daily_net_value.sum(axis=1)
        self.df_combined_data_dict['DailyNetValue'] = df_combined_daily_net_value
        self.df_combined_data_dict['DailyContracts'] = df_combined_daily_contract
        return True

    def calculate_forecast(self):
        strategy_rules = self.data_blob.get_strategy_rules()
        for symbol in self.data_blob.get_portfolio_symbol_list():
            df_forecast = None
            df_continuous = self.data_blob.get_data_continuous(symbol)
            for parameters in strategy_rules:
                rule_name = parameters['rule']
                rule_weight = parameters['weight']
                rule = importlib.import_module(f"rules.{rule_name}")
                df = rule.calculate_forecast(df_continuous, parameters)
                df = df * rule_weight
                if df_forecast is None:
                    df_forecast = df
                else:
                    df_forecast += df
            self.df_forecast_dict[symbol] = df_forecast

    def generate_combined_data(self):
        symbol_list = self.data_blob.get_portfolio_symbol_list()
        columns = self.df_forecast_dict[symbol_list[0]].columns
        for column in columns:
            df_combined = None
            for symbol in symbol_list:
                df = self.df_forecast_dict[symbol]
                df = df[column]
                if df_combined is None:
                    df_combined = df[column]
                else:
                    df_combined = pd.merge(df_combined, df[column], left_index=True, right_index=True, how='inner')
                df_combined.columns = list(range(df_combined.columns))
            df_combined.columns = symbol_list
            self.df_combined_data_dict[column] = df_combined
