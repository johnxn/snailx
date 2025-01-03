# coding=utf-8
import numpy
import pandas as pd
from data_blob import DataBlob
import importlib


class Position(object):
    def __init__(self, symbol, multiplier, commission, slippage=0):
        self.symbol = symbol
        self.multiplier = multiplier
        self.commission = commission
        self.slippage = slippage

        self.contract = 0
        self.cash = 0.0
        self.asset_value = 0.0


class StrategyRobert(object):
    def __init__(self, data_blob: DataBlob):
        self.data_blob = data_blob

        self.position_list = []
        for info in self.data_blob.get_portfolio():
            symbol = info['symbol']
            multiplier = info['multiplier']
            commission = info['commission']
            self.position_list.append(Position(symbol, multiplier, commission))

        # key: combined data type, Forecast, AdjustPrice...
        self.df_combined_data_dict = {}
        # key: symbol, ES, CL...
        self.df_continuous_with_forecast_dict = {}

    def get_df_combined_data_dict(self):
        return self.df_combined_data_dict

    def get_buy_costs_ex(self, buy_price, contract, position):
        return contract * (buy_price + position.slippage) * position.multiplier + contract * position.commission

    def get_sell_reward_ex(self, sell_price, contract, position):
        return contract * (sell_price - position.slippage) * position.multiplier - contract * position.commission
    
    def simulate(self, is_live=False):
        self.calculate_extra_data()
        self.generate_combined_data()

        strategy_parameters = self.data_blob.get_strategy_parameters()
        df_daily_account_value = self.data_blob.get_daily_account_value()
        df_daily_symbol_weight = self.data_blob.get_daily_symbol_weights()

        annual_risk_target = strategy_parameters.get('annual_risk_target', 0)
        forecast_base = strategy_parameters.get('forecast_base', 10)
        idm = strategy_parameters.get('instrument_diversification_multiplier', 1)
        fdm = strategy_parameters.get('forecast_diversification_multiplier', 1)

        df_combined_adjust_close = self.df_combined_data_dict['AdjustPrice']
        df_combined_forecast = self.df_combined_data_dict['Forecast']
        df_price_val = self.df_combined_data_dict['PriceVol']
        df_combined_daily_contract = pd.DataFrame(0, index=df_combined_adjust_close.index,
                                                  columns=df_combined_adjust_close.columns)
        df_combined_daily_net_value = pd.DataFrame(0.0, index=df_combined_adjust_close.index,
                                                   columns=df_combined_adjust_close.columns)

        merged_index = df_combined_forecast.index.union(df_daily_symbol_weight.index)
        merged_index = merged_index.union(df_daily_account_value.index)
        df_daily_account_value = df_daily_account_value.reindex(merged_index).ffill().fillna(0)
        df_daily_account_value = df_daily_account_value.reindex(df_combined_forecast.index)
        df_daily_symbol_weight = df_daily_symbol_weight.reindex(merged_index).ffill().fillna(0)
        df_daily_symbol_weight = df_daily_symbol_weight.reindex(df_combined_forecast.index)
        for date in df_combined_forecast.index:
            mark_to_market = df_daily_account_value.loc[date]['NetValue'] + df_daily_account_value.loc[date]['Cash']
            for position in self.position_list:
                symbol = position.symbol
                position_weight = df_daily_symbol_weight.loc[date][symbol]
                daily_risk_target = annual_risk_target / 16.0
                daily_position_capital = mark_to_market * daily_risk_target * position_weight
                price_vol = df_price_val.loc[date][symbol]
                forecast = df_combined_forecast.loc[date][symbol]
                if price_vol == 0:
                    need_contract = 0
                else:
                    need_contract = (daily_position_capital / price_vol) * idm * fdm * (forecast / forecast_base)
                if numpy.isnan(need_contract):
                    print(f"something is wrong!!! {symbol}, date {date}")
                need_contract = round(need_contract)
                df_combined_daily_contract.loc[date, symbol] = need_contract

                if not is_live:
                    date_tomorrow_iloc = df_combined_forecast.index.get_loc(date) + 1
                    date_tomorrow_iloc = date_tomorrow_iloc if date_tomorrow_iloc < len(
                        df_combined_forecast.index) else len(df_combined_forecast.index) - 1
                    date_tomorrow = df_combined_forecast.index[date_tomorrow_iloc]
                    price_tomorrow = df_combined_adjust_close.loc[date_tomorrow][symbol]
                    if need_contract < position.contract:
                        sell_contract = position.contract - need_contract
                        position.cash += self.get_sell_reward_ex(price_tomorrow, sell_contract, position)
                        position.contract -= sell_contract
                    elif need_contract > position.contract:
                        buy_contract = need_contract - position.contract
                        position.cash -= self.get_buy_costs_ex(price_tomorrow, buy_contract, position)
                        position.contract += buy_contract
                    position.asset_value = position.contract * price_tomorrow * position.multiplier
                    net_value = position.asset_value + position.cash
                    df_combined_daily_net_value.loc[date, symbol] = net_value
                    df_daily_account_value.loc[date_tomorrow, 'NetValue'] += net_value

        df_combined_daily_net_value['combined'] = df_combined_daily_net_value.sum(axis=1)
        self.df_combined_data_dict['DailyNetValue'] = df_combined_daily_net_value
        self.df_combined_data_dict['DailyContracts'] = df_combined_daily_contract
        self.df_combined_data_dict['DailyWeights'] = df_daily_symbol_weight
        return True
    
    def calculate_extra_data(self):
        for position in self.position_list:
            symbol = position.symbol
            multiplier = position.multiplier
            df_continuous = self.data_blob.get_data_continuous(symbol)
            self.calculate_forecast(df_continuous)
            self.calculate_daily_vol_value(df_continuous, multiplier)
            self.df_continuous_with_forecast_dict[symbol] = df_continuous

    def calculate_forecast(self, df_continuous):
        strategy_rules = self.data_blob.get_strategy_rules()
        strategy_parameters = self.data_blob.get_strategy_parameters()
        df_daily_rule_weight = self.data_blob.get_daily_rule_weights()
        merged_index = df_continuous.index.union(df_daily_rule_weight.index)
        df_daily_rule_weight = df_daily_rule_weight.reindex(merged_index).ffill().fillna(0)
        df_daily_rule_weight = df_daily_rule_weight.reindex(df_continuous.index)
        df_forecast = None
        for rule_args in strategy_rules:
            rule_args.update(strategy_parameters)
            rule_name = rule_args['rule']
            df_rule_weight = df_daily_rule_weight[rule_args['name']]
            rule = importlib.import_module(f"rules.{rule_name}")
            df = rule.calculate_forecast(df_continuous, rule_args)
            df = df * df_rule_weight
            if df_forecast is None:
                df_forecast = df
            else:
                df_forecast += df
        df_continuous['Forecast'] = df_forecast
    
    def calculate_daily_vol_value(self, df_continuous, multiplier):
        strategy_parameters = self.data_blob.get_strategy_parameters()
        volatility_lookback_window = int(strategy_parameters.get('volatility_lookback_window', 25))
        daily_price_diff = df_continuous['AdjustPrice'].diff()
        daily_price_diff_vol = daily_price_diff.rolling(window=volatility_lookback_window).std()
        daily_price_diff_vol.iloc[0] = 0 if numpy.isnan(daily_price_diff_vol.iloc[0]) else daily_price_diff_vol.iloc[0]
        daily_price_diff_vol = daily_price_diff_vol.ffill()
        df_continuous['PriceVol'] = daily_price_diff_vol * multiplier

    def generate_combined_data(self):
        symbol_list = [position.symbol for position in self.position_list]
        columns = self.df_continuous_with_forecast_dict[symbol_list[0]].columns
        for column in columns:
            df_combined = None
            for symbol in symbol_list:
                df = self.df_continuous_with_forecast_dict[symbol]
                df = df[[column]]
                if df_combined is None:
                    df_combined = df
                else:
                    df_combined = pd.merge(df_combined, df, left_index=True, right_index=True, how='inner')
                df_combined.columns = list(range(len(df_combined.columns)))
            df_combined.columns = symbol_list
            self.df_combined_data_dict[column] = df_combined
