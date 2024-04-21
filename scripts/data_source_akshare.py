# coding=utf-8
import akshare as ak
import os
import util
import pandas as pd
from data_source import DataSource

china_market_list = ["CFFEX", "INE", "CZCE", "DCE", "SHFE", "GFEX"]


class DataSourceAkshare(DataSource):
    def __init__(self):
        super().__init__()
        self.temp_data_dir = os.path.join(util.get_project_dir(), 'data/temp')
        if not os.path.exists(self.temp_data_dir):
            os.mkdir(self.temp_data_dir)

    def download_single_contract(self, symbol, contract_date):
        # 把 202409 转成 2409
        df = ak.futures_zh_daily_sina(symbol=f"{symbol}{str(contract_date)[2:]}")
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df

    def download_all_single_contracts(self, symbol_list):
        self.download_raw_contract_data()
        df_single_contract_dict = self.build_all_single_contracts_from_raw_data(symbol_list)
        return df_single_contract_dict

    def download_raw_contract_data(self):
        year_list = range(2010, 2025)
        for market in china_market_list:
            for year in year_list:
                start_date = f"{year}0101"
                end_date = f"{year + 1}0101"
                print(f"getting {market}, {start_date}, {end_date}")
                save_path = os.path.join(self.temp_data_dir, f"{market}_{start_date}_{end_date}.csv")
                if not os.path.exists(save_path):
                    try:
                        daily_df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=market)
                        daily_df.to_csv(save_path, index=False)
                        import time
                        time.sleep(0.1)
                    except Exception as es:
                        print(f"failed to get {market}, {start_date}, {end_date}, {str(es)}")

    def build_all_single_contracts_from_raw_data(self, symbol_list):
        df_single_contract_dict = {}
        for market in china_market_list:
            df_list = []
            for filename in os.listdir(self.temp_data_dir):
                if filename.startswith(market):
                    filepath = os.path.join(self.temp_data_dir, filename)
                    df = pd.read_csv(filepath)
                    df_list.append(df)
            df_all = pd.concat(df_list)
            contracts = set(df_all['symbol'])
            for contract in contracts:
                df = df_all[df_all['symbol'] == contract]
                df = df.drop(columns='symbol')
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date')
                symbol, contract_date = contract[:-4], '20' + str(contract[-4:])
                if symbol not in symbol_list:
                    continue
                df_single_contract_dict[(symbol, contract_date)] = df
        return df_single_contract_dict
