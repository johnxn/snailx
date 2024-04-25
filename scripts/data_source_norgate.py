# coding=utf-8
from data_source import DataSource
import norgatedata

padding_settings = norgatedata.PaddingType.NONE

contract_letters = "FGHJKMNQUVXZ"
contract_months = range(1, 13)

letter_to_month = dict([(contract_letters[i], contract_months[i]) for i in range(12)])
month_to_letter = dict([(contract_months[i], contract_letters[i]) for i in range(12)])


def get_symbol_and_contract_date_from_norgate_name(norgate_name):
    symbol, year_month = norgate_name.split('-')
    year = year_month[0:4]
    letter = year_month[4:]
    month = letter_to_month[letter]
    contract_date = f"{year}{month:02d}"
    return symbol, contract_date


def get_norgate_name_from_symbol_and_contract_date(symbol, contract_date):
    year = int(contract_date[0:4])
    month = int(contract_date[4:6])
    norgate_name = f'{symbol}-{year}{month_to_letter[month]}'
    return norgate_name


class DataSourceNorgate(DataSource):
    def __init__(self):
        super().__init__()

    def download_all_single_contracts(self, symbol_list):
        df_single_contract_dict = {}
        for symbol in symbol_list:
            contract_date_list = self.get_all_contract_dates(symbol)
            for contract_date in contract_date_list:
                df = self.download_single_contract(symbol, contract_date)
                df_single_contract_dict[(symbol, contract_date)] = df
        return df_single_contract_dict

    def download_single_contract(self, symbol, contract_date):
        norgate_name = get_norgate_name_from_symbol_and_contract_date(symbol, contract_date)
        df = norgatedata.price_timeseries(
            norgate_name,
            timeseriesformat='pandas-dataframe'
        )
        return df

    def get_all_contract_dates(self, symbol):
        contract_date_list = []
        contracts = norgatedata.futures_market_session_contracts(symbol)
        for contract_name in contracts:
            symbol, contract_date = get_symbol_and_contract_date_from_norgate_name(contract_name)
            contract_date_list.append(contract_date)
        return contract_date_list

    def get_active_contract_dates(self, symbol):
        # fixme
        return self.get_all_contract_dates(symbol)


if __name__ == "__main__":
    data_source = DataSourceNorgate()
    contract_date_list = data_source.get_active_contract_dates('ES')
    print(contract_date_list)
    print(data_source.download_single_contract('ES', contract_date_list[0]))
