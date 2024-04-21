# coding=utf-8

class DataSource(object):
    def __init__(self):
        pass

    def download_single_contract(self, symbol, contract_date):
        raise NotImplementedError

    def download_all_single_contracts(self, symbol_list):
        raise NotImplementedError
