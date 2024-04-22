# coding=utf-8
import os
import util

class DataSource(object):
    def __init__(self):
        self.temp_data_dir = os.path.join(util.get_project_dir(), 'data/temp')
        if not os.path.exists(self.temp_data_dir):
            os.mkdir(self.temp_data_dir)

    def download_single_contract(self, symbol, contract_date):
        raise NotImplementedError

    def download_all_single_contracts(self, symbol_list):
        raise NotImplementedError
    
    def get_active_contract_dates(self, symbol):
        raise NotImplementedError
