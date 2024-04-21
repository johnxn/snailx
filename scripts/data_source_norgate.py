# coding=utf-8
import os
import util
import pandas as pd
from data_source import DataSource


class DataSourceNorgate(DataSource):
    def __init__(self):
        super().__init__()

    def download_single_contract(self, symbol, contract_date):
        raise NotImplementedError

    def download_all_single_contracts(self, symbol_list):
        raise NotImplementedError
