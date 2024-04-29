# coding=utf-8
import util
import os
import pandas as pd
from data_broker import DataBroker
from futu_api.futu_gateway import FutuGateway


class FutuDataBroker(DataBroker):
    def __init__(self):
        super(FutuDataBroker, self).__init__()
        futu_config_file_path = os.path.join(util.get_project_dir(), 'config/futu_config.csv')
        df_futu_config = pd.read_csv(futu_config_file_path)
        settings = util.df_config_to_dict(df_futu_config)
        settings['port'] = int(settings['port'])
        self.gate_way = FutuGateway()
        self.gate_way.connect(settings)

    def get_account_value(self):
        total_value = 0
        account_list = self.gate_way.query_account_new()
        for account in account_list:
            total_value += account.total_value
        return total_value

    def destroy(self):
        super(FutuDataBroker, self).destroy()
        self.gate_way.close()
