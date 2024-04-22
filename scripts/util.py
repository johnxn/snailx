# coding=utf-8
import pandas as pd
import os


def get_project_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.dirname(current_dir))


def datetime_to_str(dt):
    return dt.strftime("%Y-%m-%d")


def str_to_datetime(str):
    return pd.to_datetime(str)


def generate_portfolio_config(portfolio_config_file_path):
    if os.path.exists(portfolio_config_file_path):
        return
    data = {
        'Symbol': ['TA', 'M', 'RB'],
        'Name': ['PTA', '豆粕', '螺纹钢'],
        'CarryContract': ['202405', '202405', '202405'],
        'CurrentContract': ['202409', '202409', '202409'],
    }
    df = pd.DataFrame(data)
    df.to_excel(portfolio_config_file_path, index=False)


