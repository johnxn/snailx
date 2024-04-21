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


