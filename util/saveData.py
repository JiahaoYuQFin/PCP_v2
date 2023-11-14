# -*- coding: utf-8 -*-
"""
Created on 2023/11/8 16:05
@author: jhyu
"""
from abc import ABC
import pandas as pd
from pathlib import Path


class SaveABC(ABC):
    def __init__(
            self,
            dir_path: str = '',
            folder_name: str = ''
    ):
        if dir_path == '':
            dir_path = Path(__file__).parent.parent / 'data'
        else:
            dir_path = Path(dir_path)

        if folder_name == '':
            self.path = dir_path
        else:
            self.path = dir_path / folder_name

        if not self.path.exists():
            self.path.mkdir()

    def toCsv(self, **kwargs):
        raise NotImplementedError

    def toPkl(self, **kwargs):
        raise NotImplementedError


class SaveRes(SaveABC):
    def np2Csv(self, arr, col_name, filename: str = 'res.csv'):
        df = pd.DataFrame(arr).T
        df.columns = col_name
        df.to_csv(self.path / filename, index=False)
        # print(f"{filename} has been saved.")

    def df2Exc(self, df: pd.DataFrame, filename: str = 'res.xlsx'):
        df.to_excel(self.path / filename, index=False)

    def df2Csv(self, df: pd.DataFrame, filename: str = 'res.csv'):
        df.to_csv(self.path / filename, index=False)

    def df2Pkl(self, df: pd.DataFrame, filename: str = 'res.csv'):
        df.to_pickle(self.path / filename)
