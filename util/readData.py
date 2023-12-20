# -*- coding: utf-8 -*-
"""
Created on 2023/11/3 13:06
@author: jhyu
"""
from abc import ABC, abstractmethod
from pathlib import Path
import pymysql
from sqlalchemy import create_engine
import dask.dataframe as dd
import pandas as pd


class ReadABC(ABC):
    def __init__(
            self,
            path: str = r'Z://tick/stock/20231101/quote',
            host: str = 'wind.quantchina.pro',
            port: int = 33071,
            usr: str = 'quantchina',
            psw: str = 'zMxq7VNYJljTFIQ8',
            db_name: str = 'wind'
    ):
        self.path = Path(path)
        # self.engine = create_engine(f"mysql://{usr}:{psw}@{host}:{port}/{db_name}")
        self.conn = pymysql.Connect(host=host, port=port, user=usr, passwd=psw, db=db_name)

    @abstractmethod
    def database(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def remote(self, **kwargs):
        raise NotImplementedError

    def close_connect(self):
        self.conn.close()
        # self.engine.dispose()


class Read4PCP(ReadABC):
    def database(self, start_dt='20231001', end_dt='20231101', und='510050.SH') -> pd.DataFrame:
        print(f"{und}'s option info is loading...")

        start_dt1 = (pd.to_datetime(start_dt, format='%Y%m%d') - pd.DateOffset(days=10)).strftime('%Y%m%d')

        sql_txt = """
        SELECT tb1.TRADE_DT `date`, tb1.S_INFO_WINDCODE `code`, tb1.S_DQ_SETTLE `settlement`, 
        tb2.S_INFO_STRIKEPRICE `strike`, tb2.S_INFO_MATURITYDATE `maturity`, tb2.S_INFO_CALLPUT `cp`, 
        tb2.S_INFO_LPRICE `lprice`, tb3.S_DQ_CLOSE `close_spot`
        FROM wind.CHINAOPTIONEODPRICES tb1
        INNER JOIN wind.CHINAOPTIONDESCRIPTION tb2 on (tb1.S_INFO_WINDCODE = tb2.S_INFO_WINDCODE) 
        AND (LEFT(tb2.S_INFO_SCCODE, 6) = LEFT('{Und_code}', 6))
        INNER JOIN wind.CHINACLOSEDFUNDEODPRICE tb3 on (tb3.S_INFO_WINDCODE = '{Und_code}') 
        AND (tb3.TRADE_DT = tb1.TRADE_DT)
        WHERE tb1.TRADE_DT BETWEEN {Start_date} AND {End_date}
        ORDER BY tb1.TRADE_DT, tb1.S_INFO_WINDCODE
        """.format(Start_date=start_dt1, End_date=end_dt, Und_code=und)

        df = pd.read_sql(sql=sql_txt, con=self.conn)
        df['cp'] = (df['cp'] == 708001000) * 1 - (df['cp'] == 708002000)
        df['texp'] = (pd.to_datetime(df['maturity'], format='%Y%m%d') -
                      pd.to_datetime(df['date'], format='%Y%m%d')).dt.days
        df['pre_settlement'] = df.groupby(['code'])['settlement'].shift(1)
        df['pre_settlement'].fillna(df['lprice'], inplace=True)
        return df.query("@start_dt <= date <= @end_dt")

    def remote(self, codes: list or tuple) -> pd.DataFrame:
        print(f"{self.path.parts[-2]}: Spot and Option's ticks are loading...")

        # 1. read etf file, and its time index
        etf_file = list(self.path.glob('sh_510050_*'))[0]
        df_spot = pd.read_parquet(etf_file)
        df_spot = df_spot.query("93000000<= time <= 145700000")[
            ['time', 'date', 'prev_close', 'ask_prc1', 'bid_prc1', 'ask_vol1', 'bid_vol1']]
        df_spot = df_spot.drop_duplicates(subset=['time'], keep='last')
        df_spot = df_spot.rename(columns={'prev_close': 'preclose_spot'}).copy()
        df_spot['ts'] = pd.to_datetime(df_spot['time'], format='%H%M%S%f')
        spot_ts = df_spot['ts'].tolist()

        # 2. read options file, and ceil the last slice of 500ms tick to 3s tick
        filtered_files = [str(file) for file in self.path.iterdir() if (file.stem[3:11] + '.SH') in codes]
        # Use Dask to read and concatenate all dataframes in parallel
        df_opt_ori = dd.read_parquet(filtered_files).compute()
        df_opt_ori = df_opt_ori.rename(columns={'symbol': 'code'})[
            ['time', 'code', 'ask_prc1', 'bid_prc1', 'ask_vol1', 'bid_vol1']]
        df_opt_ori = df_opt_ori[df_opt_ori['time'] >= 1e7].copy()
        df_opt_ori['ts'] = pd.to_datetime(df_opt_ori['time'], format='%H%M%S%f').dt.ceil('1s')
        df_opt_ori = (df_opt_ori.groupby(['code', 'ts']).last()).reset_index().drop(columns=['time'])

        df_opt = pd.DataFrame(index=pd.MultiIndex.from_product([spot_ts, codes]))
        df_opt.index.names = ['ts', 'code']
        df_opt = (df_opt.join(df_opt_ori.set_index(['ts', 'code']), how='outer')).sort_index()
        df_opt = df_opt.groupby('code').ffill().loc[spot_ts].reset_index()

        # 3. merge option and spot
        df_merge = pd.merge(df_opt, df_spot, on='ts', suffixes=('_opt', '_spot'), how='left').drop(columns=['ts'])
        df_merge['date'] = df_merge['date'].astype('str')

        return df_merge


class ReadCommodity(ReadABC):
    def database(self, start_dt='20231001', end_dt='20231101', future_name='C') -> pd.DataFrame:
        print(f"{future_name}'s option info is loading...")

        return 0
