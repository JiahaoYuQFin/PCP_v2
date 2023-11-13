# -*- coding: utf-8 -*-
"""
Created on 2023/11/1 14:40
@author: jhyu
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Union
from abc import ABC, abstractmethod

from util.market_elements import MarketElement
from util.convert_time import int_to_seconds


class PCPABC(ABC):
    """
    This class tests the put-call parity (PCP) arbitrage opportunity.
    """
    def __init__(self, **kwargs):  # ,pre_spot=4255.30, fut_margin_rate=0.12, lend_rate=0.035, borrow_rate=0.035):
        """
        ETF期权
        经手费：1.3元/张
        交易结算费：0.3元/张
        行权结算费：0.6元/张
        佣金：5～15元/张
        卖出开仓暂免
        """
        self.rl = kwargs.get('lend_rate', 0.015)
        self.rb = kwargs.get('borrow_rate', 0.015)
        self.margin_multiplier = kwargs.get('margin_multiplier', 1.2)   # 开仓保证金预存安全系数
        self.ret_threshold = 0.00
        self.contract_month = None     # 去除当天到期的合约后，None代表全部，0代表最近月，1代表次近月，以此类推

        # general option
        self.opt_multiplier = 10000
        self.opt_commission = 10
        self.exercise_commission = 5

        # index option
        self.opt_margin_adj = 0.1
        self.opt_coverage_facotr = 0.5

        # spot
        self.spot_multiplier = 100
        self.spot_commission_rate = 1e-3
        self.spot_short_margin_rate = 0.75
        self.spot_short_intr = kwargs.get('spot_short_intr_rate', 0.106)

        # future
        self.fut_multiplier = kwargs.get('fut_multiplier', 300)
        self.fut_margin_rate = kwargs.get('fut_margin_rate', 0.12)
        self.fut_commission_rate = kwargs.get('fut_commission_rate', 23e-6)

    def convert_df_to_input(self, df_opt: pd.DataFrame, df_opt_info: pd.DataFrame) -> MarketElement:
        df_opt = (df_opt.merge(df_opt_info, on=['date', 'code'], how='left')).query("texp > 0")
        df_unstack = df_opt.set_index(['time', 'maturity', 'strike', 'cp']).unstack(level='cp').swaplevel(0, 1, axis=1)
        market_dic = {
            'time': df_unstack[1].reset_index()['time'].values,
            'texp': df_unstack[1]['texp'].values,
            'strike': df_unstack[1].reset_index()['strike'].values,
            'code': df_unstack[1]['code'].tolist(),
            'call_settlement': df_unstack[1]['pre_settlement'].values,
            'put_settlement': df_unstack[-1]['pre_settlement'].values,

            'spot_close': df_opt['preclose_spot'].tolist()[0],
            'spot_ap1': df_unstack[1]['ask_prc1_spot'].values,
            'spot_bp1': df_unstack[1]['bid_prc1_spot'].values,
            'spot_av1': df_unstack[1]['ask_vol1_spot'].values,
            'spot_bv1': df_unstack[1]['bid_vol1_spot'].values,

            'call_ap1': df_unstack[1]['ask_prc1_opt'].values,
            'call_bp1': df_unstack[1]['bid_prc1_opt'].values,
            'put_ap1': df_unstack[-1]['ask_prc1_opt'].values,
            'put_bp1': df_unstack[-1]['bid_prc1_opt'].values,
            'call_av1': df_unstack[1]['ask_vol1_opt'].values.astype('int32'),
            'call_bv1': df_unstack[1]['bid_vol1_opt'].values.astype('int32'),
            'put_av1': df_unstack[-1]['ask_vol1_opt'].values.astype('int32'),
            'put_bv1': df_unstack[-1]['bid_vol1_opt'].values.astype('int32')
        }
        return MarketElement(**market_dic)

    @abstractmethod
    def get_margin(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def arbitrage_ret(self, **kwargs):
        raise NotImplementedError


class ETFpcp(PCPABC):

    def get_margin(self, settlement_price, spot, strike, cp: int):
        """
        Get the opening margin when the input price was from yesterday's, otherwise the maintenance margin.
        ETF call margin = settlement price + max(0.12*spot close - 虚值额， 0.07*spot close)
        ETF put margin = min(settlement price + max(0.12*spot close - 虚值额， 0.07*strike), strike)

        Args:
            settlement_price: yesterday's / today's settlement price of options.
            spot: yesterday's / today's closing price of spot.
            strike: strike price.
            cp: option type, 1 means call, -1 means put.
        Returns:
            margin
        """
        a = 0.12
        b = 0.07
        if np.isscalar(cp):
            if cp == 1:
                otm = np.maximum(strike - spot, 0)
                margin = settlement_price + np.maximum(a * spot - otm, b * spot)
            elif cp == -1:
                otm = np.maximum(spot - strike, 0)
                margin = np.minimum(settlement_price + np.maximum(a * spot - otm, b * strike), strike)
            else:
                raise ValueError
        else:
            otm = (cp == 1) * np.maximum(strike - spot, 0) + (cp == -1) * np.maximum(spot - strike, 0)
            margin = (
                    (cp == 1) * settlement_price + np.maximum(a * spot - otm, b * spot) +
                    (cp == -1) * np.minimum(settlement_price + np.maximum(a * spot - otm, b * strike), strike)
            )

        return margin * self.margin_multiplier

    def arbitrage_ret(
            self,
            time_vals,
            call_price,
            put_price,
            volume_call: Union[np.ndarray, float],
            volume_put,
            volume_spot,
            margin,
            spot,
            strike,
            texp: Union[np.ndarray, int, float],
            direction: int
    ):
        """
        Forward arbitrage, means +S - (C-P), say short Call and long Put. The formula is as followed,
            1) At initial date t0,
            NCFt0 = - S_ap1 * (1 + fee) + Call_bp1 - Put_ap1 - Call_margin - 2 * OptionCost
            2) At exercise date t1,
            NCFt1 = - exercise fee
            3) At releasing margin date t2, the day after delivery date,
            NCFt2 = strike + Call_margin

        Backward arbitrage, means -S + (C-P), say long Call and short Put. The formula is as followed,
            1) At initial date t0,
            NCFt0 = - S_bp1 * S_margin - Call_ap1 + Put_bp1 - Put_margin - 2 * OptionCost
            2) At exercise date t1,
            NCFt1 = - strike - exercise fee
            3) At releasing margin date t2=t1+2,
            NCFt2 = Put_margin
            4) At releasing credit account date t3=t1+4,
            NCFt3 = S_margin * S_bp1 + S_bp1 * (1 - fee - r*(t1+3-t0)/360)

        Args:
            call_price:
            put_price:
            volume_call:
            volume_put:
            volume_spot:
            margin: option's open margin.
            spot:
            strike:
            texp: calendar days to expiry.
            direction: 1 means forward arbitrage, otherwise backward arbitrage.

        Returns:
            arbitrage ret, expected profit
        """
        max_lot = np.minimum(volume_call, volume_put, volume_spot*self.spot_multiplier//self.opt_multiplier)
        num = max_lot
        if direction == 1:
            cf0 = (
                self.opt_multiplier *
                (- spot * (1 + self.spot_commission_rate) + call_price - put_price - margin) -
                2 * self.opt_commission
            )
            cf1 = - self.exercise_commission
            cf2 = self.opt_multiplier * (strike + margin)
            expt_profit = cf0 + cf1 + cf2
            occupied_cash = (cf0 * (texp + 2) + cf1 * 2) / 360
        elif direction == -1:
            cf0 = (
                    self.opt_multiplier *
                    (- spot * self.spot_short_margin_rate - call_price + put_price - margin) -
                    2 * self.opt_commission
            )
            cf1 = - (strike * self.opt_multiplier + self.exercise_commission)
            cf2 = self.opt_multiplier * margin
            cf3 = (
                    self.opt_multiplier *
                    (self.spot_short_margin_rate * spot +
                     spot * (1 - self.spot_commission_rate - self.spot_short_intr * (texp + 3) / 360))
            )
            expt_profit = cf0 + cf1 + cf2 + cf3
            occupied_cash = (cf0 * (texp + 6) + cf1 * 6 + cf2 * 4) / 360
        else:
            raise ValueError(f"Invalid direction '{direction}'. Expected one of [1, -1].")

        r = (- expt_profit / occupied_cash) * (num > 0)
        df = pd.DataFrame(np.c_[time_vals, expt_profit * (r >= self.ret_threshold) * num, -occupied_cash * num, texp],
                          columns=['time', 'profit', 'cash_used', 'texp'])
        if self.contract_month is None:
            df2 = df.groupby(['time', 'texp']).apply(
                lambda x: x['profit'].sum() / x['cash_used'].sum() if x['cash_used'].sum() != 0 else 0).to_frame(
                name='ret')
        else:
            target_texp = np.sort(np.unique(texp))[self.contract_month]
            df = df.query("texp==@target_texp")
            df2 = df.groupby('time').apply(
                lambda x: x['profit'].sum() / x['cash_used'].sum() if x['cash_used'].sum() != 0 else 0).to_frame(
                name='ret')
        return r, df2

    def get_indicator(self, ret, time_vals=None, code_vals=None):
        if (ret['ret'].values > 0).sum() <= 0:
            res = pd.DataFrame(0, columns=['avg_ret', 'positive_ret_ratio', 'avg_dur'], index=ret['texp'].unique())
            res.index.name = 'texp'
            return res
        df = ret.copy()
        df['time'] = int_to_seconds(df['time'].values)
        df['sign'] = np.sign(df['ret'])

        if self.contract_month is None:
            def calculate_blocks(group):
                group['block'] = (group['sign'] != group['sign'].shift())
                group['block'] = group['block'].cumsum()
                return group
            df = df.groupby('texp', group_keys=False).apply(calculate_blocks)

            # 计算每个连续区间的时间差
            time_diffs = df[df['sign'] == 1].groupby(['texp', 'block'])['time'].apply(lambda x: x.max() - x.min())
            # 计算平均持续时间
            average_durations = time_diffs.groupby(level=0).mean()
            average_durations.name = 'avg_dur'

            freq_occur = df.groupby(['texp'])['ret'].agg(avg_ret=lambda x: x[x > 0].mean(),
                                                         positive_ret_ratio=lambda x: (x > 0).mean())
            res = freq_occur.merge(average_durations, left_index=True, right_index=True)
        else:
            df['block'] = (df['sign'] != df['sign'].shift()).cumsum()
            time_diffs = df[df['sign'] == 1].groupby(['block'])['time'].apply(lambda x: x.max() - x.min())
            average_durations = time_diffs.mean()

            freq_occur = (df['sign'] > 0).mean()
            avg_ret = df.loc[df['sign'] > 0, 'ret'].mean()
            res = pd.DataFrame([df['texp'].tolist()[0], avg_ret, freq_occur, average_durations],
                               columns=['texp', 'avg_ret', 'positive_ret_ratio', 'avg_dur'])
            res.set_index('texp', inplace=True)
        # else:
        #     if (ret > 0).sum() <= 0:
        #         return 0, 0
        #
        #     time_vals = int_to_seconds(time_vals)
        #     df = pd.DataFrame(np.c_[time_vals, ret], columns=['time', 'ret'])
        #     df.insert(1, column='code', value=code_vals)
        #
        #     df['sign'] = np.sign(df['ret'])
        #     freq_occur = (df['sign'] > 0).sum() / df.shape[0]
        #
        #
        #
        #     ddf = dd.from_pandas(df, npartitions=5)
        return res

