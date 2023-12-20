# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 14:16
# @Author  : Jinwen Wang
# @Email   : jw4013@columbia.edu
# @File    : future.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import datetime
import os


def get_mid_prc(pre_settle_prc, settle_prc, bid, ask, vol):
    """
    计算 mid price
    :param pre_settle_prc: 前结算价
    :param settle_prc: 成交价
    :param bid: 买价
    :param ask: 卖价
    :param vol: 成交量
    :return: 中间价，float
    """
    if vol:
        # 该分钟有成交
        if bid and ask:
            # 存在买卖报价
            return settle_prc if bid <= settle_prc <= ask else (bid + ask) / 2  # 最新成交价处于买卖报价之间，取最新成交价，否则用中间价
        elif bid:
            # 仅存在买价
            return max(settle_prc, bid)  # 成交价和买价中的最大值
        elif ask:
            # 仅存在卖价
            return min(settle_prc, ask)  # 成交价和卖价的最小值
        else:
            # 不存在买价和卖价
            return settle_prc  # 返回成交价
    else:
        # 该分钟没有成交
        if bid and ask:
            # 存在买卖报价
            return (bid + ask) / 2  # 取中间价
        elif bid:
            # 仅存在买价
            return max(pre_settle_prc, bid)  # 前结算价和买价的最大值
        elif ask:
            # 仅存在卖价
            return min(pre_settle_prc, ask)  # 前结算价和卖价的最小值
        else:
            # 不存在买卖报价
            return pre_settle_prc  # 返回前结算价


def get_synthetic_futures(code, date, rf, df_contract, input_stock_path, input_future_path, output_path):
    """
    计算合成期货价格
    :param code: 标的资产代码，'510050' or '510300' or '000300'
    :param date: 处理当天的日期，格式为 '20210521'
    :param rf: 年化的无风险利率
    :param df_contract: 包含每日期权信息的 Dataframe
    :param input_stock_path: 输入的minbar股票数据路径
    :param input_future_path: minbar期货数据路径
    :param output_path: 输出结果的路径
    :return: 包含当天分钟级合成期货、升贴水、分红率的Dataframe
    """
    print('Calculating futures data of %s' % date)
    daily_options = df_contract[df_contract['日期'] == date].reset_index(drop=True)
    stock_minbar_path = os.path.join(input_stock_path, date, '1min')  # 当天stock minbar路径
    future_minbar_path = os.path.join(input_future_path, date, '1min')  # 当天future minbar路径
    stockpath = os.path.join(stock_minbar_path, 'sh_%s_%s_1min.parquet' % (code, date))  # 标的 minbar路径
    if os.path.exists(stockpath):
        df_stock = pd.read_parquet(stockpath)  # 标的 minbar数据
        """剔除掉集合竞价多余的分钟数据"""
        df_stock = df_stock.loc[df_stock['datetime'].apply(lambda x:
                                                           x[-8:] not in ['14:57:00', '14:58:00', '14:59:00'])
                                ].reset_index(drop=True)
    else:
        print("========== No %s file in %s ==========" % (code, date))
        return
    df_futures = pd.DataFrame(df_stock['datetime'])  # 储存合成期货数据

    strike_prc_list = np.sort(daily_options['行权价'].unique())  # 包含当天期权行权价的array
    for i in range(len(df_stock)):
        # 确定围绕标的价格的3个行权价
        stockprice = df_stock.at[i, 'open']  # 当前分钟的标的价格，用open的价格
        if stockprice >= 0:
            closest_index = np.argmin(np.abs(strike_prc_list - stockprice))  # 最接近标的价格的行权价
            if stockprice < strike_prc_list[0]:  # 标的价格小于最低行权价
                k1 = strike_prc_list[0]
                k2 = k1
                k3 = k1
            elif stockprice > strike_prc_list[-1]:  # 标的价格大于最高行权价
                k1 = strike_prc_list[-1]
                k2 = k1
                k3 = k1
            elif closest_index == 0:  # 最接近标的价格的是最低的行权价，且标的价格高于最低的行权价
                # 取最低和次低的行权价
                k1 = strike_prc_list[closest_index]
                k2 = strike_prc_list[closest_index + 1]
                k3 = k2
            elif closest_index == len(strike_prc_list) - 1:  # 最接近标的价格的是最高的行权价，且标的价格低于最高的行权价
                # 取最高和次高的行权价
                k1 = strike_prc_list[closest_index]
                k2 = strike_prc_list[closest_index - 1]
                k3 = k2
            else:
                k1 = strike_prc_list[closest_index]
                k2 = strike_prc_list[closest_index + 1]
                k3 = strike_prc_list[closest_index - 1]
            k_list = list({k1, k2, k3})

            if code[0:3] == '510':
                monthlist = np.sort(pd.Series([x[7: 11] for x in daily_options['交易代码']]).unique())  # 当天存续合约的四个到期月份
            else:
                monthlist = np.sort(pd.Series([x[2: 6] for x in daily_options['交易代码']]).unique())
            min_price = pd.DataFrame(index=k_list, columns=monthlist)  # 储存一分钟的不同到期日、不同行权价计算出的合成期货价格

            maturities = {}  # 储存当前分钟距离四个到期日的时间（年化）

            for k in k_list:
                for month in min_price.columns:
                    if code[0:3] == '510':
                        tradecode_call = code + 'C%sM' % month + '0%i' % int(k * 1000)  # 看涨期权交易代码
                        tradecode_put = code + 'P' + tradecode_call[7:]  # 看跌期权交易代码
                    else:
                        tradecode_call = 'IO%s-C-%i' % (month, k)
                        tradecode_put = 'IO%s-P-%i' % (month, k)
                    call_index = (daily_options['交易代码'] == tradecode_call).argmax()
                    put_index = (daily_options['交易代码'] == tradecode_put).argmax()
                    if 'T_%s' % month not in maturities.keys():
                        expire_time = str(daily_options.at[call_index, '到期日'])[0:10] + ' 15:00:00'
                        expire_time = datetime.datetime.strptime(expire_time, '%Y-%m-%d %H:%M:%S')  # 期权到期时间
                        now_time = datetime.datetime.strptime(df_stock.at[i, 'datetime'], '%Y-%m-%d %H:%M:%S')
                        T = (expire_time - now_time).total_seconds() / 60 / (365 * 24 * 60)  # 距离到期的时间（年化）
                        maturities['T_%s' % month] = T
                    if tradecode_call in daily_options['交易代码'].values:
                        if code[0:3] == '510':
                            callcode = daily_options.at[call_index, '期权代码']  # 看涨期权代码
                            putcode = daily_options.at[put_index, '期权代码']  # 看跌期权代码
                            callpath = os.path.join(stock_minbar_path,
                                                    'sh_%i_%s_1min.parquet' % (callcode, date))  # 看涨期权合约minbar路径
                            putpath = os.path.join(stock_minbar_path,
                                                   'sh_%i_%s_1min.parquet' % (putcode, date))  # 看跌期权合约minbar路径
                        else:
                            callpath = os.path.join(future_minbar_path,
                                                    'cfe_%s_%s_1min.parquet' % (tradecode_call.lower(), date))
                            putpath = os.path.join(future_minbar_path,
                                                   'cfe_%s_%s_1min.parquet' % (tradecode_put.lower(), date))
                        df_call = pd.read_parquet(callpath)
                        df_put = pd.read_parquet(putpath)

                        """############# 处理ask和bid的异常值 #############"""
                        if code[0:3] == '510':
                            df_call['first_ask_prc1'] = df_call.apply(lambda x: 0 if ((x['first_ask_vol1'] == 0 and
                                                                                      (x['first_ask_vol2'] +
                                                                                       x['first_ask_vol3'] +
                                                                                       x['first_ask_vol4'] +
                                                                                       x['first_ask_vol5']) != 0) or
                                                                                      (x['first_ask_vol1'] +
                                                                                       x['first_ask_vol2'] +
                                                                                       x['first_ask_vol3'] +
                                                                                       x['first_ask_vol4'] +
                                                                                       x['first_ask_vol5'] == 0)
                                                                                      ) else x['first_ask_prc1'], axis=1)
                            df_call['first_ask_prc1'] = df_call['first_ask_prc1'].fillna(0)

                            df_call['first_bid_prc1'] = df_call.apply(lambda x: 0 if ((x['first_bid_vol1'] == 0 and
                                                                                      (x['first_bid_vol2'] +
                                                                                       x['first_bid_vol3'] +
                                                                                       x['first_bid_vol4'] +
                                                                                       x['first_bid_vol5']) != 0) or
                                                                                      (x['first_bid_vol1'] +
                                                                                       x['first_bid_vol2'] +
                                                                                       x['first_bid_vol3'] +
                                                                                       x['first_bid_vol4'] +
                                                                                       x['first_bid_vol5'] == 0)
                                                                                      ) else x['first_bid_prc1'], axis=1)
                            df_call['first_bid_prc1'] = df_call['first_bid_prc1'].fillna(0)

                            df_put['first_ask_prc1'] = df_put.apply(lambda x: 0 if ((x['first_ask_vol1'] == 0 and
                                                                                    (x['first_ask_vol2'] +
                                                                                     x['first_ask_vol3'] +
                                                                                     x['first_ask_vol4'] +
                                                                                     x['first_ask_vol5']) != 0) or
                                                                                    (x['first_ask_vol1'] +
                                                                                     x['first_ask_vol2'] +
                                                                                     x['first_ask_vol3'] +
                                                                                     x['first_ask_vol4'] +
                                                                                     x['first_ask_vol5'] == 0)
                                                                                    ) else x['first_ask_prc1'], axis=1)
                            df_put['first_ask_prc1'] = df_put['first_ask_prc1'].fillna(0)

                            df_put['first_bid_prc1'] = df_put.apply(lambda x: 0 if ((x['first_bid_vol1'] == 0 and
                                                                                    (x['first_bid_vol2'] +
                                                                                     x['first_bid_vol3'] +
                                                                                     x['first_bid_vol4'] +
                                                                                     x['first_bid_vol5']) != 0) or
                                                                                    (x['first_bid_vol1'] +
                                                                                     x['first_bid_vol2'] +
                                                                                     x['first_bid_vol3'] +
                                                                                     x['first_bid_vol4'] +
                                                                                     x['first_bid_vol5'] == 0)
                                                                                    ) else x['first_bid_prc1'], axis=1)
                            df_put['first_bid_prc1'] = df_put['first_bid_prc1'].fillna(0)
                        """############# 处理ask和bid的异常值 #############"""

                        # 如果是中信的数据，根据合约单位对price进行调整
                        minbar_close_call = df_call['close'].max()
                        real_close_call = daily_options.at[call_index, '收盘价']
                        if minbar_close_call / real_close_call > 1000:
                            unit = daily_options.at[call_index, '合约单位']
                            for price_name in ['close', 'open', 'first_ask_prc1', 'first_bid_prc1']:
                                df_call[price_name] /= unit

                        minbar_close_put = df_put['close'].max()
                        real_close_put = daily_options.at[put_index, '收盘价']
                        if minbar_close_put / real_close_put > 1000:
                            unit = daily_options.at[put_index, '合约单位']
                            for price_name in ['close', 'open', 'first_ask_prc1', 'first_bid_prc1']:
                                df_put[price_name] /= unit

                        df_call['pre_close_prc'] = df_call['close'].shift(1).fillna(method='bfill')
                        df_put['pre_close_prc'] = df_put['close'].shift(1).fillna(method='bfill')

                        C = get_mid_prc(df_call.at[i, 'pre_close_prc'],
                                        df_call.at[i, 'open'],
                                        df_call.at[i, 'first_bid_prc1'],
                                        df_call.at[i, 'first_ask_prc1'],
                                        df_call.at[i, 'volume'])
                        P = get_mid_prc(df_put.at[i, 'pre_close_prc'],
                                        df_put.at[i, 'open'],
                                        df_put.at[i, 'first_bid_prc1'],
                                        df_put.at[i, 'first_ask_prc1'],
                                        df_put.at[i, 'volume'])

                        t = maturities['T_%s' % month]
                        q = -np.log((C + k * np.exp(-rf * t) - P) / stockprice) / t  # 分红率
                        min_price.at[k, month] = stockprice * np.exp((rf - q) * t)
            # 取三个行权价合成期货价格的均值填入sys_future
            for month in min_price.columns:
                df_futures.at[i, month] = min_price[month].mean()
                df_futures.at[i, 'premium_%s' % month] = df_futures.at[i, month] - stockprice
                t = maturities['T_%s' % month]
                df_futures.at[i, 'q_%s' % month] = rf - np.log(df_futures.at[i, month] / stockprice) / t

    # 导出当天的合成期货数据
    outfile = os.path.join(output_path, '%s.csv' % date)
    df_futures.to_csv(outfile)
    return df_futures
