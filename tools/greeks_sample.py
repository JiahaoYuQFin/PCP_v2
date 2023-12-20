# -*- coding: utf-8 -*-
# @Time    : 2021/5/26 13:55
# @Author  : Jinwen Wang
# @Email   : jw4013@columbia.edu
# @File    : greeks.py
# @Software: PyCharm

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr

#%% Greeks
def Delta(s, k, t, r, q, sigma, callput):
    """
    Calculate the delta of an option
    :param s: 标的资产价格
    :param k: 行权价
    :param t: 距离到期日时间（年化）
    :param r: 无风险利率
    :param q: 分红率
    :param sigma: 波动率
    :param callput: 认购认沽，'c' or 'p'
    :return: float
    """
    if sigma == 0:
        return 0
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    if callput == 'c':
        return np.exp(-q * t) * ndtr(d1)
    else:
        return np.exp(-q * t) * (ndtr(d1) - 1)


def Gamma(s, k, t, r, q, sigma):
    """
    Calculate the gamma of an option
    :param s: 标的资产价格
    :param k: 行权价
    :param t: 距离到期日时间（年化）
    :param r: 无风险利率
    :param q: 分红率
    :param sigma: 波动率
    :return: float
    """
    if sigma == 0:
        return 0
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return norm._pdf(d1) * np.exp(-q * t) / (s * sigma * np.sqrt(t))


def Theta(s, k, t, r, q, sigma, callput):
    """
    Calculate the theta of an option
    :param s: 标的资产价格
    :param k: 行权价
    :param t: 距离到期日时间（年化）
    :param r: 无风险利率
    :param q: 分红率
    :param sigma: 波动率
    :param callput: 认购认沽，'c' or 'p'
    :return: float
    """
    if sigma == 0:
        return 0
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if callput == 'c':
        return -s * norm._pdf(d1) * sigma * np.exp(-q * t) / 2 / np.sqrt(t) + q * s * ndtr(d1) * np.exp(-q * t) -\
               r * k * np.exp(-r * t) * ndtr(d2)
    else:
        return -s * norm._pdf(d1) * sigma * np.exp(-q * t) / 2 / np.sqrt(t) - q * s * ndtr(-d1) * np.exp(-q * t) +\
               r * k * np.exp(-r * t) * ndtr(-d2)


def Vega(s, k, t, r, q, sigma):
    """
    Calculate the vega of an option
    :param s: 标的资产价格
    :param k: 行权价
    :param t: 距离到期日时间（年化）
    :param r: 无风险利率
    :param q: 分红率
    :param sigma: 波动率
    :return: float
    """
    if sigma == 0:
        return 0
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return s * np.sqrt(t) * norm._pdf(d1) * np.exp(-q * t)


def Rho(s, k, t, r, q, sigma, callput):
    """
    Calculate the rho of an option
    :param s: 标的资产价格
    :param k: 行权价
    :param t: 距离到期日时间（年化）
    :param r: 无风险利率
    :param q: 分红率
    :param sigma: 波动率
    :param callput: 认购认沽，'c' or 'p'
    :return: float
    """
    if sigma == 0:
        return 0
    d2 = (np.log(s / k) + (r - q - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    if callput == 'c':
        return k * t * np.exp(-r * t) * ndtr(d2)
    else:
        return -k * t * np.exp(-r * t) * ndtr(-d2)