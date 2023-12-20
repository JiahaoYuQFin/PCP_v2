# -*- coding: utf-8 -*-
"""
Created on 2023/11/23 15:19
@author: jhyu
"""
import pandas as pd
import numpy as np


def synthesize_close2future(group: pd.DataFrame, rf):
    group2 = group.set_index(['cp', 'strike']).unstack(level='cp').swaplevel(0, 1, axis=1).reset_index()
    #group.set_index(['cp']).unstack(level='cp').swaplevel(0, 1, axis=1).reset_index(drop=True).copy()
    spot = group['close_spot'].tolist()[0]
    tau = group['texp'].tolist()[0]/365

    idx_kmin = group2['strike'].idxmin()
    idx_kmax = group2['strike'].idxmax()
    idx_atm = (spot - group2['strike']).abs().argmin()
    if spot < group2.loc[idx_kmin, 'strike'].to_list()[0]:
        k = np.ones(3) * group2.loc[idx_kmin, 'strike'].to_list()[0]
    elif spot > group2.loc[idx_kmax, 'strike'].to_list()[0]:
        k = np.ones(3) * group2.loc[idx_kmax, 'strike'].to_list()[0]
    elif idx_atm == idx_kmin:
        k = np.ones(3) * group2.loc[idx_kmin + 1, 'strike'].to_list()[0]
        k[0] = group2.loc[idx_atm, 'strike'].to_list()[0]
    elif idx_atm == idx_kmax:
        k = np.ones(3) * group2.loc[idx_kmax - 1, 'strike'].to_list()[0]
        k[0] = group2.loc[idx_atm, 'strike'].to_list()[0]
    else:
        k = group2.loc[idx_atm - 1: idx_atm+1, 'strike'].values

    c = group2.set_index(['strike']).loc[k, (1, 'settlement')].values
    p = group2.set_index(['strike']).loc[k, (-1, 'settlement')].values
    f = (c - p) * np.exp(rf*tau) + k
    group['fwd'] = f.mean().round(3)
    return group
