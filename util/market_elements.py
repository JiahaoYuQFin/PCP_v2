# -*- coding: utf-8 -*-
"""
Created on 2023/11/7 9:24
@author: jhyu
"""
from abc import ABC


class MarketElement(ABC):
    def __init__(self, **kwargs):
        self.time = kwargs.get('time', None)
        self.code = kwargs.get('code', None)

        self.spot_close = kwargs.get('spot_close', None)
        self.spot_ap1 = kwargs.get('spot_ap1', None)
        self.spot_bp1 = kwargs.get('spot_bp1', None)
        self.spot_av1 = kwargs.get('spot_av1', None)
        self.spot_bv1 = kwargs.get('spot_bv1', None)

        self.call_settlement = kwargs.get('call_settlement', None)
        self.put_settlement = kwargs.get('put_settlement', None)
        self.strike = kwargs.get('strike', None)
        self.texp = kwargs.get('texp', None)
        self.maturity = kwargs.get('maturity', None)

        self.call_ap1 = kwargs.get('call_ap1', None)
        self.call_bp1 = kwargs.get('call_bp1', None)
        self.put_ap1 = kwargs.get('put_ap1', None)
        self.put_bp1 = kwargs.get('put_bp1', None)

        self.call_av1 = kwargs.get('call_av1', None)
        self.call_bv1 = kwargs.get('call_bv1', None)
        self.put_av1 = kwargs.get('put_av1', None)
        self.put_bv1 = kwargs.get('put_bv1', None)
