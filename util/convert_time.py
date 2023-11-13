# -*- coding: utf-8 -*-
"""
Created on 2023/11/7 14:03
@author: jhyu
"""


def int_to_seconds(timestamp: int):
    # Assume the timestamp format is HHMMSSmmm
    hours = timestamp // 10000000
    minutes = (timestamp // 100000) % 100
    seconds = (timestamp // 1000) % 100
    milliseconds = timestamp % 1000

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds
