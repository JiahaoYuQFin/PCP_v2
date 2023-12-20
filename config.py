# -*- coding: utf-8 -*-
"""
Created on 2023/11/20 9:34
@author: jhyu
"""
from pathlib import Path


start_dt = '20231115'   # input('Start date (YYYYMMDD): ')
end_dt = '20231122' # input('End date (YYYYMMDD): ')
und_code = '510050.SH'  # input('Underlying code (XXXXXX.SH): ')

# Remote path of tick data
tickdata_dir = Path(r'Z://tick/stock')
# local root path of data
localdata_dir = Path(__file__).parent / 'data'
# local path of result
localres_dir = Path(__file__).parent / 'result'

# BSM parameters
bsm_params = {'rf': 0.015, 'q': 0.0157}
