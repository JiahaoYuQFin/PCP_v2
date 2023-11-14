# -*- coding: utf-8 -*-
"""
Created on 2023/11/6 13:46
@author: jhyu
"""
from pcp_etf import *
from util.readData import *
from util.saveData import SaveRes
from joblib import Parallel, delayed


def run(file: Path):
    data_repo = Read4PCP(path=str(file))
    today = file.parts[-2]

    # df_opt_info = data_repo.database(start_dt=today, end_dt=today, und='510050.SH')
    df_opt_info = pd.read_pickle(data_dir / 'opt_info.pkl').query("date==@today")
    opt_codes = df_opt_info['code'].tolist()
    df_opt = data_repo.remote(codes=opt_codes)

    model = ETFpcp()
    mslice = model.convert_df_to_input(df_opt=df_opt, df_opt_info=df_opt_info)
    cmargin = model.get_margin(mslice.call_settlement, mslice.spot_close, mslice.strike, 1)
    pmargin = model.get_margin(mslice.put_settlement, mslice.spot_close, mslice.strike, -1)

    fwd_ret, wr1 = model.arbitrage_ret(
        mslice.time, mslice.call_bp1, mslice.put_ap1, mslice.call_bv1, mslice.put_av1, mslice.spot_av1,
        cmargin, mslice.spot_ap1, mslice.strike, mslice.texp, 1)
    bwd_ret, wr2 = model.arbitrage_ret(
        mslice.time, mslice.call_ap1, mslice.put_bp1, mslice.call_av1, mslice.put_bv1, mslice.spot_bv1,
        pmargin, mslice.spot_bp1, mslice.strike, mslice.texp, -1)
    # SaveRes(folder_name='expected_ret').np2Csv(
    #     [int_to_seconds(mslice.time), mslice.code, mslice.texp, mslice.strike, fwd_ret, bwd_ret],
    #     ['time', 'code', 'texp', 'strike', 'fret', 'bret'], f"{today}.csv"
    # )

    df_output = model.get_indicator(wr1.reset_index(), mslice.time, mslice.code)
    df_output2 = model.get_indicator(wr2.reset_index())

    df_output = df_output.merge(df_output2, left_index=True, right_index=True, suffixes=('_f', '_b'))
    df_output.insert(0, 'date', today)
    return df_output


if __name__ == '__main__':
    # 1. input the date
    start_dt = '20150301'
    end_dt = '20150401'

    data_dir = Path(__file__).parent / 'data'
    if not (data_dir / f'opt_info.pkl').exists():
        df_info = Read4PCP().database(start_dt, end_dt, '510050.SH')
        SaveRes().df2Pkl(df_info, f"opt_info.pkl")

    # 2. input the director of tick data
    target_dir = Path(r'Z://tick/stock')
    target_dir = sorted(target_dir.glob('*'))
    res = Parallel(n_jobs=20)(
        delayed(run)(f / 'quote')
        for f in target_dir if start_dt <= f.parts[-1] <= end_dt
    )
    df = pd.concat(res, ignore_index=False, axis=0).reset_index().sort_values(['date', 'texp'])
    SaveRes(dir_path=Path(__file__).parent / 'result').df2Exc(df, 'ret.xlsx')
