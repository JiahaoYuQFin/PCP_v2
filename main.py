# -*- coding: utf-8 -*-
"""
Created on 2023/11/6 13:46
@author: jhyu
"""
from config import *
from pcp_etf import *
from util.readData import *
from util.saveData import SaveRes
from util.price_correction import synthesize_close2future
from tools.bsm import Bsm
from joblib import Parallel, delayed


def _pcp_task(file: Path):
    data_repo = Read4PCP(path=str(file))
    today = file.parts[-2]

    # df_opt_info = data_repo.database(start_dt=today, end_dt=today, und='510050.SH')
    df_opt_info = pd.read_pickle(localdata_dir / 'opt_info.pkl').query("date==@today")
    opt_codes = df_opt_info['code'].tolist()
    df_opt = data_repo.remote(codes=opt_codes)

    model = ETFpcp(contract_month=None)
    mslice = model.convert_df_to_input(df_opt=df_opt, df_opt_info=df_opt_info)
    cmargin = model.get_margin(mslice.call_settlement, mslice.spot_close, mslice.strike, 1)
    pmargin = model.get_margin(mslice.put_settlement, mslice.spot_close, mslice.strike, -1)

    fwd_ret, fwd_num, wr1 = model.arbitrage_ret(
        mslice.time, mslice.call_bp1, mslice.put_ap1, mslice.call_bv1, mslice.put_av1, mslice.spot_av1,
        cmargin, mslice.spot_ap1, mslice.strike, mslice.texp, mslice.maturity, 1)
    bwd_ret, bwd_num, wr2 = model.arbitrage_ret(
        mslice.time, mslice.call_ap1, mslice.put_bp1, mslice.call_av1, mslice.put_bv1, mslice.spot_bv1,
        pmargin, mslice.spot_bp1, mslice.strike, mslice.texp, mslice.maturity, -1)
    SaveRes(folder_name='single_ret').np2Csv(
        [mslice.time, mslice.code, mslice.texp, mslice.strike, fwd_ret, fwd_num, bwd_ret, bwd_num],
        ['time', 'code', 'texp', 'strike', 'fret', 'fnum', 'bret', 'bnum'], f"{today}.csv"
    )
    # SaveRes(folder_name='weighted_ret').np2Csv(
    #     [(wr1['time']/1e3).astype('int32').to_list(), wr1['maturity'].values, wr1['ret'].values, wr2['ret'].values],
    #     ['time', 'maturity', 'fret', 'bret'], f"{today}.csv"
    # )

    df_output = model.get_indicator(wr1)
    df_output2 = model.get_indicator(wr2)

    df_output = df_output.merge(df_output2, left_index=True, right_index=True, suffixes=('_f', '_b'))
    df_output.insert(0, 'date', today)
    return df_output


def run_pcp():

    if not (localdata_dir / f'opt_info.pkl').exists():
        df_info = Read4PCP().database(start_dt, end_dt, und_code)
        SaveRes().df2Pkl(df_info, f"opt_info.pkl")

    # 2. input the director of tick data
    target_dir = tickdata_dir
    target_dir = sorted(target_dir.glob('*'))
    res = Parallel(n_jobs=5)(
        delayed(_pcp_task)(f / 'quote')
        for f in target_dir if start_dt <= f.parts[-1] <= end_dt
    )
    df = pd.concat(res, ignore_index=False, axis=0).reset_index().sort_values(['date', 'maturity'])
    SaveRes(dir_path=Path(__file__).parent / 'result').df2Exc(df, 'multexp_ret_ts.xlsx')


def _greek_task(file: Path, df_opt_info):
    data_repo = Read4PCP(path=str(file))
    model = Bsm(0, intr=bsm_params['rf'], divr=bsm_params['q'], is_fwd=False)
    today = file.parts[-2]

    df_opt_info = df_opt_info.query("date==@today")
    opt_codes = df_opt_info['code'].tolist()
    df_opt = data_repo.remote(codes=opt_codes)
    df_opt = (df_opt.merge(df_opt_info, on=['date', 'code'], how='left'))

    price_vals = (df_opt['ask_prc1_opt'] + df_opt['bid_prc1_opt']).values / 2
    spot_vals = (df_opt['ask_prc1_spot'] + df_opt['bid_prc1_spot']).values / 2
    k_vals = df_opt['strike'].values
    texp_vals = df_opt['texp'].values / 365
    cp_vals = df_opt['cp'].values

    df_greek = df_opt[['date', 'time', 'maturity', 'strike', 'cp', 'code', 'texp']].copy()
    df_greek['iv'] = model.impvol_naive(
        price=price_vals, strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals, setval=True
    )
    df_greek['delta'] = model.delta(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['vega'] = model.vega(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['theta'] = model.theta(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['rho'] = model.rho(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['gamma'] = model.gamma(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['volga'] = model.volga(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['charm'] = model.charm(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['veta'] = model.veta(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['vanna'] = model.vanna(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['speed'] = model.speed(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['ultima'] = model.ultima(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['color'] = model.color(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['zomma'] = model.zomma(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    return df_greek


def run_greek():
    df_info = Read4PCP().database(start_dt, end_dt, und_code)
    target_dir = sorted(tickdata_dir.glob('*'))

    res = Parallel(n_jobs=1)(
        delayed(_greek_task)(f / 'quote', df_info)
        for f in target_dir if start_dt <= f.parts[-1] <= end_dt
    )
    df = pd.concat(res, ignore_index=False, axis=0).sort_values(['date', 'time', 'maturity', 'strike', 'cp'])
    SaveRes(dir_path=localres_dir).df2Csv(df, 'greeks.csv')


def run_greek_by_day():
    df_opt_day = Read4PCP().database(start_dt, end_dt, und_code)
    df_opt_day2 = df_opt_day.groupby(['date', 'maturity']).apply(synthesize_close2future, bsm_params['rf']).copy()
    model = Bsm(0, intr=bsm_params['rf'], divr=bsm_params['q'], is_fwd=False)

    price_vals = df_opt_day['settlement'].values
    spot_vals = df_opt_day['close_spot'].values
    k_vals = df_opt_day['strike'].values
    texp_vals = df_opt_day['texp'].values / 365
    texp_vals[texp_vals <= 1e-8] = 1e-10
    cp_vals = df_opt_day['cp'].values

    df_greek = df_opt_day.copy()
    df_greek['iv'] = model.impvol_naive(
        price=price_vals, strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals, setval=True
    )
    df_greek['delta'] = model.delta(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['vega'] = model.vega(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['theta'] = model.theta(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['rho'] = model.rho(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['gamma'] = model.gamma(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['volga'] = model.volga(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['charm'] = model.charm(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['veta'] = model.veta(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['vanna'] = model.vanna(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['speed'] = model.speed(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['ultima'] = model.ultima(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['color'] = model.color(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)
    df_greek['zomma'] = model.zomma(strike=k_vals, spot=spot_vals, texp=texp_vals, cp=cp_vals)

    SaveRes(dir_path=localres_dir).df2Csv(df_greek, f"{start_dt}_{end_dt}_greeks_by_day.csv")


if __name__ == '__main__':
    run_greek_by_day()
