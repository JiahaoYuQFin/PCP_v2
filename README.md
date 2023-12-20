# PCP_v2
 test the expected return and different contracts.

## `pcp_etf` Framework

## `main.py` instructions

### `config.py` parameters

start_dt: YYYYMMDD, 策略起始日期

end_dt: YYYYMMDD, 策略截止日期

und_code: XXXXXX.SH, 标的交易所代码

tickdata_dir: Path(r'Z://tick/stock'), 远程连接添加到本地映射盘符

localdata_dir: 本地存储数据的根目录

localres_dir: 本地存储策略结果的根目录

bsm_params: BSM模型的无风险利率rf、股息率q参数
