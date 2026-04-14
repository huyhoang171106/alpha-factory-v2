"""
alpha_seeds.py — Alpha Expression Seed Library v2
Sources:
  - 101 Alphas (Kakushadze 2015, arXiv)
  - SMC / Price Action concepts
  - WQ Brain community templates
  - NEW: Microstructure, Quality, Behavioural, Cross-Sectional, Regime factors
"""

import random
from typing import List

# ============================================================
# Building blocks
# ============================================================
PRICES = ["open", "high", "low", "close", "vwap"]
VOLUMES = ["volume", "adv20", "adv60"]
RETURNS = ["returns"]
FUNDAMENTALS = ["eps", "sales", "book_value", "ebitda", "debt", "operating_margin", "return_equity"]
ALT_DATA = ["sentiment", "short_interest_ratio", "analyst_estimates"]
FIELDS = PRICES + VOLUMES + RETURNS + FUNDAMENTALS + ALT_DATA

TS_OPS_1D = ["ts_rank", "ts_zscore", "ts_mean", "ts_std_dev", "ts_sum",
             "ts_min", "ts_max", "ts_arg_max", "ts_arg_min", "ts_skewness",
             "ts_decay_linear", "ts_delta", "ts_product"]
RANK_OPS = ["rank", "-rank", "zscore", "sigmoid"]
GROUP_OPS = ["group_rank", "group_zscore", "group_neutralize", "group_mean"]
GROUPS = ["sector", "industry", "subindustry", "market"]
LOOKBACKS = [3, 5, 7, 10, 14, 20, 30, 50, 60, 100, 120, 200, 252]


# ============================================================
# Seed 1: arXiv 101 Alphas (Kakushadze 2015)
# ============================================================
ARXIV_101 = [
    "rank(ts_arg_max(signed_power(if(returns < 0, ts_std_dev(returns, 20), close), 2), 5)) - 0.5", # Alpha#1
    "-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank((close - open) / open), 6)", # Alpha#2
    "-1 * ts_corr(rank(open), rank(volume), 10)", # Alpha#3
    "-1 * ts_rank(rank(low), 9)", # Alpha#4
    "rank(open - ts_sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))", # Alpha#5
    "-1 * ts_corr(open, volume, 10)", # Alpha#6
    "if(adv20 < volume, -1 * ts_rank(abs(ts_delta(close, 7)), 60) * sign(ts_delta(close, 7)), -1)", # Alpha#7
    "-1 * rank(ts_sum(open, 5) * ts_sum(returns, 5) - ts_delay(ts_sum(open, 5) * ts_sum(returns, 5), 10))", # Alpha#8
    "if(ts_min(ts_delta(close, 1), 5) > 0, ts_delta(close, 1), if(ts_max(ts_delta(close, 1), 5) < 0, ts_delta(close, 1), -1 * ts_delta(close, 1)))", # Alpha#9
    "rank(if(ts_min(ts_delta(close, 1), 4) > 0, ts_delta(close, 1), if(ts_max(ts_delta(close, 1), 4) < 0, ts_delta(close, 1), -1 * ts_delta(close, 1))))", # Alpha#10
    "(rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(ts_delta(volume, 3))", # Alpha#11
    "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))", # Alpha#12
    "-1 * rank(ts_covariance(rank(close), rank(volume), 5))", # Alpha#13
    "-1 * rank(ts_delta(returns, 3)) * ts_corr(open, volume, 10)", # Alpha#14
    "-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3)", # Alpha#15
    "-1 * rank(ts_covariance(rank(high), rank(volume), 5))", # Alpha#16
    "-1 * rank(ts_rank(close, 10)) * rank(ts_delta(ts_delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))", # Alpha#17
    "-1 * rank(ts_std_dev(abs(close - open), 5) + (close - open) + ts_corr(close, open, 10))", # Alpha#18
    "-1 * sign(close - ts_delay(close, 7) + ts_delta(close, 7)) * (1 + rank(1 + ts_sum(returns, 250)))", # Alpha#19
    "-1 * rank(open - ts_delay(high, 1)) * rank(open - ts_delay(close, 1)) * rank(open - ts_delay(low, 1))", # Alpha#20
    "if(ts_mean(close, 8) + ts_std_dev(close, 8) < ts_mean(close, 2), -1, if(ts_mean(close, 2) < ts_mean(close, 8) - ts_std_dev(close, 8), 1, if(volume / adv20 >= 1, 1, -1)))", # Alpha#21
    "-1 * ts_delta(ts_corr(high, volume, 5), 5) * rank(ts_std_dev(close, 20))", # Alpha#22
    "if(ts_mean(high, 20) < high, -1 * ts_delta(high, 2), 0)", # Alpha#23
    "if(ts_delta(ts_mean(close, 100), 100) / ts_delay(close, 100) <= 0.05, -1 * (close - ts_min(close, 100)), -1 * ts_delta(close, 3))", # Alpha#24
    "rank(-1 * returns * adv20 * vwap * (high - close))", # Alpha#25
    "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)", # Alpha#26
    "if(rank(ts_mean(ts_corr(rank(volume), rank(vwap), 6), 2)) > 0.5, -1, 1)", # Alpha#27
    "scale(ts_corr(adv20, low, 5) + (high + low) / 2 - close)", # Alpha#28
    "ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank(-1 * rank(ts_delta(close - 1, 5)))), 2), 1))))), 1), 5) + ts_rank(ts_delay(-1 * returns, 6), 5)", # Alpha#29
    "(1.0 - rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)", # Alpha#30
    "rank(rank(rank(ts_decay_linear(-1 * rank(rank(ts_delta(close, 10))), 10)))) + rank(-1 * ts_delta(close, 3)) + sign(scale(ts_corr(adv20, low, 12)))", # Alpha#31
    "scale(ts_mean(close, 7) - close) + 20 * scale(ts_corr(vwap, ts_delay(close, 5), 230))", # Alpha#32
    "rank(-1 * power(1 - open / close, 1))", # Alpha#33
    "rank(1 - rank(ts_std_dev(returns, 2) / ts_std_dev(returns, 5)) + (1 - rank(ts_delta(close, 1))))", # Alpha#34
    "ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))", # Alpha#35
    "2.21 * rank(ts_corr(close - open, ts_delay(volume, 1), 15)) + 0.7 * rank(open - close) + 0.73 * rank(ts_rank(ts_delay(-1 * returns, 6), 5)) + rank(abs(ts_corr(vwap, adv20, 6))) + 0.6 * rank((ts_mean(close, 200) - open) * (close - open))", # Alpha#36
    "rank(ts_corr(ts_delay(open - close, 1), close, 200)) + rank(open - close)", # Alpha#37
    "-1 * rank(ts_rank(close, 10)) * rank(close / open)", # Alpha#38
    "-1 * rank(ts_delta(close, 7) * (1 - rank(ts_decay_linear(volume / adv20, 9)))) * (1 + rank(ts_sum(returns, 250)))", # Alpha#39
    "-1 * rank(ts_std_dev(high, 10)) * ts_corr(high, volume, 10)", # Alpha#40
    "power(high * low, 0.5) - vwap", # Alpha#41
    "rank(vwap - close) / rank(vwap + close)", # Alpha#42
    "ts_rank(volume / adv20, 20) * ts_rank(-1 * ts_delta(close, 7), 8)", # Alpha#43
    "-1 * ts_corr(high, rank(volume), 5)", # Alpha#44
    "-1 * rank(ts_mean(ts_delay(close, 5), 20)) * ts_corr(close, volume, 2) * rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))", # Alpha#45
    "if(0.25 < (ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10, -1, if((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10 < 0, 1, -1 * (close - ts_delay(close, 1))))", # Alpha#46
    "rank(1 / close) * volume / adv20 * high * rank(high - close) / (ts_sum(high, 5) / 5) - rank(vwap - ts_delay(vwap, 5))", # Alpha#47
    "group_neutralize(ts_corr(ts_delta(close, 1), ts_delta(ts_delay(close, 1), 1), 250) * ts_delta(close, 1) / close, subindustry) / ts_sum(power(ts_delta(close, 1) / ts_delay(close, 1), 2), 250)", # Alpha#48
    "if((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10 < -0.1, 1, -1 * (close - ts_delay(close, 1)))", # Alpha#49
    "-1 * ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)", # Alpha#50
    "if((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10 < -0.05, 1, -1 * (close - ts_delay(close, 1)))", # Alpha#51
    "(-1 * ts_min(low, 5) + ts_delay(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)", # Alpha#52
    "-1 * ts_delta(((close - low) - (high - close)) / (close - low), 9)", # Alpha#53
    "-1 * (low - close) * power(open, 5) / ((low - high) * power(close, 5))", # Alpha#54
    "-1 * ts_corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)", # Alpha#55
    "-1 * rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * rank(returns * cap)", # Alpha#56
    "-1 * (close - vwap) / ts_decay_linear(rank(ts_arg_max(close, 30)), 2)", # Alpha#57
    "-1 * ts_rank(ts_decay_linear(ts_corr(group_neutralize(vwap, sector), volume, 3.92795), 7.89291), 5.50322)", # Alpha#58
    "-1 * ts_rank(ts_decay_linear(ts_corr(group_neutralize(vwap * 0.728317 + vwap * (1 - 0.728317), industry), volume, 4.25197), 16.2289), 8.19648)", # Alpha#59
    "-1 * (2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume)) - scale(rank(ts_arg_max(close, 10))))", # Alpha#60
    "rank(vwap - ts_min(vwap, 16.1219)) < rank(ts_corr(vwap, adv180, 17.9282))", # Alpha#61
    "-1 * (rank(ts_corr(vwap, ts_sum(adv20, 22.4101), 9.91009)) < rank(rank(open) + rank(open) < rank((high + low) / 2) + rank(high)))", # Alpha#62
    "-1 * (rank(ts_decay_linear(ts_delta(group_neutralize(close, industry), 2.25164), 8.22237)) - rank(ts_decay_linear(ts_corr(vwap * 0.318108 + open * (1 - 0.318108), ts_sum(adv180, 37.2467), 13.557), 12.2883)))", # Alpha#63
    "-1 * (rank(ts_corr(ts_mean(open * 0.178404 + low * (1 - 0.178404), 12.7054), ts_sum(adv120, 12.7054), 16.6208)) < rank(ts_delta((high + low) / 2 * 0.178404 + vwap * (1 - 0.178404), 3.69741)))", # Alpha#64
    "-1 * (rank(ts_corr(open * 0.00817205 + vwap * (1 - 0.00817205), ts_sum(adv60, 8.6911), 6.40374)) < rank(open - ts_min(open, 13.635)))", # Alpha#65
    "-1 * (rank(ts_decay_linear(ts_delta(vwap, 3.51013), 7.23052)) + ts_rank(ts_decay_linear((low * 0.96633 + low * (1 - 0.96633) - vwap) / (open - (high + low) / 2), 11.4157), 6.72611))", # Alpha#66
    "-1 * power(rank(high - ts_min(high, 2.14593)), rank(ts_corr(group_neutralize(vwap, sector), group_neutralize(adv20, subindustry), 6.02936)))", # Alpha#67
    "-1 * (ts_rank(ts_corr(rank(high), rank(adv15), 8.91644), 13.9333) < rank(ts_delta(close * 0.518371 + low * (1 - 0.518371), 1.06157)))", # Alpha#68
    "-1 * power(rank(ts_max(ts_delta(group_neutralize(vwap, industry), 2.72412), 4.79344)), ts_rank(ts_corr(close * 0.490655 + vwap * (1 - 0.490655), adv20, 4.92416), 9.0615))", # Alpha#69
    "-1 * power(rank(ts_delta(vwap, 1.29456)), ts_rank(ts_corr(group_neutralize(close, industry), adv50, 17.8256), 17.9171))", # Alpha#70
    "max(ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 3.43976), ts_rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), ts_rank(ts_decay_linear(power(rank(low + open - (vwap + vwap)), 2), 16.4662), 4.4388))", # Alpha#71
    "rank(ts_decay_linear(ts_corr((high + low) / 2, adv40, 8.93345), 10.1519)) / rank(ts_decay_linear(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011))", # Alpha#72
    "-1 * max(rank(ts_decay_linear(ts_delta(vwap, 4.72775), 2.91864)), ts_rank(ts_decay_linear(-1 * ts_delta(open * 0.147155 + low * (1 - 0.147155), 2.03608) / (open * 0.147155 + low * (1 - 0.147155)), 3.33829), 16.7411))", # Alpha#73
    "-1 * (rank(ts_corr(close, ts_sum(adv30, 37.4843), 15.1365)) < rank(ts_corr(rank(high * 0.0261661 + vwap * (1 - 0.0261661)), rank(volume), 11.4791)))", # Alpha#74
    "rank(ts_corr(vwap, volume, 4.24304)) < rank(ts_corr(rank(low), rank(adv50), 12.4413))", # Alpha#75
    "-1 * max(rank(ts_decay_linear(ts_delta(vwap, 1.24383), 11.8259)), ts_rank(ts_decay_linear(ts_rank(ts_corr(group_neutralize(low, sector), adv81, 8.14941), 19.569), 17.1543), 19.383))", # Alpha#76
    "min(rank(ts_decay_linear((high + low) / 2 + high - (vwap + high), 20.0451)), rank(ts_decay_linear(ts_corr((high + low) / 2, adv40, 3.1614), 5.64125)))", # Alpha#77
    "power(rank(ts_corr(ts_mean(low * 0.352233 + vwap * (1 - 0.352233), 19.7428), ts_sum(adv40, 19.7428), 6.83313)), rank(ts_corr(rank(vwap), rank(volume), 5.77492)))", # Alpha#78
    "rank(ts_delta(group_neutralize(close * 0.60733 + open * (1 - 0.60733), sector), 1.23438)) < rank(ts_rank(ts_corr(ts_rank(vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644), 14.6644))", # Alpha#79 (Fixing nest)
    "-1 * power(rank(sign(ts_delta(group_neutralize(open * 0.868128 + high * (1 - 0.868128), industry), 4.04545))), ts_rank(ts_corr(high, adv10, 5.11456), 5.53756))", # Alpha#80
    "-1 * (rank(log(ts_product(rank(power(rank(ts_corr(vwap, ts_sum(adv10, 49.6054), 8.47743)), 4)), 14.9655))) < rank(ts_corr(rank(vwap), rank(volume), 5.07914)))", # Alpha#81
    "-1 * min(rank(ts_decay_linear(ts_delta(open, 1.46063), 14.8717)), ts_rank(ts_decay_linear(ts_corr(group_neutralize(volume, sector), (open * 0.634196 + open * (1 - 0.634196)), 17.4842), 6.92131), 13.4283))", # Alpha#82
    "rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume)) / ((high - low) / (ts_sum(close, 5) / 5) / (vwap - close))", # Alpha#83
    "signed_power(ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127), ts_delta(close, 4.96796))", # Alpha#84
    "power(rank(ts_corr(high * 0.876703 + close * (1 - 0.876703), adv30, 9.61331)), rank(ts_corr(ts_rank((high + low) / 2, 3.70596), ts_rank(volume, 10.1595), 7.11408)))", # Alpha#85
    "-1 * (ts_rank(ts_corr(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) < rank(open + close - (vwap + open)))", # Alpha#86
    "-1 * max(rank(ts_decay_linear(ts_delta(close * 0.369701 + vwap * (1 - 0.369701), 1.91233), 2.65461)), ts_rank(ts_decay_linear(abs(ts_corr(group_neutralize(adv81, industry), close, 13.4132)), 4.89768), 14.4535))", # Alpha#87
    "min(rank(ts_decay_linear(rank(open) + rank(low) - (rank(high) + rank(close)), 8.06882)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8.44728), ts_rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))", # Alpha#88
    "ts_rank(ts_decay_linear(ts_corr(low * 0.967285 + low * (1 - 0.967285), adv10, 6.94279), 5.51607), 3.79744) - ts_rank(ts_decay_linear(ts_delta(group_neutralize(vwap, industry), 3.48158), 10.1466), 15.3012)", # Alpha#89
    "-1 * power(rank(close - ts_max(close, 4.66719)), ts_rank(ts_corr(group_neutralize(adv40, subindustry), low, 5.38375), 3.21856))", # Alpha#90
    "-1 * (ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(group_neutralize(close, industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(ts_decay_linear(ts_corr(vwap, adv30, 4.01303), 2.6809)))", # Alpha#91
    "min(ts_rank(ts_decay_linear((high + low) / 2 + close < low + open, 14.7221), 18.8683), ts_rank(ts_decay_linear(ts_corr(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))", # Alpha#92
    "ts_rank(ts_decay_linear(ts_corr(group_neutralize(vwap, industry), adv81, 17.4193), 19.848), 7.54455) / rank(ts_decay_linear(ts_delta(close * 0.524434 + vwap * (1 - 0.524434), 2.77377), 16.2664))", # Alpha#93
    "-1 * power(rank(vwap - ts_min(vwap, 11.5783)), ts_rank(ts_corr(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756))", # Alpha#94
    "rank(open - ts_min(open, 12.4105)) < ts_rank(power(rank(ts_corr(ts_mean((high + low) / 2, 19.1351), ts_sum(adv40, 19.1351), 12.8742)), 5), 11.7584)", # Alpha#95
    "-1 * max(ts_rank(ts_decay_linear(ts_corr(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), ts_rank(ts_decay_linear(ts_arg_max(ts_corr(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143))", # Alpha#96
    "-1 * (rank(ts_decay_linear(ts_delta(group_neutralize(low * 0.721001 + vwap * (1 - 0.721001), industry), 3.3705), 20.4523)) - ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659))", # Alpha#97
    "rank(ts_decay_linear(ts_corr(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(ts_decay_linear(ts_rank(ts_arg_min(ts_corr(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206))", # Alpha#98
    "-1 * (rank(ts_corr(ts_mean((high + low) / 2, 19.8975), ts_sum(adv60, 19.8975), 8.8136)) < rank(ts_corr(low, volume, 6.28259)))", # Alpha#99
    "-1 * (1.5 * scale(group_neutralize(group_neutralize(rank((close - low - (high - close)) / (high - low) * volume), subindustry), subindustry)) - scale(group_neutralize(ts_corr(close, rank(adv20), 5) - rank(ts_arg_min(close, 30)), subindustry))) * volume / adv20", # Alpha#100
    "(close - open) / (high - low + 0.001)" # Alpha#101
]


# ============================================================
# Seed 2: SMC / Price Action concepts
# ============================================================
SMC_ALPHAS = [
    "-rank(ts_delta(close, 5))",
    "-rank(ts_delta(close, 10))",
    "-rank(ts_delta(close, 20))",
    "rank(ts_mean(returns, 20))",
    "rank(ts_mean(returns, 60))",
    "rank(close / ts_delay(close, 20) - 1)",
    "rank(volume / ts_mean(volume, 20))",
    "rank(volume / adv20) * rank(returns)",
    "rank(close / vwap - 1)",
    "-rank(abs(close - vwap) / vwap)",
    "-rank(ts_std_dev(returns, 5) / ts_std_dev(returns, 20))",
    "rank(ts_std_dev(close, 5) / ts_std_dev(close, 60))",
    "rank((close - open) / (high - low + 0.001))",
    "-rank((high - close) / (high - low + 0.001))",
    "rank(ts_mean(volume, 5) / ts_mean(volume, 60))",
    "rank(open / ts_delay(close, 1) - 1)",
    "-rank((high - low) / ts_mean(high - low, 20))",
]


# ============================================================
# Seed 3: Community / WQ Brain popular templates
# ============================================================
COMMUNITY_ALPHAS = [
    "group_neutralize(volume/(ts_sum(volume,60)/60),sector)",
    "rank(close+ts_product(close, 5)^(0.2))",
    "ts_corr(rank(close), rank(volume/(ts_sum(volume,20)/20)), 5)",
    "rank(group_mean(ts_delta(close,5),1,subindustry)-ts_delta(close,5))",
    "-ts_rank((ts_regression(close,close,20,LAG=1,RETTYPE=3)-ts_sum(ts_delay(close,1),2)/2)/close,60)*(1-rank(volume/(ts_sum(volume,30)/30)))",
    "-ts_sum(close-min(low,ts_delay(close,1)),5)/ts_sum(max(high,ts_delay(close,1))-low,5)",
    "-rank(close-ts_max(high,5))/(ts_max(high,5)-ts_min(low,5))",
    "signed_power(Ts_Rank((vwap - ts_max(vwap, 15)), 21), ts_delta(close,5))",
    "-Ts_Rank(ts_decay_linear(ts_corr(group_neutralize(vwap, sector), volume, 4), 8), 6)",
]


# ============================================================
# Seed 4: MICROSTRUCTURE (Information Asymmetry)
# Bản chất: Informed traders để lại dấu vết trong volume & price patterns.
# Sources: Kyle (1985), Amihud (2002), Easley-O'Hara (1987)
# ============================================================
MICROSTRUCTURE_ALPHAS = [
    # Amihud illiquidity ratio: high price impact per unit volume → illiquid → premium
    "-rank(ts_mean(abs(returns) / (volume + 1), 20))",
    "-rank(ts_mean(abs(returns) / (adv20 + 1), 60))",

    # Kyle's lambda proxy: price change per unit volume → informed trading
    "rank(abs(ts_delta(close, 1)) / (volume + 1))",
    "-rank(ts_mean(abs(ts_delta(close, 1)) / (volume + 1), 10))",

    # Order flow imbalance: signed volume pressure
    "rank(ts_sum(returns * volume, 5) / (ts_sum(abs(returns) * volume, 5) + 0.001))",
    "rank(ts_sum(returns * volume, 10) / (ts_sum(abs(returns) * volume, 10) + 0.001))",

    # Price impact reversal: large volume → temporary price impact → revert
    "-rank(ts_corr(abs(returns), volume, 10))",
    "-rank(ts_corr(abs(ts_delta(close, 1)), ts_delta(volume, 1), 5))",

    # Intraday range efficiency: narrow range with high volume = informed
    "rank((high - low) / (volume + 1))",
    "-rank(ts_mean((high - low) / (adv20 + 1), 20))",

    # Trade size signal: large trades are more informed
    "rank(ts_delta(volume, 1) / (adv20 + 1))",

    # VWAP deviation persistence: sustained deviation = information
    "rank(ts_mean(close / vwap - 1, 5))",
    "-rank(abs(ts_mean(close / vwap - 1, 10)))",

    # Tick direction: consecutive up-ticks → buying pressure
    "rank(ts_sum(sign(ts_delta(close, 1)), 5))",
    "rank(ts_sum(sign(ts_delta(close, 1)) * volume, 5) / ts_sum(volume, 5))",

    # Volume surge without price move → accumulation
    "rank(volume / adv20) * (-rank(abs(returns)))",

    # Close-to-open gap vs intraday: institutional overnight vs retail intraday
    "rank(open / ts_delay(close, 1) - 1) * (-rank(close / open - 1))",

    # Participation rate trend
    "-rank(ts_delta(volume / adv20, 5))",

    # Bid-ask proxy (high-low to open)
    "-rank((high - low) / (open + 0.001))",

    # Price clustering: stocks near integers attract more uninformed attention
    "rank(close - floor(close))",

    # Reversal after extreme volume
    "(-rank(ts_delta(close, 1))) * rank(ts_delta(volume, 1))",

    # Turnover momentum
    "rank(ts_mean(volume / sharesout, 20))",
    "-rank(ts_delta(volume / sharesout, 5))",

    # Smart money divergence: price up but volume drying
    "rank(returns) * (-rank(ts_delta(volume, 5)))",

    # Compressed spread after high volume → informed trade absorbed
    "-rank(ts_corr(volume / adv20, abs(returns), 10))",
]


# ============================================================
# Seed 5: QUALITY FACTOR (Earnings Stability)
# Bản chất: High-quality = stable earnings, predictable price → outperform over time.
# Sources: Novy-Marx (2013), Asness et al (2019) "Quality Minus Junk"
# ============================================================
QUALITY_ALPHAS = [
    # Low return volatility = high quality → long
    "-rank(ts_std_dev(returns, 60))",
    "-rank(ts_std_dev(returns, 20))",

    # Trend stability: high correlation with its own trend = predictable
    "rank(ts_corr(close, ts_rank(close, 60), 20))",
    "rank(ts_corr(returns, ts_rank(returns, 20), 10))",

    # Mean reversion autocorrelation: predictable reversion = quality signal
    "-rank(ts_corr(returns, ts_delay(returns, 1), 20))",

    # Consistent price appreciation: steady rise beats volatile spike
    "rank(ts_mean(returns, 60) / (ts_std_dev(returns, 60) + 0.001))",  # Sharpe proxy
    "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",

    # Volume stability: consistent participation (not spiky)
    "-rank(ts_std_dev(volume / adv20, 20))",

    # Range stability: compressed volatility = quality
    "-rank(ts_std_dev((high - low) / close, 20))",

    # Price above trend persistence
    "rank(ts_sum(sign(close - ts_mean(close, 20)), 20))",
    "rank(ts_sum(sign(close - ts_mean(close, 60)), 20))",

    # Drawdown recovery: recover quickly from dips = strong
    "rank(close / ts_min(low, 20))",
    "rank(close - ts_min(close, 20))",

    # Earnings surprise proxy: gap between long-term and short-term trend
    "rank(ts_mean(returns, 5) - ts_mean(returns, 60))",

    # Volatility of highs vs lows: stable high-low range = quality
    "-rank(ts_std_dev(high - low, 20) / ts_mean(high - low, 20))",

    # Price to VWAP quality: consistently near VWAP = institutional backing
    "-rank(ts_std_dev(close / vwap - 1, 20))",

    # Momentum quality: smooth momentum vs choppy
    "rank(ts_mean(sign(returns), 20))",

    # Auto-correlation of returns quality
    "rank(ts_corr(returns, ts_delay(returns, 5), 10))",

    # Low skewness (not lottery stock)
    "-rank(ts_skewness(returns, 60))",

    # Fundamental stability
    "-rank(ts_std_dev(volume, 20) / ts_mean(volume, 20))",

    # Positive momentum with low variance = quality momentum
    "rank(ts_sum(sign(returns), 10)) * (-rank(ts_std_dev(returns, 10)))",

    # Price trend consistency: same direction many days in a row
    "rank(abs(ts_sum(sign(ts_delta(close, 1)), 20)) / 20)",
]


# ============================================================
# Seed 6: BEHAVIOURAL BIAS (Investor Psychology)
# Bản chất: Anchoring, disposition effect, 52-week high bias → exploitable.
# Sources: George & Hwang (2004), Shefrin & Statman (1985)
# ============================================================
BEHAVIOURAL_ALPHAS = [
    # 52-week high momentum: stocks near high → reference point anchoring → bullish
    "rank(close / ts_max(high, 252))",
    "rank(close / ts_max(close, 120))",

    # Distance from 52-week high → contrarian short
    "-rank(ts_max(high, 252) / close - 1)",

    # Disposition effect: investors hold losers, sell winners → mean reversion
    "-rank(close / ts_delay(close, 252))",  # annual loser → bounce
    "rank(-(close / ts_delay(close, 252) - 1))",

    # Recency bias: overweight recent bad news, ignore long-run view
    "rank(ts_mean(returns, 5) - ts_mean(returns, 20))",  # short > long = recent positive
    "rank(ts_mean(returns, 5) - ts_mean(returns, 60))",

    # Herding / trend chasing reversal: crowded trade → exhaustion
    "-rank(ts_mean(returns, 3)) * rank(ts_mean(volume / adv20, 3))",

    # Attention effect: high volume after big move = media attention → fade
    "(-rank(abs(returns))) * rank(volume / adv20)",

    # January effect proxy: stocks that fell in December → January bounce
    # Approximated: recent losers with tax-loss harvesting proxy
    "-rank(ts_mean(returns, 20)) * rank(ts_std_dev(returns, 5))",

    # Round number anchoring: stocks just below $10, $50, $100 boundaries
    # Proxy: close mod-distance from integers
    "rank(close - floor(close))",
    "rank(1 - (close - floor(close)))",  # near next integer = resistance

    # Overreaction to negative news: reversal after sharp decline
    "-rank(ts_min(returns, 5))",
    "-rank(ts_mean(ts_min(returns, 1), 5))",

    # Underreaction to gradual change: slow movers ignored
    "rank(ts_mean(returns, 60) - ts_mean(returns, 252))",

    # Momentum crash risk: past winner with high institutional ownership → riskier
    "-rank(ts_corr(volume / adv20, returns, 10)) * rank(ts_mean(returns, 20))",

    # Post-earnings drift proxy: recent momentum continuation
    "rank(ts_mean(returns, 5)) * rank(ts_mean(volume / adv20, 5))",

    # Lottery preference: high volatility stocks = overpriced, short
    "-rank(ts_std_dev(returns, 20)) * rank(ts_sum(sign(returns > 0.05), 20))",

    # Availability bias: stocks with recent extreme moves get attention premium
    "-rank(ts_max(abs(returns), 20))",

    # Status quo bias: under-reacting to industry trends
    "rank(ts_delta(close, 20) - group_mean(ts_delta(close, 20), 1, industry))",

    # Anchoring to opening price: sentiment signal
    "rank(close / open - 1)",
    "rank((close - open) / (0.001 + ts_mean(high - low, 20)))",

    # Fear & greed: high-low range expansion after calm
    "rank(ts_delta(high - low, 5))",
    "rank((high - low) / ts_mean(high - low, 20) - 1)",
]


# ============================================================
# Seed 7: CROSS-SECTIONAL RELATIVE VALUE
# Bản chất: Alpha từ relative performance giữa stocks trong ngành.
# Sources: Moskowitz & Grinblatt (1999), Lewellen (2002)
# ============================================================
CROSS_SECTIONAL_ALPHAS = [
    # Industry-relative momentum: outperform within industry
    "rank(ts_delta(close, 20) - group_mean(ts_delta(close, 20), 1, industry))",
    "rank(ts_delta(close, 5) - group_mean(ts_delta(close, 5), 1, sector))",
    "rank(returns - group_mean(returns, 1, subindustry))",

    # Sector rotation signal: rotating from laggard sector to leader
    "group_rank(rank(ts_mean(returns, 20)), 1, sector)",
    "group_rank(rank(ts_mean(returns, 60)), 1, industry)",

    # Value within sector: relative VWAP position
    "group_neutralize(rank(close / vwap), sector)",
    "group_neutralize(rank(close / ts_mean(close, 20)), industry)",

    # Momentum spread: leader - laggard within sector
    "rank(returns) - group_mean(rank(returns), 1, sector)",
    "rank(ts_mean(returns, 10)) - group_mean(rank(ts_mean(returns, 10)), 1, industry)",

    # Volume relative to sector: unusual participation within group
    "group_neutralize(rank(volume / adv20), sector)",
    "rank(volume / adv20) - group_mean(rank(volume / adv20), 1, industry)",

    # Cross-sector correlation: stock uncorrelated with its sector = alpha source
    "-rank(ts_corr(returns, group_mean(returns, 1, sector), 20))",
    "-rank(ts_corr(returns, group_mean(returns, 1, industry), 10))",

    # Within-industry dispersion: high dispersion → mean revert to mean
    "group_mean(rank(ts_delta(close, 5)), 1, subindustry) - rank(ts_delta(close, 5))",

    # Sector-relative volume trend
    "rank(ts_delta(volume / adv20, 5)) - group_mean(rank(ts_delta(volume / adv20, 5)), 1, sector)",

    # Industry leader signal: stock with best recent momentum leads sector
    "group_rank(rank(close / ts_delay(close, 20) - 1), 1, industry)",

    # Peer relative strength
    "rank(close / ts_delay(close, 60) - 1) - group_mean(rank(close / ts_delay(close, 60) - 1), 1, sector)",

    # Group-based volatility adjusted momentum
    "group_neutralize(rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001)), sector)",

    # Cross-sectional reversal after sector move
    "-rank(ts_delta(group_mean(returns, 1, sector), 5))",

    # Relative VWAP deviation within industry
    "rank(close / vwap) - group_mean(rank(close / vwap), 1, industry)",

    # Intra-sector value spread
    "group_neutralize(rank(close / ts_mean(close, 60)), subindustry)",
    "group_neutralize(rank(ts_mean(returns, 20)), subindustry)",

    # Sector reversal: sector laggard within laggard industry
    "-group_rank(rank(ts_mean(returns, 20)), 1, sector)",

    # Inter-sector divergence: stock diverging from group theme
    "rank(returns) * (-rank(ts_corr(returns, group_mean(returns, 1, sector), 5)))",
]


# ============================================================
# Seed 8: REGIME-CONDITIONAL SIGNALS
# Bản chất: Một số alpha chỉ work trong trending, một số chỉ trong ranging.
# Sources: Ang & Bekaert (2004), Volatility regime literature
# ============================================================
REGIME_ALPHAS = [
    # Vol regime: high vol → mean revert; low vol → trend follow
    "rank(ts_std_dev(returns, 5)) * (-rank(ts_delta(close, 5)))",   # high vol → revert
    "-rank(ts_std_dev(returns, 5)) * rank(ts_mean(returns, 20))",    # low vol → trend

    # Volume confirms trend: volume expansion + momentum is stronger signal
    "rank(ts_corr(returns, ts_delta(volume, 1), 10)) * rank(ts_mean(returns, 20))",
    "rank(volume / adv20) * rank(ts_mean(returns, 10))",

    # Breakout after compression
    "rank(ts_max(high, 5) / ts_min(low, 5)) * rank(ts_delta(volume, 5))",
    "rank(ts_delta(high - low, 5)) * rank(ts_delta(volume, 5))",

    # Trend vs range detection
    "rank(abs(ts_mean(returns, 20)) / ts_std_dev(returns, 20))",  # high = trending
    "rank(ts_corr(close, ts_rank(close, 20), 20))",               # linear trend strength

    # Trending regime momentum: double-confirm with direction + vol expansion
    "rank(ts_mean(returns, 10)) * rank(ts_delta(adv20, 5))",

    # Mean-revert when volatility spikes
    "-rank(ts_delta(close, 3)) * rank(ts_std_dev(returns, 5) / ts_std_dev(returns, 20))",

    # Quiet accumulation: low vol + increasing volume → breakout approaching
    "-rank(ts_std_dev(returns, 5)) * rank(ts_delta(volume / adv20, 10))",

    # Gap follow-through: gap + volume confirmation
    "rank(open / ts_delay(close, 1) - 1) * rank(volume / adv20)",

    # Momentum with regime filter: only when market trending
    "rank(ts_mean(returns, 20)) * rank(abs(ts_mean(returns, 20)) / ts_std_dev(returns, 20))",

    # Volume dry-up at highs → distribution (smart money selling)
    "-rank(ts_delta(volume / adv20, 5)) * rank(close / ts_max(close, 20))",

    # Volatility breakout: range expansion after compression
    "rank(ts_std_dev(returns, 5) / ts_std_dev(returns, 60))",
    "-rank(ts_std_dev(returns, 5) / ts_std_dev(returns, 60))",  # both sides worth testing

    # Trend persistence: how many days in same direction
    "rank(abs(ts_sum(sign(ts_delta(close, 1)), 10)) / 10) * rank(ts_delta(close, 10))",

    # Mean revert regime: wide range + high vol + high volume → near exhaustion
    "-rank(returns) * rank(ts_std_dev(returns, 5)) * rank(volume / adv20)",

    # Low vol squeeze → explosive move expected
    "(-rank(ts_std_dev(returns, 5))) * rank(ts_delta(volume, 5))",

    # Trend confirmed by volume flow
    "rank(ts_sum(sign(ts_delta(close, 1)) * volume, 10) / ts_sum(volume, 10))",

    # Regime: high turnover + positive returns = momentum regime
    "rank(ts_mean(returns, 20)) * rank(ts_mean(volume / sharesout, 20))",

    # Regime: corr between price and volume trend
    "rank(ts_corr(ts_rank(close, 20), ts_rank(volume, 20), 10))",
    "-rank(ts_corr(ts_rank(close, 20), ts_rank(volume, 20), 10))",  # can flip

    # Oscillation detection: bouncing inside range → pick extremes
    "rank(ts_rank(close, 20)) * (-rank(ts_corr(returns, ts_delay(returns, 1), 5)))",
]


# ============================================================
# Template mutation patterns (expanded)
# ============================================================
MUTATION_TEMPLATES = [
    "rank(ts_corr({P1}, ts_mean(volume, {D}), {W}))",
    "-rank(ts_delta({P1}, {D}))",
    "rank(ts_mean({P1}, {D1}) / ts_mean({P1}, {D2}))",
    "rank({P1} / ts_delay({P1}, {D}) - 1)",
    "ts_rank(({P1} - ts_mean({P1}, {D})) / ts_std_dev({P1}, {D}), {W})",
    "-rank(ts_std_dev({P1}, {D1}) / ts_std_dev({P1}, {D2}))",
    "rank(ts_corr({P1}, {P2}, {D}))",
    "group_neutralize(rank(ts_delta({P1}, {D})), {G})",
    "rank(ts_decay_linear(ts_delta({P1}, {D}), {W}))",
    "rank(({P1} - {P2}) / ({P1} + {P2} + 0.001))",
    "-ts_rank(ts_corr(rank({P1}), rank(volume), {D}), {W})",
    "rank(ts_covariance({P1}, volume, {D}))",
    "rank(power(ts_corr({P1}, {P2}, {D}), 2))",
    "(rank(ts_delta({P1}, {D1})) * rank(ts_delta(volume, {D2})))",
    "rank(ts_sum({P1} * volume, {D}) / ts_sum(volume, {D}))",
    # NEW: Quality templates
    "rank(ts_mean({P1}, {D1}) / (ts_std_dev({P1}, {D2}) + 0.001))",
    "-rank(ts_std_dev({P1}, {D}) / ts_mean({P1}, {D}))",
    "rank(ts_corr({P1}, ts_rank({P1}, {D}), {W}))",
    # NEW: Cross-sectional templates
    "group_neutralize(rank(ts_mean({P1}, {D})), {G})",
    "rank(ts_delta({P1}, {D})) - group_mean(rank(ts_delta({P1}, {D})), 1, {G})",
    "group_rank(rank(ts_mean({P1}, {D})), 1, {G})",
    # NEW: Regime templates
    "rank(ts_std_dev(returns, {D1})) * rank(ts_delta({P1}, {D2}))",
    "(-rank(ts_std_dev(returns, {D1}))) * rank(ts_mean({P1}, {D2}))",
    "rank({P1} / ts_mean(volume, {D})) * rank(ts_mean(returns, {W}))",
]


# ============================================================
# COMPOSITE templates (2-factor combinations)
# Bản chất: signal × confirming_filter = higher precision
# ============================================================
COMPOSITE_TEMPLATES = [
    # Momentum × Volume confirmation
    "rank(ts_mean(returns, {D1})) * rank(volume / ts_mean(volume, {D2}))",
    "rank(ts_delta(close, {D1})) * rank(ts_delta(volume, {D2}))",
    # Mean reversion × Volatility regime
    "-rank(ts_delta(close, {D1})) * rank(ts_std_dev(returns, {D2}))",
    # Quality × Momentum
    "rank(ts_mean(returns, {D1}) / (ts_std_dev(returns, {D2}) + 0.001)) * rank(ts_mean(returns, {D1}))",
    # Price position × Volume
    "rank((close - ts_min(low, {D1})) / (ts_max(high, {D1}) - ts_min(low, {D1}) + 0.001)) * rank(volume / ts_mean(volume, {D2}))",
    # VWAP deviation × momentum
    "rank(close / vwap - 1) * rank(ts_mean(returns, {D1}))",
    # Volume-weighted signal
    "rank(ts_sum(returns * volume, {D1}) / ts_sum(volume, {D1}))",
    "rank(ts_sum(ts_delta(close, 1) * volume, {D1}) / ts_sum(volume, {D1}))",
    # Smart money: volume confirms trend reversal
    "-rank(ts_delta(close, {D1})) * rank(volume / ts_mean(volume, {D2}))",
    # Microstructure × reversal
    "(-rank(returns)) * rank(abs(returns) / (volume + 1))",
]

# ============================================================
# Seed 9: FUNDAMENTAL VALUE FACTORS
# Bản chất: WQ Brain có sẵn fundamental data!
# sources: Fama-French (1992), Novy-Marx (2013), Piotroski (2000)
# ============================================================
FUNDAMENTAL_ALPHAS = [
    # === Value factors (P/E, P/B, P/S) ===
    # Lower P/E = value stock = higher expected return
    "-rank(close / (eps + 0.001))",
    "-rank(cap / (sales + 1))",                    # P/S ratio (lower = cheap)
    "-rank(cap / (book_value + 1))",               # P/B ratio — Fama-French
    "-rank(cap / (ebitda + 1))",                   # EV/EBITDA proxy

    # === Quality: stable earnings ===
    "rank(eps / (cap + 1))",                       # Earnings yield
    "rank(sales / cap)",                           # Revenue yield
    "rank(ebitda / (cap + 1))",                    # EBITDA yield

    # === Profitability (Graham, Novy-Marx) ===
    "rank(sales / book_value)",                    # Asset turnover proxy
    "rank(ebitda / (book_value + 1))",             # ROA proxy
    "rank(eps / (book_value + 1))",                # ROE proxy

    # === Growth signals ===
    "rank(ts_delta(sales, 60))",                   # Sales growth (quarterly)
    "rank(ts_delta(eps, 60))",                     # EPS growth
    "rank(ts_delta(ebitda, 60))",                  # EBITDA growth
    "-rank(ts_delta(debt, 60))",                   # Debt increase = bad

    # === Cross-sectional value relative to sector ===
    "group_neutralize(-rank(cap / (sales + 1)), sector)",
    "group_neutralize(-rank(cap / (book_value + 1)), sector)",
    "group_neutralize(rank(eps / (cap + 1)), sector)",
    "group_neutralize(rank(sales / cap), industry)",

    # === Composite value + momentum ===
    # Dogs of Dow: cheap stock + recent momentum
    "-rank(cap / (sales + 1)) * rank(ts_mean(returns, 20))",
    "rank(eps / (cap + 1)) * rank(ts_mean(returns, 60))",

    # === Piotroski F-Score proxies ===
    # Positive earnings = fundamental strength
    "rank(sign(eps))",
    "rank(sign(ts_delta(eps, 60)))",               # improving earnings
    "rank(sign(ts_delta(sales, 60)))",             # improving revenue
    "-rank(sign(ts_delta(debt, 60)))",             # declining debt = good

    # === Accruals (quality of earnings) ===
    # Low accruals = higher quality earnings
    "-rank((ebitda - ts_delay(ebitda, 60)) / (cap + 1))",

    # === Size effect (cap-based) ===
    "-rank(cap)",                                  # small cap premium
    "group_neutralize(-rank(cap), sector)",

    # === Reversal within value ===
    "-rank(ts_delta(close / (eps + 0.001), 20))",  # P/E expansion → revert
    "-rank(ts_delta(close / (book_value + 1), 20))",  # P/B expansion → revert
]


# ============================================================
# Seed 10: STATISTICAL ARBITRAGE (StatArb)
# ============================================================
STAT_ARB_ALPHAS = [
    # Beta-neutralization proxy: returns minus sector returns
    "rank(returns - group_mean(returns, 1, sector))",
    "group_neutralize(rank(returns - group_mean(returns, 1, market)), subindustry)",

    # Idiosyncratic volatility: low vol of residuals outperforms
    "-rank(ts_std_dev(returns - group_mean(returns, 1, sector), 20))",

    # Correlation spread (Cross-asset / Volume beta)
    "rank(ts_covariance(returns, volume / adv20, 20) / (ts_std_dev(volume / adv20, 20) + 0.001))",

    # Statistical pair divergence (Stock vs Sector Mean)
    "-rank(close / ts_mean(close, 20) - group_mean(close / ts_mean(close, 20), 1, sector))",
]

# ============================================================
# Seed 11: SIGNAL PROCESSING / CONTINUOUS MATH
# ============================================================
SIGNAL_PROCESSING_ALPHAS = [
    # MACD proxy: 12-day vs 26-day EMA crossover
    "rank(ts_decay_linear(close, 12) - ts_decay_linear(close, 26))",
    
    # RSI proxy: relative strength of gains vs losses
    "rank(ts_mean(max(ts_delta(close, 1), 0), 14) / (ts_mean(abs(ts_delta(close, 1)), 14) + 0.001))",
    
    # Bollinger Bands squeeze (price position within bands)
    "rank((close - (ts_mean(close, 20) - 2 * ts_std_dev(close, 20))) / (4 * ts_std_dev(close, 20) + 0.001))",
    
    # Information Ratio (Risk-adjusted return smoothness)
    "rank(ts_mean(returns, 20) / (ts_std_dev(returns, 20) + 0.001))",
    
    # High-frequency derivative filtering (Double smoothing momentum)
    "rank(ts_decay_linear(ts_decay_linear(ts_delta(close, 1), 5), 10))",
]


# ============================================================
# Seed 12: ARXIV ELITE (Kakushadze 101 — translated to FASTEXPR)
# Source: WQ-Brain/arxiv.txt — hand-picked structurally diverse set
# ============================================================
ARXIV_ELITE_ALPHAS = [
    # #1 Conditional volatility
    "rank(ts_arg_max(signed_power(((returns < 0) * ts_std_dev(returns, 20) + (returns >= 0) * close), 2), 5)) - 0.5",
    # #2 Volume-momentum divergence
    "-ts_corr(rank(ts_delta(log(volume), 2)), rank((close - open) / (open + 0.001)), 6)",
    # #5 Price-VWAP separation
    "rank(open - ts_mean(vwap, 10)) * (-abs(rank(close - vwap)))",
    # #7 ADV breakout conditional
    "((adv20 < volume) * (-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7))) + ((adv20 >= volume) * -1)",
    # #9 Momentum regime switch
    "((0 < ts_min(ts_delta(close, 1), 5)) * ts_delta(close, 1)) + ((ts_max(ts_delta(close, 1), 5) < 0) * ts_delta(close, 1)) + ((ts_min(ts_delta(close, 1), 5) <= 0) * (ts_max(ts_delta(close, 1), 5) >= 0) * -ts_delta(close, 1))",
    # #11 VWAP distance × volume change
    "(rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(ts_delta(volume, 3))",
    # #12 Volume-confirmed reversal
    "sign(ts_delta(volume, 1)) * (-ts_delta(close, 1))",
    # #17 Triple-rank composite
    "(-rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))",
    # #25 Multi-factor negative
    "rank((-returns * adv20 * vwap * (high - close)))",
    # #28 Scale-based mean reversion
    "scale(ts_corr(adv20, low, 5) + (high + low) / 2 - close)",
    # #30 Trend persistence filter
    "(1 - rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3)))) * ts_sum(volume, 5) / ts_sum(volume, 20)",
    # #33 Open-Close ratio power
    "rank(-((1 - open / (close + 0.001))))",
    # #34 Volatility ratio momentum
    "rank(1 - rank(ts_std_dev(returns, 2) / (ts_std_dev(returns, 5) + 0.001)) + (1 - rank(ts_delta(close, 1))))",
    # #35 Triple time-series intersection
    "ts_rank(volume, 32) * (1 - ts_rank((close + high - low), 16)) * (1 - ts_rank(returns, 32))",
    # #37 VWAP-based return correlation
    "rank(ts_corr(ts_delay(open - close, 1), close, 200)) + rank(open - close)",
    # #38 Rank ratio
    "(-rank(ts_rank(close, 10))) * rank(close / (open + 0.001))",
    # #40 High vol × correlation
    "(-rank(ts_std_dev(high, 10))) * ts_corr(high, volume, 10)",
    # #41 Geometric midpoint minus VWAP
    "signed_power(high * low, 0.5) - vwap",
    # #42 VWAP position ratio
    "rank(vwap - close) / (rank(vwap + close) + 0.001)",
    # #43 ADV rank crossover
    "ts_rank(volume / adv20, 20) * ts_rank(-ts_delta(close, 7), 8)",
    # #53 Stochastic delta
    "-ts_delta(((close - low) - (high - close)) / (close - low + 0.001), 9)",
    # #54 Price geometry
    "-(low - close) * power(open, 5) / ((low - high) * power(close, 5) + 0.001)",
    # #55 Normalized channel × volume
    "-ts_corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + 0.001)), rank(volume), 6)",
    # #101 Intraday position
    "(close - open) / ((high - low) + 0.001)",
]


# ============================================================
# Seed 13: ADVANCED OPS (Rare WQ Brain operators from operatorRAW.json)
# These operators are almost never used by competitors = low correlation
# ============================================================
ADVANCED_OPS_ALPHAS = [
    # trade_when: Hold position only when volume confirms
    "trade_when(rank(ts_delta(close, 5)), volume > adv20, 0)",
    # trade_when: Trade mean reversion only in high-vol regime
    "trade_when(-rank(ts_delta(close, 3)), ts_std_dev(returns, 10) > ts_mean(ts_std_dev(returns, 10), 40), 0)",

    # bucket + group_rank: Custom market-cap-bucketed relative value
    "group_rank(ts_delta(close, 5), bucket(rank(cap), range='0,1,0.2'))",
    # bucket: Volume-bucketed momentum
    "group_zscore(returns, bucket(rank(volume), range='0,1,0.25'))",

    # winsorize: Outlier-robust momentum
    "rank(winsorize(ts_delta(close, 10), std=3))",
    # winsorize + group_neutralize
    "group_neutralize(rank(winsorize(ts_zscore(returns, 20), std=4)), sector)",

    # ts_quantile: Gaussian-distributed rank (reduces outlier impact)
    "ts_quantile(close, 20) * rank(volume / adv20)",
    # ts_quantile: Cauchy-distribution tail capture
    "ts_quantile(returns, 30, driver='cauchy')",

    # hump: Turnover reducer on momentum signal
    "hump(rank(ts_delta(close, 5)), hump=0.01)",
    # hump: Stabilize volatility signal
    "hump(rank(ts_std_dev(returns, 10)), hump=0.02)",

    # ts_scale: Min-max normalized in time
    "rank(ts_scale(close, 20)) - 0.5",
    # ts_scale: Normalized volume burst
    "ts_scale(volume, 20) * rank(ts_delta(close, 1))",

    # ts_av_diff: NaN-safe mean deviation
    "rank(ts_av_diff(close, 20))",
    # ts_av_diff: Volume deviation from mean
    "-rank(ts_av_diff(volume, 10)) * rank(returns)",

    # normalize: Cross-sectional zero-mean
    "normalize(ts_delta(close, 5) * rank(volume))",

    # vector_neut: Orthogonalize signal to market beta
    "vector_neut(rank(ts_delta(close, 5)), rank(returns))",

    # days_from_last_change: Earnings/event staleness signal
    "rank(days_from_last_change(close)) * -rank(ts_delta(close, 1))",
    # jump_decay: Smooth out earnings jumps
    "rank(jump_decay(close, 20, sensitivity=0.5, force=0.1))",

    # scale: Book-size normalized composite
    "scale(rank(ts_delta(close, 10)) * rank(volume / adv20))",

    # Composite: Multiple rare ops combined
    "group_neutralize(winsorize(ts_quantile(returns, 20) * rank(ts_av_diff(close, 10)), std=3), industry)",
]


def get_all_seeds() -> List[str]:
    """Return all seed expressions from all themes (deduplicated)"""
    all_seeds = (
        ARXIV_101
        + SMC_ALPHAS
        + COMMUNITY_ALPHAS
        + MICROSTRUCTURE_ALPHAS
        + QUALITY_ALPHAS
        + BEHAVIOURAL_ALPHAS
        + CROSS_SECTIONAL_ALPHAS
        + REGIME_ALPHAS
        + FUNDAMENTAL_ALPHAS
        + STAT_ARB_ALPHAS
        + SIGNAL_PROCESSING_ALPHAS
        + ARXIV_ELITE_ALPHAS
        + ADVANCED_OPS_ALPHAS
    )
    return list(dict.fromkeys(all_seeds))  # preserve order, remove dups


def get_seeds_by_theme(theme: str) -> List[str]:
    """Return seeds for a specific theme"""
    themes = {
        "arxiv": ARXIV_101,
        "smc": SMC_ALPHAS,
        "community": COMMUNITY_ALPHAS,
        "microstructure": MICROSTRUCTURE_ALPHAS,
        "quality": QUALITY_ALPHAS,
        "behavioural": BEHAVIOURAL_ALPHAS,
        "cross_sectional": CROSS_SECTIONAL_ALPHAS,
        "regime": REGIME_ALPHAS,
        "fundamental": FUNDAMENTAL_ALPHAS,
        "stat_arb": STAT_ARB_ALPHAS,
        "signal_processing": SIGNAL_PROCESSING_ALPHAS,
        "arxiv_elite": ARXIV_ELITE_ALPHAS,
        "advanced_ops": ADVANCED_OPS_ALPHAS,
    }
    return themes.get(theme, [])


def get_random_seeds(n: int = 10) -> List[str]:
    """Return n random seeds"""
    seeds = get_all_seeds()
    return random.sample(seeds, min(n, len(seeds)))


if __name__ == "__main__":
    seeds = get_all_seeds()
    print(f"Total unique seeds: {len(seeds)}")
    print(f"\nBreakdown:")
    print(f"  arXiv 101: {len(ARXIV_101)}")
    print(f"  SMC:       {len(SMC_ALPHAS)}")
    print(f"  Community: {len(COMMUNITY_ALPHAS)}")
    print(f"  Microstructure: {len(MICROSTRUCTURE_ALPHAS)}")
    print(f"  Quality:   {len(QUALITY_ALPHAS)}")
    print(f"  Behavioural: {len(BEHAVIOURAL_ALPHAS)}")
    print(f"  Cross-sectional: {len(CROSS_SECTIONAL_ALPHAS)}")
    print(f"  Regime:    {len(REGIME_ALPHAS)}")
    print(f"  Stat_Arb:  {len(STAT_ARB_ALPHAS)}")
    print(f"  Signal_Processing: {len(SIGNAL_PROCESSING_ALPHAS)}")
    print(f"  Arxiv_Elite:   {len(ARXIV_ELITE_ALPHAS)}")
    print(f"  Advanced_Ops:  {len(ADVANCED_OPS_ALPHAS)}")
    print(f"\nSample 5:")
    for s in get_random_seeds(5):
        print(f"  {s}")
