import json
import os

# New 101 Alphas
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
    "rank(ts_delta(group_neutralize(close * 0.60733 + open * (1 - 0.60733), sector), 1.23438)) < rank(ts_rank(ts_corr(ts_rank(vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644), 14.6644))", # Alpha#79
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

# Path to elite_seed.json
seed_path = "d:/alpha-factory-private/data/elite_alphas/elite_seed.json"

if os.path.exists(seed_path):
    with open(seed_path, 'r') as f:
        existing_seeds = json.load(f)
    
    existing_exprs = {s['expression'] for s in existing_seeds}
    
    new_added = 0
    for alpha in ARXIV_101:
        if alpha not in existing_exprs:
            existing_seeds.append({
                "expression": alpha,
                "sharpe": 1.25, # Default high sharpe for classic alphas
                "fitness": 1.0,
                "turnover": 50,
                "source": "arxiv_101_import"
            })
            new_added += 1
            existing_exprs.add(alpha)
            
    if new_added > 0:
        with open(seed_path, 'w') as f:
            json.dump(existing_seeds, f, indent=2)
        print(f"Added {new_added} new alphas to elite_seed.json")
    else:
        print("No new alphas to add.")
else:
    print(f"Seed file {seed_path} not found.")
