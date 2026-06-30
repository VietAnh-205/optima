import os
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rcParams 
import seaborn as sns 
import pandas as pd

# setup rcParams  
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 200
rcParams["axes.unicode_minus"] = False
for font in ["Segoe UI", "Arial", "DejaVu Sans"]:
    try: rcParams["font.family"] = font; break
    except: pass

OUTPUT_DIR = 'output_all_province' 
os.makedirs(OUTPUT_DIR, exist_ok = True) 

def load_data(): 
    traces = pd.read_csv('output_all_traces/traces_non_1a.csv', low_memory=False)
    bill = pd.read_csv('bill.csv', usecols = ['bill_code', 'origin_province', 'destination_province', 'actual_weight', 'service']) 
    wh = pd.read_csv('warehouse.csv')
    wh_prov = wh.set_index('name')['province_name'].to_dict()
    trunk_set = set(wh['name']) 
    return traces, bill, wh_prov, trunk_set 

def  build_master(traces, bill, wh_prov):
    master = traces.merge(
        bill,
        on = 'bill_code', 
        how = 'inner'
    )

    master['first_trunk_prov'] = master['first_trunk'].map(wh_prov) 
    master['last_trunk_prov'] = master['last_trunk'].map(wh_prov) 

    return master 

def province_to_first_trunk(master): 
    sub = master.dropna(subset=['origin_province', 'first_trunk']).copy() 
    stats = (
        sub.groupby(['origin_province', 'first_trunk'])
        .agg(so_bill = ('bill_code', 'nunique'), 
            tong_kg = ('actual_weight', 'sum'))
        .reset_index()
    )
    total_by_prov = stats.groupby('origin_province')['so_bill'].transform('sum') 
    stats['pct_trong_tinh'] = (stats['so_bill'] / total_by_prov * 100).round(2) 
    stats['tong_kg'] = stats['tong_kg'].round(1) 
    stats = stats.sort_values(['origin_province', 'so_bill'], ascending = [True, False]) 

    stats.to_csv(os.path.join(OUTPUT_DIR, 'province_to_first_trunk_non_1a.csv'), index = False)

    return stats 

def last_trunk_to_province(master): 
    sub = master.dropna(subset = ['destination_province', 'last_trunk']).copy() 
    stats = (
        sub.groupby(['last_trunk', 'destination_province'])
        .agg(so_bill = ('bill_code', 'nunique'), tong_kg = ('actual_weight', 'sum'))
        .reset_index()
    )
    total_by_prov = stats.groupby('last_trunk')['so_bill'].transform('sum') 
    stats['pct_trong_tinh'] = (stats['so_bill'] / total_by_prov * 100).round(2) 
    stats['tong_kg'] = stats['tong_kg'].round(1) 
    stats = stats.sort_values(['last_trunk', 'so_bill'], ascending = [True, False]) 

    stats.to_csv(os.path.join(OUTPUT_DIR, 'last_trunk_to_province_non_1a.csv'), index = False) 

    stats2 = (
        sub.groupby(["destination_province", "last_trunk"])
        .agg(so_bill=("bill_code", "nunique"), tong_kg=("actual_weight", "sum"))
        .reset_index()
    )
    total_by_prov = stats2.groupby("destination_province")["so_bill"].transform("sum")
    stats2["pct_trong_tinh"] = (stats2["so_bill"] / total_by_prov * 100).round(2)
    stats2["tong_kg"] = stats2["tong_kg"].round(1)
    stats2 = stats2.sort_values(["destination_province", "so_bill"], ascending=[True, False])
    stats2.to_csv(os.path.join(OUTPUT_DIR, "province_to_last_trunk_non_1a.csv"), index=False)

    return stats, stats2

def trunk_province_flow(master): 
    sub = master[master['n_trunk_stops'] >= 2].copy() 

    stats = (
        sub.groupby(['origin_province', 'first_trunk', 'last_trunk', 'destination_province'])
        .agg(so_bill = ('bill_code', 'nunique'), tong_kg = ('actual_weight', 'sum'))
        .reset_index()
    )
    stats['tong_kg'] = stats['tong_kg'].round(1)

    key = stats.groupby(['origin_province', 'destination_province'])['so_bill'].transform('sum') 

    stats['pct_theo_origin_kho'] = (stats['so_bill'] / key * 100 ).round(2) 
    stats.sort_values(['origin_province', 'so_bill', 'destination_province'], ascending = [True, False, True])
    stats.to_csv(os.path.join(OUTPUT_DIR, 'trunk_province_flow_non_1a.csv'), index = False)
    return stats 

def main(): 
    traces, bill, wh_prov, trunk_set = load_data() 
    master = build_master(traces, bill, wh_prov) 
    p2ft     = province_to_first_trunk(master)
    lt2p, p2lt = last_trunk_to_province(master)
    flow     = trunk_province_flow(master)

if __name__ == '__main__': 
    main()