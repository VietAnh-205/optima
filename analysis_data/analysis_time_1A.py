import os 
import pandas as pd 

OUTPUT_DIR = "output_all_traces"

wh1a = pd.read_csv('warehouse_1A.csv') 
set_1a = set(wh1a['name'].dropna().str.strip()) 
bill = pd.read_csv('bill.csv', usecols = ['bill_code', 'actual_weight'])

chunks = [] 
for chunk in pd.read_csv('bill_schedule.csv',chunksize = 500_000):
    chunk['io_time'] = pd.to_datetime(chunk['io_time']) 
    chunks.append(chunk) 

sche = pd.concat(chunks, ignore_index=True) 
del chunks 
sche = sche.sort_values(['bill_code', 'io_time']).reset_index(drop=True) 


sche['pre_wh'] = sche.groupby('bill_code')['warehouse_name'].shift(1) 
sche_unique = sche[sche['warehouse_name'] != sche['pre_wh']].copy() 
sche_unique['wh_rank'] = sche_unique.groupby('bill_code').cumcount() 

sche_1a = sche_unique[sche_unique['warehouse_name'].isin(set_1a)] 
count_1a = sche_1a.groupby('bill_code').size().reset_index(name = 'count_1a') 
count_1a = count_1a[count_1a['count_1a'] >= 2] 
sche_unique = sche_unique[sche_unique['bill_code'].isin(set(count_1a['bill_code']))]
max_rank = sche_unique.groupby('bill_code')['wh_rank'].transform('max') 
sche_unique['wh_rank_rev'] = max_rank - sche_unique['wh_rank'] 
sche_unique = sche_unique.merge(bill, on = 'bill_code', how = 'inner')

wh_o = (sche_unique[sche_unique['wh_rank'] == 0][['bill_code', 'warehouse_name', 'io_time']].rename(columns = {'warehouse_name': 'kho_o', 'io_time': 'time_o'})) 
wh_o1a = (sche_unique[sche_unique['wh_rank'] == 1][['bill_code', 'warehouse_name', 'io_time', 'actual_weight']].rename(columns = {'warehouse_name': 'kho_o1a', 'io_time': 'time_o1a'}))
wh_d = (sche_unique[sche_unique['wh_rank_rev'] == 0][['bill_code', 'warehouse_name', 'io_time']].rename(columns = {'warehouse_name': 'kho_d', 'io_time': 'time_d'})) 
wh_d1a = (sche_unique[sche_unique['wh_rank_rev'] == 1][['bill_code', 'warehouse_name', 'io_time', 'actual_weight']].rename(columns = {'warehouse_name': 'kho_d1a', 'io_time': 'time_d1a'})) 

pair_o = (wh_o.merge(wh_o1a, on = 'bill_code', how = 'inner'))
pair_d = (wh_d1a.merge(wh_d, on = 'bill_code', how = 'inner')) 

pair_o = pair_o[~pair_o['kho_o'].isin(set_1a)]
pair_o = pair_o[pair_o['kho_o1a'].isin(set_1a)]
pair_d = pair_d[pair_d['kho_d1a'].isin(set_1a)] 
pair_d = pair_d[~pair_d['kho_d'].isin(set_1a)] 

pair_o['time'] = (((pair_o['time_o1a'] - pair_o['time_o']).dt.total_seconds())/3600).round(0).astype('Int64') 
pair_d['time'] = (((pair_d['time_d'] - pair_d['time_d1a']).dt.total_seconds())/3600).round(0).astype("Int64") 


pair_o.to_csv(os.path.join(OUTPUT_DIR, 'origin_to_1A.csv'), index = False) 
pair_d.to_csv(os.path.join(OUTPUT_DIR, '1A_to_destination.csv'), index = False) 


