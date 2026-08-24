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
sche['next_wh'] = sche.groupby('bill_code')['warehouse_name'].shift(-1) 

sche_first = sche[sche['warehouse_name'] != sche['pre_wh']].copy() 
sche_last = sche[sche['warehouse_name'] != sche['next_wh']].copy() 

sche_first['wh_rank'] = sche_first.groupby('bill_code').cumcount() 
sche_last['wh_rank'] = sche_last.groupby('bill_code').cumcount() 

sche_1a = sche_first[sche_first['warehouse_name'].isin(set_1a)] 
count_1a = sche_1a.groupby('bill_code').size().reset_index(name = 'count_1a') 
count_1a = count_1a[count_1a['count_1a'] >= 2] 
valid_bills = set(count_1a['bill_code'])

sche_first = sche_first[sche_first['bill_code'].isin(valid_bills)]
max_rank_first = sche_first.groupby('bill_code')['wh_rank'].transform('max') 
sche_first['wh_rank_rev'] = max_rank_first - sche_first['wh_rank'] 
sche_first = sche_first.merge(bill, on = 'bill_code', how = 'inner')

sche_last = sche_last[sche_last['bill_code'].isin(valid_bills)]
max_rank_last = sche_last.groupby('bill_code')['wh_rank'].transform('max') 
sche_last['wh_rank_rev'] = max_rank_last - sche_last['wh_rank'] 
sche_last = sche_last.merge(bill, on = 'bill_code', how = 'inner')

wh_o = (sche_last[sche_last['wh_rank'] == 0][['bill_code', 'warehouse_name', 'io_time', 'io_status']].rename(columns = {'warehouse_name': 'kho_o', 'io_time': 'time_o', 'io_status': 'status_o'})) 
wh_o1a = (sche_first[sche_first['wh_rank'] == 1][['bill_code', 'warehouse_name', 'io_time', 'io_status', 'actual_weight']].rename(columns = {'warehouse_name': 'kho_o1a', 'io_time': 'time_o1a', 'io_status': 'status_o1a'}))
wh_d = (sche_first[sche_first['wh_rank_rev'] == 0][['bill_code', 'warehouse_name', 'io_time','io_status']].rename(columns = {'warehouse_name': 'kho_d', 'io_time': 'time_d', 'io_status': 'status_d'})) 
wh_d1a = (sche_last[sche_last['wh_rank_rev'] == 1][['bill_code', 'warehouse_name', 'io_time', 'io_status', 'actual_weight']].rename(columns = {'warehouse_name': 'kho_d1a', 'io_time': 'time_d1a', 'io_status': 'status_d1a'})) 

pair_o = (wh_o.merge(wh_o1a, on = 'bill_code', how = 'inner'))
pair_d = (wh_d1a.merge(wh_d, on = 'bill_code', how = 'inner')) 

pair_o = pair_o[~pair_o['kho_o'].isin(set_1a)]
pair_o = pair_o[pair_o['kho_o1a'].isin(set_1a)]
pair_d = pair_d[pair_d['kho_d1a'].isin(set_1a)] 
pair_d = pair_d[~pair_d['kho_d'].isin(set_1a)] 

pair_o['time'] = (((pair_o['time_o1a'] - pair_o['time_o']).dt.total_seconds())/3600)
pair_d['time'] = (((pair_d['time_d'] - pair_d['time_d1a']).dt.total_seconds())/3600)


pair_o.to_csv(os.path.join(OUTPUT_DIR, 'origin_to_1A.csv'), index = False) 
pair_d.to_csv(os.path.join(OUTPUT_DIR, '1A_to_destination.csv'), index = False) 
print('pair_o shape', pair_o.shape)
print('pair_d shape', pair_d.shape)

import json 
with open("D:\\optima\\VietAnh\\normal_bill_code_sample.json", 'r', encoding='utf-8') as f:
    data = json.load(f) 

head_data = data.get('head', []) 
tail_data = data.get('tail', []) 
head = pd.DataFrame(data['head'], columns = ['bill_code']) 
tail = pd.DataFrame(data['tail'], columns = ['bill_code'] )
print('head shape', head.shape) 
print('tail shape', tail.shape) 

warehouse = pd.read_csv('warehouse.csv') 
set_bc = warehouse[warehouse['Bưu Cục'] == "Y"]['name'].to_list()
origin_bc = pair_o[pair_o['kho_o'].isin(set_bc)] 
des_bc = pair_d[pair_d['kho_d'].isin(set_bc)] 
print('origin_bc shape: ', origin_bc.shape) 
print('des_bc shape: ', des_bc.shape) 

oh = origin_bc[origin_bc['bill_code'].isin(set(head['bill_code']))] 
dt = des_bc[des_bc['bill_code'].isin(set(tail['bill_code']))] 
print('origin_head shape: ', oh.shape) 
print('destination_tail shape: ', dt.shape) 

oh_final = oh[((oh['status_o'] == "OUT") & (oh['status_o1a'] == 'IN'))] 
dt_final = dt[((dt['status_d'] == 'IN') & (dt['status_d1a'] == 'OUT'))] 

print("oh_final shape: ", oh_final.shape)
print("dt_final shape: ", dt_final.shape) 
print('percentage origin: ', oh_final.shape[0] / oh.shape[0]) 
print('percentage destination: ', dt_final.shape[0] / dt.shape[0])

oh_final.to_csv(os.path.join('output_all_traces', 'origin_head.csv'), index = False) 
dt_final.to_csv(os.path.join('output_all_traces', 'destination_tail.csv'), index = False)