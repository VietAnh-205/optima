import os  
import pandas as pd 
import numpy as np 

df_o = pd.read_csv('output_all_traces/origin_to_1A.csv') 
df_d = pd.read_csv('output_all_traces/1A_to_destination.csv') 

target_bills = set(df_o['bill_code']).union(set(df_d['bill_code']))

chunks = [] 
for chunk in pd.read_csv('bill_schedule.csv', chunksize=1_000_000):
    chunk = chunk[chunk['bill_code'].isin(target_bills)]
    if not chunk.empty:
        chunk['io_time'] = pd.to_datetime(chunk['io_time'])
        chunks.append(chunk)

sche = pd.concat(chunks, ignore_index=True) 
sche = sche.sort_values(['bill_code', 'io_time']).reset_index(drop=True)

sche['pre_wh'] = sche.groupby('bill_code')['warehouse_name'].shift(1)
sche_first = sche[sche['warehouse_name'] != sche['pre_wh']].copy()
sche_first['wh_rank'] = sche_first.groupby('bill_code').cumcount()
max_rank = sche_first.groupby('bill_code')['wh_rank'].transform('max')
sche_first['wh_rank_rev'] = max_rank - sche_first['wh_rank']

sche['next_wh'] = sche.groupby('bill_code')['warehouse_name'].shift(-1)
sche_last = sche[sche['warehouse_name'] != sche['next_wh']].copy()
sche_last['wh_rank'] = sche_last.groupby('bill_code').cumcount()

visit_times = pd.merge(
    sche_first[['bill_code', 'wh_rank', 'wh_rank_rev', 'warehouse_name', 'io_time', 'io_status']].rename(columns={'io_time': 'visit_in_time', 'io_status': 'in_status'}),
    sche_last[['bill_code', 'wh_rank', 'warehouse_name', 'io_time', 'io_status']].rename(columns={'io_time': 'visit_out_time', 'io_status': 'out_status'}),
    on=['bill_code', 'wh_rank', 'warehouse_name']
)


valid_visits = visit_times[(visit_times['in_status'] == 'IN') & (visit_times['out_status'] == 'OUT')].copy()
valid_visits['time_in_1a'] = ((valid_visits['visit_out_time'] - valid_visits['visit_in_time']).dt.total_seconds() / 3600).round(0).astype('Int64')

valid_visits_o1a = valid_visits[valid_visits['wh_rank'] == 1]
df_o_filter = pd.merge(df_o, valid_visits_o1a[['bill_code', 'warehouse_name', 'visit_in_time', 'visit_out_time', 'time_in_1a']], 
                       left_on=['bill_code', 'kho_o1a'], right_on=['bill_code', 'warehouse_name'], how='inner').drop(columns=['warehouse_name'])

valid_visits_d1a = valid_visits[valid_visits['wh_rank_rev'] == 1]
df_d_filter = pd.merge(df_d, valid_visits_d1a[['bill_code', 'warehouse_name', 'visit_in_time', 'visit_out_time', 'time_in_1a']],
                       left_on=['bill_code', 'kho_d1a'], right_on=['bill_code', 'warehouse_name'], how='inner').drop(columns=['warehouse_name'])

df_o_filter = df_o_filter.rename(columns={'visit_in_time': 'time_1a_in', 'visit_out_time': 'time_1a_out'})
df_d_filter = df_d_filter.rename(columns={'visit_in_time': 'time_1a_in', 'visit_out_time': 'time_1a_out'})

df_o_filter.to_csv(os.path.join('output_all_traces', 'origin_inout_1a.csv'), index = False) 
df_d_filter.to_csv(os.path.join('output_all_traces', 'destination_inout_1a.csv'), index = False)

print(f"size of o_filter.csv: {df_o_filter.shape}") 
print(f"size of d_filter.csv: {df_d_filter.shape}")