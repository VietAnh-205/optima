import os, sys 
import numpy as np 
import pandas as pd 
from typing import Tuple, Dict, List 

from .config_v2 import (
    CHUNK_SIZE, TRAIN_SPLIT, VAL_SPLIT, ARRIVAL_OFFSET_MAX_HOUR, 
    PROJECT_ROOT, MAX_SEQ_LEN, BILL_NUM_FEATURES,
)

sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None 


def load_raw_data(data_dir: str = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 3 file CSV chính. Tái sử dụng logic V1."""
    from analysis_data.end_to_end.e2e_preprocessor import load_raw_data as _load
    return _load(data_dir=data_dir, **kwargs)

def extract_1a_journey(bill_schedule_df, warehouse_1a_df) -> pd.DataFrame:
    """Trích xuất hành trình kho 1A. Tái sử dụng nguyên vẹn từ V1."""
    from analysis_data.end_to_end.e2e_preprocessor import extract_1a_journey as _extract
    return _extract(bill_schedule_df, warehouse_1a_df)


def compute_time_to_first_1a(bill_schedule_df, journey_df) -> pd.DataFrame:
    """Tính thời gian đến kho 1A đầu tiên. Tái sử dụng từ V1."""
    from analysis_data.end_to_end.e2e_preprocessor import compute_time_to_first_1a as _compute
    return _compute(bill_schedule_df, journey_df)


def _engineer_numerical_features(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 

    if 'bill_creation_date' in df.columns: 
        df['bill_creation_date'] = pd.to_datetime(df['bill_creation_date'], errors='coerce')
        c = df['bill_creation_date'] 
        df['creation_hour'] = c.dt.hour + c.dt.minute / 60.0 
        df['creation_day_of_week'] = c.dt.dayofweek 
        df['creation_month'] = c.dt.month 

    if "estimated_delivery_date" in df.columns: 
        df['estimated_delivery_date'] = pd.to_datetime(df['estimated_delivery_date']) 
        if 'bill_creation_date' in df.columns: 
            diff_days = (df['estimated_delivery_date'] - df['bill_creation_date']).total_seconds() / 86400 
            df['estimated_delivery_days'] = diff_days 
        
    if 'estimated_delivery_date' in df.columns and 'bill_creation_date' in df.columns: 
        e_dow = df['estimated_delivery_date'].dt.dayofweek 
        c_dow = df['creation_dat_of_week'] 
        df['creation_delivery_dow_diff'] = e_dow - c_dow 
        df['same_creation_delivery_dow'] = (c_dow == e_dow).astype(int) 
        e_month = df['estimated_delivery_date'].dt.month 
        c_month = df['creation_month'] 

        df['same_creation_delivery_month'] = (c_month == e_month).astype(int) 
    
    hour = df.get('creation_hour', pd.Series(np.zeros(len(df)))) 
    dow = df.get('creation_day_of_week', pd.Series(np.zeros(len(df)))) 
    month = df.get('creation_month', pd.Series(np.ones(len(df)))) 
    df["creation_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["creation_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["creation_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["creation_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["creation_month_sin"] = np.sin(2 * np.pi * month / 12)
    df["creation_month_cos"] = np.cos(2 * np.pi * month / 12)
    
    if "actual_weight" in df.columns:
        df["actual_weight"] = pd.to_numeric(df["actual_weight"], errors="coerce").clip(lower=0)
    else:
        df["actual_weight"] = 0.0
    df["log_actual_weight"] = np.log1p(df["actual_weight"])
    
    return df

def build_trajactories(bill_schedule_df: pd.DataFrame, journey_df: pd.DataFrame) -> Dict[str, List[dict]]: 
    print("Xây dựng chuỗi trajactories cho mỗi bill") 
    valid_bills = set(journey_df['bill_code']) 
    sche = bill_schedule_df[bill_schedule_df['bill_code'].isin(valid_bills)].copy() 
    sche['warehouse_name'] = sche['warehouse_name'].astype(str).str.strip() 
    sche['io_time'] = pd.to_datetime(sche['io_time'], errors='coerce') 
    sche = sche.dropna(subset=['io_time']).sort_values(['bill_code', 'io_time']) 

    # merge first 1A info 
    first_info = journey_df.set_index('bill_code')[['first_1A', 'first_1a_arrival_time']] 
    sche['first_1A'] = sche['bill_code'].map(first_info['first_1A']) 
    sche['first_1a_arrival_time'] = sche['bill_code'].map(first_info['first_1a_arrival_time']) 
    
    sche['row_id'] = sche.groupby('bill_code').cumcount() 
    is_first_arrival = (
        (sche['warehouse_name'] == sche['first_1A'])
        & (sche['io_status'] == "IN") & (sche['io_time'] == sche['first_1a_arrival_time']) 
    )
    first_arr_row = (
        sche[is_first_arrival].groupby('bill_code')['row_id'].min().rename('first_arr_row')
    )
    sche = sche.merge(first_arr_row, on='bill_code', how='inner') 

    sche = sche[sche["row_id"] < sche['first_arr_row']].copy() 

    trajactories = {} 
    for bill_code, group in sche.groupby('bill_code'): 
        events = []
        for _, row in group.iterrows(): 
            events.append({
                'warehouse_name': row['warehouse_name'], 
                'io_status': row['io_status'] if pd.notna(row['io_status']) else 'MISSING', 
                'io_time': row['io_time'], 
            })
        if len(events) >= 1: 
            trajactories[bill_code] = events 
    
    print(f" Xay dung xong {len(trajactories):,} trajatories") 
    
    return trajactories 

def build_v2_data(data_dir: str = None, bill_path: str = None, bill_schedule_path: str = None, warehouse_1a_path: str = None) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]: 
    print("Xay dung du lieu V2-Hybrid Deep Learning") 

    sche, bill, wh1a = load_raw_data(data_dir = data_dir, bill_path = bill_path, bill_schedule_path = bill_schedule_path, warehouse_1a_path=warehouse_1a_path) 
    journey_df = extract_1a_journey(sche, wh1a) 
    journey_df = compute_time_to_first_1a(sche, journey_df) 

    journey_df = journey_df[
        (journey_df['time_to_first_1a_hours'] >= 0) 
        & (journey_df['time_to_first_1a_hours'] <= ARRIVAL_OFFSET_MAX_HOUR)
    ].copy() 

    trajectories = build_trajactories(sche, journey_df) 
    valid_bills = set(trajectories.keys()) 
    journey_df = journey_df[journey_df['bill_code'].isin(valid_bills)].copy() 
    bill_df = bill.merge(journey_df[['bill_code']], on='bill_code', how='inner').drop_dulicates(subset=['bill_code']) 
    bill_df = _engineer_numerical_features(bill_df)

    labels_df = journey_df[[
        "bill_code", "first_1A", "last_1A",
        "time_to_first_1a_hours", "first_wh_out_time",
        "first_1a_arrival_time",
    ]].copy()
    
    print(f"\n[build_v2_data] Hoàn tất:")
    print(f"  Bills: {len(bill_df):,}")
    print(f"  Trajectories: {len(trajectories):,}")
    print(f"  Labels: {len(labels_df):,}")

    del sche, bill
    import gc; gc.collect()
    
    return bill_df, trajectories, labels_df

def time_based_split(
    labels_df: pd.DataFrame,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
) -> Tuple[set, set, set]:

    sorted_bills = labels_df.sort_values("first_1a_arrival_time")["bill_code"].values
    n = len(sorted_bills)
    train_idx = int(n * train_split)
    val_idx = int(n * (train_split + val_split))
    
    train_bills = set(sorted_bills[:train_idx])
    val_bills = set(sorted_bills[train_idx:val_idx])
    test_bills = set(sorted_bills[val_idx:])
    
    print(f"\n[time_based_split] Train: {len(train_bills):,} | "
          f"Val: {len(val_bills):,} | Test: {len(test_bills):,}")
    
    return train_bills, val_bills, test_bills