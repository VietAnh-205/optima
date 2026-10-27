import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import Dict, List, Tuple, Optional 

from .config_v2 import (
    BILL_CAT_FEATURES, BILL_NUM_FEATURES, MAX_SEQ_LEN, PAD_IDX, UNK_IDX, UNK_DROPOUT_RATE,
)
from .vocab_builder import VocabBuilder

class HybridBillDataset(Dataset): 
    """
    Dataset cho pipeline Hybrid.
    
    Mỗi sample trả về:
        bill_cat:  dict {feature_name: int}    — categorical indices
        bill_num:  np.array (12,)              — numerical features (standardized)
        seq_wh:    np.array (seq_len,)         — warehouse indices per event
        seq_io:    np.array (seq_len,)         — io_status indices per event
        seq_hour:  np.array (seq_len,)         — hour of day (0-24) per event
        seq_dt:    np.array (seq_len,)         — delta_t hours since prev event
        seq_cum:   np.array (seq_len,)         — cumulative hours since first event
        seq_len:   int                         — actual sequence length (before padding)
        targets:   dict {first_1A: int, last_1A: int, hours: float}
    """

    def __init__(
        self, 
        bill_df: pd.DataFrame, 
        trajectories: Dict[str, list], 
        labels_df: pd.DataFrame, 
        vocab: VocabBuilder, 
        label_encoder: Dict[str, Dict[str, int]], 
        num_scaler:Optional[object] = None, 
        is_train: bool = False, 
        unk_dropout_rate: float = UNK_DROPOUT_RATE, 
    ): 

        valid_bills = (
            set(bill_df['bill_code']) 
            & set(trajectories.keys()) 
            & set(labels_df['bill_code']) 
        ) 

        self.bill_df = bill_df[bill_df['bill_code'].isin(valid_bills)].set_index('bill_code') 
        self.labels_df = labels_df[labels_df['bill_code'].isin(valid_bills)].set_index('bill_code') 
        self.trajectories = {k:v for k, v in trajectories.items() if k in valid_bills} 
        self.bill_codes = sorted(list(valid_bills)) 

        self.vocab = vocab 
        self.label_encoder = label_encoder 
        self.num_scaler = num_scaler
        self.is_train = is_train 
        self.unk_dropout_rate = unk_dropout_rate  if is_train else 0.0 

        # pre-encode bill categorical features
        self._bill_cat_cache = {} 
        for col in BILL_CAT_FEATURES: 
            if col in self.bill_df.columns: 
                encode = vocab.encode_batch(col, self.bill_df[col].values) 
                self._bill_cat_cache[col] = dict(zip(self.bill_df.index, encode)) 
            else: 
                self._bill_cat_cache[col] = {bc: UNK_IDX for bc in self.bill_codes} 
        
        # pre-extract numerical features 
        num_cols = [c for c in BILL_NUM_FEATURES if c in self.bill_df.columns] 
        self._bill_num = self.bill_df[num_cols].fillna(0).values.astype(np.float32) 
        # Apply scaler if provided
        if self.num_scaler is not None:
            self._bill_num = self.num_scaler.transform(self._bill_num).astype(np.float32)
        self._bill_num_map = dict(zip(self.bill_df.index, range(len(self.bill_df))))
        
        print(f"  [HybridBillDataset] {len(self.bill_codes):,} samples | "
              f"is_train={is_train}")   
            
    def __len__(self) -> int: 
        return len(self.bill_codes) 
    
    def __getitem__(self, idx: int) -> dict: 
        bill_code = self.bill_codes[idx] 

        # --bill categorical indices -- 
        cat_indices = {} 
        for col in BILL_CAT_FEATURES: 
            val = self._bill_cat_cache[col].get(bill_code, UNK_IDX) 
            # word dropout: ngẫu nhiên thay thế token -> UNK khi train để model học UNK embedding tốt hơn 
            if self.is_train and self.unk_dropout_rate > 0: 
                if np.random.random() < self.unk_dropout_rate: 
                    val = UNK_IDX 
            cat_indices[col] = val 
        
        # bill numerical values 
        num_idx = self._bill_num_map.get(bill_code, 0) 
        bill_num = self._bill_num[num_idx] 

        # sequence events 
        events = self.trajectories[bill_code]
        seq_len = min(len(events), MAX_SEQ_LEN) 
        events = events[:seq_len] 
        seq_wh = np.zeros(seq_len, dtype=np.int64)
        seq_io = np.zeros(seq_len, dtype=np.int64)
        seq_hour = np.zeros(seq_len, dtype=np.float32)
        seq_dt = np.zeros(seq_len, dtype=np.float32)
        seq_cum = np.zeros(seq_len, dtype=np.float32)
        
        prev_time = None
        first_time = None
        
        for i, evt in enumerate(events):
            # Warehouse embedding index
            wh_val = self.vocab.encode("warehouse_name", evt["warehouse_name"])
            if self.is_train and self.unk_dropout_rate > 0:
                if np.random.random() < self.unk_dropout_rate:
                    wh_val = UNK_IDX
            seq_wh[i] = wh_val
            
            # IO status embedding index
            io_val = self.vocab.encode("io_status", evt.get("io_status", "MISSING"))
            seq_io[i] = io_val
            
            # Time features
            io_time = pd.Timestamp(evt["io_time"])
            if pd.isna(io_time):
                seq_hour[i] = 0.0
                seq_dt[i] = 0.0
                seq_cum[i] = 0.0
            else:
                # Giờ trong ngày (0-24, thập phân)
                seq_hour[i] = io_time.hour + io_time.minute / 60.0
                
                # Δt = hours since previous event
                if prev_time is not None:
                    dt = (io_time - prev_time).total_seconds() / 3600.0
                    seq_dt[i] = max(0.0, dt)
                
                # Cumulative time since first event
                if first_time is None:
                    first_time = io_time
                cum = (io_time - first_time).total_seconds() / 3600.0
                seq_cum[i] = max(0.0, cum)
                
                prev_time = io_time
                # ── Targets ──
        labels = self.labels_df.loc[bill_code]
        first_1a_idx = self.label_encoder["first_1A"].get(
            labels["first_1A"], 0
        )
        last_1a_idx = self.label_encoder["last_1A"].get(
            labels["last_1A"], 0
        )
        hours = float(labels["time_to_first_1a_hours"])
        
        return {
            "bill_cat": cat_indices,
            "bill_num": bill_num,
            "seq_wh": seq_wh,
            "seq_io": seq_io,
            "seq_hour": seq_hour,
            "seq_dt": seq_dt,
            "seq_cum": seq_cum,
            "seq_len": seq_len,
            "first_1A": first_1a_idx,
            "last_1A": last_1a_idx,
            "hours": hours,
            "bill_code": bill_code,
        }

    # hàm gom và đóng gói 
        # gom 512 đơn lẻ vào một batch hoàn chỉnh trước khi đưa vào gpu
def collate_fn(batch: List[dict]) -> dict: 
    batch_size = len(batch) 
    max_len = max(item['seq_len'] for item in batch) 
    max_len = min(max_len, MAX_SEQ_LEN) 

    # bill categorical: stack thanh dict of tensors 
    bill_cat = {} 
    for col in BILL_CAT_FEATURES: 
        bill_cat[col] = torch.tensor(
            [item['bill_cat'][col] for item in batch], dtype=torch.long
        ) 

    # bill numerical: stack 
    bill_num = torch.tensor(
        np.stack([item['bill_num'] for item in batch]), dtype=torch.float32 
    ) 

    # sequence: pad with zeros (PAD_INX = 0) 
        # Sequence: pad with zeros (PAD_IDX = 0)
    seq_wh = torch.zeros(batch_size, max_len, dtype=torch.long)
    seq_io = torch.zeros(batch_size, max_len, dtype=torch.long)
    seq_hour = torch.zeros(batch_size, max_len, dtype=torch.float32)
    seq_dt = torch.zeros(batch_size, max_len, dtype=torch.float32)
    seq_cum = torch.zeros(batch_size, max_len, dtype=torch.float32)
    seq_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padded (masked)
    
    for i, item in enumerate(batch):
        slen = min(item["seq_len"], max_len)
        seq_wh[i, :slen] = torch.from_numpy(item["seq_wh"][:slen])
        seq_io[i, :slen] = torch.from_numpy(item["seq_io"][:slen])
        seq_hour[i, :slen] = torch.from_numpy(item["seq_hour"][:slen])
        seq_dt[i, :slen] = torch.from_numpy(item["seq_dt"][:slen])
        seq_cum[i, :slen] = torch.from_numpy(item["seq_cum"][:slen])
        seq_mask[i, :slen] = False  # False = NOT padded (attend to this)
    
    # targets 
    first_1A = torch.tensor([item['first_1A'] for item in batch], dtype=torch.long) 
    last_1A = torch.tensor([item["last_1A"] for item in batch], dtype=torch.long)
    hours = torch.tensor([item["hours"] for item in batch], dtype=torch.float32)
    
    # Sequence lengths
    seq_lens = torch.tensor(
        [min(item["seq_len"], max_len) for item in batch], dtype=torch.long
    )
    
    # Bill codes (for debugging/evaluation)
    bill_codes = [item["bill_code"] for item in batch]
    
    return {
        "bill_cat": bill_cat,
        "bill_num": bill_num,
        "seq_wh": seq_wh,
        "seq_io": seq_io,
        "seq_hour": seq_hour,
        "seq_dt": seq_dt,
        "seq_cum": seq_cum,
        "seq_mask": seq_mask,
        "seq_lens": seq_lens,
        "first_1A": first_1A,
        "last_1A": last_1A,
        "hours": hours,
        "bill_codes": bill_codes,
    }