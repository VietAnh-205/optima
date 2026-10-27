from analysis_data.end_to_end_v2.config_v2 import BILL_CAT_FEATURES
import json 
import os 
from collections import Counter 
from typing import Dict, List, Optional
import pandas as pd 
import numpy as np 
from .config_v2 import PAD_IDX, PAD_TOKEN, UNK_TOKEN, UNK_IDX, BILL_NUM_FEATURES 

class VocabBuilder: 
    """
    Quản lý tất cả vocabularies cho pipeline V2.
    
    Cấu trúc mỗi vocab:
        {
            "<PAD>": 0,
            "<UNK>": 1,
            "giá_trị_1": 2,
            "giá_trị_2": 3,
            ...
        }
    """
    def __init__(self): 
        self.vocabs: Dict[str, Dict[str, int]] = {} 
        self.reverse_vocabs: Dict[str, Dict[int, str]] = {} 
        self._is_built = False 

    def build_from_train(self, bill_df: pd.DataFrame, trajectories: Dict[str, list], min_freq: int = 1): 
        """
        Xây dựng vocabulary từ tập train.
        
        Args:
            bill_df: DataFrame chứa bill-level categorical features
            trajectories: Dict[bill_code → List[event]]
            min_freq: Tần suất tối thiểu để giữ token (default=1, giữ tất cả)
        """

        # bill_level categorical features 
        print("\n[VocabBuilder] Xây dựng vocabulary từ tập train...")
        for col in BILL_CAT_FEATURES: 
            if col not in bill_df.columns: 
                print(f"{col} khong co trong bill, tao vocab rong")
                self.vocabs[col] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX} 
                continue 

            counter = Counter(bill_df[col].dropna().astype(str).values) 
            tokens = sorted(
                [
                    tok for tok, cnt in counter.items()
                    if cnt >= min_freq and tok not in (PAD_TOKEN, UNK_TOKEN)
                ]
            )

            vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX} 
            for i, tok in enumerate(tokens, start = 2): 
                vocab[tok] = i 
            
            self.vocabs[col] = vocab 
            print(f" {col}: {len(vocab)} tokens" 
            f"(+2 special = {len(tokens)} unique values)") 

        # sequence-level features: warehouse_name, io_status 
            # collect from trajectories 
        wh_counter = Counter() 
        io_counter = Counter() 

        for events in trajectories.values(): 
            for evt in events: 
                wh_counter[str(evt['warehouse_name'])] += 1 
                io_counter[str(evt.get('io_status', "MISSING"))] += 1 
        
        wh_tokens = sorted([
            tok for tok, cnt in wh_counter.items() 
            if cnt >= min_freq and tok not in (PAD_TOKEN, UNK_TOKEN)
        ])
        self.vocabs['warehouse_name'] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX} 
        for i, tok in enumerate(wh_tokens, start = 2): 
            self.vocabs['warehouse_name'][tok] = 1 
        print(f"  warehouse_name: {len(self.vocabs['warehouse_name'])} tokens "
              f"({len(wh_tokens)} unique warehouses)")
        
        io_tokens = sorted([
            tok for tok, cnt in io_counter.items()
            if cnt >= min_freq and tok not in (PAD_TOKEN, UNK_TOKEN)
        ]) 
        self.vocabs['io_status'] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX} 
        for i, tok in enumerate(io_tokens, start = 2): 
            self.vocabs['io_status'][tok] = i 
        print(f"  io_status: {len(self.vocabs['io_status'])} tokens" 
              f"({len(io_tokens)} unique statuses)")

        # label vocabs (first 1a, last 1a) 
            #build reverse vocabs 
        self.reverse_vocabs = {
            name: {idx: tok for tok, idx in vocab.items()}
            for name, vocab in self.vocabs.items()
        } 

        self._is_built = True 
        print(f"Vocab build xong: {len(self.vocabs)} features") 
        
        
    def encode(self, feature_name: str, value: str) -> int: 
        if feature_name not in self.vocabs: 
            raise KeyError(f"Feature '{feature_name}' not in vocabs") 
        vocab = self.vocabs[feature_name] 
        return vocab.get(str(value), UNK_IDX) 
    
    def encode_batch(self, feature_name: str, values) -> np.ndarray: 
        vocab = self.vocabs[feature_name] 
        return np.array([
            vocab.get(str(v), UNK_IDX) for v in values 
        ], dtype=np.int64) 
    
    def decode(self, feature_name: str, index: int) -> str: 
        return self.reverse_vocabs[feature_name].get(index, UNK_TOKEN) 
    
    def vocab_size(self, feature_name: str) -> int:
        return len(self.vocabs[feature_name]) 
    
    def get_all_sizes(self) -> Dict[str, int]:
        return {name: len(v) for name, v in self.vocabs.items()} 
    
    def save(self, filepath: str): 
        os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        with open(filepath, 'w', encoding='utf-8') as f: 
            json.dump(self.vocabs, f, ensure_ascii=False, indent=2) 
        print(f"[VocaBuilder] luu vocab tai: {filepath}") 
    
    @classmethod
    def load(cls, filepath: str) -> "VocabBuilder":
        # 1. Đọc file JSON đã lưu
        with open(filepath, "r", encoding="utf-8") as f:
            vocabs = json.load(f)
        
        # 2. Tạo một instance mới của class VocabBuilder
        builder = cls()
        builder.vocabs = vocabs
        
        # 3. Khôi phục lại từ điển đảo ngược (reverse_vocabs)
        builder.reverse_vocabs = {
            name: {int(idx): tok for tok, idx in vocab.items()}
            for name, vocab in vocabs.items()
        }
        
        # 4. Kích hoạt cờ sẵn sàng và trả về instance
        builder._is_built = True
        print(f"[VocabBuilder] Đã load vocab từ: {filepath}")
        return builder
