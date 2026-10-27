import os 

# path 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")) 

# data processing 
CHUNG_SIZE = 500_000 
TRAIN_SPLIT = 0.75 
VAL_SPLIT = 0.15 
ARRIVAL_OFFSET_MAX_HOUR = 72.0 

# vocabulary & embedding 
PAD_TOKEN = "<PAD>" 
UNK_TOKEN = "<UNK>" 
PAD_IDX = 0 
UNK_IDX = 1 

# dim = min(50, max(4, cardinality // k))
EMBEDDING_DIMS = {
    'origin_province': 16, 
    "origin_district": 32, 
    'destination_province': 16, 
    'destination_province': 32, 
    'service': 8, 
    'VD_type': 8, 
    'warehouse_name': 32, 
    'io_status': 4,
}

BILL_EMB_DIM = sum([
    EMBEDDING_DIMS["origin_province"],
    EMBEDDING_DIMS["origin_district"],
    EMBEDDING_DIMS["destination_province"],
    EMBEDDING_DIMS["destination_district"],
    EMBEDDING_DIMS["service"],
    EMBEDDING_DIMS["VD_type"],
]) 

# Bill categorical features
BILL_CAT_FEATURES = [
    "origin_province",
    "origin_district",
    "destination_province",
    "destination_district",
    "service",
    "VD_type",
]

# Bill numerical features
BILL_NUM_FEATURES = [
    "actual_weight",
    "log_actual_weight",
    "estimated_delivery_days",
    "creation_hour_sin",
    "creation_hour_cos",
    "creation_dow_sin",
    "creation_dow_cos",
    "creation_month_sin",
    "creation_month_cos",
    "creation_delivery_dow_diff",
    "same_creation_delivery_dow",
    "same_creation_delivery_month",
]
NUM_NUMERICAL = len(BILL_NUM_FEATURES)

# sequence (trajactory) 
MAX_SEQ_LEN = 15
TIME2VEC_DIM = 16
DELTA_T_DIM = 4
CUM_TIME_DIM = 4 

# Event dim = warehouse_emb(32) + io_emb(4) + time2vec(16) + delta_t(4) + cum_time(4) = 60
EVENT_DIM = (
    EMBEDDING_DIMS["warehouse_name"]
    + EMBEDDING_DIMS["io_status"]
    + TIME2VEC_DIM
    + DELTA_T_DIM
    + CUM_TIME_DIM
) 

# model architecture 
LSTM_HIDDEN = 64 
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2 
LSTM_BIDIRECTIONAL = True 
TRAJ_OUTPUT_DIM = LSTM_HIDDEN * (2 if LSTM_BIDIRECTIONAL else 1) 

NUMERICAL_PROJ_DIM = 16 
BILL_BRANCH_DIM = BILL_EMB_DIM + NUMERICAL_PROJ_DIM 

FUSION_HIDDEN = 256 
SHARED_DIM = 128 
HEAD_HIDDEN = 64
N_CLASSES = 25 

ATTENSION_HEADS = 4 

# training 
BATCH_SIZE = 512 
LEARNING_RATE = 1e-3 
WEIGHT_DECAY = 1e-4 
MAX_EPOCHS = 50 
PATTENCE = 7 
GRADIENT_CLIP = 1.0 
SCHEDULER = "OneCycleLR" 
USE_AMP = True # mixed precision 
HUBER_DELTA = 2.0 # huber loss delta 
UNK_DROPOUT_RATE = 0.05 
SEED = 42 
