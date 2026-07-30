"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MULTI-STRATEGY REGRESSION PIPELINE – Ghi chú giải thích           ║
╚══════════════════════════════════════════════════════════════════════════════╝

 MỤC TIÊU
   Dự đoán y_offset = số giờ từ lúc gửi hàng đến khi đến kho đích,
   áp dụng cho Top-10 cặp kho theo lưu lượng bill.

 PIPELINE (với mỗi cặp kho) – Train → Val selection → Retrain trainval → Test ONCE
   Bước 1 │ Tách Test TRƯỚC khi làm bất cứ điều gì khác (80% trainval / 20% test)
           │   → Test không bao giờ được chạm tới cho đến bước cuối cùng
   Bước 2 │ DBSCAN clustering trên y_offset – CHỈ chạy trên trainval
           │   → Phân nhóm tự động các khung giờ đến thực tế
           │   → Noise (cluster = -1) bị loại bỏ
           │   → Cụm được đánh số lại 0, 1, 2... theo thứ tự giờ tăng dần
           │   → Test không có cluster label (tránh leakage); khi cần gán cụm
           │     cho test, dùng pick_cluster_by_hour (inference-only, Cách 1)
   Bước 3 │ Split trainval → train (75%) / val (25%), stratify theo cluster
   Bước 4 │ Train Regression per cluster trên TRAIN (để phục vụ val selection)
           │   → 3 mô hình: LR (Linear), RF (RandomForest), LGBM (LightGBM)
           │   → 2 bộ features: Option 1 (chỉ giờ xuất phát), Option 2 (full)
           │   → Mỗi cụm có bộ model riêng
   Bước 5 │ Val Selection – 3 chiến lược chọn cụm × 2 feature option × 3 model
           │   Cách 1 (Dominant Cluster): Với mỗi khung giờ, chọn cụm chiếm
           │     đa số trong training → đơn giản, không cần ML, ổn định
           │   Cách 2 (ML Classifier): RandomForest classifier dự đoán cụm
           │     từ full features → linh hoạt nhưng có thể sai cụm
           │   Cách 3 (Hybrid): Dùng ML nếu xác suất >= ngưỡng (HYBRID_THRESHOLD),
           │     còn lại dùng Cách 1 → kết hợp ưu điểm cả hai
           │   → Chọn best_combo theo Val MAE thấp nhất
   Bước 6 │ Retrain final models trên TOÀN BỘ trainval (80%) với best_combo
   Bước 7 │ TEST ONCE – chỉ chạy đúng 1 lần với best_combo trên test (20%)
           │   → Không có "Dominant (Correct)" oracle scope trên test, vì test
           │     không có cluster label thật (tránh dùng nhãn oracle)
           │   → Train (diagnostic) cũng được báo cáo song song để phát hiện
           │     overfit, nhưng không dùng để chọn combo

 CÁC TRƯỜNG THÔNG TIN TRONG OUTPUT
   flow             : Luồng vận chuyển (Luồng 1: Kho gửi→1A, Luồng 2: 1A→Kho nhận)
   pair             : Cặp kho (Kho A -> Kho B)
   data_split       : Tập dữ liệu của dòng kết quả (Train (diagnostic) / Test)
   cluster_selection: Chiến lược chọn cụm của best_combo (Cách 1/2/3)
   feature_option   : Bộ features dùng cho regression (Option 1/2)
   regression_model : Mô hình hồi quy (LR / RF / LGBM)
   n_samples        : Số bill trong tập tương ứng (train/test)
   MAE              : Mean Absolute Error – sai số trung bình tuyệt đối (giờ)
   RMSE             : Root Mean Squared Error – nhạy cảm với outlier (giờ)
   R²               : Hệ số xác định – càng gần 1 càng tốt (âm = tệ hơn đoán trung bình)
   MAPE             : Mean Absolute Percentage Error – sai số % so với giá trị thực

 LƯU Ý VỀ LR
   Linear Regression không có giới hạn output. Nếu ML Classifier gán nhầm cluster,
   LR có thể extrapolate ra ngoài vùng train → giá trị vọt cực lớn.
   → Đã xử lý bằng clip output trong khoảng [y_min-2, y_max+2] của cluster tương ứng.
   RF và LGBM không bị vấn đề này vì là tree-based model (output bounded).

 LƯU MODEL
   Models được lưu vào thư mục output_plot/saved_models/<pair_slug>/
   dưới dạng .joblib để có thể tái sử dụng mà không cần train lại.
   Xem hàm save_models() và load_models() trong file này.

Output:
  - CSV:     output_plot/multi_classification_summary.csv
  - HTML:    output_plot/multi_classification.html
  - Models:  output_plot/saved_models/<pair_slug>/*.joblib
"""

import sys, os, re, hashlib
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import DBSCAN
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Cấu hình ────────────────────────────────────────────────────────────
N_TOP = 10
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 150
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_plot")
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")  
HYBRID_THRESHOLD = 0.6  

# Cache key: thay đổi bất kỳ tham số nào dưới đây sẽ tự động vô hiệu hóa model cache cũ
CONFIG_HASH = hashlib.md5(
    f"{DBSCAN_EPS}_{DBSCAN_MIN_SAMPLES}_{HYBRID_THRESHOLD}".encode()
).hexdigest()[:8]

EXCLUDE = [
    'Kho TGDĐ Bảo Hành', 'Kho Hà Tĩnh', 'Kho Kiến An', 'Kho Thường Tín',
    'Kho Amway', 'Kho Digiworld', 'Kho Thái Nguyên', 'Kho Việt Trì',
    'Kho Ninh Hòa', 'Kho An Giang',
    'Bưu cục Dự Án Vgreen Sóng Thần', 'Bưu cục Dự Án Vgreen Văn Giang'
]

TIME_SLOTS = {
    "Sáng sớm (0-6h)": (0, 6),
    "Sáng (6-12h)": (6, 12),
    "Chiều (12-18h)": (12, 18),
    "Tối (18-24h)": (18, 24),
}


# ── Helpers ──────────────────────────────────────────────────────────────
def get_top_pairs(df, col_a, col_b, sort_by, wt_col, is_dest_flow):
    """Lấy top 10 cặp tỉnh có số lượng bill lớn nhất, mỗi cặp tỉnh lấy top 5 cặp kho"""
    df = df.copy()
    if is_dest_flow:
        df = df[~df[col_b].isin(EXCLUDE)]
    else:
        df = df[~df[col_a].isin(EXCLUDE)]
    key = "so_bill" if sort_by == "bill_count" else "tong_kg"
    # Đọc danh sách kho để lấy thông tin tỉnh
    wh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "warehouse.csv")
    wh = pd.read_csv(wh_path, usecols=['name', 'province'])
    
    # Merge province A
    df = pd.merge(df, wh, left_on=[col_a], right_on=['name'], how='left')
    df = df.drop(columns=['name']).rename(columns={'province': f'{col_a}_province'})
    
    # Merge province B
    df = pd.merge(df, wh, left_on=[col_b], right_on=['name'], how='left')
    df = df.drop(columns=['name']).rename(columns={'province': f'{col_b}_province'})

    df["pair"] = df[col_a].astype(str) + " (" + df[f'{col_a}_province'].astype(str) + ") → " + df[col_b].astype(str) + " (" + df[f'{col_b}_province'].astype(str) + ")"

    # Lọc dữ liệu theo luồng
    if is_dest_flow:
        df = df[df['destination_province'] == df[f'{col_b}_province']]
    else:
        df = df[df['origin_province'] == df[f'{col_a}_province']]

    # Lấy top cặp tỉnh theo số lượng bill
    agg_prov = df.groupby([f'{col_a}_province', f'{col_b}_province']).agg(
        so_bill=('bill_code', 'nunique'),
        tong_kg=(wt_col, 'sum')
    ).reset_index()
    top_prov = agg_prov.nlargest(N_TOP, key)
    
    # Lấy top cặp kho cho mỗi cặp tỉnh
    final_pairs = []
    for _, row in top_prov.iterrows():
        prov_a = row[f'{col_a}_province']
        prov_b = row[f'{col_b}_province']
        
        sub_df = df[(df[f'{col_a}_province'] == prov_a) & (df[f'{col_b}_province'] == prov_b)]
        agg_kho = sub_df.groupby(["pair"]).agg(
            so_bill=('bill_code', 'nunique'),
            tong_kg=(wt_col, 'sum')
        ).reset_index()
        top_kho = agg_kho.nlargest(3, key)
        final_pairs.extend(top_kho['pair'].tolist())
        
    return df, final_pairs


def transform_province(series, top=10):
    s = series.copy()
    top_province = s.value_counts().nlargest(top).index.tolist()
    return s.where(s.isin(top_province), 'khác')


def get_time_slot(hour):
    """Trả về khung giờ cho 1 giá trị giờ"""
    for slot_name, (lo, hi) in TIME_SLOTS.items():
        if lo <= hour < hi:
            return slot_name
    return "Tối (18-24h)"


def compute_regression_metrics(y_true, y_pred):
    """Tính MAE, RMSE, R², MAPE"""
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE: tránh chia 0
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def _pair_slug(pair: str) -> str:
    """Chuyển tên cặp kho thành tên thư mục hợp lệ. CONFIG_HASH tự vô hiệu hóa cache khi config thay đổi."""
    base = re.sub(r'[^\w]+', '_', pair, flags=re.ASCII).strip('_')[:70]
    return f"{base}_{CONFIG_HASH}"


def save_models(pair, regression_models, clf, flow_name, extra_meta=None):
    """
    Lưu toàn bộ model của một cặp kho vào đĩa.
    Lưu kèm extra_meta (best_combo, cluster_info...) trong meta.joblib
    """
    slug = _pair_slug(pair)
    pair_dir = os.path.join(MODELS_DIR, slug)
    os.makedirs(pair_dir, exist_ok=True)
    joblib.dump(regression_models, os.path.join(pair_dir, "regression_models.joblib"))
    joblib.dump(clf, os.path.join(pair_dir, "classifier.joblib"))
    meta = {"pair": pair, "flow": flow_name, "config_hash": CONFIG_HASH}
    if extra_meta:
        meta.update(extra_meta)
    joblib.dump(meta, os.path.join(pair_dir, "meta.joblib"))
    print(f"  [Lưu model] → {pair_dir}")


def load_models(pair):
    """
    Tải lại model đã lưu. Trả về (regression_models, clf, meta) hoặc (None, None, {}) nếu chưa có.
    """
    slug = _pair_slug(pair)
    pair_dir = os.path.join(MODELS_DIR, slug)
    reg_path  = os.path.join(pair_dir, "regression_models.joblib")
    clf_path  = os.path.join(pair_dir, "classifier.joblib")
    meta_path = os.path.join(pair_dir, "meta.joblib")
    if os.path.exists(reg_path) and os.path.exists(clf_path):
        print(f"  [Load model] ← {pair_dir}")
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        return joblib.load(reg_path), joblib.load(clf_path), meta
    return None, None, {}


# ── Hàm chính ───────────────────────────────────────────────────────────
def run_multi_pipeline(df, pairs, t_col_1, t_col_2, flow_name, is_dest_flow):
    """
    Pipeline đầy đủ: DBSCAN → Train/Test → Regression per cluster → Inference → Evaluate
    """
    all_results = []       # Tổng hợp metrics cho CSV
    flow_pair_reports = [] # Chi tiết cho HTML

    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"    PAIR: {pair}")
        print(f"{'='*70}")

        sub = df.loc[df["pair"] == pair, :].copy()
        sub = sub.dropna(subset=[t_col_1, t_col_2]).copy()

        # L\u1ecdc time ngo\u1ea1i lai
        if "time" in sub.columns:
            sub = sub[((sub['time'] >= 0) & (sub['time'] <= 100))].copy()
            if not sub.empty:
                q1 = sub['time'].quantile(0.01)
                q3 = sub['time'].quantile(0.99)
                sub = sub[((sub['time'] >= q1) & (sub['time'] <= q3))]

        if sub.empty or len(sub) < 200:
            print('  Kh\u00f4ng \u0111\u1ee7 d\u1eef li\u1ec7u --> B\u1ecf qua')
            continue

        # T\u00ednh y_offset v\u00e0 departure_hour
        dt1 = pd.to_datetime(sub[t_col_1])
        dt2 = pd.to_datetime(sub[t_col_2])
        departure_date = dt1.dt.normalize()
        sub['y_offset'] = (dt2 - departure_date).dt.total_seconds() / 3600.0
        sub['departure_hour_numeric'] = dt1.dt.hour + dt1.dt.minute / 60.0
        sub['departure_hour'] = dt1.dt.hour.fillna(-1).astype(int).astype(str) + 'h'
        sub.loc[sub['departure_hour'] == "-1h", 'departure_hour'] = "N/A"
        sub = sub[((sub['y_offset'] > 0) & (sub['y_offset'] < 48))].copy()

        if sub.empty or len(sub) < 200:
            print('  Kh\u00f4ng \u0111\u1ee7 d\u1eef li\u1ec7u sau l\u1ecdc --> B\u1ecf qua')
            continue

        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=42)

        # \u2550\u2550 B\u01b0\u1edbc A: T\u00e1ch Test TR\u01af\u1edaC DBSCAN \u2550\u2550
        n_sub = len(sub)
        idx_all = np.arange(n_sub)
        idx_trainval_raw, idx_test_raw = train_test_split(idx_all, test_size=0.2, random_state=42)
        sub_trainval = sub.iloc[idx_trainval_raw].copy().reset_index(drop=True)
        sub_test     = sub.iloc[idx_test_raw].copy().reset_index(drop=True)

        # ══ Bước B: DBSCAN chỉ trên trainval ══
        y_tv_raw = sub_trainval['y_offset'].values.reshape(-1, 1)
        dbscan   = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        sub_trainval['cluster'] = dbscan.fit_predict(y_tv_raw)

        valid_mask_tv = sub_trainval['cluster'] != -1
        if valid_mask_tv.any():
            cluster_means = sub_trainval.loc[valid_mask_tv].groupby('cluster')['y_offset'].mean()
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_mapping = {old_c: new_c for new_c, old_c in enumerate(sorted_clusters)}
            cluster_mapping[-1] = -1
            sub_trainval['cluster'] = sub_trainval['cluster'].map(cluster_mapping)

        sub_tv_clean = sub_trainval[sub_trainval['cluster'] != -1].copy().reset_index(drop=True)
        n_cluster = sub_tv_clean['cluster'].nunique()
        if n_cluster < 2:
            print("  Không đủ số lượng cụm. Bỏ qua.")
            continue
        print(f"  Số mẫu trainval hợp lệ {len(sub_tv_clean):,} | Số cụm {n_cluster}")

        cluster_counts  = sub_tv_clean['cluster'].value_counts()
        dominant_cluster = cluster_counts.idxmax()
        print(f"  Cụm lớn nhất: Cụm {dominant_cluster} ({cluster_counts[dominant_cluster]:,} bills)")

        # ══ Bước C: Feature Engineering ══
        def _build_X2(sub_df):
            X2 = pd.DataFrame(index=sub_df.index)
            X2['departure_hour_numeric'] = sub_df['departure_hour_numeric']
            _cats = ['VD_type', 'service', 'day_of_week', 'creation_hour', 'departure_hour']
            for c in _cats:
                if c in sub_df.columns:
                    X2 = pd.concat([X2, pd.get_dummies(sub_df[c], prefix=c)], axis=1)
            if is_dest_flow:
                if "origin_province" in sub_df.columns:
                    X2 = pd.concat([X2, pd.get_dummies(
                        transform_province(sub_df['origin_province']), prefix='origin_prov')], axis=1)
            else:
                if "destination_province" in sub_df.columns:
                    X2 = pd.concat([X2, pd.get_dummies(
                        transform_province(sub_df['destination_province']), prefix='dest_prov')], axis=1)
            w = sub_df['actual_weight'] if 'actual_weight' in sub_df.columns else pd.Series(0, index=sub_df.index)
            X2['actual_weight'] = w.fillna(w.median())
            return X2

        X1_tv  = sub_tv_clean[['departure_hour_numeric']].copy()
        X2_tv  = _build_X2(sub_tv_clean)
        tv_cols = X2_tv.columns.tolist()
        y_tv   = sub_tv_clean['y_offset'].values
        c_tv   = sub_tv_clean['cluster'].values

        X1_test = sub_test[['departure_hour_numeric']].copy()
        X2_test = _build_X2(sub_test).reindex(columns=tv_cols, fill_value=0)
        y_test  = sub_test['y_offset'].values

        # Lọc nhãn hợp lệ
        _counts = pd.Series(c_tv).value_counts()
        _valid_labels = _counts[_counts > 1].index
        _vmask = np.isin(c_tv, _valid_labels)
        X1_tv, X2_tv, y_tv, c_tv = X1_tv[_vmask], X2_tv[_vmask], y_tv[_vmask], c_tv[_vmask]
        if len(np.unique(c_tv)) < 2:
            print("  Không đủ nhãn hợp lệ. Bỏ qua.")
            continue

        # ══ Bước D: Split trainval → train (75%) / val (25%) ══
        idx_tv = np.arange(len(y_tv))
        idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=42, stratify=c_tv)
        X1_train, X1_val = X1_tv.iloc[idx_train], X1_tv.iloc[idx_val]
        X2_train, X2_val = X2_tv.iloc[idx_train], X2_tv.iloc[idx_val]
        y_train, y_val   = y_tv[idx_train], y_tv[idx_val]
        c_train, c_val   = c_tv[idx_train], c_tv[idx_val]
        print(f"  Split: TrainVal={len(y_tv):,} | Train={len(idx_train):,} | Val={len(idx_val):,} | Test={len(y_test):,}")

        # ══ Bước E: Load cache hoặc train mới ══
        final_reg_models, final_clf, cached_meta = load_models(pair)
        best_combo = cached_meta.get('best_combo') if cached_meta else None
        cluster_info_final = cached_meta.get('cluster_info') if cached_meta else None
        # Khôi phục val_scores từ cache (nếu có) để HTML report không mất bảng Val Selection
        all_val_scores = cached_meta.get('val_scores', []) if cached_meta else []

        # Lấy classifier metrics (cần thiết cả 2 nhánh cache hit và train mới)
        if final_reg_models is not None and final_clf is not None and best_combo is not None and cluster_info_final is not None:
            print("  [Cache hit] Dùng model đã lưu và best_combo sẵn có.")
            _cp_val   = final_clf.predict_proba(X2_val)
            _cpred_val= final_clf.classes_[np.argmax(_cp_val, axis=1)]
            clf_acc   = float(accuracy_score(c_val, _cpred_val))
            clf_f1    = float(f1_score(c_val, _cpred_val, average='weighted', zero_division=0))
            hybrid_ml_rate = float(np.mean(np.max(_cp_val, axis=1) >= HYBRID_THRESHOLD))
            if not all_val_scores:
                print("  [Cache hit] Cảnh báo: cache cũ không có val_scores → Val Selection Table sẽ trống cho cặp này.")
        else:
            # Model specs
            model_specs = [
                ("LR",   LinearRegression),
                ("RF",   lambda: RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
                ("LGBM", lambda: LGBMRegressor(n_estimators=100, max_depth=15, verbose=-1, random_state=42, n_jobs=-1)),
            ]

            # ── Phase 1: Train trên TRAIN (cho val selection) ──
            print(f"\n  [Phase 1/2] Training selection models trên Train ({len(y_train):,} bills)...")
            sel_reg = {}
            for clus in sorted(np.unique(c_train)):
                mask_c = c_train == clus
                y_c    = y_train[mask_c]
                for opt_name, opt_X in [("Opt1", X1_train), ("Opt2", X2_train)]:
                    X_c = opt_X.iloc[mask_c.nonzero()[0]]
                    for mn, mf in model_specs:
                        try:
                            m = mf() if callable(mf) and not isinstance(mf, type) else mf()
                            m.fit(X_c, y_c)
                            sel_reg[(clus, mn, opt_name)] = m
                        except Exception as e:
                            print(f"     Lỗi fit {mn} cụm {clus} {opt_name}: {e}")

            sel_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                             n_jobs=-1, max_depth=15, random_state=42)
            sel_clf.fit(X2_train, c_train)

            _cp_val    = sel_clf.predict_proba(X2_val)
            _cpred_val = sel_clf.classes_[np.argmax(_cp_val, axis=1)]
            clf_acc    = float(accuracy_score(c_val, _cpred_val))
            clf_f1     = float(f1_score(c_val, _cpred_val, average='weighted', zero_division=0))
            hybrid_ml_rate = float(np.mean(np.max(_cp_val, axis=1) >= HYBRID_THRESHOLD))
            print(f"  Classifier (Val) Acc={clf_acc:.4f}, W-F1={clf_f1:.4f}")

            # Cluster info từ train
            dep_train = X1_train['departure_hour_numeric'].values
            ci_sel = {}
            for cid in np.unique(c_train):
                dh = dep_train[c_train == cid]
                ci_sel[cid] = {'min': dh.min(), 'max': dh.max(), 'mean': dh.mean(), 'count': len(dh)}
            cbs_sel = sorted(ci_sel.keys(), key=lambda c: ci_sel[c]['count'], reverse=True)

            def _pick_sel(dep_hour):
                for c in cbs_sel:
                    if ci_sel[c]['min'] <= dep_hour <= ci_sel[c]['max']:
                        return c
                return min(cbs_sel, key=lambda c: abs(ci_sel[c]['mean'] - dep_hour))

            # ── Val Selection Loop ──
            _clf_max_v  = np.max(_cp_val, axis=1)
            _clf_pred_v = _cpred_val
            dep_hour_v  = X1_val['departure_hour_numeric'].values

            chv1 = np.array([_pick_sel(h) for h in dep_hour_v])
            chv2 = _clf_pred_v.copy()
            chv3 = np.where(_clf_max_v >= HYBRID_THRESHOLD, _clf_pred_v, chv1)

            print(f"\n  Val Selection: {3*2*3} combo trên val ({len(y_val):,} bills)...")
            for cs_name, ch_v in [
                ("Cách 1 (Dominant Cluster)", chv1),
                ("Cách 2 (ML Classifier)",    chv2),
                ("Cách 3 (Hybrid)",            chv3),
            ]:
                for fo_name, X_v_opt, opt_key in [
                    ("Option 1 (Hour)", X1_val, "Opt1"),
                    ("Option 2 (Full)", X2_val, "Opt2"),
                ]:
                    for reg_name in ["LR", "RF", "LGBM"]:
                        yp = np.full(len(y_val), np.nan)
                        for cid in np.unique(ch_v):
                            idx_cv = np.where(ch_v == cid)[0]
                            key = (cid, reg_name, opt_key)
                            if key not in sel_reg:
                                key = (dominant_cluster, reg_name, opt_key)
                            if key not in sel_reg:
                                continue
                            cm = c_train == cid
                            lo = y_train[cm].min() - 2 if cm.any() else -2
                            hi = y_train[cm].max() + 2 if cm.any() else 50
                            yp[idx_cv] = np.clip(sel_reg[key].predict(X_v_opt.iloc[idx_cv]), lo, hi)
                        vv = ~np.isnan(yp)
                        if vv.sum() == 0:
                            continue
                        all_val_scores.append({
                            "cs_name": cs_name, "fo_name": fo_name, "reg_name": reg_name,
                            "val_mae":  float(np.mean(np.abs(y_val[vv] - yp[vv]))),
                            "val_rmse": float(np.sqrt(np.mean((y_val[vv] - yp[vv])**2))),
                            "n_val": int(vv.sum()),
                        })

            all_val_scores.sort(key=lambda x: x['val_mae'])
            best_combo = all_val_scores[0] if all_val_scores else {
                "cs_name": "Cách 3 (Hybrid)", "fo_name": "Option 2 (Full)", "reg_name": "LGBM",
                "val_mae": np.nan, "val_rmse": np.nan, "n_val": 0,
            }
            print(f"  → Best (Val MAE={best_combo['val_mae']:.3f}h): "
                  f"{best_combo['cs_name']} | {best_combo['fo_name']} | {best_combo['reg_name']}")

            # ── Phase 2: Retrain trên TRAINVAL (80%) ──
            print(f"\n  [Phase 2/2] Retrain final models trên TrainVal ({len(y_tv):,} bills)...")
            final_reg_models = {}
            for clus in sorted(np.unique(c_tv)):
                mask_tv = c_tv == clus
                y_c = y_tv[mask_tv]
                for opt_name, opt_X in [("Opt1", X1_tv), ("Opt2", X2_tv)]:
                    X_c = opt_X.iloc[mask_tv.nonzero()[0]]
                    for mn, mf in model_specs:
                        try:
                            m = mf() if callable(mf) and not isinstance(mf, type) else mf()
                            m.fit(X_c, y_c)
                            final_reg_models[(clus, mn, opt_name)] = m
                        except Exception as e:
                            print(f"     Lỗi retrain {mn} cụm {clus} {opt_name}: {e}")
            print(f"  Đã retrain {len(final_reg_models)} final regression models")

            final_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                               n_jobs=-1, max_depth=15, random_state=42)
            final_clf.fit(X2_tv, c_tv)

            dep_tv = X1_tv['departure_hour_numeric'].values
            cluster_info_final = {}
            for cid in np.unique(c_tv):
                dh = dep_tv[c_tv == cid]
                cluster_info_final[cid] = {'min': dh.min(), 'max': dh.max(), 'mean': dh.mean(), 'count': len(dh)}

            save_models(pair, final_reg_models, final_clf, flow_name, extra_meta={
                'best_combo': best_combo, 'cluster_info': cluster_info_final,
                'dominant_cluster': dominant_cluster,
                'clf_acc': clf_acc, 'clf_f1': clf_f1, 'hybrid_ml_rate': hybrid_ml_rate,
                'val_scores': all_val_scores,
            })

        # ── Cluster info & pick function ──
        cbs_final = sorted(cluster_info_final.keys(),
                           key=lambda c: cluster_info_final[c]['count'], reverse=True)

        def pick_cluster_by_hour(dep_hour):
            for c in cbs_final:
                if cluster_info_final[c]['min'] <= dep_hour <= cluster_info_final[c]['max']:
                    return c
            return min(cbs_final, key=lambda c: abs(cluster_info_final[c]['mean'] - dep_hour))

        slot_cluster_map = {
            f"Cụm {c}": f"dep=[{cluster_info_final[c]['min']:.1f}h–{cluster_info_final[c]['max']:.1f}h] mean={cluster_info_final[c]['mean']:.1f}h"
            for c in sorted(cluster_info_final.keys())
        }
        info_str = " | ".join([
            f"Cụm {c}: [{cluster_info_final[c]['min']:.1f}h, {cluster_info_final[c]['max']:.1f}h] n={cluster_info_final[c]['count']}"
            for c in sorted(cluster_info_final.keys())
        ])
        print(f"  Cluster info (TrainVal): {info_str}")

        # ══ TEST ONCE với best_combo ══
        print(f"\n  [Test ONCE] {best_combo['cs_name']} | {best_combo['fo_name']} | {best_combo['reg_name']}")

        _cptest  = final_clf.predict_proba(X2_test)
        _cpred_t = final_clf.classes_[np.argmax(_cptest, axis=1)]
        _cmax_t  = np.max(_cptest, axis=1)
        dep_h_t  = X1_test['departure_hour_numeric'].values

        ch_t1 = np.array([pick_cluster_by_hour(h) for h in dep_h_t])
        ch_t2 = _cpred_t.copy()
        ch_t3 = np.where(_cmax_t >= HYBRID_THRESHOLD, _cpred_t, ch_t1)
        _cs_test = {"C\u00e1ch 1 (Dominant Cluster)": ch_t1, "C\u00e1ch 2 (ML Classifier)": ch_t2, "C\u00e1ch 3 (Hybrid)": ch_t3}
        _fo_test = {"Option 1 (Hour)": (X1_test, "Opt1"), "Option 2 (Full)": (X2_test, "Opt2")}
        opt_key_t = _fo_test[best_combo['fo_name']][1]
        X_opt_t   = _fo_test[best_combo['fo_name']][0]
        chosen_t  = _cs_test[best_combo['cs_name']]

        y_pred_test = np.full(len(y_test), np.nan)
        for cid in np.unique(chosen_t):
            idx_c = np.where(chosen_t == cid)[0]
            key = (cid, best_combo['reg_name'], opt_key_t)
            if key not in final_reg_models:
                key = (dominant_cluster, best_combo['reg_name'], opt_key_t)
            if key not in final_reg_models:
                continue
            cm = c_tv == cid
            lo = y_tv[cm].min() - 2 if cm.any() else -2
            hi = y_tv[cm].max() + 2 if cm.any() else 50
            y_pred_test[idx_c] = np.clip(final_reg_models[key].predict(X_opt_t.iloc[idx_c]), lo, hi)

        vt = ~np.isnan(y_pred_test)
        test_metrics = compute_regression_metrics(y_test[vt], y_pred_test[vt]) if vt.sum() > 0 else \
                       {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
        print(f"  [Test] MAE={test_metrics['MAE']:.3f}h | RMSE={test_metrics['RMSE']:.3f}h | "
              f"R\u00b2={test_metrics['R2']:.4f} | MAPE={test_metrics['MAPE']:.1f}% (N={int(vt.sum())})")

        # Train diagnostic
        _cptrn    = final_clf.predict_proba(X2_train)
        _cpred_trn= final_clf.classes_[np.argmax(_cptrn, axis=1)]
        _cmax_trn = np.max(_cptrn, axis=1)
        dep_h_trn = X1_train['departure_hour_numeric'].values
        ch_trn1 = np.array([pick_cluster_by_hour(h) for h in dep_h_trn])
        ch_trn2 = _cpred_trn.copy()
        ch_trn3 = np.where(_cmax_trn >= HYBRID_THRESHOLD, _cpred_trn, ch_trn1)
        _cs_trn = {"C\u00e1ch 1 (Dominant Cluster)": ch_trn1, "C\u00e1ch 2 (ML Classifier)": ch_trn2, "C\u00e1ch 3 (Hybrid)": ch_trn3}
        _fo_trn = {"Option 1 (Hour)": (X1_train, "Opt1"), "Option 2 (Full)": (X2_train, "Opt2")}
        X_opt_trn   = _fo_trn[best_combo['fo_name']][0]
        opt_key_trn = _fo_trn[best_combo['fo_name']][1]
        chosen_trn  = _cs_trn[best_combo['cs_name']]

        y_pred_trn = np.full(len(y_train), np.nan)
        for cid in np.unique(chosen_trn):
            idx_c = np.where(chosen_trn == cid)[0]
            key = (cid, best_combo['reg_name'], opt_key_trn)
            if key not in final_reg_models:
                key = (dominant_cluster, best_combo['reg_name'], opt_key_trn)
            if key not in final_reg_models:
                continue
            cm = c_tv == cid
            lo = y_tv[cm].min() - 2 if cm.any() else -2
            hi = y_tv[cm].max() + 2 if cm.any() else 50
            y_pred_trn[idx_c] = np.clip(final_reg_models[key].predict(X_opt_trn.iloc[idx_c]), lo, hi)
        vtrn = ~np.isnan(y_pred_trn)
        train_metrics = compute_regression_metrics(y_train[vtrn], y_pred_trn[vtrn]) if vtrn.sum() > 0 else \
                        {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}

        # L\u01b0u k\u1ebft qu\u1ea3
        combo_info = {
            "cluster_selection": best_combo['cs_name'],
            "feature_option":    best_combo['fo_name'],
            "regression_model":  best_combo['reg_name'],
        }
        for split_name, met, n_s in [
            ("Train (diagnostic)", train_metrics, int(vtrn.sum())),
            ("Test",               test_metrics,  int(vt.sum())),
        ]:
            all_results.append({
                "data_split": split_name, "flow": flow_name, "pair": pair,
                **combo_info, "n_samples": n_s, **met,
            })

        # Th\u1ed1ng k\u00ea cluster t\u1eeb trainval
        cluster_stats = []
        for c in sorted(sub_tv_clean['cluster'].unique()):
            c_data = sub_tv_clean[sub_tv_clean['cluster'] == c]['y_offset']
            cluster_stats.append({"cluster": c, "n": len(c_data),
                                   "mean": f"{c_data.mean():.1f}h",
                                   "median": f"{c_data.median():.1f}h",
                                   "std": f"{c_data.std():.1f}h"})

        flow_pair_reports.append({
            "pair": pair,
            "n_samples": len(sub_tv_clean),
            "n_test": len(y_test),
            "n_clusters": n_cluster,
            "dominant_cluster": dominant_cluster,
            "cluster_stats": cluster_stats,
            "val_scores": all_val_scores,
            "best_combo": best_combo,
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "slot_cluster_map": slot_cluster_map,
            "clf_acc": clf_acc,
            "clf_f1": clf_f1,
            "hybrid_ml_rate": hybrid_ml_rate,
        })

    return all_results, flow_pair_reports


# ── HTML Report ──────────────────────────────────────────────────────────
def build_html_report(results_by_flow, summary_df):
    html = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Multi-Strategy Regression Pipeline</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; color: #1e293b; padding: 24px; }
  h1 { text-align: center; font-size: 24px; margin-bottom: 8px; }
  .subtitle { text-align: center; color: #64748b; font-size: 14px; margin-bottom: 24px; }
  .flow-header { background: linear-gradient(135deg, #1e40af, #3b82f6); color: white;
                   padding: 14px 24px; border-radius: 10px; font-size: 18px; font-weight: 700;
                   margin: 32px 0 16px; box-shadow: 0 2px 8px rgba(30,64,175,0.3); }
  .summary-card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 24px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .summary-card h2 { font-size: 16px; margin-bottom: 12px; color: #334155; }
  .note-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px;
                 margin-bottom: 24px; font-size: 13px; line-height: 1.6; color: #334155; }
  .note-card h2 { font-size: 16px; margin-bottom: 12px; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
  .note-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .note-section { margin-bottom: 12px; }
  .note-section b { color: #1e293b; display: block; margin-bottom: 4px; font-weight: 600; }
  .avg-card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 24px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .avg-card h2 { font-size: 16px; margin-bottom: 12px; color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
  .avg-grid { display: flex; flex-direction: column; gap: 32px; }
  .avg-row-container { display: flex; flex-direction: column; gap: 16px; 
                       background: #f8fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; }
  .avg-table-wrap { overflow-x: auto; background: white; border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
  .avg-chart-wrap { width: 100%; min-height: 320px; background: white; border-radius: 8px; 
                    border: 1px solid #e2e8f0; padding: 12px; }
  .avg-table-wrap h3 { font-size: 14px; font-weight: 700; color: #1e293b; margin: 0; padding: 12px 16px;
                       background: #f1f5f9; border-bottom: 1px solid #e2e8f0; border-radius: 8px 8px 0 0; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #ffffff; padding: 12px 16px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; color: #475569; user-select: none; white-space: nowrap; }
  th.sortable:hover { background: #f1f5f9; cursor: pointer; }
  td { padding: 10px 16px; border-bottom: 1px solid #f1f5f9; }
  tr:nth-child(even) td { background: #fafafa; }
  tr:hover td { background: #eff6ff; }
  .pair-card { background: white; border-radius: 10px; margin-bottom: 24px; overflow: hidden;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .pair-header { background: #f8fafc; padding: 14px 20px; border-bottom: 1px solid #e2e8f0;
                   display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  .pair-title { font-size: 15px; font-weight: 700; color: #1e293b; }
  .pair-meta { display: flex; gap: 10px; flex-wrap: wrap; }
  .meta-badge { background: #eff6ff; color: #1d4ed8; padding: 4px 10px; border-radius: 6px;
                  font-size: 11px; font-weight: 600; }
  .meta-badge.green { background: #ecfdf5; color: #065f46; }
  .pair-body { padding: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-container { min-height: 400px; }
  .section-title { font-size: 14px; font-weight: 700; color: #334155; margin: 16px 0 8px; }
  .highlight { background: #fef3c7; padding: 2px 6px; border-radius: 3px; font-weight: 700; }
  td.best { background: #d1fae5; font-weight: 700; color: #065f46; }
</style>
</head><body>
<h1>Multi-Strategy Regression Pipeline</h1>
<p class="subtitle">Train → Val selection (3 chiến lược chọn cụm × 2 feature options × 3 mô hình hồi quy) → Retrain trainval → Test ONCE</p>

<div class="note-card">
  <h2>Hướng dẫn đọc báo cáo</h2>
  <div class="note-grid">
    <div>
      <div class="note-section"><b>Pipeline (mỗi cặp kho)</b>
        Bước 1: Tách Test 20% TRƯỚC (không đụng tới cho đến bước cuối) →
        Bước 2: DBSCAN phân cụm y_offset CHỈ trên trainval (80%) →
        Bước 3: Split trainval → Train 75% / Val 25% →
        Bước 4: Train Regression per cluster trên Train (LR/RF/LGBM × Opt1/Opt2) →
        Bước 5: Val Selection (chọn best combo theo Val MAE thấp nhất) →
        Bước 6: Retrain final models trên toàn bộ trainval với best combo →
        Bước 7: Test ONCE – chỉ chạy đúng 1 lần trên test với best combo
      </div>
      <div class="note-section"><b>Chiến lược chọn cụm</b>
        <u>Cách 1 (Dominant Cluster)</u>: Với mỗi khung giờ, dùng cụm chiếm đa số trong training – đơn giản, ổn định<br>
        <u>Cách 2 (ML Classifier)</u>: RandomForest tự phân loại cụm từ full features – linh hoạt hơn<br>
        <u>Cách 3 (Hybrid)</u>: Dùng ML nếu xác suất ≥ ngưỡng (mặc định 60%), còn lại dùng Cách 1
      </div>
      <div class="note-section"><b>Lưu ý quan trọng</b>
        Test hoàn toàn không tham gia DBSCAN và không có nhãn cụm thật, nên
        không tồn tại eval scope "Dominant (Correct)" mang tính oracle trên test.
        Cụm cho test được suy ra bằng <u>pick_cluster_by_hour</u> (inference-only).
        "Train (diagnostic)" chỉ để so sánh phát hiện overfit, không dùng để chọn model.
      </div>
    </div>
    <div>
      <div class="note-section"><b>Feature Options</b>
        <u>Option 1 (Hour)</u>: Chỉ dùng giờ xuất phát (departure_hour_numeric)<br>
        <u>Option 2 (Full)</u>: Giờ + loại VD + dịch vụ + ngày trong tuần + giờ tạo đơn + tỉnh + trọng lượng
      </div>
      <div class="note-section"><b>Metrics</b>
        <u>MAE</u>: Sai số trung bình (giờ) – nhỏ hơn = tốt hơn<br>
        <u>RMSE</u>: Nhạy với outlier (giờ) – nhỏ hơn = tốt hơn<br>
        <u>R²</u>: Mức độ giải thích biến động – càng gần 1 càng tốt (âm = tệ hơn đoán trung bình)<br>
        <u>MAPE</u>: Sai số phần trăm – nhỏ hơn = tốt hơn<br>
        <u>MAPE</u>: Sai số phần trăm – nhỏ hơn = tốt hơn
      </div>
    </div>
  </div>
</div>
"""

    # ── Summary Table + Avg Tables ──
    if not summary_df.empty:
        # Lọc bỏ outlier cực đoan trước khi tính avg (|MAE| > 1000 coi là artifact)
        df_valid = summary_df[summary_df['MAE'].abs() < 1000].copy()

        def _avg_table(df, group_col, label, idx):
            """Tạo HTML table trung bình metrics theo 1 cột nhóm (phân tách thêm theo luồng và data_split)."""
            cols_to_group = [group_col]
            if group_col != "flow":
                cols_to_group.append("flow")
            if group_col != "data_split" and "data_split" in df.columns:
                cols_to_group.append("data_split")
                
            grp = df.groupby(cols_to_group)[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
            # Sort để dữ liệu nhóm liền kề nhau dễ nhìn
            grp = grp.sort_values(cols_to_group)
            
            header_map = {
                "flow": "Luồng",
                "data_split": "Split",
                "cluster_selection": "Chọn Cụm", 
                "feature_option": "Features", 
                "regression_model": "Model"
            }
            
            th_html = "".join([f"<th>{header_map.get(col, col)}</th>" for col in cols_to_group])
            
            grp = grp.reset_index(drop=True)

            rows = ""
            x_labels = []
            y_maes = []
            hover_texts = []

            for row_idx, r in grp.iterrows():
                row_num = row_idx + 1
                td_html = "".join([
                    f"<td><b>{r[col]}</b></td>" if col == group_col else f"<td>{r[col]}</td>"
                    for col in cols_to_group
                ])
                rows += (f"<tr><td style='color:#64748b;font-weight:600;'>#{row_num}</td>{td_html}"
                         f"<td>{r['MAE']:.3f}</td><td>{r['RMSE']:.3f}</td>"
                         f"<td>{r['R2']:.4f}</td><td>{r['MAPE']:.1f}%</td></tr>")
                x_labels.append(f"#{row_num}")
                y_maes.append(r["MAE"])
                detail = " | ".join([f"{header_map.get(col, col)}: {r[col]}" for col in cols_to_group])
                hover_texts.append(f"#{row_num} {detail}<br>MAE: {r['MAE']:.3f}h")

            table_html = (
                '<div class="avg-table-wrap">'
                f'<h3>{label}</h3>'
                '<table>'
                f'<thead><tr><th>#</th>{th_html}<th>Avg MAE</th><th>Avg RMSE</th><th>Avg R²</th><th>Avg MAPE</th></tr></thead>'
                f'<tbody>{rows}</tbody>'
                '</table></div>'
            )

            # ── Vẽ histogram phân phối MAE theo từng index ──
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import math
            
            n_groups = len(grp)
            COLS_PER_ROW = 3
            n_rows_plot = math.ceil(n_groups / COLS_PER_ROW)
            n_cols_plot = min(n_groups, COLS_PER_ROW)
            
            subplot_titles = [f"#{i+1}" for i in range(n_groups)]
            v_spacing = min(0.15, 1.0 / max(n_rows_plot - 1, 1) - 0.01) if n_rows_plot > 1 else 0.1
            fig = make_subplots(
                rows=n_rows_plot, cols=n_cols_plot,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.06,
                vertical_spacing=v_spacing,
            )
            
            import numpy as np
            
            for row_idx, (_, r) in enumerate(grp.iterrows()):
                # Lọc raw data khớp với điều kiện của dòng này
                mask = pd.Series([True] * len(df), index=df.index)
                for c in cols_to_group:
                    if c in df.columns:
                        mask = mask & (df[c] == r[c])
                raw_mae_vals = df.loc[mask, 'MAE'].dropna()
                raw_mae_vals = raw_mae_vals[raw_mae_vals.abs() < 1000]  # loại outlier
                raw_mae_list = raw_mae_vals.tolist()  # buộc dùng list để tránh binary encoding khi serialize JSON
                
                plot_row = row_idx // COLS_PER_ROW + 1
                plot_col = row_idx % COLS_PER_ROW + 1
                color = "#3B82F6"
                
                detail_label = " | ".join([f"{r[c]}" for c in cols_to_group])
                
                fig.add_trace(go.Histogram(
                    x=raw_mae_list,
                    nbinsx=15,
                    name=f"#{row_idx+1}",
                    marker_color=color,
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate=f"#{row_idx+1} {detail_label}<br>MAE: %{{x:.2f}}h<br>Count: %{{y}}<extra></extra>",
                ), row=plot_row, col=plot_col)
                if len(raw_mae_list) > 0:
                    mean_val = float(raw_mae_vals.mean())
                    fig.add_vline(
                        x=mean_val,
                        line_width=2,
                        line_dash="dash",
                        line_color="#EF4444",
                        row=plot_row,
                        col=plot_col,
                    )
                    # Annotation hiển thị giá trị mean phía trên đường
                    fig.add_annotation(
                        x=mean_val,
                        y=1.0,
                        xref=f"x{ (plot_row - 1) * COLS_PER_ROW + plot_col if (plot_row, plot_col) != (1, 1) else '' }",
                        yref=f"y{ (plot_row - 1) * COLS_PER_ROW + plot_col if (plot_row, plot_col) != (1, 1) else '' } domain",
                        text=f"Mean: {mean_val:.2f}h",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10, color="#EF4444"),
                        row=plot_row,
                        col=plot_col,
                    )
            chart_height = 320 * n_rows_plot
            fig.update_layout(
                height=chart_height,
                bargap=0.1,
                margin=dict(t=40, b=30, l=45, r=15),
                plot_bgcolor="#FAFAFA",
                paper_bgcolor="#FFFFFF",
            )
            fig.update_xaxes(title_text="MAE (h)")
            # fig.update_yaxes(title_text="Count")
            
            chart_id = f"avg_chart_{idx}"
            var_name = f"fd{idx}"
            fig_json = fig.to_json()
            script_tag = (
                f'<div class="avg-chart-wrap" id="{chart_id}"></div>\n'
                f'<script>var {var_name}='
                + fig_json
                + f';Plotly.newPlot("{chart_id}",{var_name}.data,{var_name}.layout,{{responsive:true}});</script>\n'
            )
            return f'<div class="avg-row-container">{table_html}' + script_tag + '</div>'



        avg_html = ""
        for i, (col, lbl) in enumerate([
            ("flow",               "Theo Luồng"),
            ("data_split",         "Theo Tập (Train/Val/Test)"),
            ("cluster_selection",  "Theo Chiến lược chọn cụm"),
            ("feature_option",     "Theo Feature Option"),
            ("regression_model",   "Theo Regression Model"),
        ]):
            if col in df_valid.columns:
                avg_html += _avg_table(df_valid, col, lbl, i)

        html += f"""
<div class="avg-card">
  <h2>Trung bình hiệu suất theo từng nhóm
    <span style="font-size:11px;font-weight:400;color:#64748b;"> (loại bỏ outlier MAE &gt; 1000h)</span>
  </h2>
  <div class="avg-grid">{avg_html}</div>
</div>
"""

        # ── Bảng tổng hợp tổ hợp chiến lược ──
        combo_cols = ['data_split', 'flow', 'cluster_selection', 'feature_option', 'regression_model']
        combo_cols = [c for c in combo_cols if c in df_valid.columns]  # bảo vệ nếu không có data_split
        combo_grp = df_valid.groupby(combo_cols)[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
        combo_grp = combo_grp.sort_values(combo_cols)
        
        combo_rows = ""
        for _, r in combo_grp.iterrows():
            split_td = f"<td>{r['data_split']}</td>" if 'data_split' in combo_cols else ""
            combo_rows += (f"<tr>{split_td}<td>{r['flow']}</td><td>{r['cluster_selection']}</td>"
                           f"<td>{r['feature_option']}</td><td>{r['regression_model']}</td>"
                           f"<td><b>{r['MAE']:.3f}</b></td>"
                           f"<td>{r['RMSE']:.3f}</td><td>{r['R2']:.4f}</td><td>{r['MAPE']:.1f}%</td></tr>")

        table_html = f"""
  <table id="comboSummaryTable">
    <thead><tr>
      {'<th class="sortable" onclick="sortTable(0, \'comboSummaryTable\')">Split ↕</th>' if 'data_split' in combo_cols else ''}
      <th class="sortable" onclick="sortTable({'1' if 'data_split' in combo_cols else '0'}, 'comboSummaryTable')">Luồng ↕</th>
      <th class="sortable" onclick="sortTable({'2' if 'data_split' in combo_cols else '1'}, 'comboSummaryTable')">Chọn Cụm ↕</th>
      <th class="sortable" onclick="sortTable({'3' if 'data_split' in combo_cols else '2'}, 'comboSummaryTable')">Features ↕</th>
      <th class="sortable" onclick="sortTable({'4' if 'data_split' in combo_cols else '3'}, 'comboSummaryTable')">Regression ↕</th>
      <th class="sortable" onclick="sortTable({'5' if 'data_split' in combo_cols else '4'}, 'comboSummaryTable')">Avg MAE (h) ↕</th>
      <th class="sortable" onclick="sortTable({'6' if 'data_split' in combo_cols else '5'}, 'comboSummaryTable')">Avg RMSE (h) ↕</th>
      <th class="sortable" onclick="sortTable({'7' if 'data_split' in combo_cols else '6'}, 'comboSummaryTable')">Avg R² ↕</th>
      <th class="sortable" onclick="sortTable({'8' if 'data_split' in combo_cols else '7'}, 'comboSummaryTable')">Avg MAPE ↕</th>
    </tr></thead>
    <tbody>{combo_rows}</tbody>
  </table>
"""

        html += f"""
<div class="summary-card">
  <h2>Trung bình toàn tổ hợp (Split → Flow → Cụm → Feature → Model)</h2>
  <div class="avg-row-container">
    <div class="avg-table-wrap" style="width: 100%;">
      {table_html}
    </div>
  </div>
</div>
"""

        summary_rows = ""
        for _, row in summary_df.sort_values("MAE", ascending=True).iterrows():
            mae_str  = f"{row['MAE']:.2f}"  if not pd.isna(row['MAE'])  else "N/A"
            rmse_str = f"{row['RMSE']:.2f}" if not pd.isna(row['RMSE']) else "N/A"
            r2_str   = f"{row['R2']:.4f}"   if not pd.isna(row['R2'])   else "N/A"
            mape_str = f"{row['MAPE']:.1f}%" if not pd.isna(row['MAPE']) else "N/A"
            split_td = f"<td>{row['data_split']}</td>" if 'data_split' in row.index else ""
            n_val = int(row['n_samples']) if 'n_samples' in row.index else int(row.get('n_samples_test', 0))
            summary_rows += f"""<tr>
                {split_td}
                <td><b>{row['pair']}</b></td>
                <td>{row['flow']}</td>
                <td>{row['cluster_selection']}</td>
                <td>{row['feature_option']}</td>
                <td>{row['regression_model']}</td>
                <td>{n_val:,}</td>
                <td><b>{mae_str}</b></td>
                <td>{rmse_str}</td>
                <td>{r2_str}</td>
                <td>{mape_str}</td>
            </tr>"""

        html += f"""
<div class="summary-card">
  <h2>Bảng tổng hợp so sánh tất cả tổ hợp chiến lược</h2>
  <table id="summaryTable">
    <thead><tr>
      {'<th class="sortable" onclick="sortTable(0)">Split ↕</th>' if 'data_split' in summary_df.columns else ''}
      <th class="sortable" onclick="sortTable({'1' if 'data_split' in summary_df.columns else '0'})">Cặp Kho ↕</th>
      <th class="sortable" onclick="sortTable({'2' if 'data_split' in summary_df.columns else '1'})">Luồng ↕</th>
      <th class="sortable" onclick="sortTable({'3' if 'data_split' in summary_df.columns else '2'})">Chọn Cụm ↕</th>
      <th class="sortable" onclick="sortTable({'4' if 'data_split' in summary_df.columns else '3'})">Features ↕</th>
      <th class="sortable" onclick="sortTable({'5' if 'data_split' in summary_df.columns else '4'})">Regression ↕</th>
      <th class="sortable" onclick="sortTable({'6' if 'data_split' in summary_df.columns else '5'})">N Test ↕</th>
      <th class="sortable" onclick="sortTable({'7' if 'data_split' in summary_df.columns else '6'})">MAE (h) ↕</th>
      <th class="sortable" onclick="sortTable({'8' if 'data_split' in summary_df.columns else '7'})">RMSE (h) ↕</th>
      <th class="sortable" onclick="sortTable({'9' if 'data_split' in summary_df.columns else '8'})">R² ↕</th>
      <th class="sortable" onclick="sortTable({'10' if 'data_split' in summary_df.columns else '9'})">MAPE ↕</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>
"""

    chart_idx = 0

    # ── Per-pair reports ──
    for flow_name, pair_reports in results_by_flow:
        html += f'<div class="flow-header"> {flow_name}</div>\n'

        for rep in pair_reports:
            pair = rep["pair"]
            val_scores = rep.get("val_scores", [])
            best_combo = rep.get("best_combo", {})
            c_stats = rep.get("cluster_stats", [])

            if not val_scores:
                continue

            # ── Cluster Stats Table ──
            cluster_rows = ""
            for cs in c_stats:
                cluster_rows += f"<tr><td><b>Cụm {cs['cluster']}</b></td><td>{cs['n']:,}</td><td>{cs['mean']}</td><td>{cs['median']}</td><td>{cs['std']}</td></tr>"

            # ── Combo Detail Table ──
            combo_rows = ""
            def _f(v, fmt=".2f"): return format(v, fmt) if not (isinstance(v, float) and (v != v)) else "N/A"
            for vs in val_scores:
                is_best = (vs['cs_name'] == best_combo.get('cs_name') and 
                           vs['fo_name'] == best_combo.get('fo_name') and 
                           vs['reg_name'] == best_combo.get('reg_name'))
                bg_color = "#d1fae5" if is_best else "transparent"
                fw = "bold" if is_best else "normal"
                combo_rows += f"""<tr style='background:{bg_color}; font-weight:{fw}'>
                    <td>{vs['cs_name']}</td>
                    <td>{vs['fo_name']}</td>
                    <td>{vs['reg_name']}</td>
                    <td>{_f(vs['val_mae'])}</td><td>{_f(vs['val_rmse'])}</td>
                    <td style='color:#64748b;font-size:11px'>{vs['n_val']:,}</td>
                </tr>"""

            # ── Bar Chart: MAE comparison on Val ──
            colors_map = {"LR": "#3B82F6", "RF": "#10B981", "LGBM": "#F59E0B"}
            fig = go.Figure()
            for reg_name in ["LR", "RF", "LGBM"]:
                x_vals = []
                y_vals = []
                for vs in val_scores:
                    if vs['reg_name'] == reg_name:
                        label = f"{vs['cs_name']}<br>{vs['fo_name']}"
                        x_vals.append(label)
                        y_vals.append(vs['val_mae'])
                if x_vals:
                    fig.add_trace(go.Bar(
                        x=x_vals, y=y_vals,
                        name=reg_name,
                        marker_color=colors_map.get(reg_name, "#888"),
                        hovertemplate=f"{reg_name}<br>%{{x}}<br>Val MAE: %{{y:.2f}}h<extra></extra>"
                    ))

            fig.update_layout(
                title="MAE trên tập Validation (để chọn Best Combo)",
                barmode="group",
                height=450, width=1100,
                plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
                font=dict(family="Segoe UI, Arial", size=10),
                margin=dict(t=60, b=120, l=50, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                yaxis_title="Val MAE (giờ)"
            )

            id_chart = f"chart_{chart_idx}"
            slot_info = " | ".join([f"{k}: {v}" for k, v in rep['slot_cluster_map'].items()])
            fig_json = fig.to_json()
            
            # ── Final Test Metrics display ──
            tm = rep.get('test_metrics', {})
            trm = rep.get('train_metrics', {})
            
            html += f"""
<div class="pair-card">
  <div class="pair-header">
    <span class="pair-title">{pair}</span>
    <div class="pair-meta">
      <span class="meta-badge">{rep.get('n_samples', 0):,} trainval samples</span>
      <span class="meta-badge">{rep.get('n_test', 0):,} test samples</span>
      <span class="meta-badge">{rep.get('n_clusters', 0)} cụm</span>
      <span class="meta-badge green">Dominant: Cụm {rep.get('dominant_cluster', -1)}</span>
      <span class="meta-badge">Clf Acc={rep.get('clf_acc', 0):.3f}</span>
      <span class="meta-badge">Hybrid ML: {rep.get('hybrid_ml_rate', 0)*100:.1f}%</span>
    </div>
  </div>
  <div class="pair-body">
    <div style="margin-bottom: 20px; padding: 15px; background: #e0f2fe; border-left: 4px solid #0284c7; border-radius: 4px;">
      <h3 style="margin-bottom: 10px; color: #0369a1;">Kết quả Final Test (chỉ chạy 1 lần) trên {rep.get('n_test', 0):,} bills</h3>
      <p><b>Best Combo:</b> {best_combo.get('cs_name')} | {best_combo.get('fo_name')} | {best_combo.get('reg_name')}</p>
      <div style="display: flex; gap: 40px; margin-top: 10px;">
        <div>
          <b>Test Metrics:</b> MAE={_f(tm.get('MAE'))}h, RMSE={_f(tm.get('RMSE'))}h, R²={_f(tm.get('R2'), '.4f')}, MAPE={_f(tm.get('MAPE'), '.1f')}%
        </div>
        <div>
          <b>Train(Diag) Metrics:</b> MAE={_f(trm.get('MAE'))}h, RMSE={_f(trm.get('RMSE'))}h, R²={_f(trm.get('R2'), '.4f')}, MAPE={_f(trm.get('MAPE'), '.1f')}%
        </div>
      </div>
    </div>
    
    <p class="section-title"> Khung giờ → Cụm dominant (Cách 1): {slot_info}</p>

    <div class="grid-2">
      <div>
        <p class="section-title">Thống kê Cụm</p>
        <table>
          <thead><tr><th>Cụm</th><th>Số bill</th><th>Mean</th><th>Median</th><th>Std</th></tr></thead>
          <tbody>{cluster_rows}</tbody>
        </table>
      </div>
      <div id="{id_chart}" class="chart-container"></div>
    </div>

    <p class="section-title"> Chi tiết Val Selection (Tuning)</p>
    <table>
      <thead><tr>
        <th>Chọn Cụm</th><th>Features</th><th>Regression</th>
        <th>Val MAE</th><th>Val RMSE</th>
        <th>N Val</th>
      </tr></thead>
      <tbody>{combo_rows}</tbody>
    </table>
  </div>
</div>
<script>
var d_{chart_idx} = {fig_json};
Plotly.newPlot('{id_chart}', d_{chart_idx}.data, d_{chart_idx}.layout, {{responsive: true}});
</script>
"""
            chart_idx += 1

    # Sort table script
    html += """
<script>
function sortTable(n, tableId) {
  if (!tableId) tableId = "summaryTable";
  var table = document.getElementById(tableId);
  if (!table) return;
  var tbody = table.querySelector("tbody");
  var rows = Array.from(tbody.querySelectorAll("tr"));

  // Toggle direction: nếu click cùng cột thì đảo chiều
  var prevCol = parseInt(table.dataset.sortCol ?? "-1");
  var prevDir = table.dataset.sortDir || "asc";
  var dir = (prevCol === n && prevDir === "asc") ? "desc" : "asc";
  table.dataset.sortDir = dir;
  table.dataset.sortCol = n;

  // Cập nhật icon trên header
  var ths = table.querySelectorAll("thead th");
  ths.forEach(function(th, idx) {
    var txt = th.textContent.replace(/ [\u2191\u2193\u2195]/g, '').trim();
    th.textContent = txt + (idx === n ? (dir === "asc" ? " \u2191" : " \u2193") : " \u2195");
  });

  rows.sort(function(rowA, rowB) {
    var cellA = rowA.querySelectorAll("td")[n];
    var cellB = rowB.querySelectorAll("td")[n];
    if (!cellA || !cellB) return 0;
    var a = (cellA.innerText || cellA.textContent).replace(/,/g, '').replace(/%/g, '').trim();
    var b = (cellB.innerText || cellB.textContent).replace(/,/g, '').replace(/%/g, '').trim();
    var aNum = parseFloat(a);
    var bNum = parseFloat(b);
    var cmp = (!isNaN(aNum) && !isNaN(bNum)) ? (aNum - bNum) : a.toLowerCase().localeCompare(b.toLowerCase());
    return dir === "asc" ? cmp : -cmp;
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}
</script>
</body></html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    bill_path = os.path.join(BASE_DIR, "bill.csv")
    traces_dir = os.path.join(BASE_DIR, "output_all_traces")

    print("Đọc dữ liệu bill...")
    bill_df = pd.read_csv(bill_path, usecols=["bill_code", "VD_type", "service",
                                                "receiving_date", "actual_weight",
                                                "origin_province", "destination_province",
                                                "bill_creation_date"])
    bill_df['receiving_date'] = pd.to_datetime(bill_df['receiving_date'], errors="coerce")
    day_maps = {0: "T2", 1: "T3", 2: "T4", 3: "T5", 4: "T6", 5: "T7", 6: "CN"}
    bill_df['day_of_week'] = bill_df['receiving_date'].dt.dayofweek.map(day_maps)
    bill_df['bill_creation_date'] = pd.to_datetime(bill_df['bill_creation_date'], errors="coerce")
    bill_df['creation_hour'] = bill_df['bill_creation_date'].dt.hour.fillna(-1).astype(int).astype(str) + 'h'
    bill_df.loc[bill_df["creation_hour"] == "-1h", "creation_hour"] = "N/A"

    print("Đọc dữ liệu traces...")
    df_o = pd.read_csv(os.path.join(traces_dir, "origin_head.csv"))
    df_d = pd.read_csv(os.path.join(traces_dir, "destination_tail.csv"))

    merge_cols = ["bill_code", "VD_type", "service", "day_of_week", "actual_weight",
                  "origin_province", "destination_province", "creation_hour"]
    bill_merge = bill_df[merge_cols]

    df_o = df_o.drop(columns=["actual_weight"], errors="ignore").merge(bill_merge, on="bill_code", how="left")
    df_d = df_d.drop(columns=["actual_weight"], errors="ignore").merge(bill_merge, on="bill_code", how="left")

    all_results = []
    results_by_flow = []

    # ── Luồng 1: Origin → 1A ──
    print("\n" + "█" * 70)
    print("  LUỒNG 1: KHO GỬI → KHO 1A NGUỒN")
    print("█" * 70)
    df_o_proc, pairs_o = get_top_pairs(df_o, "kho_o", "kho_o1a", "tong_kg", "actual_weight", False)
    res_o, rep_o = run_multi_pipeline(df_o_proc, pairs_o, "time_o", "time_o1a",
                                       "Kho gửi → Kho 1A", is_dest_flow=False)
    all_results.extend(res_o)
    results_by_flow.append(("Luồng 1: Kho gửi → Kho 1A nguồn", rep_o))

    # ── Luồng 2: 1A → Dest ──
    print("\n" + "█" * 70)
    print("  LUỒNG 2: KHO 1A ĐÍCH → KHO NHẬN")
    print("█" * 70)
    df_d_proc, pairs_d = get_top_pairs(df_d, "kho_d1a", "kho_d", "tong_kg", "actual_weight", True)
    res_d, rep_d = run_multi_pipeline(df_d_proc, pairs_d, "time_d1a", "time_d",
                                       "Kho 1A → Kho nhận", is_dest_flow=True)
    all_results.extend(res_d)
    results_by_flow.append(("Luồng 2: Kho 1A đích → Kho nhận", rep_d))

    # ── Xuất kết quả ──
    df_sum = pd.DataFrame(all_results)
    if not df_sum.empty:
        csv_path = os.path.join(OUTPUT_DIR, "multi_classification_summary.csv")
        df_sum.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n Đã lưu CSV: {csv_path}")

    print("\n Đang tạo HTML Dashboard...")
    html_content = build_html_report(results_by_flow, df_sum if not df_sum.empty else pd.DataFrame())
    html_path = os.path.join(OUTPUT_DIR, "multi_classification.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" Đã lưu HTML: {html_path}")

    print("\n HOÀN TẤT!")