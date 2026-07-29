"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MULTI-STRATEGY REGRESSION PIPELINE – Ghi chú giải thích           ║
╚══════════════════════════════════════════════════════════════════════════════╝

 MỤC TIÊU
   Dự đoán y_offset = số giờ từ lúc gửi hàng đến khi đến kho đích,
   áp dụng cho Top-10 cặp kho theo lưu lượng bill.

 PIPELINE (với mỗi cặp kho)
   Bước 1 │ DBSCAN clustering trên y_offset
           │   → Phân nhóm tự động các khung giờ đến thực tế
           │   → Noise (cluster = -1) bị loại bỏ
           │   → Cụm được đánh số lại 0, 1, 2... theo thứ tự giờ tăng dần
   Bước 2 │ Train/Test split 80/20 (stratify theo cluster)
   Bước 3 │ Train Regression per cluster
           │   → 3 mô hình: LR (Linear), RF (RandomForest), LGBM (LightGBM)
           │   → 2 bộ features: Option 1 (chỉ giờ xuất phát), Option 2 (full)
           │   → Mỗi cụm có bộ model riêng
   Bước 4 │ Inference – 3 chiến lược chọn cụm
           │   Cách 1 (Dominant Cluster): Với mỗi khung giờ, chọn cụm chiếm
           │     đa số trong training → đơn giản, không cần ML, ổn định
           │   Cách 2 (ML Classifier): RandomForest classifier dự đoán cụm
           │     từ full features → linh hoạt nhưng có thể sai cụm
           │   Cách 3 (Hybrid): Dùng ML nếu xác suất >= 90%, còn lại dùng Cách 1
           │     → kết hợp ưu điểm cả hai
   Bước 5 │ Đánh giá trên 2 scope
           │   All      : Toàn bộ test set
           │   Dominant : Chỉ bill thuộc cụm lớn nhất (dominant cluster)

 CÁC TRƯỜNG THÔNG TIN TRONG OUTPUT
   flow             : Luồng vận chuyển (Luồng 1: Kho gửi→1A, Luồng 2: 1A→Kho nhận)
   pair             : Cặp kho (Kho A -> Kho B)
   cluster_selection: Chiến lược chọn cụm (Cách 1/2/3)
   feature_option   : Bộ features dùng cho regression (Option 1/2)
   regression_model : Mô hình hồi quy (LR / RF / LGBM)
   eval_scope       : Phạm vi đánh giá (All / Dominant)
   n_samples_test   : Số bill trong tập test
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

import sys, os, re
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
N_TOP = 6
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 150
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_plot")
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")  # Thư mục lưu model
HYBRID_THRESHOLD = 0.9  # Ngưỡng xác suất cho chiến lược Hybrid

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
    """Chuyển tên cặp kho thành tên thư mục hợp lệ"""
    return re.sub(r'[^\w]+', '_', pair).strip('_')[:80]


def save_models(pair, regression_models, clf, flow_name):
    """
    Lưu toàn bộ model của một cặp kho vào đĩa.
    Cấu trúc: saved_models/<pair_slug>/regression_models.joblib
                                       classifier.joblib
                                       meta.joblib
    """
    slug = _pair_slug(pair)
    pair_dir = os.path.join(MODELS_DIR, slug)
    os.makedirs(pair_dir, exist_ok=True)
    joblib.dump(regression_models, os.path.join(pair_dir, "regression_models.joblib"))
    joblib.dump(clf, os.path.join(pair_dir, "classifier.joblib"))
    joblib.dump({"pair": pair, "flow": flow_name}, os.path.join(pair_dir, "meta.joblib"))
    print(f"  [Lưu model] → {pair_dir}")


def load_models(pair):
    """
    Tải lại model đã lưu. Trả về (regression_models, clf) hoặc (None, None) nếu chưa có.
    """
    slug = _pair_slug(pair)
    pair_dir = os.path.join(MODELS_DIR, slug)
    reg_path = os.path.join(pair_dir, "regression_models.joblib")
    clf_path = os.path.join(pair_dir, "classifier.joblib")
    if os.path.exists(reg_path) and os.path.exists(clf_path):
        print(f"  [Load model] ← {pair_dir}")
        return joblib.load(reg_path), joblib.load(clf_path)
    return None, None


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

        # Lọc time ngoại lai
        if "time" in sub.columns:
            sub = sub[((sub['time'] >= 0) & (sub['time'] <= 100))].copy()
            if not sub.empty:
                q1 = sub['time'].quantile(0.01)
                q3 = sub['time'].quantile(0.99)
                sub = sub[((sub['time'] >= q1) & (sub['time'] <= q3))]

        if sub.empty or len(sub) < 200:
            print('  Không đủ dữ liệu --> Bỏ qua')
            continue

        # Tính y_offset và departure_hour
        dt1 = pd.to_datetime(sub[t_col_1])
        dt2 = pd.to_datetime(sub[t_col_2])
        departure_date = dt1.dt.normalize()
        sub['y_offset'] = (dt2 - departure_date).dt.total_seconds() / 3600.0
        sub['departure_hour_numeric'] = dt1.dt.hour + dt1.dt.minute / 60.0
        sub['departure_hour'] = dt1.dt.hour.fillna(-1).astype(int).astype(str) + 'h'
        sub.loc[sub['departure_hour'] == "-1h", 'departure_hour'] = "N/A"
        sub = sub[((sub['y_offset'] > 0) & (sub['y_offset'] < 48))].copy()

        if sub.empty or len(sub) < 200:
            print('  Không đủ dữ liệu sau lọc --> Bỏ qua')
            continue

        if len(sub) > 20000:
            sub = sub.sample(20000, random_state=42)

        # ══ DBSCAN ══
        y_vals = sub['y_offset'].values.reshape(-1, 1)
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        sub['cluster'] = dbscan.fit_predict(y_vals)

        # Sắp xếp lại tên cụm
        valid_mask = sub['cluster'] != -1
        if valid_mask.any():
            cluster_means = sub.loc[valid_mask].groupby("cluster")["y_offset"].mean()
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_mapping = {old_c: new_c for new_c, old_c in enumerate(sorted_clusters)}
            cluster_mapping[-1] = -1
            sub["cluster"] = sub["cluster"].map(cluster_mapping) 

        sub_clean = sub[sub['cluster'] != -1].copy()
        n_cluster = sub_clean['cluster'].nunique()
        if n_cluster < 2:
            print("  Không đủ số lượng cụm. Bỏ qua.")
            continue
        print(f"  Số mẫu hợp lệ {len(sub_clean):,} | Số cụm {n_cluster}")

        # ── Xác định cụm lớn nhất (dominant cluster) ──
        cluster_counts = sub_clean['cluster'].value_counts()
        dominant_cluster = cluster_counts.idxmax()
        print(f"  Cụm lớn nhất: Cụm {dominant_cluster} ({cluster_counts[dominant_cluster]:,} bills)")

        # ── Feature Engineering ──
        # Option 1: Chỉ giờ xuất phát
        X_opt1 = sub_clean[['departure_hour_numeric']].copy()

        # Option 2: Full features
        X_opt2 = pd.DataFrame(index=sub_clean.index)
        X_opt2['departure_hour_numeric'] = sub_clean['departure_hour_numeric']

        cat_cols = ['VD_type', 'service', 'day_of_week', 'creation_hour', 'departure_hour']
        for c in cat_cols:
            if c in sub_clean.columns:
                dummies = pd.get_dummies(sub_clean[c], prefix=c)
                X_opt2 = pd.concat([X_opt2, dummies], axis=1)

        if is_dest_flow:
            if "origin_province" in sub_clean.columns:
                op_trans = transform_province(sub_clean['origin_province'])
                dummies = pd.get_dummies(op_trans, prefix='origin_prov')
                X_opt2 = pd.concat([X_opt2, dummies], axis=1)
        else:
            if "destination_province" in sub_clean.columns:
                dp_trans = transform_province(sub_clean['destination_province'])
                dummies = pd.get_dummies(dp_trans, prefix='dest_prov')
                X_opt2 = pd.concat([X_opt2, dummies], axis=1)

        w = sub_clean['actual_weight']
        X_opt2['actual_weight'] = w.fillna(w.median())

        y_target = sub_clean['y_offset'].values
        clusters = sub_clean['cluster'].values

        # ── Lọc labels hợp lệ (> 1 sample) ──
        counts = pd.Series(clusters).value_counts()
        valid_labels = counts[counts > 1].index
        valid_mask = np.isin(clusters, valid_labels)
        X_opt1 = X_opt1[valid_mask]
        X_opt2 = X_opt2[valid_mask]
        y_target = y_target[valid_mask]
        clusters = clusters[valid_mask]

        if len(np.unique(clusters)) < 2:
            print("  Không đủ nhãn hợp lệ sau khi làm sạch. Bỏ qua.")
            continue

        # ══ TRAIN / TEST SPLIT ══
        indices = np.arange(len(y_target))
        idx_train, idx_test = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=clusters
        )

        X1_train, X1_test = X_opt1.iloc[idx_train], X_opt1.iloc[idx_test]
        X2_train, X2_test = X_opt2.iloc[idx_train], X_opt2.iloc[idx_test]
        y_train, y_test = y_target[idx_train], y_target[idx_test]
        c_train, c_test = clusters[idx_train], clusters[idx_test]
        dep_hour_test = X1_test['departure_hour_numeric'].values

        # ══ TRAINING hoặc LOAD MODEL ══
        regression_models, clf = load_models(pair)

        if regression_models is not None and clf is not None:
            # ── Đã có model lưu sẵn → bỏ qua training, dùng lại ──
            print(f"  [Cache hit] Dùng model đã lưu, bỏ qua training.")
            clf_proba_test = clf.predict_proba(X2_test)
            clf_pred_test = clf.classes_[np.argmax(clf_proba_test, axis=1)]
            clf_max_proba_test = np.max(clf_proba_test, axis=1)
        else:
            # ── Chưa có model → train mới ──

            # --- 1. Fit Regression per Cluster ---
            regression_models = {}  # key: (cluster, model_name, option) -> model
            model_specs = [("LR", LinearRegression)]
            model_specs.append(("RF", lambda: RandomForestRegressor(
                n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)))
            model_specs.append(("LGBM", lambda: LGBMRegressor(
                n_estimators=100, max_depth=15, verbose=-1, random_state=42, n_jobs=-1)))

            unique_clusters = sorted(np.unique(c_train))
            print(f"\n  Training regression models cho {len(unique_clusters)} cụm...")

            for clus in unique_clusters:
                mask_c = c_train == clus
                if mask_c.sum() < 5:
                    continue

                y_c = y_train[mask_c]

                for opt_name, opt_X_train in [("Opt1", X1_train), ("Opt2", X2_train)]:
                    X_c = opt_X_train.iloc[mask_c.nonzero()[0]] if isinstance(mask_c, np.ndarray) else opt_X_train[mask_c]

                    for model_name, model_factory in model_specs:
                        try:
                            model = model_factory() if callable(model_factory) and not isinstance(model_factory, type) else model_factory()
                            model.fit(X_c, y_c)
                            regression_models[(clus, model_name, opt_name)] = model
                        except Exception as e:
                            print(f"     Lỗi fit {model_name} cho cụm {clus} {opt_name}: {e}")

            print(f"  Đã train {len(regression_models)} regression models")

            # --- 2. Fit Classifier (cho Cách 2 & 3) ---
            clf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                n_jobs=-1, max_depth=15, random_state=42
            )
            clf.fit(X2_train, c_train)
            clf_proba_test = clf.predict_proba(X2_test)
            clf_pred_test = clf.classes_[np.argmax(clf_proba_test, axis=1)]
            clf_max_proba_test = np.max(clf_proba_test, axis=1)

        clf_acc = accuracy_score(c_test, clf_pred_test)
        clf_f1 = f1_score(c_test, clf_pred_test, average='weighted', zero_division=0)
        print(f"  Classifier Acc={clf_acc:.4f}, W-F1={clf_f1:.4f}")

        # --- 3. Bảng lookup khung giờ → cụm lớn nhất (Cách 1) ---
        train_dep_hours = X1_train['departure_hour_numeric'].values
        slot_cluster_map = {}
        for slot_name, (lo, hi) in TIME_SLOTS.items():
            slot_mask = (train_dep_hours >= lo) & (train_dep_hours < hi)
            if slot_mask.sum() > 0:
                slot_clusters = c_train[slot_mask]
                slot_cluster_map[slot_name] = pd.Series(slot_clusters).value_counts().idxmax()
            else:
                slot_cluster_map[slot_name] = dominant_cluster

        print(f"  Khung giờ → Cụm dominant: {slot_cluster_map}")

        # ══ INFERENCE ══
        print(f"\n  Inference trên {len(idx_test):,} bills test...")

        # Bước 1: Chọn cụm cho mỗi bill test
        # Cách 1 (Dominant Cluster): Với mỗi khung giờ, lấy cụm chiếm đa số trong training
        chosen_c1 = np.array([
            slot_cluster_map.get(get_time_slot(h), dominant_cluster)
            for h in dep_hour_test
        ])

        # Cách 2: ML Classifier
        chosen_c2 = clf_pred_test.copy()

        # Cách 3: Hybrid (ML nếu >= 90%, else Cách 1)
        chosen_c3 = np.where(
            clf_max_proba_test >= HYBRID_THRESHOLD,
            clf_pred_test,
            chosen_c1
        )
        hybrid_ml_rate = np.mean(clf_max_proba_test >= HYBRID_THRESHOLD)
        print(f"  Hybrid: {hybrid_ml_rate*100:.1f}% dùng ML, {(1-hybrid_ml_rate)*100:.1f}% dùng Cách 1")

        # Bước 2: Dự đoán y_offset
        cluster_selections = {
            "Cách 1 (Dominant Cluster)": chosen_c1,
            "Cách 2 (ML Classifier)": chosen_c2,
            "Cách 3 (Hybrid)": chosen_c3,
        }

        feature_options = {
            "Option 1 (Hour)": (X1_test, "Opt1"),
            "Option 2 (Full)": (X2_test, "Opt2"),
        }

        reg_model_names = ["LR", "RF", "LGBM"]

        pair_combo_results = []

        for cs_name, chosen_clusters in cluster_selections.items():
            for fo_name, (X_test_opt, opt_key) in feature_options.items():
                for reg_name in reg_model_names:
                    # Dự đoán y_offset cho mỗi bill
                    y_pred = np.full(len(y_test), np.nan)

                    # ── Tối ưu inference: Predict theo batch từng cụm (Vectorized) ──
                    # Precompute clip bounds cho từng cluster để tránh tính lại trong loop
                    clip_bounds = {}
                    for clus_id in np.unique(chosen_clusters):
                        cm = c_train == clus_id
                        if cm.any():
                            clip_bounds[clus_id] = (y_train[cm].min() - 2, y_train[cm].max() + 2)
                        else:
                            clip_bounds[clus_id] = (-2, 50)

                    # Duyệt qua các cụm được chọn trong test set
                    for clus_id in np.unique(chosen_clusters):
                        idx_clus = np.where(chosen_clusters == clus_id)[0]
                        if len(idx_clus) == 0:
                            continue

                        # Xác định model key
                        key = (clus_id, reg_name, opt_key)
                        used_clus = clus_id
                        if key not in regression_models:
                            # Fallback sang dominant cluster
                            key = (dominant_cluster, reg_name, opt_key)
                            used_clus = dominant_cluster
                        
                        if key not in regression_models:
                            continue

                        # Predict nguyên batch cho tất cả các bill thuộc cụm này
                        X_batch = X_test_opt.iloc[idx_clus]
                        y_raw_batch = regression_models[key].predict(X_batch)

                        # Clip và gán kết quả
                        lo, hi = clip_bounds.get(used_clus, (-2, 50))
                        y_pred[idx_clus] = np.clip(y_raw_batch, lo, hi)

                    # Đánh giá
                    valid = ~np.isnan(y_pred)
                    if valid.sum() == 0:
                        continue

                    # Eval Cách 1: Toàn bộ test
                    metrics_all = compute_regression_metrics(y_test[valid], y_pred[valid])

                    # Eval Cách 2: Chỉ bill thuộc cụm lớn nhất
                    dom_mask = valid & (c_test == dominant_cluster)
                    metrics_dom = compute_regression_metrics(y_test[dom_mask], y_pred[dom_mask]) if dom_mask.sum() > 0 else {
                        "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan
                    }

                    combo_label = f"{cs_name} | {fo_name} | {reg_name}"

                    for eval_scope, metrics in [("All", metrics_all), ("Dominant", metrics_dom)]:
                        result = {
                            "flow": flow_name,
                            "pair": pair,
                            "cluster_selection": cs_name,
                            "feature_option": fo_name,
                            "regression_model": reg_name,
                            "eval_scope": eval_scope,
                            "n_samples_test": int(valid.sum()) if eval_scope == "All" else int(dom_mask.sum()),
                            **metrics
                        }
                        all_results.append(result)

                    pair_combo_results.append({
                        "combo_label": combo_label,
                        "cs_name": cs_name,
                        "fo_name": fo_name,
                        "reg_name": reg_name,
                        "metrics_all": metrics_all,
                        "metrics_dom": metrics_dom,
                    })

        # Thống kê cluster
        cluster_stats = []
        for c in sorted(sub_clean['cluster'].unique()):
            c_data = sub_clean[sub_clean['cluster'] == c]['y_offset']
            cluster_stats.append({
                "cluster": c, "n": len(c_data),
                "mean": f"{c_data.mean():.1f}h",
                "median": f"{c_data.median():.1f}h",
                "std": f"{c_data.std():.1f}h"
            })

        # Print best combo
        if pair_combo_results:
            best = min(pair_combo_results, key=lambda x: x['metrics_all']['MAE'] if not np.isnan(x['metrics_all']['MAE']) else 999)
            print(f"\n   Best combo (All, MAE): {best['combo_label']}")
            print(f"    MAE={best['metrics_all']['MAE']:.2f}h, RMSE={best['metrics_all']['RMSE']:.2f}h, "
                  f"R²={best['metrics_all']['R2']:.4f}, MAPE={best['metrics_all']['MAPE']:.1f}%")

        flow_pair_reports.append({
            "pair": pair,
            "n_samples": len(sub_clean),
            "n_clusters": n_cluster,
            "dominant_cluster": dominant_cluster,
            "cluster_stats": cluster_stats,
            "combo_results": pair_combo_results,
            "slot_cluster_map": slot_cluster_map,
            "clf_acc": clf_acc,
            "clf_f1": clf_f1,
            "hybrid_ml_rate": hybrid_ml_rate,
        })

        # ── Lưu model ra đĩa ──
        save_models(pair, regression_models, clf, flow_name)

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
  .avg-row-container { display: flex; flex-direction: column; gap: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 24px; }
  .avg-row-container:last-child { border-bottom: none; padding-bottom: 0; }
  .avg-table-wrap { overflow-x: auto; }
  .avg-chart-wrap { width: 100%; min-height: 320px; }
  .avg-table-wrap h3 { font-size: 12px; font-weight: 700; color: #475569; margin-bottom: 6px;
                        border-left: 3px solid #3b82f6; padding-left: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #f8fafc; padding: 8px 10px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; color: #475569; user-select: none; }
  th.sortable:hover { background: #e2e8f0; cursor: pointer; }
  td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; }
  tr:hover td { background: #f8fafc; }
  .pair-card { background: white; border-radius: 10px; margin-bottom: 20px; overflow: hidden;
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
<p class="subtitle">So sánh tổ hợp: 3 chiến lược chọn cụm × 2 feature options × 3 mô hình hồi quy × 2 evaluation scopes</p>

<div class="note-card">
  <h2>Hướng dẫn đọc báo cáo</h2>
  <div class="note-grid">
    <div>
      <div class="note-section"><b>Pipeline (mỗi cặp kho)</b>
        Bước 1: DBSCAN phân cụm y_offset → Bước 2: Train/Test 80/20 →
        Bước 3: Train Regression per cluster (LR/RF/LGBM × Opt1/Opt2) →
        Bước 4: Inference 3 chiến lược → Bước 5: Đánh giá All & Dominant
      </div>
      <div class="note-section"><b>Chiến lược chọn cụm</b>
        <u>Cách 1 (Dominant Cluster)</u>: Với mỗi khung giờ, dùng cụm chiếm đa số trong training – đơn giản, ổn định<br>
        <u>Cách 2 (ML Classifier)</u>: RandomForest tự phân loại cụm từ full features – linh hoạt hơn<br>
        <u>Cách 3 (Hybrid)</u>: Dùng ML nếu xác suất ≥ 90%, còn lại dùng Cách 1
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
        <u>All</u>: Đánh giá toàn bộ test | <u>Dominant</u>: Chỉ cụm lớn nhất
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
            """Tạo HTML table trung bình metrics theo 1 cột nhóm (phân tách thêm theo luồng và eval_scope)."""
            cols_to_group = [group_col]
            if group_col != "flow":
                cols_to_group.append("flow")
            if group_col != "eval_scope":
                cols_to_group.append("eval_scope")
                
            grp = df.groupby(cols_to_group)[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
            # Sort để dữ liệu nhóm liền kề nhau dễ nhìn
            grp = grp.sort_values(cols_to_group)
            
            header_map = {
                "flow": "Luồng", 
                "eval_scope": "Scope", 
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
            COLS_PER_ROW = 4
            n_rows_plot = math.ceil(n_groups / COLS_PER_ROW)
            n_cols_plot = min(n_groups, COLS_PER_ROW)
            
            subplot_titles = [f"#{i+1}" for i in range(n_groups)]
            fig = make_subplots(
                rows=n_rows_plot, cols=n_cols_plot,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.06,
                vertical_spacing=0.15,
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
            ("cluster_selection",  "Theo Chiến lược chọn cụm"),
            ("feature_option",     "Theo Feature Option"),
            ("regression_model",   "Theo Regression Model"),
            ("eval_scope",         "Theo Eval Scope"),
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
        combo_cols = ['flow', 'cluster_selection', 'feature_option', 'regression_model', 'eval_scope']
        combo_grp = df_valid.groupby(combo_cols)[['MAE', 'RMSE', 'R2', 'MAPE']].mean().reset_index()
        combo_grp = combo_grp.sort_values(combo_cols)
        
        combo_rows = ""
        for _, r in combo_grp.iterrows():
            combo_rows += (f"<tr><td>{r['flow']}</td><td>{r['cluster_selection']}</td>"
                           f"<td>{r['feature_option']}</td><td>{r['regression_model']}</td>"
                           f"<td>{r['eval_scope']}</td><td><b>{r['MAE']:.3f}</b></td>"
                           f"<td>{r['RMSE']:.3f}</td><td>{r['R2']:.4f}</td><td>{r['MAPE']:.1f}%</td></tr>")

        table_html = f"""
  <table id="comboSummaryTable">
    <thead><tr>
      <th class="sortable" onclick="sortTable(0, 'comboSummaryTable')">Luồng ↕</th>
      <th class="sortable" onclick="sortTable(1, 'comboSummaryTable')">Chọn Cụm ↕</th>
      <th class="sortable" onclick="sortTable(2, 'comboSummaryTable')">Features ↕</th>
      <th class="sortable" onclick="sortTable(3, 'comboSummaryTable')">Regression ↕</th>
      <th class="sortable" onclick="sortTable(4, 'comboSummaryTable')">Eval Scope ↕</th>
      <th class="sortable" onclick="sortTable(5, 'comboSummaryTable')">Avg MAE (h) ↕</th>
      <th class="sortable" onclick="sortTable(6, 'comboSummaryTable')">Avg RMSE (h) ↕</th>
      <th class="sortable" onclick="sortTable(7, 'comboSummaryTable')">Avg R² ↕</th>
      <th class="sortable" onclick="sortTable(8, 'comboSummaryTable')">Avg MAPE ↕</th>
    </tr></thead>
    <tbody>{combo_rows}</tbody>
  </table>
"""

        html += f"""
<div class="summary-card">
  <h2>Trung bình toàn tổ hợp (Flow → Cụm → Feature → Model → Scope)</h2>
  <div class="avg-row-container">
    <div class="avg-table-wrap" style="width: 100%;">
      {table_html}
    </div>
  </div>
</div>
"""

        summary_rows = ""
        for _, row in summary_df.sort_values("MAE", ascending=True).iterrows():
            mae_str = f"{row['MAE']:.2f}" if not pd.isna(row['MAE']) else "N/A"
            rmse_str = f"{row['RMSE']:.2f}" if not pd.isna(row['RMSE']) else "N/A"
            r2_str = f"{row['R2']:.4f}" if not pd.isna(row['R2']) else "N/A"
            mape_str = f"{row['MAPE']:.1f}%" if not pd.isna(row['MAPE']) else "N/A"
            summary_rows += f"""<tr>
                <td><b>{row['pair']}</b></td>
                <td>{row['flow']}</td>
                <td>{row['cluster_selection']}</td>
                <td>{row['feature_option']}</td>
                <td>{row['regression_model']}</td>
                <td>{row['eval_scope']}</td>
                <td>{int(row['n_samples_test']):,}</td>
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
      <th class="sortable" onclick="sortTable(0)">Cặp Kho ↕</th>
      <th class="sortable" onclick="sortTable(1)">Luồng ↕</th>
      <th class="sortable" onclick="sortTable(2)">Chọn Cụm ↕</th>
      <th class="sortable" onclick="sortTable(3)">Features ↕</th>
      <th class="sortable" onclick="sortTable(4)">Regression ↕</th>
      <th class="sortable" onclick="sortTable(5)">Eval Scope ↕</th>
      <th class="sortable" onclick="sortTable(6)">N Test ↕</th>
      <th class="sortable" onclick="sortTable(7)">MAE (h) ↕</th>
      <th class="sortable" onclick="sortTable(8)">RMSE (h) ↕</th>
      <th class="sortable" onclick="sortTable(9)">R² ↕</th>
      <th class="sortable" onclick="sortTable(10)">MAPE ↕</th>
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
            combos = rep["combo_results"]
            c_stats = rep["cluster_stats"]

            if not combos:
                continue

            # ── Cluster Stats Table ──
            cluster_rows = ""
            for cs in c_stats:
                cluster_rows += f"<tr><td><b>Cụm {cs['cluster']}</b></td><td>{cs['n']:,}</td><td>{cs['mean']}</td><td>{cs['median']}</td><td>{cs['std']}</td></tr>"

            # ── Combo Detail Table ──
            combo_rows = ""
            for combo in combos:
                m_all = combo['metrics_all']
                m_dom = combo['metrics_dom']
                combo_rows += f"""<tr>
                    <td>{combo['cs_name']}</td>
                    <td>{combo['fo_name']}</td>
                    <td>{combo['reg_name']}</td>
                    <td><b>{m_all['MAE']:.2f}</b></td><td>{m_all['RMSE']:.2f}</td>
                    <td>{m_all['R2']:.4f}</td><td>{m_all['MAPE']:.1f}%</td>
                    <td><b>{m_dom['MAE']:.2f}</b></td><td>{m_dom['RMSE']:.2f}</td>
                    <td>{m_dom['R2']:.4f}</td><td>{m_dom['MAPE']:.1f}%</td>
                </tr>"""

            # ── Bar Chart: MAE comparison ──
            # Grouped by cluster_selection, colored by regression model
            colors_map = {"LR": "#3B82F6", "RF": "#10B981", "LGBM": "#F59E0B"}

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["MAE - Toàn bộ test (All)", "MAE - Cụm lớn nhất (Dominant)"],
                horizontal_spacing=0.12
            )

            # Collect data for chart
            for scope_idx, scope_key in enumerate(["metrics_all", "metrics_dom"]):
                col = scope_idx + 1
                reg_names_seen = set()
                for reg_name in ["LR", "RF", "LGBM"]:
                    x_vals = []
                    y_vals = []
                    for combo in combos:
                        if combo['reg_name'] == reg_name:
                            label = f"{combo['cs_name']}<br>{combo['fo_name']}"
                            x_vals.append(label)
                            y_vals.append(combo[scope_key]['MAE'])
                    if x_vals:
                        fig.add_trace(go.Bar(
                            x=x_vals, y=y_vals,
                            name=reg_name,
                            marker_color=colors_map.get(reg_name, "#888"),
                            showlegend=(scope_idx == 0),
                            legendgroup=reg_name,
                            hovertemplate=f"{reg_name}<br>%{{x}}<br>MAE: %{{y:.2f}}h<extra></extra>"
                        ), row=1, col=col)

                fig.update_yaxes(title_text="MAE (giờ)", row=1, col=col)

            fig.update_layout(
                barmode="group",
                height=450, width=1100,
                plot_bgcolor="#FAFAFA", paper_bgcolor="#FFFFFF",
                font=dict(family="Segoe UI, Arial", size=10),
                margin=dict(t=60, b=120, l=50, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )

            id_chart = f"chart_{chart_idx}"
            # KHÔNG tăng chart_idx ở đây — phải dùng cùng giá trị trong script bên dưới

            # Slot map info
            slot_info = " | ".join([f"{k}: Cụm {v}" for k, v in rep['slot_cluster_map'].items()])

            fig_json = fig.to_json()
            html += f"""
<div class="pair-card">
  <div class="pair-header">
    <span class="pair-title">{pair}</span>
    <div class="pair-meta">
      <span class="meta-badge">{rep['n_samples']:,} samples</span>
      <span class="meta-badge">{rep['n_clusters']} cụm</span>
      <span class="meta-badge green">Dominant: Cụm {rep['dominant_cluster']}</span>
      <span class="meta-badge">Clf Acc={rep['clf_acc']:.3f}</span>
      <span class="meta-badge">Hybrid ML: {rep['hybrid_ml_rate']*100:.1f}%</span>
    </div>
  </div>
  <div class="pair-body">
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

    <p class="section-title"> Chi tiết tất cả tổ hợp</p>
    <table>
      <thead><tr>
        <th>Chọn Cụm</th><th>Features</th><th>Regression</th>
        <th colspan="4" style="text-align:center;background:#dbeafe">── All ──</th>
        <th colspan="4" style="text-align:center;background:#dcfce7">── Dominant ──</th>
      </tr>
      <tr>
        <th></th><th></th><th></th>
        <th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE</th>
        <th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE</th>
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
            chart_idx += 1  # Tăng SAU khi đã emit script với giá trị hiện tại

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
    df_o_proc, pairs_o = get_top_pairs(df_o, "kho_o", "kho_o1a", "bill_count", "actual_weight", False)
    res_o, rep_o = run_multi_pipeline(df_o_proc, pairs_o, "time_o", "time_o1a",
                                       "Kho gửi → Kho 1A", is_dest_flow=False)
    all_results.extend(res_o)
    results_by_flow.append(("Luồng 1: Kho gửi → Kho 1A nguồn", rep_o))

    # ── Luồng 2: 1A → Dest ──
    print("\n" + "█" * 70)
    print("  LUỒNG 2: KHO 1A ĐÍCH → KHO NHẬN")
    print("█" * 70)
    df_d_proc, pairs_d = get_top_pairs(df_d, "kho_d1a", "kho_d", "bill_count", "actual_weight", True)
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
