"""
Pipeline dự đoán Giờ đến (Offset từ ngày gửi).

Với mỗi cặp kho:
  1. DBSCAN phân cụm trên y_offset
  2. Train/Test split 80/20 (stratify theo cluster)
  3. Training: Fit regression per cluster (LR, RF, LGBM) x 2 feature options
  4. Inference: 3 chiến lược chọn cụm x 2 feature options x 3 regression models
  5. Đánh giá: MAE, RMSE, R², MAPE trên 2 scope (All, Dominant Cluster)

Output:
  - CSV:  output_plot/multi_regression_summary.csv
  - HTML: output_plot/multi_regression.html
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, LightGBMRegressor
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
def get_top_pairs(df, col_a, col_b, sort_by, is_dest_flow):
    """Lấy top N cặp kho (giữ nguyên logic classification.py)"""
    df = df.copy()
    if is_dest_flow:
        df = df[~df[col_b].isin(EXCLUDE)]
    else:
        df = df[~df[col_a].isin(EXCLUDE)]

    df['pair'] = df[col_a].astype(str) + ' -> ' + df[col_b].astype(str)
    check_col = col_b if is_dest_flow else col_a
    agg = df.groupby(['pair', check_col]).agg(
        so_bill=('bill_code', 'nunique'),
        tong_kg=('actual_weight', 'sum')
    ).reset_index()
    key = "so_bill" if sort_by == 'bill_count' else "tong_kg"
    top = agg.nlargest(N_TOP, key)
    return df, top['pair'].tolist()


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

        # ══ TRAINING ══

        # --- 1. Fit Regression per Cluster ---
        regression_models = {}  # key: (cluster, model_name, option) -> model
        model_specs = [("LR", LinearRegression)]
        model_specs.append(("RF", lambda: RandomForestRegressor(
            n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)))
        model_specs.append(("LGBM", lambda: LightGBMRegressor(
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
        # Dùng Option 2 features cho classifier
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

        print(f"  Khung giờ → Cụm: {slot_cluster_map}")

        # ══ INFERENCE ══
        print(f"\n  Inference trên {len(idx_test):,} bills test...")

        # Bước 1: Chọn cụm cho mỗi bill test
        # Cách 1: Khung giờ
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
            "Cách 1 (Khung giờ)": chosen_c1,
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

                    for i in range(len(y_test)):
                        clus = chosen_clusters[i]
                        key = (clus, reg_name, opt_key)
                        if key in regression_models:
                            x_i = X_test_opt.iloc[[i]]
                            y_pred[i] = regression_models[key].predict(x_i)[0]
                        else:
                            # Fallback: dùng dominant cluster model
                            fallback_key = (dominant_cluster, reg_name, opt_key)
                            if fallback_key in regression_models:
                                x_i = X_test_opt.iloc[[i]]
                                y_pred[i] = regression_models[fallback_key].predict(x_i)[0]

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
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #f8fafc; padding: 8px 10px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; color: #475569; user-select: none; }
  th:hover { background: #e2e8f0; cursor: pointer; }
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
</style>
</head><body>
<h1> Multi-Strategy Regression Pipeline</h1>
<p class="subtitle">So sánh tổ hợp: 3 chiến lược chọn cụm × 2 feature options × 3 mô hình hồi quy × 2 evaluation scopes</p>
"""

    # ── Summary Table ──
    if not summary_df.empty:
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
  <h2> Bảng tổng hợp so sánh tất cả tổ hợp chiến lược</h2>
  <table id="summaryTable">
    <thead><tr>
      <th onclick="sortTable(0)">Cặp Kho ↕</th>
      <th onclick="sortTable(1)">Luồng ↕</th>
      <th onclick="sortTable(2)">Chọn Cụm ↕</th>
      <th onclick="sortTable(3)">Features ↕</th>
      <th onclick="sortTable(4)">Regression ↕</th>
      <th onclick="sortTable(5)">Eval Scope ↕</th>
      <th onclick="sortTable(6)">N Test ↕</th>
      <th onclick="sortTable(7)">MAE (h) ↕</th>
      <th onclick="sortTable(8)">RMSE (h) ↕</th>
      <th onclick="sortTable(9)">R² ↕</th>
      <th onclick="sortTable(10)">MAPE ↕</th>
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
            chart_idx += 1

            # Slot map info
            slot_info = " | ".join([f"{k}: Cụm {v}" for k, v in rep['slot_cluster_map'].items()])

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
    <p class="section-title"> Khung giờ → Cụm (Cách 1): {slot_info}</p>

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
var d_{chart_idx} = {fig.to_json()};
Plotly.newPlot('{id_chart}', d_{chart_idx}.data, d_{chart_idx}.layout, {{responsive: true}});
</script>
"""

    # Sort table script
    html += """
<script>
function sortTable(n) {
  var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
  table = document.getElementById("summaryTable");
  switching = true;
  dir = "asc";
  while (switching) {
    switching = false;
    rows = table.rows;
    for (i = 1; i < (rows.length - 1); i++) {
      shouldSwitch = false;
      x = rows[i].getElementsByTagName("TD")[n];
      y = rows[i + 1].getElementsByTagName("TD")[n];
      let xContent = (x.innerText || x.textContent).replace(/,/g, '').replace(/%/g, '');
      let yContent = (y.innerText || y.textContent).replace(/,/g, '').replace(/%/g, '');
      let xNum = parseFloat(xContent);
      let yNum = parseFloat(yContent);
      let isNum = !isNaN(xNum) && !isNaN(yNum);
      if (dir == "asc") {
        if (isNum ? xNum > yNum : xContent.toLowerCase() > yContent.toLowerCase()) { shouldSwitch = true; break; }
      } else if (dir == "desc") {
        if (isNum ? xNum < yNum : xContent.toLowerCase() < yContent.toLowerCase()) { shouldSwitch = true; break; }
      }
    }
    if (shouldSwitch) {
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true; switchcount++;
    } else {
      if (switchcount == 0 && dir == "asc") { dir = "desc"; switching = true; }
    }
  }
}
</script>
</body></html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    df_o_proc, pairs_o = get_top_pairs(df_o, "kho_o", "kho_o1a", "bill_count", False)
    res_o, rep_o = run_multi_pipeline(df_o_proc, pairs_o, "time_o", "time_o1a",
                                       "Kho gửi → Kho 1A", is_dest_flow=False)
    all_results.extend(res_o)
    results_by_flow.append(("Luồng 1: Kho gửi → Kho 1A nguồn", rep_o))

    # ── Luồng 2: 1A → Dest ──
    print("\n" + "█" * 70)
    print("  LUỒNG 2: KHO 1A ĐÍCH → KHO NHẬN")
    print("█" * 70)
    df_d_proc, pairs_d = get_top_pairs(df_d, "kho_d1a", "kho_d", "bill_count", True)
    res_d, rep_d = run_multi_pipeline(df_d_proc, pairs_d, "time_d1a", "time_d",
                                       "Kho 1A → Kho nhận", is_dest_flow=True)
    all_results.extend(res_d)
    results_by_flow.append(("Luồng 2: Kho 1A đích → Kho nhận", rep_d))

    # ── Xuất kết quả ──
    df_sum = pd.DataFrame(all_results)
    if not df_sum.empty:
        csv_path = os.path.join(OUTPUT_DIR, "multi_regression_summary.csv")
        df_sum.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n Đã lưu CSV: {csv_path}")

    print("\n Đang tạo HTML Dashboard...")
    html_content = build_html_report(results_by_flow, df_sum if not df_sum.empty else pd.DataFrame())
    html_path = os.path.join(OUTPUT_DIR, "multi_regression.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" Đã lưu HTML: {html_path}")

    print("\n HOÀN TẤT!")
