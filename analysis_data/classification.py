import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd  
import numpy as np 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cluster import DBSCAN 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 

N_TOP = 10 
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 150
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_plot")
CONFIDENCE_THRESHOLD = 0.8

EXCLUDE = [
    'Kho TGDĐ Bảo Hành',
    'Kho Hà Tĩnh',
    'Kho Kiến An',
    'Kho Thường Tín',
    'Kho Amway',
    'Kho Digiworld',
    'Kho Thái Nguyên',
    'Kho Việt Trì',
    'Kho Ninh Hòa',
    'Kho An Giang',
    'Bưu cục Dự Án Vgreen Sóng Thần',
    'Bưu cục Dự Án Vgreen Văn Giang'
]

def get_top_pairs(df, col_a, col_b, sort_by, is_dest_flow): 
    df = df.copy()  
    if is_dest_flow: 
        df = df[~df[col_b].isin(EXCLUDE)] 
    else: 
        df = df[~df[col_a].isin(EXCLUDE)]
        
    df['pair'] = df[col_a].astype(str) + ' -> ' + df[col_b].astype(str) 
    check_col = col_b if is_dest_flow else col_a 
    agg = df.groupby(['pair', check_col]).agg(
        so_bill = ('bill_code', 'nunique'),
        tong_kg = ('actual_weight', 'sum')
    ).reset_index() 
    key = "so_bill" if sort_by == 'bill_count' else "tong_kg"
    top = agg.nlargest(N_TOP, key) 
    return df, top['pair'].tolist() 

def transform_province(series, top = 10): 
    s = series.copy() 
    top_province = s.value_counts().nlargest(top).index.tolist() 
    return s.where(s.isin(top_province), 'khác') 

def run_rf_pipeline(df, pairs, t_col_1, t_col_2, flow_name, is_dest_flow): 
    all_summary = [] 
    all_importances = [] 
    flow_reports = [] 

    for pair in pairs: 
        print(f"\n{'='*70}") 
        print(f"    PAIR: {pair}") 
        print(f"{'='*70}")
        sub = df.loc[df["pair"] == pair, :].copy()
        sub = sub.dropna(subset=[t_col_1, t_col_2]).copy() 
 
        if "time" in sub.columns: 
            sub = sub[((sub['time'] >= 0) & (sub['time'] <= 100))].copy() 
            q1 = sub['time'].quantile(0.01) 
            q3 = sub['time'].quantile(0.99) 
            sub = sub[((sub['time'] >= q1) & (sub['time'] <= q3))] 
        
        if sub.empty or len(sub) < 200: 
            print('Không đủ dữ liệu --> Bỏ quan') 
            continue 
            
        dt1 = pd.to_datetime(sub[t_col_1]) 
        dt2 = pd.to_datetime(sub[t_col_2])
        departure_date = dt1.dt.normalize() 
        sub['y_offset']  = (dt2 - departure_date).dt.total_seconds() / 3600.0 
        sub['departure_hour'] = dt1.dt.hour.fillna(-1).astype(int).astype(str) + 'h'
        sub.loc[sub['departure_hour'] == "-1h", 'departure_hour'] = "N/A"
        sub = sub[((sub['y_offset'] > 0) & (sub['y_offset'] < 48))].copy()

        if sub.empty or len(sub) < 200: 
            print('Không đủ dữ liệu sau khi lọc --> Bỏ quan') 
            continue 
            
        if len(sub) > 20000: 
            sub = sub.sample(20000, random_state = 42) 
        
        y_vals = sub['y_offset'].values.reshape(-1,1) 
        dbscan = DBSCAN(eps = DBSCAN_EPS, min_samples = DBSCAN_MIN_SAMPLES) 
        sub['cluster'] = dbscan.fit_predict(y_vals) 

        valid_mask = sub['cluster'] != -1  
        if valid_mask.any():
            cluster_means = sub.loc[valid_mask].groupby("cluster")["y_offset"].mean()
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_mapping = {old_c: new_c for new_c, old_c in enumerate(sorted_clusters)}
            cluster_mapping[-1] = -1
            sub["cluster"] = sub["cluster"].map(cluster_mapping)

        sub_clean = sub[sub['cluster'] != -1].copy()
        n_cluster = sub_clean['cluster'].nunique() 
        if n_cluster >= 2: 
            print(f"     Số mẫu hợp lệ {len(sub_clean):,} | số cụm {n_cluster}") 
        else : 
            print("      Không đủ số lượng cụm để phân loại") 
            continue 
        
        # Feature engineering 

        X_df = pd.DataFrame(index = sub_clean.index) 
        cat_cols = ['VD_type', 'service', 'day_of_week', 'creation_hour', 'departure_hour'] 
        # cat_cols = ['VD_type', 'service', 'day_of_week', 'creation_hour'] 

        for c in cat_cols: 
            dummies = pd.get_dummies(sub_clean[c], prefix = c) 
            X_df = pd.concat([X_df, dummies], axis = 1) 
        if is_dest_flow: 
            if "origin_province" in sub_clean.columns: 
                op_trans = transform_province(sub_clean['origin_province']) 
                dummies = pd.get_dummies(op_trans, prefix = 'origin_prov') 
                X_df = pd.concat([X_df, dummies], axis = 1) 
        else : 
            if "destination_province" in sub_clean.columns: 
                dp_trans = transform_province(sub_clean['destination_province']) 
                dummies = pd.get_dummies(dp_trans, prefix = 'dest_prov') 
                X_df = pd.concat([X_df, dummies], axis = 1) 
        w = sub_clean['actual_weight'] 
        X_df['actual_weight'] = w.fillna(w.median()) 
        y = sub_clean['cluster'].values 

        counts = pd.Series(y).value_counts()
        valid_labels = counts[counts > 1].index
        valid_mask = np.isin(y, valid_labels)
        X_df = X_df[valid_mask]
        y = y[valid_mask]

        if len(np.unique(y)) < 2:
            print("  Không đủ nhãn hợp lệ sau khi làm sạch. Bỏ qua.")
            continue
        
        # model randomforestclassifier 
            
        X_train, X_test, y_train, y_test  = train_test_split(X_df, y, test_size = 0.2, random_state = 42, stratify = y) 
        rf = RandomForestClassifier(
            n_estimators = 100, 
            class_weight = "balanced", 
            n_jobs = -1, 
            max_depth = 15,
            bootstrap = True,
            oob_score = True
        )
        rf.fit(X_train, y_train) 
        print(f"   OOB Score: {rf.oob_score_:.4f}")

        y_proba = rf.predict_proba(X_test)
        max_proba = np.max(y_proba, axis=1)
        max_idx = np.argmax(y_proba, axis=1)
        raw_pred = rf.classes_[max_idx]

        y_pred = np.where(max_proba >= CONFIDENCE_THRESHOLD, raw_pred, -1)
        coverage_rate = np.mean(max_proba >= CONFIDENCE_THRESHOLD)
        null_rate = np.mean(max_proba < CONFIDENCE_THRESHOLD)
        n_null = np.sum(max_proba < CONFIDENCE_THRESHOLD)

        acc = accuracy_score(y_test, y_pred) 
        f1_w = f1_score(y_test, y_pred, average = 'weighted') 
        f1_m = f1_score(y_test, y_pred, average= "macro") 
        rec_m = recall_score(y_test, y_pred, average="macro", zero_division=0)
        rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        pre_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
        pre_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        print(f"   Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"   Coverage Rate (>= {CONFIDENCE_THRESHOLD}): {coverage_rate:.4f} ({coverage_rate*100:.2f}%)")
        print(f"   Null Rate (< {CONFIDENCE_THRESHOLD}): {null_rate:.4f} ({null_rate*100:.2f}%) [{n_null:,}/{len(y_test):,} mẫu test]")
        print(f"   Accuracy score: {acc:.4f}") 
        print(f"   F1 score: {f1_w:.4f}") 
        print(f"   Recall macro: {rec_m:.4f}")
        print(f"   Recall weighted: {rec_w:.4f}")
        print(f"   Precision macro: {pre_m:.4f}")
        print(f"   Precision weighted: {pre_w:.4f}")
        classes = np.unique(y)
        clc_report = classification_report(y_test, y_pred, labels=classes, output_dict = True, zero_division=0) 
        cm = confusion_matrix(y_test, y_pred, labels=classes) 

        importances = rf.feature_importances_ 
        feat_name = X_df.columns 
        feat_importances = pd.DataFrame(
           { "feature": feat_name, 
            "importance": importances}
        ).sort_values("importance", ascending = False) 

        # Lưu importance vào list tổng
        for rank, row in enumerate(feat_importances.itertuples(), 1):
            all_importances.append({
                "flow": flow_name,
                "pair": pair,
                "feature": row.feature,
                "importance_mdi": row.importance,
                "importance_rank": rank
            })
            
        all_summary.append({
            "flow": flow_name,
            "pair": pair,
            "n_samples": len(X_df),
            "n_features": X_df.shape[1],
            "n_clusters": len(np.unique(y)),
            "coverage_rate": coverage_rate,
            "null_rate": null_rate,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "recall_macro": rec_m,
            "recall_weighted": rec_w,
            "precision_macro": pre_m,
            "precision_weighted": pre_w
        })
        
        flow_reports.append({
            "pair": pair,
            "coverage_rate": coverage_rate,
            "null_rate": null_rate,
            "accuracy": acc,
            "f1": f1_w,
            "f1_macro": f1_m,
            "recall_macro": rec_m,
            "recall_weighted": rec_w,
            "precision_macro": pre_m,
            "precision_weighted": pre_w,
            "n_samples": len(X_df),
            "fi_df": feat_importances, 
            "cm": cm,
            "classes": classes,
            "report": clc_report
        })

    return all_summary, all_importances, flow_reports    


def build_html_report(results_by_flow, summary_df):
    html = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Random Forest - Cluster Classification</title>
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
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #f8fafc; padding: 10px 12px; text-align: left; font-weight: 600;
       border-bottom: 2px solid #e2e8f0; color: #475569; user-select: none; }
  th:hover { background: #e2e8f0; cursor: pointer; }
  td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
  tr:hover td { background: #f8fafc; }
  .pair-card { background: white; border-radius: 10px; margin-bottom: 20px; overflow: hidden;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .pair-header { background: #f8fafc; padding: 14px 20px; border-bottom: 1px solid #e2e8f0;
                   display: flex; justify-content: space-between; align-items: center; }
  .pair-title { font-size: 15px; font-weight: 700; color: #1e293b; }
  .pair-meta { display: flex; gap: 16px; }
  .meta-badge { background: #eff6ff; color: #1d4ed8; padding: 4px 10px; border-radius: 6px;
                  font-size: 12px; font-weight: 600; }
  .pair-body { padding: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-container { height: 350px; }
</style>
</head><body>
<h1> Random Forest: Phân loại cụm giờ đến</h1>
<p class="subtitle">Sử dụng Feature Engineering & Random Forest Classifier</p>
"""
    # ── Summary Table ──
    if not summary_df.empty:
        summary_rows = ""
        for _, row in summary_df.sort_values("f1_weighted", ascending=False).iterrows():
            summary_rows += f"""<tr>
                <td><b>{row['pair']}</b></td>
                <td>{row['flow']}</td>
                <td>{int(row['n_samples']):,}</td>
                <td>{int(row['n_clusters'])}</td>
                <td><b>{row['coverage_rate']*100:.1f}%</b></td>
                <td><span style="color:#e11d48;font-weight:600">{row['null_rate']*100:.1f}%</span></td>
                <td>{row['accuracy']:.4f}</td>
                <td><b>{row['f1_weighted']:.4f}</b></td>
                <td><b>{row['f1_macro']:.4f}</b></td>
                <td><b>{row['recall_macro']:.4f}</b></td>
                <td><b>{row['recall_weighted']:.4f}</b></td>
                <td><b>{row['precision_macro']:.4f}</b></td>
                <td><b>{row['precision_weighted']:.4f}</b></td>
            </tr>"""

        html += f"""
<div class="summary-card">
  <h2> Hiệu suất mô hình các cặp kho</h2>
  <table id="summaryTable">
    <thead><tr>
      <th onclick="sortTable(0)" title="Nhấn để sắp xếp">Cặp Kho ↕</th>
      <th onclick="sortTable(1)" title="Nhấn để sắp xếp">Luồng ↕</th>
      <th onclick="sortTable(2)" title="Nhấn để sắp xếp">Số mẫu ↕</th>
      <th onclick="sortTable(3)" title="Nhấn để sắp xếp">Số cụm ↕</th>
      <th onclick="sortTable(4)" title="Nhấn để sắp xếp">Coverage (≥{CONFIDENCE_THRESHOLD}) ↕</th>
      <th onclick="sortTable(5)" title="Nhấn để sắp xếp">Null (<{CONFIDENCE_THRESHOLD}) ↕</th>
      <th onclick="sortTable(6)" title="Nhấn để sắp xếp">Accuracy ↕</th>
      <th onclick="sortTable(7)" title="Nhấn để sắp xếp">Weighted F1 ↕</th>
      <th onclick="sortTable(8)" title="Nhấn để sắp xếp">Macro F1 ↕</th>
      <th onclick="sortTable(9)" title="Nhấn để sắp xếp">Macro Recall ↕</th>
      <th onclick="sortTable(10)" title="Nhấn để sắp xếp">Weighted Recall ↕</th>
      <th onclick="sortTable(11)" title="Nhấn để sắp xếp">Macro Precision ↕</th>
      <th onclick="sortTable(12)" title="Nhấn để sắp xếp">Weighted Precision ↕</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</div>
"""
    chart_idx = 0
    # ── Detailed Reports ──
    for flow_name, reports in results_by_flow:
        html += f'<div class="flow-header"> {flow_name}</div>\n'
        for rep in reports:
            pair = rep["pair"]
            fi_df = rep["fi_df"]
            cm = rep["cm"]
            classes = rep["classes"]
            
            # Confusion Matrix Plot
            z_data = cm[::-1].tolist()
            fig_cm = go.Figure(data=go.Heatmap(
                z=z_data,
                x=[f"Cụm {c}" for c in classes],
                y=[f"Cụm {c}" for c in classes][::-1],
                colorscale='Blues',
                texttemplate="%{z}",
                showscale=False
            ))
            fig_cm.update_layout(
                title="Confusion Matrix (Tập Test)",
                xaxis_title="Dự đoán (Predicted)",
                yaxis_title="Thực tế (True)",
                margin=dict(l=40, r=20, t=40, b=40),
                height=350, font=dict(size=11)
            )
            
            # Feature Importance Plot
            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"][::-1],
                y=fi_df["feature"][::-1],
                orientation='h',
                marker_color="#3B82F6"
            ))
            fig_fi.update_layout(
                title="Feature Importance (MDI) - Tất cả các biến",
                margin=dict(l=150, r=20, t=40, b=40),
                height=max(350, len(fi_df) * 20), font=dict(size=11)
            )
            
            id_cm = f"cm_{chart_idx}"
            id_fi = f"fi_{chart_idx}"
            chart_idx += 1

            html += f"""
<div class="pair-card">
  <div class="pair-header">
    <span class="pair-title">{pair}</span>
    <div class="pair-meta">
      <span class="meta-badge">{rep['n_samples']:,} samples</span>
      <span class="meta-badge">Acc: {rep['accuracy']:.3f}</span>
      <span class="meta-badge">W-F1: {rep['f1']:.3f}</span>
      <span class="meta-badge">M-F1: {rep['f1_macro']:.3f}</span>
      <span class="meta-badge">M-Rec: {rep['recall_macro']:.3f}</span>
      <span class="meta-badge">W-Rec: {rep['recall_weighted']:.3f}</span>
      <span class="meta-badge">M-Pre: {rep['precision_macro']:.3f}</span>
      <span class="meta-badge">W-Pre: {rep['precision_weighted']:.3f}</span>
    </div>
  </div>
  <div class="pair-body">
    <div class="grid-2">
      <div id="{id_cm}" class="chart-container"></div>
      <div id="{id_fi}" class="chart-container"></div>
    </div>
  </div>
</div>
<script>
Plotly.newPlot('{id_cm}', {fig_cm.to_json()}.data, {fig_cm.to_json()}.layout, {{responsive: true}});
Plotly.newPlot('{id_fi}', {fig_fi.to_json()}.data, {fig_fi.to_json()}.layout, {{responsive: true}});
</script>
"""
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
      
      let xContent = x.innerText || x.textContent;
      let yContent = y.innerText || y.textContent;
      
      xContent = xContent.replace(/,/g, '');
      yContent = yContent.replace(/,/g, '');
      
      let xNum = parseFloat(xContent);
      let yNum = parseFloat(yContent);
      let isNum = !isNaN(xNum) && !isNaN(yNum);

      if (dir == "asc") {
        if (isNum ? xNum > yNum : xContent.toLowerCase() > yContent.toLowerCase()) {
          shouldSwitch = true;
          break;
        }
      } else if (dir == "desc") {
        if (isNum ? xNum < yNum : xContent.toLowerCase() < yContent.toLowerCase()) {
          shouldSwitch = true;
          break;
        }
      }
    }
    if (shouldSwitch) {
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true;
      switchcount ++;      
    } else {
      if (switchcount == 0 && dir == "asc") {
        dir = "desc";
        switching = true;
      }
    }
  }
}
</script>
</body></html>"""
    return html

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    bill_path = os.path.join(BASE_DIR, "bill.csv") if os.path.exists(os.path.join(BASE_DIR, "bill.csv")) else "bill.csv"
    traces_dir = os.path.join(BASE_DIR, "output_all_traces") if os.path.exists(os.path.join(BASE_DIR, "output_all_traces")) else "output_all_traces"

    print("Đọc dữ liệu bill...")
    bill_df = pd.read_csv(bill_path, usecols=["bill_code", "VD_type", "service",
                                                "receiving_date", "actual_weight",
                                                "origin_province", "destination_province",
                                                "bill_creation_date"])
    bill_df['receiving_date'] = pd.to_datetime(bill_df['receiving_date'], errors="coerce") 
    day_maps = {0:"T2", 1:"T3", 2:"T4", 3:"T5", 4:"T6", 5:"T7", 6:"CN"} 
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
    all_summary = []
    all_importances = []
    results_by_flow = []

    # ── Luồng 1: Origin → 1A ──
    print("\n" + "=" * 70)
    print("  LUỒNG 1: KHO GỬI → KHO 1A NGUỒN")
    print("=" * 70)
    df_o_proc, pairs_o = get_top_pairs(df_o, "kho_o", "kho_o1a", "bill_count", False)
    sum_o, imp_o, rep_o = run_rf_pipeline(df_o_proc, pairs_o, "time_o", "time_o1a", "Kho gửi → Kho 1A", is_dest_flow=False)
    
    all_summary.extend(sum_o)
    all_importances.extend(imp_o)
    results_by_flow.append(("Luồng 1: Kho gửi → Kho 1A nguồn", rep_o))

    # ── Luồng 2: 1A → Dest ──
    print("\n" + "=" * 70)
    print("  LUỒNG 2: KHO 1A ĐÍCH → KHO NHẬN")
    print("=" * 70)
    df_d_proc, pairs_d = get_top_pairs(df_d, "kho_d1a", "kho_d", "bill_count", True)
    sum_d, imp_d, rep_d = run_rf_pipeline(df_d_proc, pairs_d, "time_d1a", "time_d", "Kho 1A → Kho nhận", is_dest_flow=True)

    all_summary.extend(sum_d)
    all_importances.extend(imp_d)
    results_by_flow.append(("Luồng 2: Kho 1A đích → Kho nhận", rep_d))

    # ── Xuất Kết quả ──
    df_sum = pd.DataFrame(all_summary)
    if not df_sum.empty:
        sum_path = os.path.join(OUTPUT_DIR, "rf_classification_summary.csv")
        df_sum.to_csv(sum_path, index=False, encoding="utf-8-sig")
        print(f"\n Đã lưu Summary: {sum_path}")

    df_imp = pd.DataFrame(all_importances)
    if not df_imp.empty:
        imp_path = os.path.join(OUTPUT_DIR, "rf_feature_importance.csv")
        df_imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
        print(f" Đã lưu Feature Importance: {imp_path}")

    # Build HTML
    print(" Đang tạo HTML Dashboard...")
    html_content = build_html_report(results_by_flow, df_sum)
    html_path = os.path.join(OUTPUT_DIR, "rf_classification.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" Đã lưu HTML Dashboard: {html_path}")

    print("\n HOÀN TẤT!")