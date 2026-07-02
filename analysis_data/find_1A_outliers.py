import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

INPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output_all_traces")
OUTPUT_DIR = "output_plot"
N_TOP      = 10
OUTLIER_PCT = 0.99       


def top_pairs(df, col_a, col_b, sort_by, wt_col="actual_weight"):
    df = df.copy()
    df["pair"] = df[col_a].astype(str) + " → " + df[col_b].astype(str)
    agg = (df.groupby("pair")
             .agg(so_bill=("bill_code", "nunique"), tong_kg=(wt_col, "sum"))
             .reset_index())
    key = "so_bill" if sort_by == "bill_count" else "tong_kg"
    return df, agg.nlargest(N_TOP, key)


def find_outliers_for(df, pairs, time_col, label_prefix):

    records = []
    for pair in pairs:
        sub = df[df["pair"] == pair].copy()
        if sub.empty:
            continue

        cap = sub["time"].quantile(OUTLIER_PCT)
        outliers = sub[sub["time"] > cap].copy()
        if outliers.empty:
            continue

        outliers["pair"]       = pair
        outliers["p99_cap_h"]  = round(cap, 1)
        outliers["pct_label"]  = f">{OUTLIER_PCT*100:.0f}%"
        outliers["hour_at_1a"] = pd.to_datetime(outliers[time_col]).dt.hour
        records.append(outliers)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    result = result.sort_values(["pair", "hour_at_1a", "time"], ascending=[True, True, False])
    return result


def build_summary_table(df_out, time_col):
    if df_out.empty:
        return pd.DataFrame()
    grp = (df_out.groupby(["pair", "hour_at_1a"])
                 .agg(n_outlier=("bill_code", "count"),
                      min_time=("time", "min"),
                      med_time=("time", "median"),
                      max_time=("time", "max"),
                      p99_cap=("p99_cap_h", "first"))
                 .reset_index())
    grp["hour_label"] = grp["hour_at_1a"].apply(lambda h: f"{h}-{h+1}h")
    return grp


def build_html_report(out_origin, out_dest, sum_o, sum_d):
    def df_to_html(df, title):
        if df.empty:
            return f"<p><i>Không có dữ liệu outlier.</i></p>"


        summary_html = sum_o.to_html(index=False, classes="tbl", border=0) if "origin" in title.lower() else sum_d.to_html(index=False, classes="tbl", border=0)

        cols_show = ["pair", "bill_code", "time", "p99_cap_h", "hour_at_1a"]

        for c in df.columns:
            if c not in cols_show and "time" in c.lower() and c != "time":
                cols_show.append(c)
        cols_show = [c for c in cols_show if c in df.columns]

        detail_html = df[cols_show].to_html(index=False, classes="tbl", border=0)
        return f"""
        <h3> outlier theo cặp kho + khung giờ</h3>
        {summary_html}
        <br>
        <details>
          <summary style="cursor:pointer;font-weight:600;color:#2563EB;">
             Xem chi tiết từng bill outlier ({len(df):,} bill)
          </summary>
          <div style="overflow-x:auto;margin-top:8px;">{detail_html}</div>
        </details>
        """

    def pair_detail_tabs(df_out, time_col_name):
        if df_out.empty:
            return ""
        pairs = df_out["pair"].unique().tolist()
        tabs_html = ""
        for pair in pairs:
            sub = df_out[df_out["pair"] == pair]
            cap = sub["p99_cap_h"].iloc[0]
            cols_show = ["bill_code", "time", "p99_cap_h", "hour_at_1a"]
            for c in sub.columns:
                if c not in cols_show and c not in ["pair", "pct_label"] and "time" in c.lower():
                    cols_show.append(c)
            cols_show = [c for c in cols_show if c in sub.columns]

            tabs_html += f"""
            <div class="pair-block">
              <h4> {pair} &nbsp;<span style="color:#6B7280;font-weight:400;font-size:13px;">— {len(sub):,} bill outlier | P99 ngưỡng = {cap:.1f}h</span></h4>
              <div style="overflow-x:auto;">{sub[cols_show].to_html(index=False, classes='tbl', border=0)}</div>
            </div>"""
        return tabs_html

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo Outlier Bill</title>
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif; margin:0; padding:20px; background:#f8fafc; color:#1f2937; }}
  h1   {{ color:#1e3a5f; border-bottom:3px solid #2563EB; padding-bottom:10px; }}
  h2   {{ color:#2563EB; margin-top:30px; border-left:4px solid #2563EB; padding-left:10px; }}
  h3   {{ color:#374151; }}
  h4   {{ color:#1e3a5f; background:#EFF6FF; padding:8px 12px; border-radius:6px; margin:18px 0 6px; }}
  .tbl {{ border-collapse:collapse; width:100%; font-size:12px; }}
  .tbl th {{ background:#2563EB; color:#fff; padding:7px 10px; text-align:left; position:sticky; top:0; }}
  .tbl td {{ border-bottom:1px solid #E5E7EB; padding:5px 10px; }}
  .tbl tr:hover td {{ background:#EFF6FF; }}
  .pair-block {{ margin:16px 0; padding:12px; background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  summary:hover {{ opacity:.8; }}
  .tabs {{ display:flex; gap:10px; margin-bottom:20px; }}
  .tab-btn {{ padding:8px 20px; border:none; border-radius:6px; cursor:pointer;
               background:#E5E7EB; color:#374151; font-size:14px; font-weight:600; }}
  .tab-btn.active {{ background:#2563EB; color:#fff; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}
</style>
<script>
function switchTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[onclick*="'+id+'"]').classList.add('active');
}}
</script>
</head>
<body>
<h1> Bill Outlier — Top {N_TOP} cặp kho</h1>
<p style="color:#6B7280;">Ngưỡng outlier: <b>time &gt; P{OUTLIER_PCT*100:.0f}</b> của từng cặp kho tương ứng.</p>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('tab-origin')">Kho gửi → Kho 1A ({len(out_origin):,} bill)</button>
  <button class="tab-btn" onclick="switchTab('tab-dest')">Kho 1A → Kho nhận ({len(out_dest):,} bill)</button>
</div>

<div id="tab-origin" class="tab-content active">
  <h2>Kho gửi → Kho 1A (origin_to_1A)</h2>
  {pair_detail_tabs(out_origin, "time_o1a")}
</div>

<div id="tab-dest" class="tab-content">
  <h2>Kho 1A → Kho nhận (1A_to_destination)</h2>
  {pair_detail_tabs(out_dest, "time_d1a")}
</div>

</body></html>"""
    return html


if __name__ == "__main__":
    print("Đọc dữ liệu...")
    df_o = pd.read_csv(os.path.join(INPUT_DIR, "origin_to_1A.csv"))
    df_d = pd.read_csv(os.path.join(INPUT_DIR, "1A_to_destination.csv"))
    print(f"  origin_to_1A : {len(df_o):,} bill")
    print(f"  1A_to_dest   : {len(df_d):,} bill")

    print("\nXử lý chiều origin (kho gửi → kho 1A)...")
    df_o_pairs, top_o_bill = top_pairs(df_o, "kho_o", "kho_o1a", "bill_count")
    pairs_o = top_o_bill["pair"].tolist()

    out_o = find_outliers_for(df_o_pairs, pairs_o, "time_o1a", "origin")
    sum_o = build_summary_table(out_o, "time_o1a")
    print(f"  → Tìm thấy {len(out_o):,} bill outlier trong top {N_TOP} cặp kho")

    print("\nXử lý chiều destination (kho 1A → kho nhận)...")
    df_d_pairs, top_d_bill = top_pairs(df_d, "kho_d1a", "kho_d", "bill_count")
    pairs_d = top_d_bill["pair"].tolist()

    out_d = find_outliers_for(df_d_pairs, pairs_d, "time_d1a", "dest")
    sum_d = build_summary_table(out_d, "time_d1a")
    print(f"  → Tìm thấy {len(out_d):,} bill outlier trong top {N_TOP} cặp kho")

    csv_o = os.path.join(OUTPUT_DIR, "outliers_origin.csv")
    csv_d = os.path.join(OUTPUT_DIR, "outliers_dest.csv")
    out_o.to_csv(csv_o, index=False)
    out_d.to_csv(csv_d, index=False)
    print(f"\nĐã lưu:")
    print(f"  ==> {csv_o}")
    print(f"  ==> {csv_d}")

    html_path = os.path.join(OUTPUT_DIR, "outliers_report.html")
    html = build_html_report(out_o, out_d, sum_o, sum_d)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ==> {html_path}")

    print("\nXong!")
