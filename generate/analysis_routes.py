"""
analysis_routes.py
==================
Phân tích phân phối cung đường (routes) cho từng cặp kho (bưu cục nguồn → bưu cục đích).

Mục tiêu:
  - Với mỗi cặp (first_trunk, last_trunk) có >= MIN_BILL bill, liệt kê tất cả
    các cung đường (trunk_route) mà hàng đã đi, kèm tỉ lệ % của từng cung.
  - Xuất 2 CSV được sắp xếp rõ ràng để nhìn thấy quy luật ngay trong Excel.

Output:
  output_routes/
    ├── route_distribution.csv   # Chi tiết từng route của từng cặp
    └── pair_summary.csv         # Tổng hợp mỗi cặp: entropy, % cung chính, ...
"""

import sys
import os
import io
import math
import warnings

import numpy as np
import pandas as pd

# Đảm bảo stdout UTF-8 trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

# ─── Cấu hình ────────────────────────────────────────────────────────────────
TRACES_CSV   = os.path.join("output_all_traces", "bill_trunk_traces.csv")
WAREHOUSE_CSV = "warehouse.csv"
OUTPUT_DIR   = "output_routes"
MIN_BILL     = 100      # Chỉ phân tích cặp có >= MIN_BILL bill
CHUNKSIZE    = 500_000  # Đọc file lớn theo chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Step 1: Load dữ liệu ─────────────────────────────────────────────────────
def step1_load(traces_csv: str, warehouse_csv: str):
    """
    Đọc bill_trunk_traces.csv theo chunk (file ~300MB).
    Chỉ giữ các cột cần thiết.
    """
    print("▶ Step 1: Đọc dữ liệu...")

    wh = pd.read_csv(warehouse_csv)
    # Map kho → tỉnh (dùng làm metadata bổ sung)
    wh_prov = wh.set_index("name")["province_name"].to_dict()

    cols = ["bill_code", "first_trunk", "last_trunk", "trunk_route", "n_trunk_stops"]
    chunks = []
    for chunk in pd.read_csv(traces_csv, usecols=cols, chunksize=CHUNKSIZE):
        chunks.append(chunk)    
    traces = pd.concat(chunks, ignore_index=True)

    print(f"  ✓ Đọc xong: {len(traces):,} bills")
    return traces, wh_prov


# ─── Step 2: Lọc cặp >= MIN_BILL ─────────────────────────────────────────────
def step2_filter_pairs(traces: pd.DataFrame) -> pd.DataFrame:
    """
    Giữ lại chỉ các bills thuộc cặp (first_trunk, last_trunk) có >= MIN_BILL bill.
    """
    print(f"▶ Step 2: Lọc cặp có >= {MIN_BILL} bill...")

    # Chỉ xét bill có ít nhất 2 trạm (first ≠ last có nghĩa)
    df = traces[traces["n_trunk_stops"] >= 2].copy()
    df = df[df["first_trunk"] != df["last_trunk"]]

    pair_counts = df.groupby(["first_trunk", "last_trunk"])["bill_code"].nunique()
    valid_pairs = pair_counts[pair_counts >= MIN_BILL].index
    df = df.set_index(["first_trunk", "last_trunk"]).loc[valid_pairs].reset_index()

    print(f"  ✓ {len(valid_pairs):,} cặp hợp lệ | {len(df):,} bills")
    return df


# ─── Step 3: Tính route_distribution ─────────────────────────────────────────
def step3_route_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group theo (first_trunk, last_trunk, trunk_route) → đếm bill, tính %.
    Thêm: rank trong cặp, số trạm trung gian.
    """
    print("▶ Step 3: Tính phân phối route cho từng cặp...")

    grp = (
        df.groupby(["first_trunk", "last_trunk", "trunk_route"])
        ["bill_code"].nunique()
        .rename("so_bill")
        .reset_index()
    )

    # Tổng bill của từng cặp (first, last)
    pair_total = grp.groupby(["first_trunk", "last_trunk"])["so_bill"].transform("sum")
    grp["pct_trong_cap"] = (grp["so_bill"] / pair_total * 100).round(2)

    # Số trạm = số "→" + 1
    grp["n_stops"] = grp["trunk_route"].str.count("→") + 1

    # Rank trong cặp (1 = phổ biến nhất)
    grp["rank_trong_cap"] = (
        grp.groupby(["first_trunk", "last_trunk"])["so_bill"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Sắp xếp: theo cặp, trong cặp theo rank tăng dần
    grp = grp.sort_values(
        ["first_trunk", "last_trunk", "rank_trong_cap"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "route_distribution.csv")
    grp.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig để Excel đọc đúng tiếng Việt
    print(f"  ✓ Đã lưu {out_path} ({len(grp):,} dòng)")
    return grp


# ─── Step 4: Tính pair_summary ────────────────────────────────────────────────
def _entropy(series: pd.Series) -> float:
    """Shannon entropy (bits) của phân phối tần suất."""
    p = series / series.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def step4_pair_summary(dist: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp theo từng cặp (first_trunk, last_trunk):
      - Tổng bill
      - Số routes khác nhau
      - % cung chính (route #1)
      - % top 2 routes
      - Entropy (đa dạng hóa)
      - Route chính (tên cung phổ biến nhất)
    """
    print("▶ Step 4: Tính pair_summary...")

    records = []
    for (ft, lt), sub in dist.groupby(["first_trunk", "last_trunk"]):
        sub = sub.sort_values("so_bill", ascending=False)
        total       = sub["so_bill"].sum()
        n_routes    = len(sub)
        top1_bill   = sub.iloc[0]["so_bill"]
        top1_route  = sub.iloc[0]["trunk_route"]
        top1_stops  = sub.iloc[0]["n_stops"]
        pct_top1    = round(top1_bill / total * 100, 2)
        pct_top2    = round(sub.head(2)["so_bill"].sum() / total * 100, 2)
        ent         = round(_entropy(sub["so_bill"]), 4)

        records.append({
            "first_trunk":          ft,
            "last_trunk":           lt,
            "so_bill_tong":         total,
            "so_route_khac_nhau":   n_routes,
            "route_chinh":          top1_route,
            "so_tram_route_chinh":  top1_stops,
            "pct_route_chinh":      pct_top1,
            "pct_top2_route":       pct_top2,
            "entropy_bits":         ent,
        })

    summary = (
        pd.DataFrame(records)
        .sort_values("so_bill_tong", ascending=False)
        .reset_index(drop=True)
    )

    out_path = os.path.join(OUTPUT_DIR, "pair_summary.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Đã lưu {out_path} ({len(summary):,} cặp)")
    return summary


# ─── Step 5: Xuất thêm bảng "flat" tiện đọc trên Excel ──────────────────────
def step5_excel_friendly(dist: pd.DataFrame, summary: pd.DataFrame):
    """
    Tạo bảng kết hợp: mỗi cặp được liệt kê kèm tổng bill,
    rồi ngay dưới là các route của cặp đó.

    Format:
      first_trunk | last_trunk | so_bill_tong | trunk_route | so_bill_route | pct | rank
    ─────────────────────────────────────────────────────────────────────────────────────
      Kho A       | Kho B      | 6935         | A → B       | 5800          | 83.6| 1
      (same)      | (same)     | (same)       | A → X → B   | 1135          | 16.4| 2
      ...

    Sắp xếp: cặp theo so_bill_tong DESC, route theo pct DESC.
    """
    print("▶ Step 5: Tạo bảng Excel-friendly...")

    # Gắn so_bill_tong vào dist để sort theo tổng bill của cặp
    pair_total_map = summary.set_index(["first_trunk", "last_trunk"])["so_bill_tong"]
    dist2 = dist.copy()
    dist2["so_bill_tong"] = dist2.set_index(["first_trunk", "last_trunk"]).index.map(pair_total_map).values

    dist2 = dist2.sort_values(
        ["so_bill_tong", "first_trunk", "last_trunk", "rank_trong_cap"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    # Cột cuối: so_bill_tong chỉ hiện ở dòng đầu của mỗi cặp (dễ đọc hơn)
    dist2["so_bill_cap"] = dist2["so_bill_tong"]

    out_cols = [
        "first_trunk", "last_trunk", "so_bill_cap",
        "trunk_route", "so_bill", "pct_trong_cap", "n_stops", "rank_trong_cap"
    ]
    flat = dist2[out_cols].rename(columns={
        "so_bill_cap":   "so_bill_tong_cap",
        "so_bill":       "so_bill_route",
        "pct_trong_cap": "pct_route",
        "n_stops":       "so_tram",
        "rank_trong_cap":"rank",
    })

    out_path = os.path.join(OUTPUT_DIR, "route_distribution_flat.csv")
    flat.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Đã lưu {out_path} (bảng flat, {len(flat):,} dòng)")
    return flat


# ─── Step 6: In báo cáo tóm tắt ra terminal ──────────────────────────────────
def step6_report(summary: pd.DataFrame):
    print()
    print("=" * 65)
    print("BÁO CÁO TÓM TẮT")
    print("=" * 65)

    total_pairs = len(summary)
    print(f"Tổng cặp (first_trunk, last_trunk) có >= {MIN_BILL} bill: {total_pairs:,}")

    # Phân loại theo số routes
    bins = [1, 2, 3, 6, 11, 9999]
    labels = ["1 route (cố định)", "2 routes", "3–5 routes", "6–10 routes", ">10 routes"]
    summary["nhom_route"] = pd.cut(summary["so_route_khac_nhau"], bins=bins, labels=labels, right=False)
    group_cnt = summary["nhom_route"].value_counts().sort_index()
    print()
    print("Phân loại cặp theo số cung đường:")
    for label, cnt in group_cnt.items():
        pct = cnt / total_pairs * 100
        print(f"  {label:<25} : {cnt:5,} cặp ({pct:.1f}%)")

    # Top 10 cặp nhiều bill nhất
    print()
    print("Top 10 cặp nhiều bill nhất:")
    top10 = summary.head(10)[
        ["first_trunk", "last_trunk", "so_bill_tong",
         "so_route_khac_nhau", "pct_route_chinh", "entropy_bits"]
    ]
    print(top10.to_string(index=False))

    # Cặp có entropy cao nhất (đi lộn xộn nhiều đường)
    print()
    print("Top 5 cặp có NHIỀU đường đi nhất (entropy cao):")
    top_ent = summary.nlargest(5, "entropy_bits")[
        ["first_trunk", "last_trunk", "so_bill_tong",
         "so_route_khac_nhau", "pct_route_chinh", "entropy_bits"]
    ]
    print(top_ent.to_string(index=False))

    # Cặp cố định 1 đường hoàn toàn (pct_route_chinh = 100)
    fixed = summary[summary["pct_route_chinh"] == 100.0]
    print()
    print(f"Số cặp luôn đi 1 cung duy nhất (pct_route_chinh=100%): {len(fixed):,}")

    print()
    print("Output files:")
    for f in ["route_distribution.csv", "route_distribution_flat.csv", "pair_summary.csv"]:
        path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path}  ({size_mb:.1f} MB)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    traces, wh_prov = step1_load(TRACES_CSV, WAREHOUSE_CSV)
    df              = step2_filter_pairs(traces)
    del traces      # giải phóng RAM

    dist    = step3_route_distribution(df)
    summary = step4_pair_summary(dist)
    _       = step5_excel_friendly(dist, summary)
    step6_report(summary)


if __name__ == "__main__":
    main()
