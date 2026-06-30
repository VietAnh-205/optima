# -*- coding: utf-8 -*-
"""
Phân tích trung chuyển giữa 26 kho đường trục (warehouse_1A.csv)
=================================================================
Input:
    - bill.csv            : Thông tin bill (2.5M rows)
    - bill_schedule.csv   : Lịch trình IN/OUT tại các kho (16M rows)
    - warehouse_1A.csv    : Danh sách 26 kho đường trục

Output (thư mục output_1A/):
    - bill_trunk_traces.csv       : Trace qua kho đường trục của từng bill
    - trunk_legs.csv              : Các chặng trunk liên tiếp
    - matrix_count.csv            : Ma trận số lượng kho A → kho B
    - matrix_pct.csv              : Ma trận % kho A → kho B
    - detail_legs.csv             : Thống kê chi tiết từng chặng
    - first_last_stats.csv        : Thống kê first_trunk → last_trunk (cấp bill)
    - summary_origin.csv          : Tổng hợp kho theo vai trò gửi
    - summary_dest.csv            : Tổng hợp kho theo vai trò nhận
    - heatmap_legs.png            : Heatmap chặng liên tiếp
    - heatmap_first_last.png      : Heatmap first→last
    - top_routes_bar.png          : Bar chart top tuyến
    - sankey.html                 : Sankey diagram (nếu có plotly)
"""

import sys
import os

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ==================== Cấu hình matplotlib ====================
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 200
rcParams["figure.figsize"] = (14, 10)
rcParams["axes.unicode_minus"] = False
for font in ["Segoe UI", "Arial", "DejaVu Sans"]:
    try:
        rcParams["font.family"] = font
        break
    except Exception:
        continue

OUTPUT_DIR = "output_1A"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
#  BƯỚC 1: ĐỌC DỮ LIỆU
# ===========================================================================
def step1_load_data():
    print("=" * 70)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print("=" * 70)

    # 1a. Danh sách 26 kho đường trục
    wh_1a = pd.read_csv("warehouse_1A.csv")
    trunk_set = set(wh_1a["name"].dropna())
    print(f"  warehouse_1A : {len(trunk_set)} kho đường trục")
    for i, name in enumerate(sorted(trunk_set), 1):
        t = wh_1a.loc[wh_1a["name"] == name, "type"].values[0]
        p = wh_1a.loc[wh_1a["name"] == name, "province_name"].values[0]
        print(f"    {i:>2}. {name:<35s}  [{t}]  ({p})")

    # 1b. bill.csv (chỉ lấy cột cần thiết)
    print("\n  Đang đọc bill.csv ...")
    bill = pd.read_csv(
        "bill.csv",
        usecols=["bill_code", "actual_weight", "service",
                 "origin_province", "destination_province"],
    )
    print(f"  bill         : {bill.shape[0]:>12,} rows")

    # 1c. bill_schedule.csv — file lớn, chỉ đọc cột cần
    print("  Đang đọc bill_schedule.csv (có thể mất vài phút) ...")
    schedule = pd.read_csv(
        "bill_schedule.csv",
        usecols=["bill_code", "io_status", "io_time", "warehouse_name"],
    )
    schedule["io_time"] = pd.to_datetime(schedule["io_time"], errors="coerce")
    print(f"  bill_schedule: {schedule.shape[0]:>12,} rows")

    return trunk_set, wh_1a, bill, schedule


# ===========================================================================
#  BƯỚC 2: LỌC SCHEDULE CHỈ GIỮ 26 KHO ĐƯỜNG TRỤC
# ===========================================================================
def step2_filter_trunk_schedule(schedule, trunk_set):
    print("\n" + "=" * 70)
    print("BƯỚC 2: LỌC SCHEDULE CHỈ GIỮ 26 KHO ĐƯỜNG TRỤC")
    print("=" * 70)

    trunk_sched = schedule[schedule["warehouse_name"].isin(trunk_set)].copy()
    trunk_sched = trunk_sched.sort_values(["bill_code", "io_time"])

    n_bills = trunk_sched["bill_code"].nunique()
    print(f"  Records liên quan 26 kho   : {len(trunk_sched):>12,}")
    print(f"  Bills đi qua ≥ 1 kho ĐT  : {n_bills:>12,}")
    print(f"  Tổng bill trong schedule   : {schedule['bill_code'].nunique():>12,}")
    print(f"  Tỷ lệ                      : {n_bills / schedule['bill_code'].nunique() * 100:.1f}%")

    # Thống kê nhanh: kho nào xuất hiện nhiều nhất
    print("\n  Số records theo kho:")
    wh_counts = trunk_sched["warehouse_name"].value_counts()
    for name, cnt in wh_counts.items():
        print(f"    {name:<35s}  {cnt:>10,}")

    return trunk_sched


# ===========================================================================
#  BƯỚC 3: XÂY DỰNG TRACE KHO ĐƯỜNG TRỤC CHO TỪNG BILL
# ===========================================================================
def step3_build_traces(trunk_sched):
    print("\n" + "=" * 70)
    print("BƯỚC 3: XÂY DỰNG TRACE KHO ĐƯỜNG TRỤC CHO TỪNG BILL")
    print("=" * 70)

    # Unique kho per bill (giữ thứ tự thời gian, bỏ lặp liên tiếp)
    # Trước tiên, drop duplicate liên tiếp (cùng bill + cùng kho)
    trunk_sched = trunk_sched.sort_values(["bill_code", "io_time"])
    trunk_unique = trunk_sched.drop_duplicates(
        subset=["bill_code", "warehouse_name"], keep="first"
    )

    # Nhóm theo bill: trunk_route, first_trunk, last_trunk, n_trunk_stops
    grp = trunk_unique.groupby("bill_code")["warehouse_name"]

    first_trunk = grp.first().rename("first_trunk")
    last_trunk = grp.last().rename("last_trunk")
    trunk_route = grp.apply(lambda x: " → ".join(x)).rename("trunk_route")
    n_trunk = grp.nunique().rename("n_trunk_stops")

    traces = pd.concat([first_trunk, last_trunk, trunk_route, n_trunk], axis=1).reset_index()

    print(f"  Tổng bill có trace kho ĐT  : {len(traces):>12,}")
    print(f"\n  Phân phối số kho ĐT (n_trunk_stops):")
    dist = traces["n_trunk_stops"].value_counts().sort_index()
    for stops, cnt in dist.items():
        pct = cnt / len(traces) * 100
        print(f"    {stops} kho ĐT : {cnt:>10,}  ({pct:.1f}%)")

    print(f"\n  Top 20 trunk_route phổ biến:")
    top_routes = traces["trunk_route"].value_counts().head(20)
    for i, (route, cnt) in enumerate(top_routes.items(), 1):
        print(f"    {i:>2}. [{cnt:>8,}]  {route}")

    # Lưu
    traces.to_csv(os.path.join(OUTPUT_DIR, "bill_trunk_traces.csv"), index=False)
    print(f"\n  ✓ Đã lưu bill_trunk_traces.csv ({len(traces):,} rows)")

    return traces


# ===========================================================================
#  BƯỚC 4: TÁCH CÁC CHẶNG TRUNK LIÊN TIẾP (LEGS)
# ===========================================================================
def step4_extract_legs(traces):
    print("\n" + "=" * 70)
    print("BƯỚC 4: TÁCH CÁC CHẶNG TRUNK LIÊN TIẾP (LEGS)")
    print("=" * 70)

    # Chỉ bill có ≥ 2 kho đường trục mới có leg
    multi = traces[traces["n_trunk_stops"] >= 2].copy()
    print(f"  Bills có ≥ 2 kho ĐT: {len(multi):>10,}")

    # Tách route thành list
    multi["nodes"] = multi["trunk_route"].str.split(" → ")

    # Explode
    exploded = multi[["bill_code", "nodes"]].explode("nodes")
    exploded["pos"] = exploded.groupby("bill_code").cumcount()
    exploded = exploded.rename(columns={"nodes": "origin_trunk"})

    # Self-join: pos_i → pos_i+1
    shifted = exploded.copy()
    shifted["pos"] = shifted["pos"] - 1
    shifted = shifted.rename(columns={"origin_trunk": "dest_trunk"})

    legs = exploded.merge(
        shifted[["bill_code", "pos", "dest_trunk"]],
        on=["bill_code", "pos"],
        how="inner",
    )
    legs = legs[["bill_code", "origin_trunk", "dest_trunk"]].reset_index(drop=True)

    # Loại bỏ self-loops (nếu có lỗi dữ liệu)
    legs = legs[legs["origin_trunk"] != legs["dest_trunk"]]

    print(f"  Tổng legs (chặng trunk liên tiếp): {len(legs):>10,}")

    legs.to_csv(os.path.join(OUTPUT_DIR, "trunk_legs.csv"), index=False)
    print(f"  ✓ Đã lưu trunk_legs.csv")

    return legs


# ===========================================================================
#  BƯỚC 5: MA TRẬN CHUYỂN ĐỔI KHO A → KHO B (THEO CHẶNG)
# ===========================================================================
def step5_leg_matrix(legs, trunk_set):
    print("\n" + "=" * 70)
    print("BƯỚC 5: MA TRẬN CHUYỂN ĐỔI KHO A → KHO B (THEO CHẶNG)")
    print("=" * 70)

    # Ma trận count
    matrix_count = legs.groupby(["origin_trunk", "dest_trunk"]).size().unstack(fill_value=0)

    # Đảm bảo đủ 26 kho (thêm hàng/cột 0 nếu kho không xuất hiện)
    all_trunks = sorted(trunk_set)
    matrix_count = matrix_count.reindex(index=all_trunks, columns=all_trunks, fill_value=0)

    # Ma trận phần trăm (theo hàng)
    row_sums = matrix_count.sum(axis=1)
    matrix_pct = matrix_count.div(row_sums.replace(0, np.nan), axis=0) * 100
    matrix_pct = matrix_pct.fillna(0).round(2)

    print(f"  Kích thước ma trận: {matrix_count.shape[0]} × {matrix_count.shape[1]}")
    print(f"  Kho có gửi đi (row sum > 0): {(row_sums > 0).sum()}")
    print(f"  Kho có nhận (col sum > 0)  : {(matrix_count.sum(axis=0) > 0).sum()}")

    matrix_count.to_csv(os.path.join(OUTPUT_DIR, "matrix_count.csv"))
    matrix_pct.to_csv(os.path.join(OUTPUT_DIR, "matrix_pct.csv"))
    print("  ✓ Đã lưu matrix_count.csv + matrix_pct.csv")

    # Bảng chi tiết
    detail = (
        legs.groupby(["origin_trunk", "dest_trunk"])
        .agg(so_chang=("bill_code", "size"), so_bill_unique=("bill_code", "nunique"))
        .reset_index()
        .sort_values("so_chang", ascending=False)
    )
    detail["pct_trong_tong"] = (detail["so_chang"] / detail["so_chang"].sum() * 100).round(2)
    total_by_origin = detail.groupby("origin_trunk")["so_chang"].transform("sum")
    detail["pct_theo_origin"] = (detail["so_chang"] / total_by_origin * 100).round(2)
    detail.to_csv(os.path.join(OUTPUT_DIR, "detail_legs.csv"), index=False)
    print("  ✓ Đã lưu detail_legs.csv")

    print(f"\n  Top 30 chặng phổ biến nhất:")
    for i, (_, r) in enumerate(detail.head(30).iterrows(), 1):
        print(
            f"    {i:>2}. {r['origin_trunk']:<30s} → {r['dest_trunk']:<30s}  "
            f"{r['so_chang']:>8,} legs  ({r['pct_theo_origin']:>6.2f}% từ origin)"
        )

    return matrix_count, matrix_pct, detail


# ===========================================================================
#  BƯỚC 6: THỐNG KÊ FIRST_TRUNK → LAST_TRUNK (CẤP BILL)
# ===========================================================================
def step6_first_last_stats(traces, bill):
    print("\n" + "=" * 70)
    print("BƯỚC 6: THỐNG KÊ FIRST_TRUNK → LAST_TRUNK (CẤP BILL)")
    print("=" * 70)

    # Chỉ bill có ≥ 2 kho (tức first ≠ last)
    fl = traces[traces["n_trunk_stops"] >= 2].copy()
    fl = fl[fl["first_trunk"] != fl["last_trunk"]]  # loại trường hợp quay lại kho cũ
    print(f"  Bills có first_trunk ≠ last_trunk: {len(fl):,}")

    # Merge thêm info
    fl = fl.merge(
        bill[["bill_code", "actual_weight", "service",
              "origin_province", "destination_province"]],
        on="bill_code", how="left",
    )

    stats = (
        fl.groupby(["first_trunk", "last_trunk"])
        .agg(
            so_bill=("bill_code", "nunique"),
            tong_kg=("actual_weight", "sum"),
            kg_trung_binh=("actual_weight", "mean"),
        )
        .reset_index()
        .sort_values("so_bill", ascending=False)
    )
    total_by_first = stats.groupby("first_trunk")["so_bill"].transform("sum")
    stats["pct_theo_origin"] = (stats["so_bill"] / total_by_first * 100).round(2)
    stats["tong_kg"] = stats["tong_kg"].round(1)
    stats["kg_trung_binh"] = stats["kg_trung_binh"].round(2)

    stats.to_csv(os.path.join(OUTPUT_DIR, "first_last_stats.csv"), index=False)
    print("  ✓ Đã lưu first_last_stats.csv")

    print(f"\n  Top 30 tuyến first→last phổ biến nhất:")
    for i, (_, r) in enumerate(stats.head(30).iterrows(), 1):
        print(
            f"    {i:>2}. {r['first_trunk']:<30s} → {r['last_trunk']:<30s}  "
            f"{r['so_bill']:>8,} bills  ({r['pct_theo_origin']:>6.2f}%)"
        )

    # Phân bổ chi tiết từ mỗi origin
    print("\n" + "-" * 70)
    print("  PHÂN BỔ % TỪ MỖI KHO ĐƯỜNG TRỤC ORIGIN")
    print("-" * 70)
    for origin in sorted(stats["first_trunk"].unique()):
        sub = stats[stats["first_trunk"] == origin].sort_values("pct_theo_origin", ascending=False)
        total_bills = sub["so_bill"].sum()
        print(f"\n  ▶ {origin}  ({total_bills:,} bills)")
        for _, r in sub.head(10).iterrows():
            bar = "█" * int(r["pct_theo_origin"] / 2)
            print(f"    → {r['last_trunk']:<30s} {r['so_bill']:>8,} bills  ({r['pct_theo_origin']:>6.2f}%)  {bar}")
        if len(sub) > 10:
            print(f"    ... và {len(sub) - 10} kho đích khác")

    return stats, fl


# ===========================================================================
#  BƯỚC 7: TỔNG HỢP TỪNG KHO ĐƯỜNG TRỤC
# ===========================================================================
def step7_summary(stats):
    print("\n" + "=" * 70)
    print("BƯỚC 7: TỔNG HỢP TỪNG KHO ĐƯỜNG TRỤC")
    print("=" * 70)

    origin_summary = (
        stats.groupby("first_trunk")
        .agg(tong_bill_gui=("so_bill", "sum"),
             so_kho_dich=("last_trunk", "nunique"),
             tong_kg=("tong_kg", "sum"))
        .reset_index()
        .rename(columns={"first_trunk": "kho_duong_truc"})
        .sort_values("tong_bill_gui", ascending=False)
    )

    dest_summary = (
        stats.groupby("last_trunk")
        .agg(tong_bill_nhan=("so_bill", "sum"),
             so_kho_nguon=("first_trunk", "nunique"),
             tong_kg=("tong_kg", "sum"))
        .reset_index()
        .rename(columns={"last_trunk": "kho_duong_truc"})
        .sort_values("tong_bill_nhan", ascending=False)
    )

    print("\n  --- Kho đường trục theo vai trò GỬI ĐI (Origin) ---")
    for _, r in origin_summary.iterrows():
        print(f"    {r['kho_duong_truc']:<35s}  {r['tong_bill_gui']:>10,} bills → {r['so_kho_dich']:>2} kho đích  ({r['tong_kg']:>12,.0f} kg)")

    print("\n  --- Kho đường trục theo vai trò NHẬN (Destination) ---")
    for _, r in dest_summary.iterrows():
        print(f"    {r['kho_duong_truc']:<35s}  {r['tong_bill_nhan']:>10,} bills ← {r['so_kho_nguon']:>2} kho nguồn  ({r['tong_kg']:>12,.0f} kg)")

    origin_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_origin.csv"), index=False)
    dest_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_dest.csv"), index=False)
    print("\n  ✓ Đã lưu summary_origin.csv + summary_dest.csv")

    return origin_summary, dest_summary


# ===========================================================================
#  BƯỚC 8: HEATMAP
# ===========================================================================
def step8_heatmap(matrix_pct, matrix_count, stats, trunk_set):
    print("\n" + "=" * 70)
    print("BƯỚC 8: VẼ BIỂU ĐỒ HEATMAP")
    print("=" * 70)

    # --- 8a. Heatmap chặng liên tiếp (legs) ---
    # Lọc chỉ các kho có dữ liệu
    active_origins = matrix_count.index[matrix_count.sum(axis=1) > 0]
    active_dests = matrix_count.columns[matrix_count.sum(axis=0) > 0]
    all_active = sorted(set(active_origins) | set(active_dests))

    heatmap_data = matrix_pct.reindex(index=all_active, columns=all_active).fillna(0)

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "% hang tu Origin Trunk"},
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Ma tran chuyen doi: Kho Duong Truc A -> Kho Duong Truc B (% theo Origin)\n"
        f"26 kho tu warehouse_1A — {len(all_active)} kho co du lieu",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Kho Dich (Dest Trunk)", fontsize=11)
    ax.set_ylabel("Kho Nguon (Origin Trunk)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_legs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu {path}")

    # --- 8b. Heatmap first→last (cấp bill) ---
    pivot_count = stats.pivot_table(
        index="first_trunk", columns="last_trunk",
        values="so_bill", aggfunc="sum", fill_value=0
    )
    pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100
    # Reindex đầy đủ
    all_fl = sorted(set(pivot_count.index) | set(pivot_count.columns))
    pivot_pct = pivot_pct.reindex(index=all_fl, columns=all_fl).fillna(0)

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        pivot_pct,
        annot=True, fmt=".1f", cmap="YlGnBu",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "% bill tu First Trunk"},
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Ma tran First Trunk -> Last Trunk (% bill theo Origin)\n"
        f"26 kho tu warehouse_1A",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Last Trunk (Kho cuoi cung)", fontsize=11)
    ax.set_ylabel("First Trunk (Kho dau tien)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_first_last.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu {path}")


# ===========================================================================
#  BƯỚC 9: BAR CHART TOP ROUTES
# ===========================================================================
def step9_top_routes_bar(stats):
    print("\n" + "=" * 70)
    print("BƯỚC 9: VẼ BAR CHART TOP TUYẾN")
    print("=" * 70)

    top_n = 30
    top = stats.head(top_n).copy()
    top["route_label"] = top["first_trunk"] + " → " + top["last_trunk"]

    fig, ax = plt.subplots(figsize=(14, 12))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top)))
    bars = ax.barh(
        range(len(top)), top["so_bill"],
        color=colors, edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["route_label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("So luong Bill", fontsize=11)
    ax.set_title(
        f"Top {top_n} Tuyen Kho Duong Truc Pho Bien Nhat\n"
        "(First Trunk → Last Trunk, 26 kho warehouse_1A)",
        fontsize=13, fontweight="bold"
    )

    for bar, pct in zip(bars, top["pct_theo_origin"]):
        ax.text(
            bar.get_width() + 200,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}  ({pct:.1f}%)",
            va="center", fontsize=7,
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "top_routes_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu {path}")


# ===========================================================================
#  BƯỚC 10: SANKEY DIAGRAM
# ===========================================================================
def step10_sankey(stats):
    print("\n" + "=" * 70)
    print("BƯỚC 10: SANKEY DIAGRAM")
    print("=" * 70)

    try:
        import plotly.graph_objects as go

        sankey_data = stats.head(40).copy()
        all_nodes = list(set(sankey_data["first_trunk"]) | set(sankey_data["last_trunk"]))
        node_idx = {name: i for i, name in enumerate(all_nodes)}

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20, thickness=20,
                    label=all_nodes,
                    color="rgba(31, 119, 180, 0.8)",
                ),
                link=dict(
                    source=[node_idx[r["first_trunk"]] for _, r in sankey_data.iterrows()],
                    target=[node_idx[r["last_trunk"]] for _, r in sankey_data.iterrows()],
                    value=sankey_data["so_bill"].tolist(),
                    color="rgba(31, 119, 180, 0.3)",
                ),
            )
        )
        fig.update_layout(
            title="Sankey: Luong hang giua 26 kho duong truc (Top 40 tuyen)",
            font=dict(size=10), width=1400, height=900,
        )
        path = os.path.join(OUTPUT_DIR, "sankey.html")
        fig.write_html(path)
        print(f"  ✓ Đã lưu {path}")
    except ImportError:
        print("  ⚠ plotly chưa được cài đặt → bỏ qua Sankey.")


# ===========================================================================
#  BƯỚC 11: TÓM TẮT
# ===========================================================================
def step11_summary(traces, legs, stats, matrix_count):
    print("\n" + "=" * 70)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 70)

    total_bills = len(traces)
    bills_multi = (traces["n_trunk_stops"] >= 2).sum()

    print(f"  Tổng bill đi qua ≥ 1 kho ĐT (26 kho)   : {total_bills:>12,}")
    print(f"  Bill đi qua ≥ 2 kho ĐT                   : {bills_multi:>12,} ({bills_multi/total_bills*100:.1f}%)")
    print(f"  Tổng chặng liên tiếp (legs)               : {len(legs):>12,}")
    print(f"  Số tuyến first→last duy nhất              : {len(stats):>12,}")
    print(f"  Số kho có gửi (origin)                    : {(matrix_count.sum(axis=1) > 0).sum():>12,}")
    print(f"  Số kho có nhận (dest)                     : {(matrix_count.sum(axis=0) > 0).sum():>12,}")

    if len(stats) > 0:
        top = stats.iloc[0]
        print(f"\n  Tuyến phổ biến nhất: {top['first_trunk']} → {top['last_trunk']}")
        print(f"    → {int(top['so_bill']):,} bills ({top['pct_theo_origin']:.1f}% từ origin)")

    print(f"\n  Các file output (thư mục {OUTPUT_DIR}/):")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    📄 {f:<40s} ({size_mb:.1f} MB)")

    print("\n✅ Hoàn tất phân tích 26 kho đường trục!")


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    # 1. Load
    trunk_set, wh_1a, bill, schedule = step1_load_data()

    # 2. Lọc schedule
    trunk_sched = step2_filter_trunk_schedule(schedule, trunk_set)
    del schedule  # giải phóng memory

    # 3. Build traces
    traces = step3_build_traces(trunk_sched)
    del trunk_sched

    # 4. Extract legs
    legs = step4_extract_legs(traces)

    # 5. Ma trận chuyển đổi (theo chặng)
    matrix_count, matrix_pct, detail = step5_leg_matrix(legs, trunk_set)

    # 6. First→Last stats (cấp bill)
    stats, fl_df = step6_first_last_stats(traces, bill)

    # 7. Tổng hợp
    step7_summary(stats)

    # 8. Heatmap
    step8_heatmap(matrix_pct, matrix_count, stats, trunk_set)

    # 9. Bar chart
    step9_top_routes_bar(stats)

    # 10. Sankey
    step10_sankey(stats)

    # 11. Tóm tắt
    step11_summary(traces, legs, stats, matrix_count)


if __name__ == "__main__":
    main()
