# -*- coding: utf-8 -*-
"""
Phân tích luồng hàng hóa theo Tỉnh/Thành phố
==============================================
Câu hỏi trả lời:
  Q1. Từ tỉnh A, bao nhiêu % hàng vào kho đường trục nào (kho đầu tiên)?
  Q2. Từ kho đường trục X, bao nhiêu % hàng tiếp tục đi tới kho Y ở tỉnh khác?
  Q3. Từ tỉnh A → kho X → kho Y → tỉnh B : ma trận tỉnh-tỉnh đi qua kho nào?
"""

import sys, os
if sys.platform == "win64":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── matplotlib setup ───────────────────────────────────────────────────────
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 200
rcParams["axes.unicode_minus"] = False
for font in ["Segoe UI", "Arial", "DejaVu Sans"]:
    try: rcParams["font.family"] = font; break
    except: pass

OUTPUT_DIR = "output_province"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP = "=" * 72


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 1: ĐỌC DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════════
def load_data():
    print(SEP)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print(SEP)

    # 1a. Bill traces (đã tính sẵn)
    print("  Đọc output_1A/bill_trunk_traces.csv ...")
    traces = pd.read_csv(
        "output_1A/bill_trunk_traces.csv",
        usecols=["bill_code", "first_trunk", "last_trunk",
                 "trunk_route", "n_trunk_stops"],
        low_memory=False,
    )
    print(f"  → {len(traces):,} bills có trace kho ĐT")

    # 1b. Bill gốc (tỉnh gửi / nhận, cân nặng)
    print("  Đọc bill.csv ...")
    bill = pd.read_csv(
        "bill.csv",
        usecols=["bill_code", "origin_province", "destination_province",
                 "actual_weight", "service"],
    )
    print(f"  → {len(bill):,} bills")

    # 1c. Warehouse_1A (tên kho → tỉnh)
    wh = pd.read_csv("warehouse_1A.csv")
    wh_prov = wh.set_index("name")["province_name"].to_dict()
    trunk_set = set(wh["name"].dropna())
    print(f"  → {len(trunk_set)} kho đường trục, tỉnh: {wh['province_name'].nunique()}")

    return traces, bill, wh_prov, trunk_set


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 2: GỘP TRACES VỚI THÔNG TIN TỈNH
# ══════════════════════════════════════════════════════════════════════════
def build_master(traces, bill, wh_prov):
    print(f"\n{SEP}")
    print("BƯỚC 2: GỘP DỮ LIỆU → MASTER DATAFRAME")
    print(SEP)

    # Merge
    master = traces.merge(
        bill[["bill_code", "origin_province", "destination_province",
              "actual_weight", "service"]],
        on="bill_code", how="inner",
    )

    # Thêm tỉnh của kho đầu / kho cuối
    master["first_trunk_prov"] = master["first_trunk"].map(wh_prov)
    master["last_trunk_prov"]  = master["last_trunk"].map(wh_prov)

    n = len(master)
    has_ft = master["first_trunk"].notna().sum()
    has_lt = master["last_trunk"].notna().sum()
    print(f"  Master bills      : {n:>12,}")
    print(f"  Có first_trunk    : {has_ft:>12,}  ({has_ft/n:.1%})")
    print(f"  Có last_trunk     : {has_lt:>12,}  ({has_lt/n:.1%})")

    # Thống kê nhanh tỉnh
    print(f"\n  Top 10 tỉnh gửi (origin_province):")
    for prov, cnt in master["origin_province"].value_counts().head(10).items():
        print(f"    {prov:<30s} {cnt:>10,}")
    print(f"\n  Top 10 tỉnh nhận (destination_province):")
    for prov, cnt in master["destination_province"].value_counts().head(10).items():
        print(f"    {prov:<30s} {cnt:>10,}")

    return master


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 3 – Q1: TỈNH GỬI → KHO ĐẦU (FIRST TRUNK)
# ══════════════════════════════════════════════════════════════════════════
def q1_province_to_first_trunk(master):
    print(f"\n{SEP}")
    print("BƯỚC 3 (Q1): TỈNH GỬI → KHO ĐẦU TIÊN (FIRST TRUNK)")
    print(SEP)

    sub = master.dropna(subset=["origin_province", "first_trunk"]).copy()

    stats = (
        sub.groupby(["origin_province", "first_trunk"])
        .agg(so_bill=("bill_code", "nunique"), tong_kg=("actual_weight", "sum"))
        .reset_index()
    )
    total_by_prov = stats.groupby("origin_province")["so_bill"].transform("sum")
    stats["pct_trong_tinh"] = (stats["so_bill"] / total_by_prov * 100).round(2)
    stats["tong_kg"] = stats["tong_kg"].round(1)
    stats = stats.sort_values(["origin_province", "so_bill"], ascending=[True, False])

    stats.to_csv(os.path.join(OUTPUT_DIR, "province_to_first_trunk.csv"), index=False)
    print(f"  ✓ Đã lưu province_to_first_trunk.csv  ({len(stats):,} rows)")

    # In chi tiết từng tỉnh
    print(f"\n  Chi tiết từng tỉnh gửi → kho đầu:")
    for prov, grp in stats.groupby("origin_province"):
        total = grp["so_bill"].sum()
        dominant = grp.iloc[0]
        print(f"\n  ▶ {prov:<28s}  ({total:>8,} bills)")
        for _, r in grp.iterrows():
            bar = "█" * int(r["pct_trong_tinh"] / 2)
            print(
                f"    → {r['first_trunk']:<30s}"
                f"  {r['so_bill']:>8,} bills"
                f"  ({r['pct_trong_tinh']:>6.2f}%)"
                f"  {bar}"
            )

    return stats


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 4 – Q1b: TỈNH NHẬN ← KHO CUỐI (LAST TRUNK)
# ══════════════════════════════════════════════════════════════════════════
def q1b_last_trunk_to_province(master):
    print(f"\n{SEP}")
    print("BƯỚC 4 (Q1b): KHO CUỐI (LAST TRUNK) → TỈNH NHẬN")
    print(SEP)

    sub = master.dropna(subset=["destination_province", "last_trunk"]).copy()

    stats = (
        sub.groupby(["last_trunk", "destination_province"])
        .agg(so_bill=("bill_code", "nunique"), tong_kg=("actual_weight", "sum"))
        .reset_index()
    )
    total_by_trunk = stats.groupby("last_trunk")["so_bill"].transform("sum")
    stats["pct_trong_kho"] = (stats["so_bill"] / total_by_trunk * 100).round(2)
    stats["tong_kg"] = stats["tong_kg"].round(1)
    stats = stats.sort_values(["last_trunk", "so_bill"], ascending=[True, False])

    # Chiều ngược: tỉnh nhận → kho cuối
    stats2 = (
        sub.groupby(["destination_province", "last_trunk"])
        .agg(so_bill=("bill_code", "nunique"), tong_kg=("actual_weight", "sum"))
        .reset_index()
    )
    total_by_prov = stats2.groupby("destination_province")["so_bill"].transform("sum")
    stats2["pct_trong_tinh"] = (stats2["so_bill"] / total_by_prov * 100).round(2)
    stats2["tong_kg"] = stats2["tong_kg"].round(1)
    stats2 = stats2.sort_values(["destination_province", "so_bill"], ascending=[True, False])

    stats.to_csv(os.path.join(OUTPUT_DIR, "last_trunk_to_province.csv"), index=False)
    stats2.to_csv(os.path.join(OUTPUT_DIR, "province_to_last_trunk.csv"), index=False)
    print(f"  ✓ Đã lưu last_trunk_to_province.csv  ({len(stats):,} rows)")
    print(f"  ✓ Đã lưu province_to_last_trunk.csv   ({len(stats2):,} rows)")

    # In chi tiết
    print(f"\n  Chi tiết từng kho cuối → tỉnh nhận:")
    for trunk, grp in stats.groupby("last_trunk"):
        total = grp["so_bill"].sum()
        print(f"\n  ▶ {trunk:<30s}  ({total:>8,} bills nhận)")
        for _, r in grp.head(10).iterrows():
            bar = "█" * int(r["pct_trong_kho"] / 2)
            print(
                f"    → Tỉnh {r['destination_province']:<25s}"
                f"  {r['so_bill']:>8,} bills"
                f"  ({r['pct_trong_kho']:>6.2f}%)"
                f"  {bar}"
            )
        if len(grp) > 10:
            print(f"    ... và {len(grp)-10} tỉnh khác")

    return stats, stats2


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 5 – Q2: KHO X → KHO Y KÈM TỈNH NGUỒN/ĐÍCH
# ══════════════════════════════════════════════════════════════════════════
def q2_trunk_province_flow(master, wh_prov):
    print(f"\n{SEP}")
    print("BƯỚC 5 (Q2): LUỒNG KHO X → KHO Y KÈM TỈNH NGUỒN + TỈNH ĐÍCH")
    print(SEP)

    # Lấy bills có ≥2 kho đường trục
    sub = master[
        master["n_trunk_stops"].fillna(0) >= 2
    ].dropna(subset=["first_trunk", "last_trunk", "origin_province", "destination_province"]).copy()

    # Dùng first→last làm đại diện (chặng quan trọng nhất)
    stats = (
        sub.groupby(["origin_province", "first_trunk",
                     "last_trunk", "destination_province"])
        .agg(so_bill=("bill_code", "nunique"), tong_kg=("actual_weight", "sum"))
        .reset_index()
    )
    stats["tong_kg"] = stats["tong_kg"].round(1)

    # % theo (tỉnh gửi, kho đầu)
    key = stats.groupby(["origin_province", "first_trunk"])["so_bill"].transform("sum")
    stats["pct_theo_origin_kho"] = (stats["so_bill"] / key * 100).round(2)
    stats = stats.sort_values("so_bill", ascending=False)

    stats.to_csv(os.path.join(OUTPUT_DIR, "trunk_province_flow.csv"), index=False)
    print(f"  ✓ Đã lưu trunk_province_flow.csv  ({len(stats):,} rows)")

    print(f"\n  Top 30 luồng (tỉnh gửi, kho đầu) → (kho cuối, tỉnh nhận):")
    for i, (_, r) in enumerate(stats.head(30).iterrows(), 1):
        print(
            f"  {i:>2}. [{r['origin_province']:<18s}]"
            f" {r['first_trunk']:<25s} →"
            f" {r['last_trunk']:<25s}"
            f" [{r['destination_province']:<18s}]"
            f"  {r['so_bill']:>8,} bills"
            f"  ({r['pct_theo_origin_kho']:>6.2f}%)"
        )

    return stats


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 6 – Q3: MA TRẬN TỈNH GỬI × TỈNH NHẬN
# ══════════════════════════════════════════════════════════════════════════
def q3_province_matrix(master):
    print(f"\n{SEP}")
    print("BƯỚC 6 (Q3): MA TRẬN TỈNH GỬI × TỈNH NHẬN")
    print(SEP)

    sub = master.dropna(subset=["origin_province", "destination_province"]).copy()
    # Chỉ cross-province (loại same-province nếu muốn, hoặc giữ all)
    # Giữ tất cả để đầy đủ

    pair_stats = (
        sub.groupby(["origin_province", "destination_province"])
        .agg(
            so_bill=("bill_code", "nunique"),
            tong_kg=("actual_weight", "sum"),
            # kho đầu phổ biến nhất
        )
        .reset_index()
    )

    # Top kho đầu cho mỗi cặp tỉnh
    top_kho = (
        sub.groupby(["origin_province", "destination_province", "first_trunk"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates(["origin_province", "destination_province"])
        .rename(columns={"first_trunk": "top_first_trunk", "cnt": "top_first_trunk_cnt"})
    )
    top_kho2 = (
        sub.groupby(["origin_province", "destination_province", "last_trunk"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates(["origin_province", "destination_province"])
        .rename(columns={"last_trunk": "top_last_trunk", "cnt": "top_last_trunk_cnt"})
    )

    pair_stats = (
        pair_stats
        .merge(top_kho[["origin_province", "destination_province",
                         "top_first_trunk", "top_first_trunk_cnt"]],
               on=["origin_province", "destination_province"], how="left")
        .merge(top_kho2[["origin_province", "destination_province",
                          "top_last_trunk", "top_last_trunk_cnt"]],
               on=["origin_province", "destination_province"], how="left")
    )

    total_by_orig = pair_stats.groupby("origin_province")["so_bill"].transform("sum")
    pair_stats["pct_theo_tinh_gui"] = (pair_stats["so_bill"] / total_by_orig * 100).round(2)
    pair_stats["tong_kg"] = pair_stats["tong_kg"].round(1)
    pair_stats = pair_stats.sort_values("so_bill", ascending=False)

    # Ma trận pivot count
    matrix_count = pair_stats.pivot_table(
        index="origin_province", columns="destination_province",
        values="so_bill", aggfunc="sum", fill_value=0
    )
    matrix_pct = matrix_count.div(matrix_count.sum(axis=1), axis=0) * 100
    matrix_pct = matrix_pct.round(2)

    pair_stats.to_csv(os.path.join(OUTPUT_DIR, "province_pair_via_trunk.csv"), index=False)
    matrix_count.to_csv(os.path.join(OUTPUT_DIR, "province_matrix_count.csv"))
    matrix_pct.to_csv(os.path.join(OUTPUT_DIR, "province_matrix_pct.csv"))
    print(f"  ✓ province_pair_via_trunk.csv  ({len(pair_stats):,} cặp tỉnh)")
    print(f"  ✓ province_matrix_count.csv    ({matrix_count.shape[0]}×{matrix_count.shape[1]})")
    print(f"  ✓ province_matrix_pct.csv")

    print(f"\n  Top 30 cặp tỉnh (gửi → nhận) phổ biến nhất:")
    for i, (_, r) in enumerate(pair_stats.head(30).iterrows(), 1):
        print(
            f"  {i:>2}. {r['origin_province']:<22s} → {r['destination_province']:<22s}"
            f"  {r['so_bill']:>8,} bills"
            f"  ({r['pct_theo_tinh_gui']:>6.2f}%)"
            f"  | kho đầu: {str(r['top_first_trunk']):<28s}"
            f"  kho cuối: {str(r['top_last_trunk'])}"
        )

    return pair_stats, matrix_count, matrix_pct


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 7: HEATMAP – TỈNH GỬI → KHO ĐẦU
# ══════════════════════════════════════════════════════════════════════════
def plot_province_to_first_trunk(p2ft_stats):
    print(f"\n{SEP}")
    print("BƯỚC 7: HEATMAP – TỈNH GỬI → KHO ĐẦU TIÊN")
    print(SEP)

    pivot = p2ft_stats.pivot_table(
        index="origin_province", columns="first_trunk",
        values="pct_trong_tinh", aggfunc="sum", fill_value=0
    )
    # Sắp xếp: tỉnh nhiều bill nhất lên đầu
    prov_order = p2ft_stats.groupby("origin_province")["so_bill"].sum().sort_values(ascending=False).index
    trunk_order = p2ft_stats.groupby("first_trunk")["so_bill"].sum().sort_values(ascending=False).index
    pivot = pivot.reindex(index=prov_order, columns=trunk_order).fillna(0)

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns)*0.9),
                                    max(12, len(pivot)*0.35)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.4, linecolor="white", ax=ax,
        cbar_kws={"label": "% bill tu tinh vao kho"},
        annot_kws={"size": 7},
    )
    ax.set_title("Phan tram bill: Tinh Gui → Kho Duong Truc Dau Tien (%)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Kho Duong Truc Dau (First Trunk)", fontsize=10)
    ax.set_ylabel("Tinh/Thanh pho Gui", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_prov_to_first_trunk.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu {path}")


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 8: HEATMAP – MA TRẬN TỈNH × TỈNH (TOP N)
# ══════════════════════════════════════════════════════════════════════════
def plot_province_matrix(matrix_count, matrix_pct):
    print(f"\n{SEP}")
    print("BƯỚC 8: HEATMAP – MA TRẬN TỈNH GỬI × TỈNH NHẬN (TOP 25)")
    print(SEP)

    TOP_N = 25
    top_origins = matrix_count.sum(axis=1).nlargest(TOP_N).index
    top_dests   = matrix_count.sum(axis=0).nlargest(TOP_N).index
    data = matrix_pct.reindex(index=top_origins, columns=top_dests).fillna(0)
    data = data.loc[
        data.sum(axis=1).sort_values(ascending=False).index,
        data.sum(axis=0).sort_values(ascending=False).index,
    ]

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="YlGnBu",
        linewidths=0.3, linecolor="white", ax=ax,
        cbar_kws={"label": "% bill tu tinh gui"},
        annot_kws={"size": 7},
    )
    ax.set_title(f"Ma tran Tinh Gui × Tinh Nhan (% theo hang, Top {TOP_N})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tinh Nhan (Destination Province)", fontsize=10)
    ax.set_ylabel("Tinh Gui (Origin Province)", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_prov_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu {path}")


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 9: BAR CHART – TOP CẶP TỈNH
# ══════════════════════════════════════════════════════════════════════════
def plot_top_province_pairs(pair_stats):
    print(f"\n{SEP}")
    print("BƯỚC 9: BAR CHART – TOP 30 CẶP TỈNH PHỔ BIẾN")
    print(SEP)

    top = pair_stats.head(30).copy()
    top["label"] = top["origin_province"] + " → " + top["destination_province"]

    fig, ax = plt.subplots(figsize=(14, 11))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(top)))
    bars = ax.barh(range(len(top)), top["so_bill"],
                   color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("So luong Bill", fontsize=10)
    ax.set_title("Top 30 Cap Tinh Pho Bien Nhat\n(Tinh Gui → Tinh Nhan)",
                 fontsize=13, fontweight="bold")

    for bar, row in zip(bars, top.itertuples()):
        ax.text(bar.get_width() + 500,
                bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width()):,}  ({row.pct_theo_tinh_gui:.1f}%)",
                va="center", fontsize=7)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "top_province_pairs_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Đã lưu {path}")


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 10: SANKEY – TỈNH → KHO ĐẦU → KHO CUỐI → TỈNH
# ══════════════════════════════════════════════════════════════════════════
def plot_sankey_province(flow_stats):
    print(f"\n{SEP}")
    print("BƯỚC 10: SANKEY DIAGRAM (TỈNH → KHO → KHO → TỈNH)")
    print(SEP)

    try:
        import plotly.graph_objects as go

        top = flow_stats.head(50).copy()

        # Nodes: tỉnh gửi + kho đầu + kho cuối + tỉnh nhận
        nodes_orig  = [f"[GUI] {p}" for p in top["origin_province"].unique()]
        nodes_ft    = [f"[KHO-DAU] {k}" for k in top["first_trunk"].unique()]
        nodes_lt    = [f"[KHO-CUOI] {k}" for k in top["last_trunk"].unique()]
        nodes_dest  = [f"[NHAN] {p}" for p in top["destination_province"].unique()]
        all_nodes   = nodes_orig + nodes_ft + nodes_lt + nodes_dest
        node_idx    = {n: i for i, n in enumerate(all_nodes)}

        sources, targets, values, labels = [], [], [], []

        for _, r in top.iterrows():
            n_o  = f"[GUI] {r['origin_province']}"
            n_ft = f"[KHO-DAU] {r['first_trunk']}"
            n_lt = f"[KHO-CUOI] {r['last_trunk']}"
            n_d  = f"[NHAN] {r['destination_province']}"

            # tỉnh gửi → kho đầu
            sources.append(node_idx[n_o]);  targets.append(node_idx[n_ft]); values.append(int(r["so_bill"]))
            # kho đầu → kho cuối (nếu khác)
            if r["first_trunk"] != r["last_trunk"]:
                sources.append(node_idx[n_ft]); targets.append(node_idx[n_lt]); values.append(int(r["so_bill"]))
            # kho cuối → tỉnh nhận
            sources.append(node_idx[n_lt]); targets.append(node_idx[n_d]);  values.append(int(r["so_bill"]))

        colors_node = (
            ["rgba(31,119,180,0.8)"] * len(nodes_orig) +
            ["rgba(255,127,14,0.8)"] * len(nodes_ft)   +
            ["rgba(44,160,44,0.8)"]  * len(nodes_lt)   +
            ["rgba(214,39,40,0.8)"]  * len(nodes_dest)
        )

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(pad=15, thickness=18, label=all_nodes, color=colors_node),
            link=dict(source=sources, target=targets, value=values,
                      color="rgba(100,100,100,0.25)"),
        ))
        fig.update_layout(
            title="Sankey: Tinh Gui → Kho Dau → Kho Cuoi → Tinh Nhan (Top 50 luong)",
            font=dict(size=9), width=1400, height=900,
        )
        path = os.path.join(OUTPUT_DIR, "sankey_province.html")
        fig.write_html(path)
        print(f"  ✓ Đã lưu {path}")
    except ImportError:
        print("  ⚠ plotly chưa cài → bỏ qua Sankey.")


# ══════════════════════════════════════════════════════════════════════════
#  BƯỚC 11: TÓM TẮT
# ══════════════════════════════════════════════════════════════════════════
def summarize(master, p2ft_stats, pair_stats, matrix_count):
    print(f"\n{SEP}")
    print("TÓM TẮT")
    print(SEP)

    print(f"  Bills trong master                  : {len(master):>12,}")
    print(f"  Tỉnh gửi unique                     : {master['origin_province'].nunique():>12,}")
    print(f"  Tỉnh nhận unique                    : {master['destination_province'].nunique():>12,}")
    print(f"  Cặp tỉnh-tỉnh duy nhất              : {len(pair_stats):>12,}")
    print(f"  Kho đầu unique                      : {p2ft_stats['first_trunk'].nunique():>12,}")

    print(f"\n  Files output (output_province/):")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            mb = os.path.getsize(fp) / 1e6
            print(f"    📄 {f:<45s} ({mb:.2f} MB)")

    print(f"\n✅ Hoàn tất phân tích tỉnh/thành phố!")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    traces, bill, wh_prov, trunk_set = load_data()
    master   = build_master(traces, bill, wh_prov)

    p2ft     = q1_province_to_first_trunk(master)
    lt2p, p2lt = q1b_last_trunk_to_province(master)
    flow     = q2_trunk_province_flow(master, wh_prov)
    pair_stats, matrix_count, matrix_pct = q3_province_matrix(master)

    plot_province_to_first_trunk(p2ft)
    plot_province_matrix(matrix_count, matrix_pct)
    plot_top_province_pairs(pair_stats)
    plot_sankey_province(flow)

    summarize(master, p2ft, pair_stats, matrix_count)


if __name__ == "__main__":
    main()
