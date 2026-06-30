import sys
import os

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows
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

# OUTPUT_DIR = "output_1A" 
OUTPUT_DIR = "output_all_traces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from prompt_toolkit.eventloop import async_generator

def step1_load_data():
    # wh_1a = pd.read_csv('warehouse_1A.csv')
    wh = pd.read_csv('warehouse.csv')
    trunk_set = set(wh['name'].dropna()) 
    
    bill = pd.read_csv('bill.csv', usecols = ['bill_code', 'actual_weight', 'service', 'origin_province', 'destination_province', 'receiving_date', 'actual_delivery_date'],)
    schedule = pd.read_csv('bill_schedule.csv', usecols = ['bill_code', 'io_status', 'io_time', 'warehouse_name'],)
    schedule['io_time'] = pd.to_datetime(schedule['io_time']) 
    return trunk_set, wh, bill, schedule

def step2_filter_trunk_schedule(schedule, trunk_set): 
    # trunk_sched = schedule[schedule['warehouse_name'].isin(trunk_set)].copy() 
    # trunk_sched = trunk_sched.sort_values(['bill_code', 'io_time']) 
    trunk_sched = schedule.copy()
    return trunk_sched 

def step3_build_traces(trunk_sched, bill):

    trunk_sched = trunk_sched.sort_values(["bill_code", "io_time"])
    
    # Identify blocks of consecutive rows for the same bill at the same warehouse
    block_id = ((trunk_sched["bill_code"] != trunk_sched["bill_code"].shift()) | 
                (trunk_sched["warehouse_name"] != trunk_sched["warehouse_name"].shift())).cumsum()
    
    # Calculate arrival (min) and departure (max) time for each block
    blocks = trunk_sched.groupby(block_id).agg(
        bill_code=("bill_code", "first"),
        warehouse_name=("warehouse_name", "first"),
        arrival_time=("io_time", "min"),
        departure_time=("io_time", "max")
    )
    
    # Format warehouse name and times: tên kho-thời gian đến-thời gian đi
    blocks["arrival_str"] = blocks["arrival_time"].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
    blocks["departure_str"] = blocks["departure_time"].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
    blocks["node_str"] = blocks["warehouse_name"].astype(str) + "-" + blocks["arrival_str"] + "-" + blocks["departure_str"]
    
    grp = blocks.groupby("bill_code")

    first_trunk = grp["warehouse_name"].first().rename("first_trunk")
    last_trunk = grp["warehouse_name"].last().rename("last_trunk")
    trunk_route = grp["node_str"].apply(lambda x: " → ".join(x)).rename("trunk_route")
    n_trunk = grp["warehouse_name"].nunique().rename("n_trunk_stops")

    traces = pd.concat([first_trunk, last_trunk, trunk_route, n_trunk], axis=1).reset_index()
    
    # Merge with receiving_date and actual_delivery_date from bill
    traces = traces.merge(
        bill[['bill_code', 'receiving_date', 'actual_delivery_date']], 
        on='bill_code', 
        how='left'
    )
    # Rename columns to match user's request
    traces = traces.rename(columns={
        'receiving_date': 'thoi_gian_khach_giao_hang',
        'actual_delivery_date': 'thoi_gian_khach_nhan'
    })

    traces.to_csv(os.path.join(OUTPUT_DIR, "bill_trunk_traces.csv"), index=False)
    return traces 


# def step4_extract_legs(traces):

#     # Chỉ bill có ≥ 2 kho đường trục mới có leg
#     multi = traces[traces["n_trunk_stops"] >= 2].copy()
#     # Tách route thành list
#     multi["nodes"] = multi["trunk_route"].str.split(" → ")

#     # Explode
#     exploded = multi[["bill_code", "nodes"]].explode("nodes")
#     exploded["pos"] = exploded.groupby("bill_code").cumcount()
#     exploded = exploded.rename(columns={"nodes": "origin_trunk"})

#     # Self-join: pos_i → pos_i+1
#     shifted = exploded.copy()
#     shifted["pos"] = shifted["pos"] - 1
#     shifted = shifted.rename(columns={"origin_trunk": "dest_trunk"})

#     legs = exploded.merge(
#         shifted[["bill_code", "pos", "dest_trunk"]],
#         on=["bill_code", "pos"],
#         how="inner",
#     )
#     legs = legs[["bill_code", "origin_trunk", "dest_trunk"]].reset_index(drop=True)


#     legs.to_csv(os.path.join(OUTPUT_DIR, "trunk_legs.csv"), index=False)

#     return legs


# def step5_leg_matrix(legs, trunk_set):
#     # Ma trận count
#     matrix_count = legs.groupby(["origin_trunk", "dest_trunk"]).size().unstack(fill_value=0)

#     # Đảm bảo đủ 26 kho (thêm hàng/cột 0 nếu kho không xuất hiện)
#     all_trunks = sorted(trunk_set)
#     matrix_count = matrix_count.reindex(index=all_trunks, columns=all_trunks, fill_value=0)

#     # Ma trận phần trăm (theo hàng)
#     row_sums = matrix_count.sum(axis=1)
#     matrix_pct = matrix_count.div(row_sums.replace(0, np.nan), axis=0) * 100
#     matrix_pct = matrix_pct.fillna(0).round(2)

#     matrix_count.to_csv(os.path.join(OUTPUT_DIR, "matrix_count.csv"))
#     matrix_pct.to_csv(os.path.join(OUTPUT_DIR, "matrix_pct.csv"))


#     # Bảng chi tiết
#     detail = (
#         legs.groupby(["origin_trunk", "dest_trunk"])
#         .agg(so_chang=("bill_code", "size"), so_bill_unique=("bill_code", "nunique"))
#         .reset_index()
#         .sort_values("so_chang", ascending=False)
#     )
#     detail["pct_trong_tong"] = (detail["so_chang"] / detail["so_chang"].sum() * 100).round(2)
#     total_by_origin = detail.groupby("origin_trunk")["so_chang"].transform("sum")
#     detail["pct_theo_origin"] = (detail["so_chang"] / total_by_origin * 100).round(2)
#     detail.to_csv(os.path.join(OUTPUT_DIR, "detail_legs.csv"), index=False)
#     return matrix_count, matrix_pct, detail


def step6_first_last_stats(traces, bill):
    # Chỉ bill có ≥ 2 kho (tức first ≠ last)
    fl = traces[traces["n_trunk_stops"] >= 2].copy()
    fl = fl[fl["first_trunk"] != fl["last_trunk"]]  # loại trường hợp quay lại kho cũ
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
    return stats, fl

def step7_summary(stats):
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

    origin_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_origin.csv"), index=False)
    dest_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_dest.csv"), index=False)


    return origin_summary, dest_summary

    
# def step8_heatmap(matrix_pct, matrix_count, stats, trunk_set):

#     # --- 8a. Heatmap chặng liên tiếp (legs) ---
#     # Lọc chỉ các kho có dữ liệu
#     active_origins = matrix_count.index[matrix_count.sum(axis=1) > 0]
#     active_dests = matrix_count.columns[matrix_count.sum(axis=0) > 0]
#     all_active = sorted(set(active_origins) | set(active_dests))

#     heatmap_data = matrix_pct.reindex(index=all_active, columns=all_active).fillna(0)

#     fig, ax = plt.subplots(figsize=(18, 14))
#     sns.heatmap(
#         heatmap_data,
#         annot=True, fmt=".1f", cmap="YlOrRd",
#         linewidths=0.5, linecolor="white", ax=ax,
#         cbar_kws={"label": "% hang tu Origin Trunk"},
#         annot_kws={"size": 7},
#     )
#     ax.set_title(
#         "Ma tran chuyen doi: Kho Duong Truc A -> Kho Duong Truc B (% theo Origin)\n"
#         f"26 kho tu warehouse_1A — {len(all_active)} kho co du lieu",
#         fontsize=13, fontweight="bold"
#     )
#     ax.set_xlabel("Kho Dich (Dest Trunk)", fontsize=11)
#     ax.set_ylabel("Kho Nguon (Origin Trunk)", fontsize=11)
#     plt.xticks(rotation=45, ha="right", fontsize=7)
#     plt.yticks(fontsize=7)
#     plt.tight_layout()
#     path = os.path.join(OUTPUT_DIR, "heatmap_legs.png")
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()
#     print(f"  ✓ Đã lưu {path}")

#     # --- 8b. Heatmap first→last (cấp bill) ---
#     pivot_count = stats.pivot_table(
#         index="first_trunk", columns="last_trunk",
#         values="so_bill", aggfunc="sum", fill_value=0
#     )
#     pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100
#     # Reindex đầy đủ
#     all_fl = sorted(set(pivot_count.index) | set(pivot_count.columns))
#     pivot_pct = pivot_pct.reindex(index=all_fl, columns=all_fl).fillna(0)

#     fig, ax = plt.subplots(figsize=(18, 14))
#     sns.heatmap(
#         pivot_pct,
#         annot=True, fmt=".1f", cmap="YlGnBu",
#         linewidths=0.5, linecolor="white", ax=ax,
#         cbar_kws={"label": "% bill tu First Trunk"},
#         annot_kws={"size": 7},
#     )
#     ax.set_title(
#         "Ma tran First Trunk -> Last Trunk (% bill theo Origin)\n"
#         f"26 kho tu warehouse_1A",
#         fontsize=13, fontweight="bold"
#     )
#     ax.set_xlabel("Last Trunk (Kho cuoi cung)", fontsize=11)
#     ax.set_ylabel("First Trunk (Kho dau tien)", fontsize=11)
#     plt.xticks(rotation=45, ha="right", fontsize=7)
#     plt.yticks(fontsize=7)
#     plt.tight_layout()
#     path = os.path.join(OUTPUT_DIR, "heatmap_first_last.png")
#     plt.savefig(path, bbox_inches="tight")
#     plt.close()

def step9_top_routes_bar(stats):
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

# def step10_sankey(stats):
#     try:
#         import plotly.graph_objects as go

#         sankey_data = stats.head(40).copy()
#         all_nodes = list(set(sankey_data["first_trunk"]) | set(sankey_data["last_trunk"]))
#         node_idx = {name: i for i, name in enumerate(all_nodes)}

#         fig = go.Figure(
#             go.Sankey(
#                 arrangement="snap",
#                 node=dict(
#                     pad=20, thickness=15,
#                     label=all_nodes,
#                     color="rgba(31, 119, 180, 0.8)",
#                 ),
#                 link=dict(
#                     source=[node_idx[r["first_trunk"]] for _, r in sankey_data.iterrows()],
#                     target=[node_idx[r["last_trunk"]] for _, r in sankey_data.iterrows()],
#                     value=sankey_data["so_bill"].tolist(),
#                     color="rgba(31, 119, 180, 0.3)",
#                 ),
#             )
#         )
#         fig.update_layout(
#             title="Sankey: Luong hang giua 26 kho duong truc (Top 40 tuyen)",
#             font=dict(size=10), width=1500, height=1000,
#         )
#         path = os.path.join(OUTPUT_DIR, "sankey.html")
#         fig.write_html(path)
#     except ImportError:
#         print("   plotly chưa được cài đặt → bỏ qua Sankey.")


def main():
    # 1. Load
    trunk_set, wh_1a, bill, schedule = step1_load_data()

    # 2. Lọc schedule
    trunk_sched = step2_filter_trunk_schedule(schedule, trunk_set)
    del schedule  # giải phóng memory

    # 3. Build traces
    traces = step3_build_traces(trunk_sched, bill)
    del trunk_sched

    # 4. Extract legs
    # legs = step4_extract_legs(traces)

    # 5. Ma trận chuyển đổi (theo chặng)
    # matrix_count, matrix_pct, detail = step5_leg_matrix(legs, trunk_set)

    # 6. First→Last stats (cấp bill)
    stats, fl_df = step6_first_last_stats(traces, bill)

    # 7. Tổng hợp
    step7_summary(stats)

    # 8. Heatmap
    # step8_heatmap(matrix_pct, matrix_count, stats, trunk_set)

    # 9. Bar chart
    step9_top_routes_bar(stats)

    # 10. Sankey
    # step10_sankey(stats)


if __name__ == "__main__":
    main()