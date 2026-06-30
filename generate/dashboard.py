"""
dashboard.py
============
Tạo dashboard.html — trực quan hóa tổng quát toàn bộ mạng lưới cung đường bưu cục.
4 tab:
  1. Tổng quan  — phân phối, top kho, biểu đồ ổn định
  2. Bản đồ nhiệt — luồng bill giữa các tỉnh
  3. Phân tích kho — scatter anomaly + bảng kho bất thường
  4. Hub — những kho trung chuyển quan trọng nhất
"""

import sys, os, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from collections import Counter

def strip_timestamps(route_str):
    """Remove -ArrTime-DepTime from each node in a route string."""
    return re.sub(r'-\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', str(route_str))

OUTPUT_DIR = "output_routes"
ps = pd.read_csv(os.path.join(OUTPUT_DIR, "pair_stats.csv"))
rd = pd.read_csv(os.path.join(OUTPUT_DIR, "route_distribution.csv"))

ps["route_chinh"] = ps["route_chinh"].apply(strip_timestamps)
rd["trunk_route"] = rd["trunk_route"].apply(strip_timestamps)

# ps = pd.read_csv(os.path.join(OUTPUT_DIR, "pair_non_1a.csv"))
# rd = pd.read_csv(os.path.join(OUTPUT_DIR, "rdis_non_1a.csv"))

if "first_trunk_prov" not in rd.columns:
    wh = pd.read_csv("warehouse.csv")
    wh_prov = wh.set_index("name")["province_name"].to_dict()
    rd["first_trunk_prov"] = rd["first_trunk"].map(wh_prov)
    rd["last_trunk_prov"]  = rd["last_trunk"].map(wh_prov)    

rd["first_trunk_prov"] = rd["first_trunk_prov"].fillna("(Không rõ tỉnh)")
rd["last_trunk_prov"]  = rd["last_trunk_prov"].fillna("(Không rõ tỉnh)")

print(f"pair_stats: {len(ps):,} cap  |  route_distribution: {len(rd):,} routes")

# ── 1. Thống kê tổng quan ──────────────────────────────────────────────────
total_pairs  = len(ps)
pct_95       = round((ps["pct_route_chinh_bill"] >= 95).sum() / total_pairs * 100, 1)
pct_90       = round((ps["pct_route_chinh_bill"] >= 90).sum() / total_pairs * 100, 1)
pct_70       = round((ps["pct_route_chinh_bill"] >= 70).sum() / total_pairs * 100, 1)
n_1route     = int((ps["so_route_theo_cap"] == 1).sum())
n_src_wh     = ps["first_trunk"].nunique()
n_dst_wh     = ps["last_trunk"].nunique()

# Phân phối nhom_co_dinh
nhom_dist = ps["nhom_co_dinh"].value_counts().sort_index()
nhom_labels = nhom_dist.index.tolist()
nhom_vals   = nhom_dist.values.tolist()

# Phân phối nhom_so_route
nroute_dist   = ps["nhom_so_route"].value_counts().sort_index()
nroute_labels = nroute_dist.index.tolist()
nroute_vals   = nroute_dist.values.tolist()

# Đọc thêm dữ liệu phân tích tỉnh
p2ft = pd.read_csv("output_all_province/province_to_first_trunk.csv")
p2lt = pd.read_csv("output_all_province/province_to_last_trunk.csv")
tpf = pd.read_csv("output_all_province/trunk_province_flow.csv")
# p2ft = pd.read_csv("output_all_province/province_to_first_trunk_non_1a.csv")
# p2lt = pd.read_csv("output_all_province/province_to_last_trunk_non_1a.csv")
# tpf = pd.read_csv("output_all_province/trunk_province_flow_non_1a.csv")

# ── 2. Province heatmap ───────────────────────────────────────────────────
# Tính toán heatmap trực tiếp từ trunk_province_flow (tpf) và pair_stats (ps) 
# để lấy số liệu chính xác theo origin_province và destination_province.

# Kết hợp tpf và ps để lấy pct_route_chinh_bill của từng cặp kho
merged_heat = tpf.merge(ps[["first_trunk", "last_trunk", "pct_route_chinh_bill"]], 
                        on=["first_trunk", "last_trunk"], how="left")
merged_heat["pct_route_chinh_bill"] = merged_heat["pct_route_chinh_bill"].fillna(100)
merged_heat["bill_route_chinh"] = (merged_heat["so_bill"] * merged_heat["pct_route_chinh_bill"] / 100).round()

prov_heat = (
    merged_heat.groupby(["origin_province", "destination_province"])
    .agg(
         tong_bill=("so_bill", "sum"),
         bill_route_chinh=("bill_route_chinh", "sum"),
         so_cap=("first_trunk", "count")
    )
    .reset_index()
)
prov_heat["avg_pct"] = (prov_heat["bill_route_chinh"] / prov_heat["tong_bill"] * 100).round(1).fillna(0)

# Chỉ lấy top 25 tỉnh nguồn + đích theo tổng bill (giữ heatmap readable)
top_prov_src = (prov_heat.groupby("origin_province")["tong_bill"]
                .sum().nlargest(25).index.tolist())
top_prov_dst = (prov_heat.groupby("destination_province")["tong_bill"]
                .sum().nlargest(25).index.tolist())
ph_filtered = prov_heat[
    prov_heat["origin_province"].isin(top_prov_src) &
    prov_heat["destination_province"].isin(top_prov_dst)
]

heatmap_data = {
    "src_labels": top_prov_src,
    "dst_labels": top_prov_dst,
    "cells": []
}
for _, row in ph_filtered.iterrows():
    si = top_prov_src.index(row["origin_province"])
    di = top_prov_dst.index(row["destination_province"])
    heatmap_data["cells"].append({
        "si": si, "di": di,
        "bills": int(row["tong_bill"]),
        "bills_chinh": int(row["bill_route_chinh"]),
        "n": int(row["so_cap"]),
        "pct": row["avg_pct"]
    })

# ── 3. Warehouse anomaly profile ──────────────────────────────────────────
wh_profile = (
    ps.groupby("first_trunk")
    .agg(
        avg_dom   =("pct_route_chinh_bill", "mean"),
        n_dest    =("last_trunk", "count"),
        tong_bill =("tong_bill", "sum"),
        tong_kg   =("tong_kg", "sum"),
        prov      =("first_trunk_prov", "first"),
    )
    .reset_index()
)
wh_profile["avg_dom"]   = wh_profile["avg_dom"].round(1)
wh_profile["tong_bill"] = wh_profile["tong_bill"].astype(int)

# Scatter data
scatter_data = wh_profile.rename(columns={
    "first_trunk": "name", "avg_dom": "dom",
    "n_dest": "n", "tong_bill": "bills", "prov": "prov"
})[["name","dom","n","bills","prov"]].to_dict("records")

# ── 4. Hub analysis ───────────────────────────────────────────────────────
hub_cnt = Counter()
rank1_routes = rd[rd["rank_trong_cap"] == 1]
for route in rank1_routes["trunk_route"]:
    parts = str(route).split(" → ")
    for node in parts[1:-1]:
        hub_cnt[node] += 1

top_hubs = [{"name": k, "count": v} for k, v in hub_cnt.most_common(30)]

# Phân loại kho: chỉ nguồn, chỉ đích, vừa nguồn vừa đích, hub thuần
all_src = set(ps["first_trunk"])
all_dst = set(ps["last_trunk"])
all_hub = set(hub_cnt.keys())
role_data = {
    "only_src":  len(all_src - all_dst - all_hub),
    "only_dst":  len(all_dst - all_src - all_hub),
    "src_dst":   len((all_src & all_dst) - all_hub),
    "hub":       len(all_hub & all_src & all_dst),
    "hub_only":  len(all_hub - all_src - all_dst),
}

# Top 50 cặp kho bất thường (low pct, high bills)
chaotic_pairs = (
    ps[ps["tong_bill"] >= 100]
    .sort_values("pct_route_chinh_bill")
    .head(50)
    .sort_values(["tong_bill", "pct_route_chinh_bill"], ascending=[False, True])
    [["first_trunk","last_trunk","pct_route_chinh_bill",
               "so_route_theo_cap","tong_bill","route_chinh"]]
    .rename(columns={"first_trunk":"from","last_trunk":"to",
                     "pct_route_chinh_bill":"pct","so_route_theo_cap":"n_routes",
                     "tong_bill":"bills","route_chinh":"top_route"})
    .to_dict("records")
)

# ── 5. Chuẩn bị dữ liệu cho Tab Chi tiết Route ───────────────────────────
prov_a = {}
for prov, grp in rd.groupby("first_trunk_prov"):
    whs = sorted(grp["first_trunk"].unique().tolist())
    prov_a[prov] = whs

# Aggregation to avoid MemoryError with 2.5M rows
agg_df = (
    rd.groupby(["first_trunk", "last_trunk", "trunk_route"])
    .agg(
        so_bill=("so_bill", "sum"),
        tong_kg=("tong_kg", "sum")
    )
    .reset_index()
)
# Re-calculate pct and rank
pair_totals = agg_df.groupby(["first_trunk", "last_trunk"]).agg(
    pair_bill=("so_bill", "sum"), pair_kg=("tong_kg", "sum")
).reset_index()
agg_df = agg_df.merge(pair_totals, on=["first_trunk", "last_trunk"])
agg_df["pct_bill"] = (agg_df["so_bill"] / agg_df["pair_bill"] * 100).round(2)
agg_df["pct_kg"] = (agg_df["tong_kg"] / agg_df["pair_kg"] * 100).round(2)
agg_df["stops"] = agg_df["trunk_route"].str.count("→") + 1
agg_df["n_routes"] = agg_df.groupby(["first_trunk", "last_trunk"])["trunk_route"].transform("count")
agg_df["rank"] = (
    agg_df.groupby(["first_trunk", "last_trunk"])["so_bill"]
    .rank(method="first", ascending=False).astype(int)
)
agg_df["tong_kg"] = agg_df["tong_kg"].round(1)

embed_df = agg_df.rename(columns={
    "first_trunk": "a", "last_trunk": "b", "trunk_route": "route",
    "so_bill": "bills", "tong_kg": "kg"
})

data_dict = {}
for (a, b), sub in embed_df.groupby(["a", "b"]):
    rows = sub.sort_values("rank")[
        ["route", "bills", "kg", "pct_bill", "pct_kg", "stops", "rank", "n_routes"]
    ].to_dict("records")
    if a not in data_dict:
        data_dict[a] = {}
    data_dict[a][b] = rows

a_to_provb = {}
for a_wh in embed_df["a"].unique():
    sub = rd[rd["first_trunk"] == a_wh][["last_trunk", "last_trunk_prov"]].drop_duplicates()
    mapping = {}
    for _, row in sub.iterrows():
        p = row["last_trunk_prov"]
        w = row["last_trunk"]
        if p not in mapping:
            mapping[p] = []
        mapping[p].append(w)
    for p in mapping:
        mapping[p] = sorted(mapping[p])
    a_to_provb[a_wh] = mapping

wh_to_prov_a = rd.drop_duplicates("first_trunk").set_index("first_trunk")["first_trunk_prov"].to_dict()
wh_to_prov_b = rd.drop_duplicates("last_trunk").set_index("last_trunk")["last_trunk_prov"].to_dict()

route_detail_data = {
    "data_json": data_dict,
    "prov_a": prov_a,
    "a_to_provb": a_to_provb,
    "all_provs_a": sorted(prov_a.keys()),
    "all_wh_a": sorted(rd["first_trunk"].unique().tolist()),
    "all_wh_b": sorted(rd["last_trunk"].unique().tolist()),
    "wh_to_prov_a": wh_to_prov_a,
    "wh_to_prov_b": wh_to_prov_b
}

prov_src_dict = {}
for p, grp in p2ft.groupby("origin_province"):
    prov_src_dict[p] = grp[["first_trunk", "so_bill", "tong_kg", "pct_trong_tinh"]].to_dict("records")

prov_dst_dict = {}
for p, grp in p2lt.groupby("destination_province"):
    prov_dst_dict[p] = grp[["last_trunk", "so_bill", "tong_kg", "pct_trong_tinh"]].to_dict("records")

prov_flow_dict = {}
for (o, d), sub in tpf.groupby(["origin_province", "destination_province"]):
    if o not in prov_flow_dict: prov_flow_dict[o] = {}
    prov_flow_dict[o][d] = sub.sort_values("so_bill", ascending=False)[["first_trunk", "last_trunk", "so_bill", "tong_kg", "pct_theo_origin_kho"]].to_dict("records")

# ── Đóng gói JSON ──────────────────────────────────────────────────────────
js_data = json.dumps({
    "stats": {
        "total_pairs": total_pairs, "n_src": n_src_wh, "n_dst": n_dst_wh,
        "pct_95": pct_95, "pct_90": pct_90, "pct_70": pct_70, "n_1route": n_1route,
    },
    "nhom_dist":   {"labels": nhom_labels,   "vals": nhom_vals},
    "nroute_dist": {"labels": nroute_labels, "vals": nroute_vals},
    "heatmap":     heatmap_data,
    "scatter":     scatter_data,
    "pairs_scatter": (
        ps[ps["tong_bill"] >= 100]
        .rename(columns={
            "first_trunk": "from_wh", "last_trunk": "to_wh", 
            "pct_route_chinh_bill": "pct", "so_route_theo_cap": "n_routes"
        })
        [["from_wh", "to_wh", "pct", "n_routes"]]
        .to_dict("records")
    ),
    "top_hubs":    top_hubs,
    "role_data":   role_data,
    "chaotic_pairs": chaotic_pairs,
    "route_detail": route_detail_data,
    
    "prov_src": prov_src_dict,
    "prov_dst": prov_dst_dict,
    "prov_flow": prov_flow_dict,
}, ensure_ascii=False)

print("Du lieu da san sang, tao HTML...")

html = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard — Mạng lưới cung đường bưu cục </title>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#f8fafc; --panel:#ffffff; --card:#ffffff; --border:#e2e8f0;
  --accent:#6366f1; --green:#10b981; --yellow:#f59e0b; --red:#ef4444;
  --blue:#0ea5e9; --text:#334155; --muted:#64748b; --radius:12px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
a{color:inherit;text-decoration:none}

/* HEADER */
.hdr{background:linear-gradient(135deg,#ffffff,#f8fafc);border-bottom:1px solid var(--border);
  padding:16px 28px;display:flex;align-items:center;justify-content:space-between}
.hdr h1{font-size:1.2rem;font-weight:700}
.hdr h1 em{color:var(--accent);font-style:normal}
.hdr .sub{font-size:.75rem;color:var(--muted)}

/* TABS */
.tabs{display:flex;gap:2px;padding:12px 28px 0;background:var(--panel);
  border-bottom:1px solid var(--border)}
.tab{padding:10px 20px;font-size:.82rem;font-weight:500;cursor:pointer;
  border-radius:8px 8px 0 0;color:var(--muted);transition:all .2s;border-bottom:2px solid transparent}
.tab:hover{color:var(--text);background:rgba(0,0,0,.04)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);background:rgba(99,102,241,.08)}

/* CONTENT */
.tab-content{display:none;padding:24px 28px}
.tab-content.active{display:block}

/* STAT CARDS */
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.mc{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:14px 18px;transition:transform .2s}
.mc:hover{transform:translateY(-2px)}
.mc .v{font-size:1.7rem;font-weight:700;color:var(--accent)}
.mc .l{font-size:.72rem;color:var(--muted);margin-top:3px;line-height:1.4}
.mc.green .v{color:var(--green)}
.mc.yellow .v{color:var(--yellow)}
.mc.red .v{color:var(--red)}
.mc.blue .v{color:var(--blue)}

/* GRID LAYOUTS */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px}
.grid-70-30{display:grid;grid-template-columns:1fr .45fr;gap:16px;margin-bottom:16px}

/* CARD */
.ccard{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:18px 22px;margin-bottom:16px}
.ccard h3{font-size:.88rem;font-weight:600;margin-bottom:4px}
.ccard .subtitle{font-size:.73rem;color:var(--muted);margin-bottom:14px}
.chart-wrap{position:relative}
.chart-wrap canvas{width:100%!important}

/* HEATMAP */
#heatmap-canvas{display:block;border-radius:8px;cursor:crosshair}
.hm-tooltip{position:fixed;background:#ffffff;border:1px solid var(--border);
  border-radius:8px;padding:8px 12px;font-size:.78rem;pointer-events:none;
  display:none;z-index:999;line-height:1.6;box-shadow:0 4px 12px rgba(0,0,0,0.1)}

/* TABLE */
.tbl-wrap{overflow-x:auto;max-height:380px;overflow-y:auto}
table{width:100%;border-collapse:collapse;font-size:.78rem}
thead th{background:rgba(99,102,241,.08);color:var(--muted);font-size:.67rem;
  text-transform:uppercase;letter-spacing:.5px;padding:8px 10px;
  text-align:left;border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:1}
tbody tr{border-bottom:1px solid rgba(0,0,0,.03);transition:background .12s}
tbody tr:hover{background:rgba(0,0,0,.03)}
tbody td{padding:7px 10px;vertical-align:middle}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.68rem;font-weight:600}
.bg-g{background:rgba(0,200,150,.15);color:var(--green)}
.bg-y{background:rgba(245,166,35,.15);color:var(--yellow)}
.bg-r{background:rgba(240,80,110,.15);color:var(--red)}

/* SECTION TITLE */
.section-title{font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.7px;color:var(--muted);margin-bottom:12px;padding-bottom:6px;
  border-bottom:1px solid var(--border)}

/* ROUTE TAB LAYOUT */
.layout-route{display:grid;grid-template-columns:300px 1fr;gap:20px;height:calc(100vh - 200px);align-items:start}
.sidebar{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:20px 16px;display:flex;flex-direction:column;gap:16px;overflow-y:auto;max-height:100%}
.sel-group{display:flex;flex-direction:column;gap:6px}
.sel-group label{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
.search-input{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 10px;font-size:.85rem;font-family:inherit;outline:none;transition:border-color .2s;width:100%}
.search-input:focus{border-color:var(--accent)}
.search-input:disabled{opacity:.4;cursor:default}
.divider{height:1px;background:var(--border);margin:2px 0}

.pcard{background:linear-gradient(135deg,rgba(124,109,250,.1),rgba(0,200,150,.06));border:1px solid rgba(124,109,250,.25);border-radius:10px;padding:12px 14px;display:none;font-size:.82rem}
.pcard.show{display:block}
.pcard-title{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:10px}
.pcard-row{display:flex;justify-content:space-between;padding:3px 0;}
.pcard-row .k{color:var(--muted)}
.pcard-row .v{font-weight:600}

.content-route{overflow-y:auto;display:flex;flex-direction:column;gap:20px;max-height:100%}
.empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);gap:10px;text-align:center;min-height:300px}
.empty svg{opacity:.2}
.rb{width:22px;height:22px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:700}
.r1{background:rgba(16,185,129,.15);color:var(--green)}
.r2{background:rgba(99,102,241,.15);color:var(--accent)}
.rn{background:rgba(0,0,0,.06);color:var(--muted)}
.bar-wrap{display:flex;align-items:center;gap:7px}
.bar{height:5px;border-radius:3px;background:rgba(0,0,0,.06);flex:1;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--green))}
.bar-fill.kg{background:linear-gradient(90deg,var(--green),var(--yellow))}
.route-cell{max-width:380px;line-height:1.5;color:var(--text)}
.arr{color:var(--accent);font-weight:600}

/* SCATTER */
#scatter-canvas{cursor:crosshair}
.legend-row{display:flex;align-items:center;gap:6px;font-size:.75rem;color:var(--muted)}
.legend-dot{width:10px;height:10px;border-radius:50%}

/* HUB BAR */
.hub-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:.78rem}
.hub-label{width:220px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  text-align:right;color:var(--text)}
.hub-bar{height:16px;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--blue));
  transition:width .6s}
.hub-count{color:var(--muted);min-width:50px}

/* ROLE DONUT */
.role-grid{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.role-legend{display:flex;flex-direction:column;gap:8px}
.role-item{display:flex;align-items:center;gap:8px;font-size:.78rem}
.role-dot{width:12px;height:12px;border-radius:3px}

@media(max-width:900px){
  .grid-2,.grid-3,.grid-70-30,.layout-route{grid-template-columns:1fr;height:auto}
  .tabs{flex-wrap:wrap}
}
</style>
</head>
<body>

<div class="hdr">
  <h1>📦 <em>Dashboard</em> — Mạng lưới cung đường bưu cục 

  </h1>
  <div class="sub" id="hdr-sub">Đang tải...</div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab(0)">📊 Tổng quan</div>
  <div class="tab" onclick="switchTab(1)">🗺 Phân tích tỉnh</div>
  <div class="tab" onclick="switchTab(2)">🔍 Phân tích kho</div>
  <div class="tab" onclick="switchTab(3)">🔗 Hub & Bất thường</div>
  <div class="tab" onclick="switchTab(4)">🛣 Chi tiết Route</div>
</div>

<!-- ══════════════════ TAB 0: TỔNG QUAN ══════════════════ -->
<div class="tab-content active" id="tab-0">
  <div class="metrics" id="metrics-bar"></div>

  <div class="grid-2">
    <div class="ccard">
      <h3>Phân loại theo độ ổn định route chính</h3>
      <div class="subtitle">% bill đi theo route phổ biến nhất của mỗi cặp</div>
      <div class="chart-wrap" style="height:260px">
        <canvas id="chart-nhom"></canvas>
      </div>
    </div>
    <div class="ccard">
      <h3>Phân loại theo số route khác nhau</h3>
      <div class="subtitle">Mỗi cặp kho dùng bao nhiêu cung đường khác nhau</div>
      <div class="chart-wrap" style="height:260px">
        <canvas id="chart-nroute"></canvas>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="ccard">
      <h3>Top 20 kho gửi (theo tổng bill đi)</h3>
      <div class="subtitle">Kho có lưu lượng phát hàng lớn nhất</div>
      <div class="chart-wrap" style="height:340px">
        <canvas id="chart-top-src"></canvas>
      </div>
    </div>
    <div class="ccard">
      <h3>Top 20 kho nhận (theo tổng bill đến)</h3>
      <div class="subtitle">Kho có lưu lượng nhận hàng lớn nhất</div>
      <div class="chart-wrap" style="height:340px">
        <canvas id="chart-top-dst"></canvas>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════ TAB 1: HEATMAP ══════════════════ -->
<div class="tab-content" id="tab-1">
  <div class="ccard">
    <h3>Bản đồ nhiệt luồng hàng giữa các tỉnh/thành (Top 25)</h3>
    <div class="subtitle">Màu = tổng số bill. Hover để xem chi tiết. Hàng = tỉnh gửi, Cột = tỉnh nhận. <b style="color:var(--accent)">Click vào ô để xem chi tiết!</b></div>
    <div style="overflow-x:auto;">
      <canvas id="heatmap-canvas"></canvas>
    </div>
  </div>
  <div class="hm-tooltip" id="hm-tip"></div>
  
  <div class="grid-2">
    <div class="ccard">
      <h3>Chi tiết Kho theo Tỉnh</h3>
      <div class="subtitle">Chọn một tỉnh để xem danh sách các kho nguồn và đích của tỉnh đó.</div>
      <input type="text" id="sel-prov" class="sel-group" style="width:100%;margin-bottom:10px;padding:8px" placeholder="🔍 Nhập hoặc chọn tỉnh..." list="dl-provs" onclick="this.value=''; this.dispatchEvent(new Event('input'));">
      <datalist id="dl-provs"></datalist>
      <div class="grid-2">
        <div>
          <h4 style="margin-bottom:8px;color:var(--accent);font-size:.75rem">Kho Nguồn (Gửi)</h4>
          <div class="tbl-wrap" style="max-height: 250px;">
            <table>
              <thead><tr><th>Kho</th><th>Bills</th><th>% Tỉnh</th></tr></thead>
              <tbody id="tbl-prov-src"></tbody>
            </table>
          </div>
        </div>
        <div>
          <h4 style="margin-bottom:8px;color:var(--green);font-size:.75rem">Kho Đích (Nhận)</h4>
          <div class="tbl-wrap" style="max-height: 250px;">
            <table>
              <thead><tr><th>Kho</th><th>Bills</th><th>% Tỉnh</th></tr></thead>
              <tbody id="tbl-prov-dst"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="ccard">
      <h3>Chi tiết Luồng Kho theo Cặp Tỉnh</h3>
      <div class="subtitle">Chọn Tỉnh Gửi và Tỉnh Nhận, hoặc <b style="color:var(--accent)">Click vào một ô trên Bản đồ nhiệt</b>.</div>
      <div style="display:flex;gap:10px;margin-bottom:10px;">
        <input type="text" id="sel-flow-src" class="sel-group" style="flex:1;padding:8px" placeholder="Tỉnh Gửi" list="dl-provs" onclick="this.value=''; this.dispatchEvent(new Event('input'));">
        <input type="text" id="sel-flow-dst" class="sel-group" style="flex:1;padding:8px" placeholder="Tỉnh Nhận" list="dl-provs" onclick="this.value=''; this.dispatchEvent(new Event('input'));">
      </div>
      <div class="tbl-wrap" style="max-height: 250px;">
        <table>
          <thead><tr><th>Kho Gửi</th><th>Kho Nhận</th><th>Bills</th><th>% Cặp Tỉnh</th></tr></thead>
          <tbody id="tbl-prov-flow"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════ TAB 2: PHÂN TÍCH KHO ══════════════════ -->
<div class="tab-content" id="tab-2">
  <div class="ccard">
    <h3>Scatter: Profiling các cặp kho (Từ Kho Gửi -> Kho Nhận)</h3>
    <div class="subtitle">
      X = Tỉ lệ % route chính (cao → cung đường cố định). 
      Y = Số lượng routes khác nhau. Kích thước = tổng bill. (Chỉ lấy các cặp >= 100 bills). 
      <b style="color:var(--accent)">Click vào bong bóng của cặp kho bất kỳ để xem chi tiết!</b>
    </div>
    <div class="chart-wrap" style="height:420px;margin-bottom:20px;">
      <canvas id="chart-scatter-pairs"></canvas>
    </div>
  </div>
  <div class="ccard">
    <h3>Scatter: Profiling từng kho gửi</h3>
    <div class="subtitle">
      X = % bill đi route chính trung bình (cao → ổn định). 
      Y = số kho đích. Kích thước = tổng bill. Hover để xem chi tiết.
    </div>
    <div class="chart-wrap" style="height:420px">
      <canvas id="chart-scatter"></canvas>
    </div>
  </div>


</div>

<!-- ══════════════════ TAB 3: HUB & BẤT THƯỜNG ══════════════════ -->
<div class="tab-content" id="tab-3">
  <div class="ccard">
    <h3>🔴 Top 50 cặp kho BẤT THƯỜNG (nhiều route, không có cung cố định)</h3>
    <div class="subtitle">
      Các cặp này có lưu lượng cao nhưng hàng đi theo nhiều đường khác nhau — 
      dấu hiệu lộ trình chưa tối ưu hoặc nhiều nhà vận chuyển khác nhau.
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Kho GỬI</th><th>Kho NHẬN</th>
          <th>% Route chính</th>
          <th style="cursor:pointer;color:var(--text);border-bottom:1px dashed var(--muted)" onclick="sortChaotic('n_routes')" title="Click để sắp xếp">Số routes ↕</th>
          <th style="cursor:pointer;color:var(--text);border-bottom:1px dashed var(--muted)" onclick="sortChaotic('bills')" title="Click để sắp xếp">Total Bills ↕</th>
          <th>Route chính</th>
        </tr></thead>
        <tbody id="tbl-chaotic"></tbody>
      </table>
    </div>
  </div>

  <div class="ccard" style="max-width:550px; margin: 0 auto 16px;">
    <h3>Vai trò các kho trong mạng lưới</h3>
    <div class="subtitle">Phân loại theo chức năng: nguồn / đích / trung chuyển</div>
    <div style="height:250px;margin-top:8px">
      <canvas id="chart-role"></canvas>
    </div>
  </div>
</div>

<!-- ══════════════════ TAB 4: CHI TIẾT ROUTE ══════════════════ -->
<div class="tab-content" id="tab-4">
  <div class="layout-route">
    <!-- SIDEBAR -->
    <div class="sidebar">
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--yellow);padding:2px 0 6px">
        Chọn Tuyến Kho
      </div>
      <div class="sel-group">
        <label for="sel-quick-a">Kho gửi</label>
        <input type="text" id="sel-quick-a" list="dl-a" class="search-input" placeholder="-- Gõ tìm kho gửi --" onclick="this.value=''; this.dispatchEvent(new Event('input'));">
        <datalist id="dl-a"></datalist>
      </div>
      <div class="sel-group">
        <label for="sel-quick-b">Kho nhận</label>
        <input type="text" id="sel-quick-b" list="dl-b" class="search-input" placeholder="-- Gõ tìm kho nhận --" onclick="this.value=''; this.dispatchEvent(new Event('input'));" disabled>
        <datalist id="dl-b"></datalist>
      </div>

      <div class="pcard" id="pcard" style="margin-top:16px;">
        <div class="pcard-title">Thông tin cặp</div>
        <div class="pcard-row"><span class="k">Tỉnh gửi</span><span class="v" id="pi-pa">—</span></div>
        <div class="pcard-row"><span class="k">Tỉnh nhận</span><span class="v" id="pi-pb">—</span></div>
        <div class="pcard-row"><span class="k">Tổng bill</span><span class="v" id="pi-bills">—</span></div>
        <div class="pcard-row"><span class="k">Tổng KG</span><span class="v" id="pi-kg">—</span></div>
        <div class="pcard-row"><span class="k">Số routes</span><span class="v" id="pi-nroutes">—</span></div>
        <div class="pcard-row"><span class="k">Route chính</span><span class="v" id="pi-pct">—</span></div>
        <div style="margin-top:9px"><span id="pi-badge"></span></div>
      </div>
    </div>

    <!-- CONTENT -->
    <div class="content-route" id="content-route">
      <div class="empty">
        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
          <polyline points="9 22 9 12 15 12 15 22"/>
        </svg>
        <p>Tìm chọn kho gửi → kho nhận<br>để xem phân tích cung đường</p>
      </div>
    </div>
  </div>
</div>

<div class="hm-tooltip" id="scatter-tip" style="display:none"></div>

<script>
const D = """ + js_data + """;

// ── Tab switching ─────────────────────────────────────────────────────────
const tabs    = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
let chartsInit = [false,false,false,false,false];
function switchTab(i) {
  tabs.forEach((t,j)   => t.classList.toggle('active', j===i));
  contents.forEach((c,j) => c.classList.toggle('active', j===i));
  if (!chartsInit[i]) { initTab(i); chartsInit[i]=true; }
}

// ── Helpers ───────────────────────────────────────────────────────────────
const fmt = n => (n||0).toLocaleString('vi-VN');
function pctBadge(p) {
  const cls = p>=90?'bg-g':p>=70?'bg-y':'bg-r';
  return `<span class="badge ${cls}">${p.toFixed(1)}%</span>`;
}

window.viewRouteDetail = function(a, b) {
  switchTab(4);
  const selA = document.getElementById('sel-quick-a');
  const selB = document.getElementById('sel-quick-b');
  selA.value = a;
  selA.dispatchEvent(new Event('input'));
  selB.value = b;
  selB.dispatchEvent(new Event('input'));
  window.scrollTo({top: 0, behavior: 'smooth'});
};

// ── Header ────────────────────────────────────────────────────────────────
document.getElementById('hdr-sub').textContent =
  `${fmt(D.stats.total_pairs)} cặp kho  |  ${D.stats.n_src} kho gửi  |  ${D.stats.n_dst} kho nhận`;

// ── Metrics ───────────────────────────────────────────────────────────────
const metricsBar = document.getElementById('metrics-bar');
[
  [D.stats.total_pairs.toLocaleString('vi-VN'), 'Tổng cặp kho phân tích', ''],
  [D.stats.pct_95+'%', 'Cặp có route chính >95% bill', 'green'],
  [D.stats.pct_90+'%', 'Cặp gần cố định (>90%)', 'green'],
  [D.stats.pct_70+'%', 'Cặp ổn định vừa phải (>70%)', 'yellow'],
  [D.stats.n_1route.toLocaleString('vi-VN'), 'Cặp chỉ có đúng 1 route', 'blue'],
  [D.stats.n_src, 'Kho/bưu cục gửi', ''],
].forEach(([v,l,c]) => {
  metricsBar.innerHTML += `<div class="mc ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
});

// ── Chart helpers ─────────────────────────────────────────────────────────
const COLORS_MULTI = [
  '#00c896','#7c6dfa','#f5a623','#f0506e','#38bdf8','#a78bfa','#34d399','#fb923c'
];

function barH(id, labels, data, color='#7c6dfa') {
  return new Chart(document.getElementById(id), {
    type:'bar',
    data:{ labels, datasets:[{ data, backgroundColor:color,
      borderRadius:5, borderSkipped:false }] },
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false},
        tooltip:{callbacks:{label:c=>` ${fmt(c.parsed.x)}`}} },
      scales:{
        x:{ ticks:{color:'#64748b'}, grid:{color:'rgba(0,0,0,.04)'} },
        y:{ ticks:{color:'#334155',font:{size:10}}, grid:{display:false} }
      }
    }
  });
}

function doughnut(id, labels, data) {
  return new Chart(document.getElementById(id), {
    type:'doughnut',
    data:{ labels, datasets:[{ data, backgroundColor:COLORS_MULTI,
      borderColor:'#131929', borderWidth:2 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{position:'right',
        labels:{color:'#dde3f0',font:{size:11},padding:12,
          boxWidth:12,usePointStyle:true}} }
    }
  });
}

function fmtKg(k){ return k >= 1000 ? (k/1000).toFixed(1)+'t' : k.toFixed(1)+'kg'; }

// ── Init Tab 0 ────────────────────────────────────────────────────────────
function initTab0() {
  doughnut('chart-nhom', D.nhom_dist.labels, D.nhom_dist.vals);
  doughnut('chart-nroute', D.nroute_dist.labels, D.nroute_dist.vals);

  // Top src/dst — compute from scatter data
  const sorted_src = [...D.scatter].sort((a,b)=>b.bills-a.bills).slice(0,20);
  barH('chart-top-src',
    sorted_src.map(r=>r.name),
    sorted_src.map(r=>r.bills),
    '#7c6dfa');

  // Top dst: need separate — build from chaotic + anomaly data
  const topDst = D.top_hubs.slice(0,20);
  barH('chart-top-dst',
    topDst.map(r=>r.name),
    topDst.map(r=>r.count),
    '#00c896');

  // Top 20 pairs table
  const tbody = document.getElementById('tbl-top-pairs');
  let h = '';
  (D.top20_pairs || []).forEach((r, i) => {
    h += `<tr style="cursor:pointer" onclick="viewRouteDetail('${r.from.replace(/'/g,"\\'")}','${r.to.replace(/'/g,"\\'")}')" title="Click để xem chi tiết">
      <td>${i+1}</td>
      <td style="color:var(--accent);font-weight:600">${r.from}</td>
      <td style="color:var(--muted);font-size:.72rem">${r.prov_from||''}</td>
      <td style="color:var(--green);font-weight:600">${r.to}</td>
      <td style="color:var(--muted);font-size:.72rem">${r.prov_to||''}</td>
      <td>${r.n_routes}</td>
      <td>${fmt(r.bills)}</td>
      <td>${fmtKg(r.kg||0)}</td>
    </tr>`;
  });
  tbody.innerHTML = h;
}

// ── Init Tab 1: Heatmap ───────────────────────────────────────────────────
function initTab1() {
  const canvas = document.getElementById('heatmap-canvas');
  const tip    = document.getElementById('hm-tip');
  const hm     = D.heatmap;
  const nR = hm.src_labels.length, nC = hm.dst_labels.length;

  const EXTRA_TOP_PAD = 70; // Thêm khoảng trống ở trên cùng cho nhãn chéo
  const LABEL_W = 150, LABEL_H = 20, CELL = 30, PAD = 5;
  const W = LABEL_W + nC * (CELL + PAD) + PAD + 10;
  const H = LABEL_H + EXTRA_TOP_PAD + nR * (CELL + PAD) + PAD + 10;
  canvas.width  = W;
  canvas.height = H;
  canvas.style.maxWidth = 'none';

  const ctx = canvas.getContext('2d');

  // Build lookup
  const lookup = {};
  hm.cells.forEach(c => { lookup[c.si+'_'+c.di] = c; });

  const maxBill = Math.max(...hm.cells.map(c=>c.bills));

  function colorScale(v) {
    const t = Math.pow(v / maxBill, 0.4);
    const r = Math.round(12 + t*(124-12));
    const g = Math.round(9  + t*(109-9));
    const b = Math.round(26 + t*(250-26));
    return `rgba(${r},${g},${b},${0.15 + t*0.85})`;
  }

  function draw() {
    ctx.clearRect(0,0,W,H);
    ctx.font = '10px Inter,sans-serif';

    // Dời vùng nhãn dọc xuống dưới để không bị cắt mép trên. Dành nhiều padding hơn cho label trên cùng
    
    // Column labels (rotated)
    ctx.fillStyle = '#6b7591';
    ctx.save();
    hm.dst_labels.forEach((lbl,di) => {
      const x = LABEL_W + di*(CELL+PAD) + CELL/2;
      ctx.save();
      // dời nhãn lên trên một chút so với ô lưới đầu tiên, tăng Extra_Top_Pad
      ctx.translate(x, LABEL_H + EXTRA_TOP_PAD - 8); 
      ctx.rotate(-Math.PI/3);
      ctx.textAlign='left'; // Align left sau khi rotate để chuỗi đi lên trên và ra ngoài
      ctx.fillText(lbl.slice(0,18), 0, 0);
      ctx.restore();
    });
    ctx.restore();

    // Row labels + cells
    hm.src_labels.forEach((lbl,si) => {
      const y = LABEL_H + EXTRA_TOP_PAD + si*(CELL+PAD);
      ctx.fillStyle='#aab4cc';
      ctx.textAlign='right';
      ctx.fillText(lbl.slice(0,22), LABEL_W-6, y+CELL/2+4);

      hm.dst_labels.forEach((_,di) => {
        const cell = lookup[si+'_'+di];
        const x = LABEL_W + di*(CELL+PAD);
        ctx.fillStyle = cell ? colorScale(cell.bills) : 'rgba(0,0,0,.03)';
        ctx.beginPath();
        ctx.roundRect(x, y, CELL, CELL, 3);
        ctx.fill();
        if (cell && cell.bills > maxBill*0.05) {
          ctx.fillStyle='rgba(255,255,255,.9)';
          ctx.textAlign='center';
          ctx.font='8px Inter';
          const txt = cell.bills>1000 ? (cell.bills/1000).toFixed(1)+'k' : cell.bills;
          ctx.fillText(txt, x+CELL/2, y+CELL/2+3);
          ctx.font='10px Inter,sans-serif';
        }
      });
    });
  }
  draw();

  // Tooltip
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const di = Math.floor((mx - LABEL_W) / (CELL+PAD));
    const si = Math.floor((my - LABEL_H - EXTRA_TOP_PAD) / (CELL+PAD));
    if (si>=0 && si<nR && di>=0 && di<nC) {
      const cell = lookup[si+'_'+di];
      if (cell) {
        tip.style.display='block';
        tip.style.left=(e.clientX+14)+'px';
        tip.style.top =(e.clientY-10)+'px';
        tip.innerHTML=`<b>${hm.src_labels[si]}</b> → <b>${hm.dst_labels[di]}</b><br>
          Tổng số bill: <b style="color:var(--accent)">${fmt(cell.bills)}</b><br>
          Số cặp kho (phục vụ tuyến này): ${cell.n}`;
        return;
      }
    }
    tip.style.display='none';
  });
  canvas.addEventListener('mouseleave', () => tip.style.display='none');
  
  // --- New Province Logic ---
  const selProv = document.getElementById('sel-prov');
  const selFlowSrc = document.getElementById('sel-flow-src');
  const selFlowDst = document.getElementById('sel-flow-dst');
  const dlProvs = document.getElementById('dl-provs');
  
  // Populate datalist with all unique provinces
  const allProvs = [...new Set([...Object.keys(D.prov_src || {}), ...Object.keys(D.prov_dst || {})])].sort();
  allProvs.forEach(p => {
    dlProvs.innerHTML += `<option value="${p}">`;
  });

  selProv.addEventListener('input', () => {
    const p = selProv.value;
    const srcBody = document.getElementById('tbl-prov-src');
    const dstBody = document.getElementById('tbl-prov-dst');
    srcBody.innerHTML = ''; dstBody.innerHTML = '';
    
    if (D.prov_src && D.prov_src[p]) {
      D.prov_src[p].forEach(r => {
        srcBody.innerHTML += `<tr><td>${r.first_trunk}</td><td>${fmt(r.so_bill)}</td><td>${r.pct_trong_tinh.toFixed(1)}%</td></tr>`;
      });
    }
    if (D.prov_dst && D.prov_dst[p]) {
      D.prov_dst[p].forEach(r => {
        dstBody.innerHTML += `<tr><td>${r.last_trunk}</td><td>${fmt(r.so_bill)}</td><td>${r.pct_trong_tinh.toFixed(1)}%</td></tr>`;
      });
    }
  });

  function updateFlowTable() {
    const src = selFlowSrc.value;
    const dst = selFlowDst.value;
    const tbody = document.getElementById('tbl-prov-flow');
    tbody.innerHTML = '';
    
    if (src && dst && D.prov_flow && D.prov_flow[src] && D.prov_flow[src][dst]) {
      D.prov_flow[src][dst].forEach(r => {
        tbody.innerHTML += `<tr style="cursor:pointer" onclick="viewRouteDetail('${r.first_trunk}', '${r.last_trunk}')" title="Click để xem chi tiết tuyến này">
          <td style="color:var(--accent)">${r.first_trunk}</td><td style="color:var(--green)">${r.last_trunk}</td>
          <td>${fmt(r.so_bill)}</td><td>${r.pct_theo_origin_kho.toFixed(1)}%</td>
        </tr>`;
      });
    } else if (src && dst) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--muted)">Không có dữ liệu luồng kho giữa 2 tỉnh này.</td></tr>';
    }
  }

  selFlowSrc.addEventListener('input', updateFlowTable);
  selFlowDst.addEventListener('input', updateFlowTable);
  
  // Click on Heatmap to select Province Pair
  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const di = Math.floor((mx - LABEL_W) / (CELL+PAD));
    const si = Math.floor((my - LABEL_H - EXTRA_TOP_PAD) / (CELL+PAD));
    if (si>=0 && si<nR && di>=0 && di<nC) {
      const srcProv = hm.src_labels[si];
      const dstProv = hm.dst_labels[di];
      selFlowSrc.value = srcProv;
      selFlowDst.value = dstProv;
      updateFlowTable();
      
      // Auto scroll to the flow table
      document.getElementById('sel-flow-src').scrollIntoView({behavior: 'smooth', block: 'center'});
    }
  });
  
  // Change cursor on hover for clickable cells
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const di = Math.floor((mx - LABEL_W) / (CELL+PAD));
    const si = Math.floor((my - LABEL_H - EXTRA_TOP_PAD) / (CELL+PAD));
    if (si>=0 && si<nR && di>=0 && di<nC && lookup[si+'_'+di]) {
      canvas.style.cursor = 'pointer';
    } else {
      canvas.style.cursor = 'default';
    }
  });
}

// ── Init Tab 2: Scatter ───────────────────────────────────────────────────
function initTab2() {
  // --- Scatter Pairs Chart ---
  const dPairs = D.pairs_scatter.map(r => ({
    x: r.pct, y: r.n_routes,
    r: Math.max(3, Math.min(18, Math.sqrt(r.bills/200))),
    from: r.from_wh, to: r.to_wh, bills: r.bills
  }));
  
  new Chart(document.getElementById('chart-scatter-pairs'), {
    type: 'bubble',
    data: {
      datasets: [{
        label: 'Cặp kho',
        data: dPairs,
        backgroundColor: 'rgba(56, 189, 248, 0.5)',
        borderColor: '#38bdf8',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      onClick: (event, elements, chart) => {
        if (elements.length > 0) {
          const dataIndex = elements[0].index;
          const datasetIndex = elements[0].datasetIndex;
          const d = chart.data.datasets[datasetIndex].data[dataIndex];
          if (typeof viewRouteDetail === 'function') {
            viewRouteDetail(d.from, d.to);
          }
        }
      },
      onHover: (event, elements, chart) => {
        event.native.target.style.cursor = elements[0] ? 'pointer' : 'default';
      },
      plugins: {
        legend: {display: false},
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              return [`${d.from} → ${d.to}`, `% Route chính: ${d.x.toFixed(1)}%`, `Số routes: ${d.y}`, `Bills: ${fmt(d.bills)}`];
            }
          }
        }
      },
      scales: {
        x: { title: {display:true, text:'Tỉ lệ % route chính', color:'#6b7591'}, ticks:{color:'#6b7591', callback:v=>v+'%'}, grid:{color:'rgba(255,255,255,.04)'} },
        y: { title: {display:true, text:'Số routes', color:'#6b7591'}, ticks:{color:'#6b7591'}, grid:{color:'rgba(255,255,255,.04)'} }
      }
    }
  });

  // Scatter chart
  const PROVS = [...new Set(D.scatter.map(r=>r.prov))].sort();
  const COL_MAP = {};
  PROVS.forEach((p,i) => COL_MAP[p] = COLORS_MULTI[i % COLORS_MULTI.length]);

  const datasets = PROVS.map(prov => ({
    label: prov,
    data: D.scatter.filter(r=>r.prov===prov).map(r=>({
      x: r.dom, y: r.n,
      r: Math.max(4, Math.min(20, Math.sqrt(r.bills/500))),
      name: r.name, bills: r.bills
    })),
    backgroundColor: COL_MAP[prov]+'99',
    borderColor: COL_MAP[prov],
    borderWidth:1,
  }));

  new Chart(document.getElementById('chart-scatter'), {
    type:'bubble',
    data:{ datasets },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{callbacks:{
          label: ctx => {
            const d=ctx.raw;
            return [`${d.name}`,`Avg dominance: ${d.x}%`,`Số đích: ${d.y}`,`Bills: ${fmt(d.bills)}`];
          }
        }}
      },
      scales:{
        x:{ title:{display:true,text:'Avg % route chính (càng cao càng ổn định)',color:'#6b7591'},
          ticks:{color:'#6b7591',callback:v=>v+'%'}, grid:{color:'rgba(255,255,255,.04)'},
          min:50,max:101 },
        y:{ title:{display:true,text:'Số kho đích khác nhau',color:'#6b7591'},
          ticks:{color:'#6b7591'}, grid:{color:'rgba(255,255,255,.04)'} }
      }
    }
  });


}

// ── Init Tab 3: Hub ───────────────────────────────────────────────────────
function initTab3() {

  // Role donut
  const rd2 = D.role_data;
  new Chart(document.getElementById('chart-role'), {
    type:'doughnut',
    data:{
      labels:['Chỉ gửi','Chỉ nhận','Gửi+Nhận','Hub+Gửi+Nhận','Hub thuần'],
      datasets:[{ data:[rd2.only_src,rd2.only_dst,rd2.src_dst,rd2.hub,rd2.hub_only],
        backgroundColor:['#7c6dfa','#00c896','#38bdf8','#f5a623','#f0506e'],
        borderColor:'#131929',borderWidth:2 }]
    },
    options:{ responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:'bottom',
        labels:{color:'#dde3f0',font:{size:10},padding:8,boxWidth:10,usePointStyle:true}}}
    }
  });

  // Chaotic pairs table
  renderChaoticTable();
}

let chaoticSortCol = '';
let chaoticSortAsc = false;

window.sortChaotic = function(col) {
  if (chaoticSortCol === col) {
    chaoticSortAsc = !chaoticSortAsc;
  } else {
    chaoticSortCol = col;
    chaoticSortAsc = false;
  }
  D.chaotic_pairs.sort((a, b) => {
    let valA = a[col], valB = b[col];
    if (valA < valB) return chaoticSortAsc ? -1 : 1;
    if (valA > valB) return chaoticSortAsc ? 1 : -1;
    return 0;
  });
  renderChaoticTable();
};

function renderChaoticTable() {
  const tbody = document.getElementById('tbl-chaotic');
  let html = '';
  D.chaotic_pairs.forEach(r => {
    const cls = r.pct<50?'bg-r':r.pct<70?'bg-y':'bg-g';
    html += `<tr style="cursor:pointer" onclick="viewRouteDetail('${r.from}', '${r.to}')" title="Click để xem chi tiết tuyến này">
      <td style="font-weight:500;color:var(--red)">${r.from}</td>
      <td style="font-weight:500;color:var(--blue)">${r.to}</td>
      <td><span class="badge ${cls}">${r.pct.toFixed(1)}%</span></td>
      <td>${r.n_routes}</td>
      <td>${fmt(r.bills)}</td>
      <td style="color:var(--muted);font-size:.72rem;max-width:250px;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${r.top_route}</td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

// ── Init Tab 4: Route Detail ───────────────────────────────────────────────
function initTab4() {
  const RDATA = D.route_detail;
  
  const pcard    = document.getElementById('pcard');
  const contentRoute  = document.getElementById('content-route');

  const selQuickA = document.getElementById('sel-quick-a');
  const selQuickB = document.getElementById('sel-quick-b');
  const dlA = document.getElementById('dl-a');
  const dlB = document.getElementById('dl-b');

  RDATA.all_wh_a.forEach(w => {
    const o = document.createElement('option'); o.value = w;
    dlA.appendChild(o);
  });

  selQuickA.addEventListener('input', () => {
    const a = selQuickA.value;
    selQuickB.value = '';
    selQuickB.disabled = true;
    dlB.innerHTML = '';
    pcard.classList.remove('show');
    showEmpty();

    if (!RDATA.all_wh_a.includes(a)) return;

    selQuickB.disabled = false;
    const bList = Object.keys(RDATA.data_json[a] || {}).sort();
    bList.forEach(w => {
      const o = document.createElement('option'); o.value = w;
      dlB.appendChild(o);
    });
  });

  selQuickB.addEventListener('input', () => {
    const a = selQuickA.value, b = selQuickB.value;
    pcard.classList.remove('show');
    
    if (!a || !b || !RDATA.data_json[a] || !RDATA.data_json[a][b]) { 
      showEmpty(); 
      return; 
    }

    const pa = RDATA.wh_to_prov_a[a] || '';
    const pb = RDATA.wh_to_prov_b[b] || '';
    renderPCard(RDATA.data_json[a][b], pa, pb);
    renderContentRoute(a, b, RDATA.data_json[a][b], pa, pb);
  });

  function showEmpty() {
    contentRoute.innerHTML = `<div class="empty">
      <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
        <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
      <p>Tìm chọn kho gửi → kho nhận<br>để xem phân tích cung đường</p>
    </div>`;
  }

  // Parse a node string like "Kho Name-2026-01-03 16:26:08-2026-01-03 16:26:44"
  // into {name, arr, dep}. Names can contain "-" so we match timestamps from the end.
  function parseNode(nodeStr) {
    const ts = /\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}/g;
    const matches = [...nodeStr.matchAll(ts)];
    if (matches.length >= 2) {
      const arr = matches[matches.length - 2][0];
      const dep = matches[matches.length - 1][0];
      // Name is everything before the first matched timestamp, minus trailing "-"
      const nameEnd = matches[matches.length - 2].index - 1;
      const name = nodeStr.slice(0, nameEnd);
      return { name, arr, dep };
    }
    return { name: nodeStr, arr: '', dep: '' };
  }

  function renderTimeline(routeFull, timeSend, timeRecv) {
    const nodes = routeFull.split(' → ').map(parseNode);
    let h = '<div class="timeline">';
    // Khách gửi
    h += `<div class="tl-node">
      <div class="tl-line-wrap"><div class="tl-dot customer"></div><div class="tl-connector"></div></div>
      <div class="tl-info"><div class="tl-name">📦 Khách Gửi Hàng</div>
        <div class="tl-times">Thời gian gửi: <b>${timeSend || '—'}</b></div></div></div>`;
    // Các kho
    nodes.forEach((n, i) => {
      const isLast = i === nodes.length - 1;
      h += `<div class="tl-node">
        <div class="tl-line-wrap"><div class="tl-dot"></div>${isLast ? '' : '<div class="tl-connector"></div>'}</div>
        <div class="tl-info"><div class="tl-name">🏭 ${n.name}</div>
          <div class="tl-times">Đến: <b>${n.arr || '—'}</b> &nbsp;|&nbsp; Đi: <b>${n.dep || '—'}</b></div></div></div>`;
    });
    // Khách nhận
    h += `<div class="tl-node">
      <div class="tl-line-wrap"><div class="tl-dot customer"></div></div>
      <div class="tl-info"><div class="tl-name">🎁 Khách Nhận Hàng</div>
        <div class="tl-times">Thời gian nhận: <b>${timeRecv || '—'}</b></div></div></div>`;
    h += '</div>';
    return h;
  }

  function showRouteModal(a, b, routeClean) {
    const key = a + '||' + b + '||' + routeClean;
    const bills = D.route_bills[key];
    if (!bills || bills.length === 0) { alert('Không tìm thấy dữ liệu bill mẫu cho cung đường này.'); return; }

    let h = `<div class="modal-overlay" onclick="if(event.target===this)this.remove()">
      <div class="modal-box">
        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button>
        <div class="modal-title">Chi tiết cung đường: Top ${bills.length} bill (theo KG)</div>
        <div style="font-size:.78rem;color:var(--muted);margin-bottom:16px;line-height:1.5">
          <b>${a}</b> <span class="arr">→</span> <b>${b}</b><br>
          Cung đường: ${routeClean.replace(/→/g,'<span class="arr"> → </span>')}
        </div>`;
    bills.forEach((bill, idx) => {
      h += `<div style="border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:12px;background:${idx===0?'rgba(99,102,241,.04)':'transparent'}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <span style="font-weight:600;color:var(--accent)">Bill #${idx+1}</span>
          <span class="badge bg-g">${bill.kg} kg</span>
        </div>
        ${renderTimeline(bill.route, bill.ts, bill.tr)}
      </div>`;
    });
    h += '</div></div>';
    document.body.insertAdjacentHTML('beforeend', h);
  }

  function renderPCard(rows, pa, pb) {
    const tot = rows.reduce((s,r)=>s+r.bills,0);
    const totKg = rows.reduce((s,r)=>s+r.kg,0);
    const p1 = rows[0].pct_bill, nr = rows[0].n_routes;
    document.getElementById('pi-pa').textContent     = pa;
    document.getElementById('pi-pb').textContent     = pb;
    document.getElementById('pi-bills').textContent  = fmt(tot);
    document.getElementById('pi-kg').textContent     = fmtKg(totKg);
    document.getElementById('pi-nroutes').textContent= nr + ' route' + (nr>1?'s':'');
    document.getElementById('pi-pct').textContent    = p1 + '%';
    let badge = '';
    if      (p1 > 95) badge = '<span class="badge bg-green">Cố định (&gt;95%)</span>';
    else if (p1 > 90) badge = '<span class="badge bg-green">Gần cố định (&gt;90%)</span>';
    else if (p1 > 70) badge = '<span class="badge bg-yellow">Bán cố định (70-90%)</span>';
    else              badge = '<span class="badge bg-red">Phân tán (&lt;70%)</span>';
    document.getElementById('pi-badge').innerHTML = badge;
    pcard.classList.add('show');
  }

  let chartBill = null, chartKg = null;
  let currentA = '', currentB = '';

  function renderContentRoute(a, b, rows, pa, pb) {
    currentA = a; currentB = b;
    const labels = rows.map((_,i) => 'Route #'+(i+1));
    const cols   = rows.map((_,i) => i===0 ? 'rgba(16,185,129,.85)' : i===1
      ? 'rgba(99,102,241,.75)' : 'rgba(0,0,0,.18)');
    const makeChart = (data) => ({
      type:'bar',
      data:{ labels, datasets:[{ data, backgroundColor:cols, borderRadius:6, borderSkipped:false }] },
      options:{
        indexAxis:'y', responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{display:false},
          tooltip:{ callbacks:{ label: c => ' '+c.parsed.x.toFixed(2)+'%' } } },
        scales:{
          x:{ max:100, ticks:{color:'#64748b',callback:v=>v+'%'}, grid:{color:'rgba(0,0,0,.04)'} },
          y:{ ticks:{color:'#64748b',font:{size:10}}, grid:{display:false} }
        }
      }
    });

    const tableRows = rows.map((r,i) => {
      const rc = i===0?'r1':i===1?'r2':'rn';
      const rt = r.route.replace(/→/g,'<span class="arr"> → </span>');
      const safeRoute = r.route.replace(/'/g, "\\'").replace(/"/g, '&quot;');
      return `<tr>
        <td><span class="rb ${rc}">${r.rank}</span></td>
        <td class="route-cell" onclick="showRouteModal(currentA, currentB, '${safeRoute}')" title="🔍 Click để xem chi tiết top 5 bill của cung đường này">${rt}</td>
        <td>${fmt(r.bills)}</td>
        <td>
          <div class="bar-wrap">
            <div class="bar"><div class="bar-fill" style="width:${r.pct_bill}%"></div></div>
            <span style="min-width:40px;text-align:right">${r.pct_bill}%</span>
          </div>
        </td>
        <td>${fmtKg(r.kg)}</td>
        <td>
          <div class="bar-wrap">
            <div class="bar"><div class="bar-fill kg" style="width:${r.pct_kg}%"></div></div>
            <span style="min-width:40px;text-align:right">${r.pct_kg}%</span>
          </div>
        </td>
        <td style="color:var(--muted);text-align:center">${r.stops}</td>
      </tr>`;
    }).join('');

    const tot = rows.reduce((s,r)=>s+r.bills,0);
    contentRoute.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:-5px;padding:0 4px">
        <h2 style="font-size:1.1rem;font-weight:600;margin:0;color:var(--text)">${a} <span style="color:var(--accent)">→</span> ${b}</h2>
        <div style="background:rgba(99,102,241,.15);color:var(--accent);padding:6px 14px;border-radius:20px;font-size:.85rem;font-weight:700;border:1px solid rgba(99,102,241,.3)">
          Tổng số bill: ${fmt(tot)}
        </div>
      </div>
      <div class="grid-2">
        <div class="ccard">
          <h3>% Số bill theo route</h3>
          <div class="chart-wrap" style="height:200px"><canvas id="ch-bill"></canvas></div>
        </div>
        <div class="ccard">
          <h3>% Khối lượng theo route</h3>
          <div class="chart-wrap" style="height:200px"><canvas id="ch-kg"></canvas></div>
        </div>
      </div>
      <div class="ccard">
        <h3>Chi tiết ${rows.length} cung đường
          <span>${pa} → ${pb} &nbsp;|&nbsp; ${a} → ${b}</span>
        </h3>
        <div class="subtitle">🔍 Click vào cung đường để xem chi tiết top 5 bill với thời gian đầy đủ</div>
        <div class="tbl-wrap"><table>
          <thead><tr>
            <th>Rank</th><th>Cung đường (click để xem)</th>
            <th>Số bill</th><th>% Bill</th>
            <th>Tổng KG</th><th>% KG</th><th>Trạm</th>
          </tr></thead>
          <tbody>${tableRows}</tbody>
        </table></div>
      </div>`;

    if (chartBill) chartBill.destroy();
    if (chartKg)   chartKg.destroy();
    chartBill = new Chart(document.getElementById('ch-bill'), makeChart(rows.map(r=>r.pct_bill)));
    chartKg   = new Chart(document.getElementById('ch-kg'),   makeChart(rows.map(r=>r.pct_kg)));
  }
}

// ── Init dispatcher ───────────────────────────────────────────────────────
function initTab(i) {
  if (i===0) initTab0();
  else if (i===1) initTab1();
  else if (i===2) initTab2();
  else if (i===3) initTab3();
  else if (i===4) initTab4();
}
initTab(0);
chartsInit[0] = true;
</script>
</body>
</html>"""

out = os.path.join(OUTPUT_DIR, "dashboard.html")
# out = os.path.join(OUTPUT_DIR, "dashboard_non_1a.html")

with open(out, "w", encoding="utf-8") as f:
    f.write(html)

mb = os.path.getsize(out)/1024/1024
print(f"  OK: dashboard.html  ({mb:.1f} MB)")
print("Xong!")
