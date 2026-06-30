
import sys, os, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

CSV_PATH   = os.path.join("output_routes", "route_distribution.csv")
OUTPUT_DIR = "output_routes"

df = pd.read_csv(CSV_PATH)

if "first_trunk_prov" not in df.columns:
    wh = pd.read_csv("warehouse.csv")
    wh_prov = wh.set_index("name")["province_name"].to_dict()
    df["first_trunk_prov"] = df["first_trunk"].map(wh_prov)
    df["last_trunk_prov"]  = df["last_trunk"].map(wh_prov)    

df["first_trunk_prov"] = df["first_trunk_prov"].fillna("(Không rõ tỉnh)")
df["last_trunk_prov"]  = df["last_trunk_prov"].fillna("(Không rõ tỉnh)")

# ── Thống kê tổng quan ────────────────────────────────────────────────────
rank1 = df[df["rank_trong_cap"] == 1]
total_pairs   = len(rank1)
co_dinh_90    = int((rank1["pct_bill_theo_cap"] > 90).sum())
only_1_route  = int((rank1["so_route_theo_cap"] == 1).sum())

stats_html = {
    "total_pairs":    total_pairs,
    "co_dinh_90":     co_dinh_90,
    "pct_co_dinh_90": round(co_dinh_90 / total_pairs * 100, 1),
    "only_1_route":   only_1_route,
}

# ── Xây dựng cấu trúc dữ liệu cho HTML ──────────────────────────────────
# Cần 2 cấu trúc:
#   1. prov_to_wh:  {prov: [wh1, wh2, ...]}  — kho theo tỉnh
#   2. data:        {a: {b: [route_rows]}}     — routes theo cặp kho

# 1. Tỉnh → danh sách kho nguồn (chỉ kho có data)
prov_a = {}
for prov, grp in df.groupby("first_trunk_prov"):
    whs = sorted(grp["first_trunk"].unique().tolist())
    prov_a[prov] = whs

# Tỉnh → danh sách kho đích (có thể nhận từ kho nào đó)
prov_b = {}
for prov, grp in df.groupby("last_trunk_prov"):
    whs = sorted(grp["last_trunk"].unique().tolist())
    prov_b[prov] = whs

# 2. Data routes: {a: {b: [rows]}}
embed_df = df[[
    "first_trunk", "last_trunk", "trunk_route",
    "so_bill", "tong_kg",
    "pct_bill_theo_cap", "pct_weight_theo_cap",
    "n_stops", "rank_trong_cap", "so_route_theo_cap",
]].copy()
embed_df.columns = [
    "a", "b", "route",
    "bills", "kg",
    "pct_bill", "pct_kg",
    "stops", "rank", "n_routes",
]

data_dict = {}
for (a, b), sub in embed_df.groupby(["a", "b"]):
    rows = sub.sort_values("rank")[
        ["route", "bills", "kg", "pct_bill", "pct_kg", "stops", "rank", "n_routes"]
    ].to_dict("records")
    if a not in data_dict:
        data_dict[a] = {}
    data_dict[a][b] = rows

# Tỉnh B có thể nhận từ kho A cụ thể
# {wh_a: {prov_b: [wh_b, ...]}}
a_to_provb = {}
for a_wh, grp in embed_df.groupby("a"):
    sub = df[df["first_trunk"] == a_wh][["last_trunk", "last_trunk_prov"]].drop_duplicates()
    mapping = {}
    for _, row in sub.iterrows():
        p = row["last_trunk_prov"]
        w = row["last_trunk"]
        if p not in mapping:
            mapping[p] = []
        mapping[p].append(w)
    # sắp xếp
    for p in mapping:
        mapping[p] = sorted(mapping[p])
    a_to_provb[a_wh] = mapping

data_json    = json.dumps(data_dict,    ensure_ascii=False)
prov_a_json  = json.dumps(prov_a,       ensure_ascii=False)
a_to_provb_json = json.dumps(a_to_provb, ensure_ascii=False)
all_provs_a  = json.dumps(sorted(prov_a.keys()), ensure_ascii=False)
# Danh sách tất cả kho nguồn và kho đích (để chọn thẳng không qua tỉnh)
all_wh_a = json.dumps(sorted(df["first_trunk"].unique().tolist()), ensure_ascii=False)
all_wh_b = json.dumps(sorted(df["last_trunk"].unique().tolist()),  ensure_ascii=False)

# Map kho → tỉnh (để sync ngược lên province dropdown khi chọn thẳng)
wh_to_prov_a = json.dumps(df.drop_duplicates("first_trunk")
                           .set_index("first_trunk")["first_trunk_prov"].to_dict(),
                           ensure_ascii=False)
wh_to_prov_b = json.dumps(df.drop_duplicates("last_trunk")
                           .set_index("last_trunk")["last_trunk_prov"].to_dict(),
                           ensure_ascii=False)

stats_json   = json.dumps(stats_html)

# ── Generate HTML ─────────────────────────────────────────────────────────
print("Tao HTML...")

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route Explorer — Phân tích cung đường bưu cục</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0e1117; --panel:#161b27; --card:#1e2333; --border:#2a2f45;
  --accent:#7c6dfa; --green:#00c896; --yellow:#f5a623; --red:#f0506e;
  --text:#dde3f0; --muted:#6b7591;
  --radius:12px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}}

/* HEADER */
.hdr{{background:linear-gradient(135deg,#161b27,#0e1117);border-bottom:1px solid var(--border);
  padding:18px 28px;display:flex;align-items:center;justify-content:space-between;}}
.hdr h1{{font-size:1.25rem;font-weight:700;letter-spacing:-.3px}}
.hdr h1 em{{color:var(--accent);font-style:normal}}
.hdr .sub{{font-size:.78rem;color:var(--muted)}}

/* STAT CARDS */
.stats{{display:flex;gap:14px;padding:18px 28px;flex-wrap:wrap}}
.sc{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:14px 20px;flex:1;min-width:140px;transition:transform .2s}}
.sc:hover{{transform:translateY(-2px)}}
.sc .v{{font-size:1.8rem;font-weight:700;color:var(--accent)}}
.sc .l{{font-size:.75rem;color:var(--muted);margin-top:3px;line-height:1.4}}

/* LAYOUT */
.layout{{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 172px)}}

/* SIDEBAR */
.sidebar{{background:var(--panel);border-right:1px solid var(--border);
  padding:20px 16px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}}

.sel-group{{display:flex;flex-direction:column;gap:6px}}
.sel-group label{{font-size:.7rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.6px;color:var(--muted)}}
.sel-group select{{
  background:var(--card);color:var(--text);border:1px solid var(--border);
  border-radius:8px;padding:9px 32px 9px 10px;font-size:.85rem;font-family:inherit;
  appearance:none;cursor:pointer;outline:none;transition:border-color .2s;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='%236b7591'%3E%3Cpath d='M7 10l5 5 5-5z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center;
}}
.sel-group select:focus{{border-color:var(--accent)}}
.sel-group select:disabled{{opacity:.4;cursor:default}}

.divider{{height:1px;background:var(--border);margin:2px 0}}

/* PAIR INFO CARD */
.pcard{{background:linear-gradient(135deg,rgba(124,109,250,.1),rgba(0,200,150,.06));
  border:1px solid rgba(124,109,250,.25);border-radius:10px;padding:12px 14px;
  display:none;font-size:.82rem}}
.pcard.show{{display:block}}
.pcard-title{{font-size:.7rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;color:var(--muted);margin-bottom:10px}}
.pcard-row{{display:flex;justify-content:space-between;padding:3px 0;}}
.pcard-row .k{{color:var(--muted)}}
.pcard-row .v{{font-weight:600}}
.badge{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.7rem;font-weight:600}}
.bg-green{{background:rgba(0,200,150,.15);color:var(--green)}}
.bg-yellow{{background:rgba(245,166,35,.15);color:var(--yellow)}}
.bg-red{{background:rgba(240,80,110,.15);color:var(--red)}}

/* CONTENT */
.content{{padding:22px 26px;overflow-y:auto;display:flex;flex-direction:column;gap:20px}}
.empty{{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;color:var(--muted);gap:10px;text-align:center}}
.empty svg{{opacity:.2}}

/* CARDS */
.ccard{{background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:18px 22px}}
.ccard h3{{font-size:.9rem;font-weight:600;margin-bottom:14px}}
.ccard h3 span{{color:var(--muted);font-weight:400;font-size:.8rem}}
.chart-wrap{{position:relative;height:240px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}

/* TABLE */
.tbl-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.8rem}}
thead th{{background:rgba(124,109,250,.12);color:var(--muted);font-size:.68rem;
  text-transform:uppercase;letter-spacing:.5px;padding:9px 11px;
  text-align:left;border-bottom:1px solid var(--border);position:sticky;top:0}}
tbody tr{{border-bottom:1px solid rgba(255,255,255,.035);transition:background .12s}}
tbody tr:hover{{background:rgba(255,255,255,.025)}}
tbody td{{padding:8px 11px;vertical-align:middle}}
tbody tr:first-child td{{color:var(--green);font-weight:500}}

.rb{{width:22px;height:22px;border-radius:50%;display:inline-flex;
  align-items:center;justify-content:center;font-size:.72rem;font-weight:700}}
.r1{{background:rgba(0,200,150,.18);color:var(--green)}}
.r2{{background:rgba(124,109,250,.18);color:var(--accent)}}
.rn{{background:rgba(255,255,255,.06);color:var(--muted)}}

.bar-wrap{{display:flex;align-items:center;gap:7px}}
.bar{{height:5px;border-radius:3px;background:rgba(124,109,250,.15);flex:1;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--accent),var(--green))}}
.bar-fill.kg{{background:linear-gradient(90deg,var(--green),var(--yellow))}}
.route-cell{{max-width:380px;line-height:1.5;color:var(--text)}}
.arr{{color:var(--accent);font-weight:600}}

@media(max-width:860px){{
  .layout{{grid-template-columns:1fr;height:auto}}
  .two{{grid-template-columns:1fr}}
  .stats{{flex-direction:column}}
}}
</style>
</head>
<body>

<div class="hdr">
  <h1>Route <em>Explorer</em> — Phân tích cung đường bưu cục</h1>
  <div class="sub">Dữ liệu: {total_pairs:,} cặp bưu cục &nbsp</div>
</div>

<div class="stats" id="stats-bar"></div>

<div class="layout">
  <!-- SIDEBAR -->
  <div class="sidebar">
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
      letter-spacing:.7px;color:var(--yellow);padding:2px 0 6px">
      Chọn Kho (Không cần chọn tỉnh)
    </div>

    <div class="sel-group">
      <label for="sel-quick-a">Kho gửi</label>
      <select id="sel-quick-a">
        <option value="">-- Tất cả kho gửi --</option>
      </select>
    </div>

    <div class="sel-group">
      <label for="sel-quick-b">Kho nhận</label>
      <select id="sel-quick-b" disabled>
        <option value="">-- Tất cả kho nhận --</option>
      </select>
    </div>

    <div class="divider"></div>
    
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
      letter-spacing:.7px;color:var(--accent);padding:2px 0 6px">
      📤 Bưu cục GỬI
    </div>

    <div class="sel-group">
      <label for="sel-prov-a">Tỉnh / Thành phố</label>
      <select id="sel-prov-a">
        <option value="">-- Chọn tỉnh gửi --</option>
      </select>
    </div>

    <div class="sel-group">
      <label for="sel-a">Kho / Bưu cục</label>
      <select id="sel-a" disabled>
        <option value="">-- Chọn kho gửi --</option>
      </select>
    </div>

    <div class="divider"></div>

    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
      letter-spacing:.7px;color:var(--green);padding:6px 0">
      📥 Bưu cục NHẬN
    </div>

    <div class="sel-group">
      <label for="sel-prov-b">Tỉnh / Thành phố</label>
      <select id="sel-prov-b" disabled>
        <option value="">-- Chọn tỉnh nhận --</option>
      </select>
    </div>

    <div class="sel-group">
      <label for="sel-b">Kho / Bưu cục</label>
      <select id="sel-b" disabled>
        <option value="">-- Chọn kho nhận --</option>
      </select>
    </div>

    <div class="pcard" id="pcard">
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
  <div class="content" id="content">
    <div class="empty">
      <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
        <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
      <p>Chọn tỉnh gửi → kho gửi<br>→ tỉnh nhận → kho nhận<br>để xem phân tích cung đường</p>
    </div>
  </div>
</div>

<script>
// ── DATA ──────────────────────────────────────────────────────────────────
const DATA       = {data_json};
const PROV_A     = {prov_a_json};
const A_TO_PROVB = {a_to_provb_json};
const STATS      = {stats_json};
const ALL_PROVS_A = {all_provs_a};
const ALL_WH_A    = {all_wh_a};
const ALL_WH_B    = {all_wh_b};
const WH_TO_PROV_A = {wh_to_prov_a};
const WH_TO_PROV_B = {wh_to_prov_b};

// ── STATS BANNER ──────────────────────────────────────────────────────────
const statsBar = document.getElementById('stats-bar');
const statItems = [
  [STATS.total_pairs.toLocaleString('vi-VN'), 'Tổng cặp bưu cục '],
  [STATS.pct_co_dinh_90 + '%', 'Cặp có route chính >90% bill'],
  [STATS.only_1_route.toLocaleString('vi-VN'), 'Cặp chỉ có đúng 1 route'],
  [STATS.co_dinh_90.toLocaleString('vi-VN'), 'Cặp tuyến gần cố định (>90%)'],
];
statsBar.innerHTML = statItems.map(([v,l]) =>
  `<div class="sc"><div class="v">${{v}}</div><div class="l">${{l}}</div></div>`
).join('');

// ── DOM refs ─────────────────────────────────────────────────────────────
const selProvA = document.getElementById('sel-prov-a');
const selA     = document.getElementById('sel-a');
const selProvB = document.getElementById('sel-prov-b');
const selB     = document.getElementById('sel-b');
const pcard    = document.getElementById('pcard');
const content  = document.getElementById('content');

const selQuickA = document.getElementById('sel-quick-a');
const selQuickB = document.getElementById('sel-quick-b');

// Populate kho gửi (tất cả)
ALL_WH_A.forEach(w => {{
  const o = document.createElement('option');
  o.value = w; o.textContent = w;
  selQuickA.appendChild(o);
}});

// Chọn kho gửi → populate kho nhận (chỉ những kho có route từ A)
selQuickA.addEventListener('change', () => {{
  const a = selQuickA.value;
  selQuickB.innerHTML = '<option value="">-- Tất cả kho nhận --</option>';
  selQuickB.disabled = !a;
  pcard.classList.remove('show');
  showEmpty();
  if (!a) return;

  // Lấy danh sách kho nhận có route từ a
  const bList = Object.keys(DATA[a] || {{}}).sort();
  bList.forEach(w => {{
    const o = document.createElement('option');
    o.value = w; o.textContent = w;
    selQuickB.appendChild(o);
  }});

  // Sync ngược lên province dropdown (chỉ để hiển thị thông tin)
  const prov = WH_TO_PROV_A[a];
  if (prov) selProvA.value = prov;
}});

// Chọn kho nhận → render ngay
selQuickB.addEventListener('change', () => {{
  const a = selQuickA.value, b = selQuickB.value;
  pcard.classList.remove('show');
  if (!a || !b || !DATA[a] || !DATA[a][b]) {{ showEmpty(); return; }}

  // Sync province display
  const pa = WH_TO_PROV_A[a] || '';
  const pb = WH_TO_PROV_B[b] || '';
  renderPCard(DATA[a][b], pa, pb);
  renderContent(a, b, DATA[a][b], pa, pb);
}});



// ── Populate tỉnh A ───────────────────────────────────────────────────────
ALL_PROVS_A.forEach(p => {{
  const o = document.createElement('option'); o.value = p; o.textContent = p;
  selProvA.appendChild(o);
}});

// ── Chọn tỉnh A → populate kho A ─────────────────────────────────────────
selProvA.addEventListener('change', () => {{
  const prov = selProvA.value;
  resetFrom('a');
  if (!prov) return;
  selA.disabled = false;
  selA.innerHTML = '<option value="">-- Chọn kho gửi --</option>';
  (PROV_A[prov] || []).forEach(w => {{
    const o = document.createElement('option'); o.value = w; o.textContent = w;
    selA.appendChild(o);
  }});
}});

// ── Chọn kho A → populate tỉnh B ─────────────────────────────────────────
selA.addEventListener('change', () => {{
  const a = selA.value;
  resetFrom('b');
  if (!a) return;
  const provBMap = A_TO_PROVB[a] || {{}};
  const provBList = Object.keys(provBMap).sort();
  selProvB.disabled = false;
  selProvB.innerHTML = '<option value="">-- Chọn tỉnh nhận --</option>';
  provBList.forEach(p => {{
    const o = document.createElement('option'); o.value = p; o.textContent = p;
    selProvB.appendChild(o);
  }});
}});

// ── Chọn tỉnh B → populate kho B ─────────────────────────────────────────
selProvB.addEventListener('change', () => {{
  const a = selA.value, prov = selProvB.value;
  resetFrom('wh-b');
  if (!prov) return;
  const whs = (A_TO_PROVB[a] || {{}})[prov] || [];
  selB.disabled = false;
  selB.innerHTML = '<option value="">-- Chọn kho nhận --</option>';
  whs.forEach(w => {{
    const o = document.createElement('option'); o.value = w; o.textContent = w;
    selB.appendChild(o);
  }});
}});

// ── Chọn kho B → render ──────────────────────────────────────────────────
selB.addEventListener('change', () => {{
  const a = selA.value, b = selB.value;
  pcard.classList.remove('show');
  if (!a || !b || !DATA[a] || !DATA[a][b]) {{ showEmpty(); return; }}
  const rows = DATA[a][b];
  const pa = selProvA.value, pb = selProvB.value;
  renderPCard(rows, pa, pb);
  renderContent(a, b, rows, pa, pb);
}});

// ── Helpers ───────────────────────────────────────────────────────────────
function resetFrom(from) {{
  if (from === 'a') {{
    // Reset toàn bộ từ kho gửi trở xuống
    selA.disabled = true;    selA.innerHTML    = '<option value="">-- Chọn kho gửi --</option>';
    selProvB.disabled = true; selProvB.innerHTML = '<option value="">-- Chọn tỉnh nhận --</option>';
    selB.disabled = true;    selB.innerHTML    = '<option value="">-- Chọn kho nhận --</option>';
    pcard.classList.remove('show');
    showEmpty();
  }} else if (from === 'b') {{
    // Reset từ tỉnh nhận trở xuống
    selProvB.disabled = true; selProvB.innerHTML = '<option value="">-- Chọn tỉnh nhận --</option>';
    selB.disabled = true;    selB.innerHTML    = '<option value="">-- Chọn kho nhận --</option>';
    pcard.classList.remove('show');
    showEmpty();
  }} else if (from === 'wh-b') {{
    // Chỉ reset kho nhận
    selB.disabled = true; selB.innerHTML = '<option value="">-- Chọn kho nhận --</option>';
    pcard.classList.remove('show');
    showEmpty();
  }}
}}

function showEmpty() {{
  content.innerHTML = `<div class="empty">
    <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
      <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>
      <polyline points="9 22 9 12 15 12 15 22"/>
    </svg>
    <p>Chọn tỉnh gửi → kho gửi<br>→ tỉnh nhận → kho nhận<br>để xem phân tích cung đường</p>
  </div>`;
}}

function fmt(n)  {{ return (n || 0).toLocaleString('vi-VN'); }}
function fmtKg(k){{ return k >= 1000 ? (k/1000).toFixed(1)+'t' : k.toFixed(1)+'kg'; }}

function renderPCard(rows, pa, pb) {{
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
}}

let chartBill = null, chartKg = null;

function renderContent(a, b, rows, pa, pb) {{
  const labels = rows.map((_,i) => 'Route #'+(i+1));
  const cols   = rows.map((_,i) => i===0 ? 'rgba(0,200,150,.85)' : i===1
    ? 'rgba(124,109,250,.75)' : 'rgba(255,255,255,.18)');
  const makeChart = (data) => ({{
    type:'bar',
    data:{{ labels, datasets:[{{ data, backgroundColor:cols, borderRadius:6, borderSkipped:false }}] }},
    options:{{
      indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{display:false}},
        tooltip:{{ callbacks:{{ label: c => ' '+c.parsed.x.toFixed(2)+'%' }} }} }},
      scales:{{
        x:{{ max:100, ticks:{{color:'#6b7591',callback:v=>v+'%'}}, grid:{{color:'rgba(255,255,255,.04)'}} }},
        y:{{ ticks:{{color:'#6b7591',font:{{size:10}}}}, grid:{{display:false}} }}
      }}
    }}
  }});

  const tableRows = rows.map((r,i) => {{
    const rc = i===0?'r1':i===1?'r2':'rn';
    const rt = r.route.replace(/→/g,'<span class="arr"> → </span>');
    return `<tr>
      <td><span class="rb ${{rc}}">${{r.rank}}</span></td>
      <td class="route-cell">${{rt}}</td>
      <td>${{fmt(r.bills)}}</td>
      <td>
        <div class="bar-wrap">
          <div class="bar"><div class="bar-fill" style="width:${{r.pct_bill}}%"></div></div>
          <span style="min-width:40px;text-align:right">${{r.pct_bill}}%</span>
        </div>
      </td>
      <td>${{fmtKg(r.kg)}}</td>
      <td>
        <div class="bar-wrap">
          <div class="bar"><div class="bar-fill kg" style="width:${{r.pct_kg}}%"></div></div>
          <span style="min-width:40px;text-align:right">${{r.pct_kg}}%</span>
        </div>
      </td>
      <td style="color:var(--muted);text-align:center">${{r.stops}}</td>
    </tr>`;
  }}).join('');

  content.innerHTML = `
    <div class="two">
      <div class="ccard">
        <h3>% Số bill theo route <span>(${{a}} → ${{b}})</span></h3>
        <div class="chart-wrap"><canvas id="ch-bill"></canvas></div>
      </div>
      <div class="ccard">
        <h3>% Khối lượng theo route</h3>
        <div class="chart-wrap"><canvas id="ch-kg"></canvas></div>
      </div>
    </div>
    <div class="ccard">
      <h3>Chi tiết ${{rows.length}} cung đường
        <span>${{pa}} → ${{pb}} &nbsp;|&nbsp; ${{a}} → ${{b}}</span>
      </h3>
      <div class="tbl-wrap"><table>
        <thead><tr>
          <th>Rank</th><th>Cung đường</th>
          <th>Số bill</th><th>% Bill</th>
          <th>Tổng KG</th><th>% KG</th><th>Trạm</th>
        </tr></thead>
        <tbody>${{tableRows}}</tbody>
      </table></div>
    </div>`;

  if (chartBill) chartBill.destroy();
  if (chartKg)   chartKg.destroy();
  chartBill = new Chart(document.getElementById('ch-bill'), makeChart(rows.map(r=>r.pct_bill)));
  chartKg   = new Chart(document.getElementById('ch-kg'),   makeChart(rows.map(r=>r.pct_kg)));
}}
</script>
</body>
</html>"""

out = os.path.join(OUTPUT_DIR, "route_explorer.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
mb = os.path.getsize(out)/1024/1024
print(f" route_explorer.html  ({mb:.1f} MB)")
