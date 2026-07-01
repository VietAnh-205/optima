"""
filter_direct_to_1A.py
======================
Lọc các bill đi THẲNG từ kho gửi → kho 1A (không qua kho trung gian nào).

Định nghĩa "đi thẳng":
  - Trong chuỗi kho unique (loại trùng liên tiếp) theo thứ tự thời gian:
      vị trí 0 = kho gửi, vị trí 1 = kho tiếp theo
  - Điều kiện: kho ở vị trí 1 PHẢI nằm trong danh sách warehouse_1A.csv
  - Tức là: kho_gửi → kho_1A (không qua kho nào ở giữa)

Thời gian vận chuyển:
  - Bắt đầu : thời gian OUT cuối cùng ở kho gửi (trước khi đến kho 1A)
  - Kết thúc : thời gian IN đầu tiên ở kho 1A
  - Kết quả  : làm tròn đến giờ (round)

Output: output_all_traces/direct_to_1A.csv
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# ─── Config ────────────────────────────────────────────────────────────────────
WAREHOUSE_1A  = "warehouse_1A.csv"
BILL_SCHEDULE = "bill_schedule.csv"
OUTPUT_DIR    = "output_all_traces"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "direct_to_1A.csv")
CHUNKSIZE     = 500_000
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. Load danh sách kho 1A
    print("Đọc danh sách kho 1A...")
    wh_1a = pd.read_csv(WAREHOUSE_1A)
    set_1a = set(wh_1a["name"].dropna().str.strip())
    print(f"   -> {len(set_1a)} kho 1A")

    # 2. Load bill_schedule theo chunk
    print("Đọc bill_schedule.csv theo chunk...")
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(
        BILL_SCHEDULE,
        usecols=["bill_code", "io_status", "io_time", "warehouse_name"],
        chunksize=CHUNKSIZE,
    ):
        chunk["io_time"] = pd.to_datetime(chunk["io_time"])
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"   -> Đã đọc {total_rows:,} dòng...", end="\r")

    sched = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"\n   -> Tổng: {len(sched):,} dòng bill_schedule")

    # 3. Sắp xếp
    print("Sắp xếp lịch trình...")
    sched = sched.sort_values(["bill_code", "io_time"]).reset_index(drop=True)

    # 4. Xây dựng chuỗi kho unique (loại trùng liên tiếp)
    print("Xây dựng chuỗi kho unique...")
    sched["prev_wh"] = sched.groupby("bill_code")["warehouse_name"].shift(1)
    sched_unique = sched[sched["warehouse_name"] != sched["prev_wh"]].copy()
    sched_unique["wh_rank"] = sched_unique.groupby("bill_code").cumcount()

    # 5. Lấy kho thứ 0 (kho gửi) và kho thứ 1 (kho tiếp theo)
    wh_first = (
        sched_unique[sched_unique["wh_rank"] == 0]
        [["bill_code", "warehouse_name"]]
        .rename(columns={"warehouse_name": "kho_gui"})
    )
    wh_second = (
        sched_unique[sched_unique["wh_rank"] == 1]
        [["bill_code", "warehouse_name"]]
        .rename(columns={"warehouse_name": "kho_thu_2"})
    )

    pair = wh_first.merge(wh_second, on="bill_code", how="inner")

    # 6. Lọc bill đi thẳng kho gửi → kho 1A
    print("Lọc bill đi thẳng kho gửi -> kho 1A...")
    direct = pair[pair["kho_thu_2"].isin(set_1a)].copy()
    direct = direct.rename(columns={"kho_thu_2": "kho_1A"})
    print(f"   -> {len(direct):,} bill đi thẳng")

    # 7. Tính thời gian vận chuyển
    print("Tính thời gian vận chuyển...")
    direct_bills = set(direct["bill_code"])
    sched_direct = sched[sched["bill_code"].isin(direct_bills)].copy()
    sched_direct = sched_direct.sort_values(["bill_code", "io_time"])

    # Gán kho_gui và kho_1A vào
    sched_direct = sched_direct.merge(
        direct[["bill_code", "kho_gui", "kho_1A"]],
        on="bill_code",
        how="left"
    )

    # IN đầu tiên ở kho_1A
    mask_in_1a = (
        (sched_direct["warehouse_name"] == sched_direct["kho_1A"]) &
        (sched_direct["io_status"] == "IN")
    )
    in_1a_first = (
        sched_direct[mask_in_1a]
        .groupby("bill_code")["io_time"]
        .first()
        .rename("time_in_1A")
    )

    # Gắn time_in_1A để lọc OUT kho gửi trước thời điểm đó
    sched_direct = sched_direct.merge(in_1a_first, on="bill_code", how="left")

    # OUT cuối cùng ở kho gửi (trước khi IN tại kho_1A)
    mask_out_gui = (
        (sched_direct["warehouse_name"] == sched_direct["kho_gui"]) &
        (sched_direct["io_status"] == "OUT") &
        (sched_direct["io_time"] < sched_direct["time_in_1A"])
    )
    out_gui_last = (
        sched_direct[mask_out_gui]
        .groupby("bill_code")["io_time"]
        .last()
        .rename("time_out_gui")
    )

    # 8. Ghép kết quả
    result = (
        direct
        .merge(in_1a_first, on="bill_code", how="left")
        .merge(out_gui_last, on="bill_code", how="left")
    )

    # 9. Tính thời gian (làm tròn theo giờ)
    result["thoi_gian_van_chuyen_giay"] = (
        result["time_in_1A"] - result["time_out_gui"]
    ).dt.total_seconds()

    result["thoi_gian_van_chuyen_gio"] = (
        result["thoi_gian_van_chuyen_giay"] / 3600
    ).round(0).astype("Int64")

    # 10. Loại bill thiếu dữ liệu hoặc thời gian âm
    n_before = len(result)
    result = result.dropna(subset=["time_out_gui", "time_in_1A"])
    result = result[result["thoi_gian_van_chuyen_giay"] >= 0]
    n_after = len(result)
    print(f"   -> Loại {n_before - n_after} bill thiếu dữ liệu / thời gian âm")

    # 11. Format và lưu
    result["time_out_gui"] = result["time_out_gui"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["time_in_1A"]   = result["time_in_1A"].dt.strftime("%Y-%m-%d %H:%M:%S")

    result = result[[
        "bill_code",
        "kho_gui",
        "kho_1A",
        "time_out_gui",
        "time_in_1A",
        "thoi_gian_van_chuyen_gio",
    ]].sort_values("bill_code").reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n==> Đã lưu {len(result):,} bill -> {OUTPUT_FILE}")

    # 12. Thống kê nhanh
    print("\n===== THỐNG KÊ NHANH =====")
    print(f"Tổng bill đi thẳng kho gửi -> kho 1A : {len(result):,}")
    print(f"Thời gian TB (giờ)                    : {result['thoi_gian_van_chuyen_gio'].mean():.1f}")
    print(f"Thời gian Min (giờ)                   : {result['thoi_gian_van_chuyen_gio'].min()}")
    print(f"Thời gian Max (giờ)                   : {result['thoi_gian_van_chuyen_gio'].max()}")
    print()
    print("Top 10 tuyến kho_gui -> kho_1A phổ biến nhất:")
    top_routes = (
        result.groupby(["kho_gui", "kho_1A"])
        .size()
        .reset_index(name="so_bill")
        .sort_values("so_bill", ascending=False)
        .head(10)
    )
    print(top_routes.to_string(index=False))

    print()
    print("Phân phối thời gian vận chuyển (giờ):")
    print(result["thoi_gian_van_chuyen_gio"].describe().to_string())


if __name__ == "__main__":
    main()








import os


import pandas as pd

OUTPUT_DIR  = "output_all_traces"


wh1a   = pd.read_csv('warehouse_1A.csv')
set_1a = set(wh1a['name'].dropna().str.strip())
chunks = []
for chunk in pd.read_csv('bill_schedule.csv', chunksize=500_000,
                         usecols=['bill_code','io_status','io_time','warehouse_name']):
    chunk['io_time'] = pd.to_datetime(chunk['io_time'])
    chunks.append(chunk)

sche = pd.concat(chunks, ignore_index=True)
del chunks

sche = sche.sort_values(['bill_code', 'io_time']).reset_index(drop=True)

sche['pre_wh']  = sche.groupby('bill_code')['warehouse_name'].shift(1)
sche_unique     = sche[sche['warehouse_name'] != sche['pre_wh']].copy()
del sche

sche_unique['wh_rank'] = sche_unique.groupby('bill_code').cumcount()

# Thêm rank tính từ cuối (max_rank - wh_rank) để lấy rank cuối và gần cuối
max_rank = sche_unique.groupby('bill_code')['wh_rank'].transform('max')
sche_unique['wh_rank_rev'] = max_rank - sche_unique['wh_rank']
# wh_rank_rev == 0  → kho cuối   (wh_d)
# wh_rank_rev == 1  → kho gần cuối (wh_d1a)

# Chỉ giữ bill có ≥ 2 kho unique (để có rank 0 và rank 1)
count_stops = sche_unique.groupby('bill_code')['wh_rank'].max()
valid_bills  = set(count_stops[count_stops >= 1].index)   # max_rank >= 1 → ≥ 2 kho
sche_unique  = sche_unique[sche_unique['bill_code'].isin(valid_bills)].copy()
print(f"    -> {len(sche_unique):,} dòng sche_unique (bill >= 2 kho)")

# ── 4. Trích xuất các mốc kho & io_time theo rank ────────────────────────────
def get_rank(rank_col, rank_val, new_wh_col, new_time_col):
    """Lấy warehouse_name và io_time tại wh_rank == rank_val."""
    sub = sche_unique[sche_unique[rank_col] == rank_val][
        ['bill_code', 'warehouse_name', 'io_time']
    ].rename(columns={
        'warehouse_name': new_wh_col,
        'io_time':        new_time_col,
    })
    return sub

wh_o    = get_rank('wh_rank',     0, 'kho_o',    'time_o')     # kho gửi
wh_o1a  = get_rank('wh_rank',     1, 'kho_o1a',  'time_o1a')   # kho tiếp theo (phải là 1A)
wh_d    = get_rank('wh_rank_rev', 0, 'kho_d',    'time_d')     # kho đích (cuối cùng)
wh_d1a  = get_rank('wh_rank_rev', 1, 'kho_d1a',  'time_d1a')   # kho trước đích (phải là 1A)

# ── 5. Ghép vào một bảng ──────────────────────────────────────────────────────
result = (
    wh_o
    .merge(wh_o1a,  on='bill_code', how='inner')
    .merge(wh_d1a,  on='bill_code', how='inner')
    .merge(wh_d,    on='bill_code', how='inner')
)

# ── 6. Lọc điều kiện ──────────────────────────────────────────────────────────
# Điều kiện 1: kho tiếp theo kho gửi (rank=1) phải là kho 1A
cond_o1a_is_1a = result['kho_o1a'].isin(set_1a)

# Điều kiện 2: kho trước đích (rank_rev=1) phải là kho 1A
cond_d1a_is_1a = result['kho_d1a'].isin(set_1a)

# Điều kiện 3: kho_o1a và kho_d1a phải liên tiếp nhau
# → nghĩa là không có kho nào ở giữa rank của o1a và d1a
# → tương đương: với bill có đúng 2 "biên" thì kho_d là cuối và phải đủ liên tiếp
# → cụ thể: kho_d1a đi thẳng tới kho_d (không kho ở giữa) = đã đảm bảo vì wh_rank_rev

result = result[cond_o1a_is_1a & cond_d1a_is_1a].copy()
print(f"[5] {len(result):,} bill sau lọc (kho_o→1A và 1A→kho_d đều thẳng)")

# ── 7. Tính thời gian ─────────────────────────────────────────────────────────
# time_o_to_o1a  = time_o1a - time_o     (rank 1 - rank 0)
# time_d1a_to_d  = time_d   - time_d1a   (rank_rev 0 - rank_rev 1)
result['time_o_to_o1a_giay']  = (result['time_o1a'] - result['time_o']).dt.total_seconds()
result['time_d1a_to_d_giay']  = (result['time_d']   - result['time_d1a']).dt.total_seconds()

result['time_o_to_o1a_gio']   = (result['time_o_to_o1a_giay'] / 3600).round(0).astype('Int64')
result['time_d1a_to_d_gio']   = (result['time_d1a_to_d_giay'] / 3600).round(0).astype('Int64')

# Loại thời gian âm
n_before = len(result)
result = result[
    (result['time_o_to_o1a_giay'] >= 0) &
    (result['time_d1a_to_d_giay'] >= 0)
]
print(f"[6] Loại {n_before - len(result):,} bill thời gian âm → còn {len(result):,}")

# ── 8. Format & lưu ───────────────────────────────────────────────────────────
for col in ['time_o', 'time_o1a', 'time_d1a', 'time_d']:
    result[col] = result[col].dt.strftime("%Y-%m-%d %H:%M:%S")

result = result[[
    'bill_code',
    'kho_o',   'kho_o1a',
    'time_o',  'time_o1a',  'time_o_to_o1a_gio',
    'kho_d1a', 'kho_d',
    'time_d1a','time_d',    'time_d1a_to_d_gio',
]].sort_values('bill_code').reset_index(drop=True)

result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n==> Lưu {len(result):,} bill -> {OUTPUT_FILE}")

# ── 9. Thống kê nhanh ─────────────────────────────────────────────────────────
print("\n===== THỐNG KÊ =====")
print(f"Tổng bill hợp lệ               : {len(result):,}")
print(f"TB time kho_o -> kho_o1a (giờ) : {result['time_o_to_o1a_gio'].mean():.1f}")
print(f"TB time kho_d1a -> kho_d (giờ) : {result['time_d1a_to_d_gio'].mean():.1f}")

print("\nTop 10 tuyến kho_o -> kho_o1a:")
print(
    result.groupby(['kho_o','kho_o1a']).size()
    .reset_index(name='so_bill')
    .sort_values('so_bill', ascending=False)
    .head(10).to_string(index=False)
)

print("\nTop 10 tuyến kho_d1a -> kho_d:")
print(
    result.groupby(['kho_d1a','kho_d']).size()
    .reset_index(name='so_bill')
    .sort_values('so_bill', ascending=False)
    .head(10).to_string(index=False)
)
