import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

MIN_BILL   = 100
CHUNKSIZE  = 500_000
OUTPUT_DIR = "output_routes"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(traces, warehouse, bill): 
    wh = pd.read_csv(warehouse)
    wh_prov = wh.set_index('name')['province_name'].to_dict() 
    
    chunks = []
    traces_reader = pd.read_csv(traces, chunksize= CHUNKSIZE) 
    for chunk in traces_reader: 
        chunks.append(chunk)
    
    routes = pd.concat(chunks, ignore_index = True) 
    bill_df = pd.read_csv(bill, usecols = ['bill_code', 'origin_province', 'actual_weight'])
    routes = routes.merge(bill_df, on = 'bill_code', how = 'left')   
    return routes, wh_prov

def filter_bill(traces):
    df = traces[traces["n_trunk_stops"] >= 2].copy()
    # df = df[df["first_trunk"] != df["last_trunk"]]
    # pair_count = df.groupby(["first_trunk", "last_trunk"])["bill_code"].nunique()
    # valid_pairs = pair_count[pair_count >= MIN_BILL].index
    # df = df.set_index(["first_trunk", "last_trunk"]).loc[valid_pairs].reset_index()
    return df

def route_distribution(df, wh_prov): 
    grp = (
        df.groupby(['first_trunk', 'last_trunk', 'trunk_route'])
        .agg(
            so_bill = ('bill_code', 'nunique'), 
            tong_kg = ('actual_weight','sum')
        ).
        reset_index()
    )

    grp["first_trunk_prov"] = grp["first_trunk"].map(wh_prov)
    grp["last_trunk_prov"]  = grp["last_trunk"].map(wh_prov)

    grp["tong_kg"] = grp["tong_kg"].round(1)

    grp["so_route_theo_cap"] = (
        grp.groupby(["first_trunk", "last_trunk"])["trunk_route"].transform("count")
    )

    pair_total_bill   = grp.groupby(["first_trunk", "last_trunk"])["so_bill"].transform("sum")
    pair_total_weight = grp.groupby(["first_trunk", "last_trunk"])["tong_kg"].transform("sum")

    grp["pct_bill_theo_cap"]   = (grp["so_bill"] / pair_total_bill   * 100).round(2)
    grp["pct_weight_theo_cap"] = (grp["tong_kg"] / pair_total_weight * 100).round(2)

    grp["n_stops"] = grp["trunk_route"].str.count("→") + 1

    grp["rank_trong_cap"] = (
        grp.groupby(["first_trunk", "last_trunk"])["so_bill"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    grp = grp.sort_values(
        ["first_trunk", "last_trunk", "rank_trong_cap"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    grp.to_csv(os.path.join(OUTPUT_DIR, "route_distribution.csv"), index=False, encoding="utf-8-sig")
    print(f"  OK: route_distribution.csv  ({len(grp):,} dong)")
    return grp


def pair_stats(grp):

    df    = grp.copy()
    rank1 = df[df["rank_trong_cap"] == 1].copy()

    ps = rank1[[
        "first_trunk", "last_trunk",
        "first_trunk_prov", "last_trunk_prov",  
        "so_route_theo_cap",
        "pct_bill_theo_cap", "pct_weight_theo_cap",
        "trunk_route", "n_stops",
    ]].rename(columns={
        "pct_bill_theo_cap":   "pct_route_chinh_bill",
        "pct_weight_theo_cap": "pct_route_chinh_kg",
        "trunk_route":         "route_chinh",
        "n_stops":             "so_tram_route_chinh",
    })

    pair_bill_total = df.groupby(["first_trunk", "last_trunk"])["so_bill"].sum().rename("tong_bill")
    pair_kg_total   = df.groupby(["first_trunk", "last_trunk"])["tong_kg"].sum().rename("tong_kg")
    ps = ps.merge(pair_bill_total, on=["first_trunk", "last_trunk"])
    ps = ps.merge(pair_kg_total,   on=["first_trunk", "last_trunk"])
    ps["tong_kg"] = ps["tong_kg"].round(1)
    ps = ps.sort_values("tong_bill", ascending=False).reset_index(drop=True)

    bins_pct   = [0, 50, 70, 90, 95, 100.001]
    labels_pct = ["<50%", "50-70%", "70-90%", "90-95%", ">95%"]
    ps["nhom_co_dinh"] = pd.cut(
        ps["pct_route_chinh_bill"], bins=bins_pct, labels=labels_pct, right=False
    )

    bins_n   = [1, 2, 3, 4, 6, 11, 9999]
    labels_n = ["1 route", "2 routes", "3 routes", "4-5 routes", "6-10 routes", ">10 routes"]
    ps["nhom_so_route"] = pd.cut(
        ps["so_route_theo_cap"], bins=bins_n, labels=labels_n, right=False
    )
    
    ps.to_csv(os.path.join(OUTPUT_DIR, "pair_stats.csv"), index=False)
    print(f"  OK: pair_stats.csv  ({len(ps):,} cap)")
    return ps


if __name__ == "__main__":
    routes, wh_prov = load_data(
        "output_all_traces/bill_trunk_traces.csv",
        "warehouse.csv",
        "bill.csv"
    )
    # df  = filter_bill(routes)


    grp = route_distribution(routes, wh_prov)
    ps  = pair_stats(grp)