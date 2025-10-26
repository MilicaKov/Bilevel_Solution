# precompute_flex.py
# ------------------------------------------------------------
# Build baseline SoC, per-hour shiftability categories, the
# flexibility matrix F^{flex}_{t,k}, and export:
#   1) Flex_matrix.xlsx (sheets: Flex_matrix, baseline_flex)
#   2) data/time_series.xlsx sheet "baseline_flex"
#      with columns: t, Bbase, Fflex1, wflex
# ------------------------------------------------------------

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# -----------------------------
# CONFIG (edit paths if needed)
# -----------------------------
# Input files (your originals)
BASELINE_XLSX = r"C:\Users\HP\source\repos\Function_Solution\Inputs\Baseline.xlsx"
BEV_PARAM_XLSX = r"C:\Users\HP\source\repos\Function_Solution\Inputs\BEV_parameters.xlsx"

# Time step in hours
TIME_STEP_H = 1.0

# Outputs
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TIMESERIES_XLSX = OUT_DIR / "time_series.xlsx"   # MILP reads this file
OUT_FLEX_XLSX = Path("flex_matrix.xlsx")             # full matrix + baseline_flex

# -----------------------------
# Load inputs & basic checks
# -----------------------------
baseline = pd.read_excel(BASELINE_XLSX)   # required cols: t, charger_id, pbase_kw
bev_parameters = pd.read_excel(BEV_PARAM_XLSX)

required_baseline_cols = {"t", "charger_id", "pbase_kw"}
required_param_cols = {"charger_id", "eta", "p_max", "e0", "t_dep", "e_target"}

missing = required_baseline_cols - set(baseline.columns)
if missing:
    raise ValueError(f"Baseline is missing columns: {missing}")

missing_p = required_param_cols - set(bev_parameters.columns)
if missing_p:
    raise ValueError(f"BEV parameters file is missing columns: {missing_p}")

# normalize types
baseline = baseline.copy()
baseline["t"] = pd.to_numeric(baseline["t"], errors="coerce").fillna(0).astype(int)
baseline["charger_id"] = baseline["charger_id"].astype(str)
baseline["pbase_kw"] = pd.to_numeric(baseline["pbase_kw"], errors="coerce").fillna(0.0)

bev_parameters = bev_parameters.copy()
bev_parameters["charger_id"] = bev_parameters["charger_id"].astype(str)

# map of per-charger parameters
bev_params_map = bev_parameters.set_index("charger_id").to_dict(orient="index")

# Sanity: every charger_id in baseline must be in BEV params
missing_ids = set(baseline["charger_id"].unique()) - set(bev_params_map.keys())
if missing_ids:
    raise KeyError(
        f"charger_id(s) present in Baseline but missing in BEV_parameters: {sorted(missing_ids)}"
    )

# Coerce/validate eta and p_max
for b, p in bev_params_map.items():
    eta = float(p["eta"])
    pmax = float(p["p_max"])
    if not (eta > 0 and pmax > 0):
        raise ValueError(f"For charger_id '{b}', eta and p_max must be > 0 (got eta={eta}, p_max={pmax}).")

# ---------------------------------------
# Helper: compute end-of-hour baseline SoC
# ---------------------------------------
def compute_soc_end_hour(
    df: pd.DataFrame, params_map: dict, time_step_h: float = 1.0
) -> pd.DataFrame:
    """
    Returns copy of df with 'Eb_t' = energy at END of hour t.
      - Start from E0 per charger.
      - For t > t_dep: keep last SoC.
      - If pbase_kw > 0: Eb_t += eta * p * Δ (apply efficiency).
      - If pbase_kw == 0 and t ≤ t_dep: carry forward.
      - Never reset to E0 mid-day.
    """
    out = df.sort_values(["charger_id", "t"]).reset_index(drop=True).copy()
    out["eta"]   = out["charger_id"].map(lambda b: float(params_map[b]["eta"]))
    out["t_dep"] = out["charger_id"].map(lambda b: int(params_map[b]["t_dep"]))

    out["E0"]    = out["charger_id"].map(lambda b: float(params_map[b]["e0"]))

    Eb = []
    last_E = {}

    for _, row in out.iterrows():
        b = row["charger_id"]
        t = int(row["t"])
        p = float(row["pbase_kw"])
        eta = float(row["eta"])
        t_dep = int(row["t_dep"])
        if b not in last_E:
            last_E[b] = float(row["E0"])

        if t > t_dep:
            Eb.append(last_E[b])            # after departure: hold
            continue

        if p > 0.0:
            last_E[b] = last_E[b] + eta * p * time_step_h
        Eb.append(last_E[b])                # carry forward (even if p == 0)

    out["Eb_t"] = Eb
    return out.drop(columns=["eta", "E0"])

# -------------------------------------------------------
# Compute per-row category (integer) and dt_min (ceil hrs)
# -------------------------------------------------------
def compute_shiftability_end_hour(df_with_soc: pd.DataFrame, params_map: dict) -> pd.DataFrame:
    """
    Δt_min = ceil(max(0, (E_target - Eb_t)/(η * Pmax)))
    cat_bt = floor(max(0, t_dep - t - Δt_min))
    Forced to 0 when pbase_kw == 0 OR t > t_dep.
    """
    out = df_with_soc.copy()
    out["E_target"] = out["charger_id"].map(lambda b: float(params_map[b]["e_target"]))
    out["Pmax"]     = out["charger_id"].map(lambda b: float(params_map[b]["p_max"]))
    out["eta"]      = out["charger_id"].map(lambda b: float(params_map[b]["eta"]))
    out["t_dep"]    = out["charger_id"].map(lambda b: int(params_map[b]["t_dep"]))

    denom = (out["eta"] * out["Pmax"]).replace(0, np.nan)
    dt_cont = (out["E_target"] - out["Eb_t"]) / denom
    dt_cont = dt_cont.clip(lower=0).fillna(np.inf)
    dt_min = np.ceil(dt_cont)

    absent = (out["pbase_kw"] <= 0.0) | (out["t"] > out["t_dep"])
    dt_min = np.where(absent, 0.0, dt_min)

    cat_cont = (out["t_dep"].astype(float) - out["t"].astype(float) - dt_min)
    cat_cont = np.where(absent, 0.0, np.maximum(0.0, cat_cont))

    out["dt_min"] = dt_min.astype(float)
    out["cat_bt"] = np.floor(cat_cont).astype(int)
    return out

# -------------------------------------------------------
# Build Fflex: F^{flex}_{t,k} = Σ_b P^{base}_{b,t} * 1{cat_{b,t} ≥ k}
# -------------------------------------------------------
def build_flex_matrix(shift_df: pd.DataFrame, K: Optional[int] = None) -> pd.DataFrame:
    if K is None:
        K = int(shift_df["cat_bt"].max()) if len(shift_df) else 0
    t_vals = shift_df["t"].astype(int)
    result = {}
    for k in range(K + 1):
        mask = (shift_df["cat_bt"] >= k).astype(int)
        series = (shift_df["pbase_kw"] * mask).groupby(t_vals).sum()
        result[k] = series
    Fflex = pd.DataFrame(result).sort_index()
    Fflex.index.name = "t"
    Fflex.columns = [f"k={c}" for c in Fflex.columns]
    Fflex.columns.name = "delay bucket"
    return Fflex

# -----------------------------
# Run pipeline
# -----------------------------
if __name__ == "__main__":
    # 1) End-of-hour SoC on baseline
    soc_df = compute_soc_end_hour(baseline, bev_params_map, time_step_h=TIME_STEP_H)

    # 2) Shiftability and categories
    shift_df = compute_shiftability_end_hour(soc_df, bev_params_map)

    # 3) Flexibility matrix
    Fflex = build_flex_matrix(shift_df)

    # 4) Diagnostics (optional)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print("\n--- Shiftability (head) ---")
        cols = ["t", "charger_id", "pbase_kw", "Eb_t", "dt_min", "cat_bt"]
        print(shift_df[cols].round(3).head(40).to_string(index=False))

        print("\n--- Flexibility Matrix (head) ---")
        print(Fflex.round(3).head(24).to_string())

    # 5) Build baseline_flex summary (also used for Excel export)
    Bbase_series = baseline.groupby("t")["pbase_kw"].sum().astype(float).sort_index()
    if "k=1" not in Fflex.columns:
        Fflex["k=1"] = 0.0
    f1 = Fflex.reindex(Bbase_series.index, fill_value=0.0)["k=1"]

    basefx = pd.DataFrame({
        "t": Bbase_series.index.astype(int),
        "Bbase": Bbase_series.values,
        "Fflex1": f1.values,
        "wflex": 0.0
    })

    # 6) Save both Fflex (full) and baseline_flex (summary) to one Excel file
    with pd.ExcelWriter(OUT_FLEX_XLSX, engine="openpyxl") as writer:
        Fflex.to_excel(writer, sheet_name="flex_matrix", index=True)
        basefx.to_excel(writer, sheet_name="baseline_flex", index=False)
    print(f"\nSaved {OUT_FLEX_XLSX.resolve()} with sheets: 'flex_matrix' and 'baseline_flex'")

    # 7) Also write/replace 'baseline_flex' in data/time_series.xlsx for the MILP
    mode = "a" if OUT_TIMESERIES_XLSX.exists() else "w"
    with pd.ExcelWriter(OUT_TIMESERIES_XLSX, engine="openpyxl",
                        mode=mode, if_sheet_exists="replace") as xlw:
        basefx.to_excel(xlw, sheet_name="baseline_flex", index=False)

    print(f"Wrote sheet 'baseline_flex' to: {OUT_TIMESERIES_XLSX.resolve()}")
    print("Columns: t, Bbase, Fflex1, wflex")
