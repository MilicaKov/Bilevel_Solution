# Function_Solution.py
# Bilevel (deferral accounting) single-MIQCP via dualization
# Python 3.9+, Gurobi 12+ ready

import pandas as pd
from pathlib import Path
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

# =========================
# EDIT THESE PATHS IF NEEDED
# =========================
NETWORK_XLSX    = Path(r"C:\Users\HP\source\repos\Function_Solution\Inputs\network.xlsx")
TIMESERIES_XLSX = Path(r"C:\Users\HP\source\repos\Function_Solution\Inputs\time_series.xlsx")
FLEET_XLSX      = Path(r"C:\Users\HP\source\repos\Function_Solution\Inputs\fleet.xlsx")

# Flex matrix produced by precompute script (use CSV or XLSX)
FLEX_CSV  = Path(r"C:\Users\HP\source\repos\Function_Solution\Precomputation_Flexibility_Matrix\Precomputation_Flexibility_Matrix\flex_matrix.csv")
FLEX_XLSX = Path(r"C:\Users\HP\source\repos\Function_Solution\Precomputation_Flexibility_Matrix\Precomputation_Flexibility_Matrix\flex_matrix.xlsx")
FLEX_SHEET = "Flex_matrix"   # sheet name in the Excel output

# =========================
# SOLVER & SWITCHES
# =========================
SOLVER_NAME = "gurobi"       # or "highs" if you don't have Gurobi
SOLVER_OPTS = {"MIPGap": 1e-3, "TimeLimit": 600} if SOLVER_NAME == "gurobi" else {"time_limit": 600}

USE_SWITCH_LIMIT = False     # limit number of ladder switches
ZMAX = 4

USE_NETWORK = True           # << set False to isolate deferral feasibility (no grid physics)
USE_ENS     = True           # energy-not-served penalty active

# =========================
# HELPERS
# =========================
def require(df: pd.DataFrame, cols, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")

def load_flex_matrix() -> pd.DataFrame:
    """
    Load flex matrix from CSV or XLSX.
    Returns a DataFrame indexed by 't' with columns 'k=0','k=1',...
    """
    if FLEX_CSV.exists():
        df = pd.read_csv(FLEX_CSV)
        used = FLEX_CSV
    elif FLEX_XLSX.exists():
        df = pd.read_excel(FLEX_XLSX, sheet_name=FLEX_SHEET, engine="openpyxl")
        used = FLEX_XLSX
    else:
        raise FileNotFoundError(
            "Flex matrix not found. Expected one of:\n"
            f"  - {FLEX_CSV}\n"
            f"  - {FLEX_XLSX}\n"
            "Run precompute_flex.py first, or update the paths above."
        )

    # Ensure 't' becomes the index
    if "t" in df.columns:
        df = df.set_index("t")
    else:
        first = df.columns[0]
        if str(first).lower() in {"t", "index", "unnamed: 0"}:
            df = df.set_index(first)
        else:
            raise ValueError("Flex matrix must have a 't' column (or first column).")

    # Keep only k=... columns
    kcols = [c for c in df.columns if isinstance(c, str) and c.startswith("k=")]
    if not kcols:
        raise ValueError("No 'k=' columns found in the flex matrix.")
    df = df[kcols].sort_index()
    df.index = df.index.astype(int)

    print(f"[info] Loaded flex matrix from: {used}")
    return df

# =========================
# READ EXCEL / CSV
# =========================
net = pd.ExcelFile(NETWORK_XLSX)
ts  = pd.ExcelFile(TIMESERIES_XLSX)
fl  = pd.ExcelFile(FLEET_XLSX)

buses = pd.read_excel(net, "buses")
lines = pd.read_excel(net, "lines")

prices = pd.read_excel(ts, "prices")
inel   = pd.read_excel(ts, "inelastic_load")
pv     = pd.read_excel(ts, "pv")
ladder = pd.read_excel(ts, "tariff_ladder")
basefx = pd.read_excel(ts, "baseline_flex")  # needs at least: t, Bbase

trucks = pd.read_excel(fl, "trucks")
avail  = pd.read_excel(fl, "availability")

# Flex matrix (supports CSV or XLSX)
Fflex_df = load_flex_matrix()

# Validate required columns
require(buses,  ["bus_id","vmin","vmax","v0bar_flag"], "buses")
require(lines,  ["from_bus","to_bus","r","x","Smax"], "lines")
require(prices, ["t","pi"], "prices")
require(inel,   ["t","bus_id","ell","qinel"], "inelastic_load")
require(pv,     ["t","bus_id","gPV"], "pv")
require(ladder, ["n","lambda_bar"], "tariff_ladder")
require(basefx, ["t","Bbase"], "baseline_flex")
require(trucks, ["truck_id","node","Pmax","eta","Emin","Emax","Einit","Etarget","tdep"], "trucks")
require(avail,  ["truck_id","t","delta"], "availability")

# Basic sanity: nonnegative prices & ladder (keeps dual costs sane)
if (prices["pi"] < 0).any():
    raise ValueError("prices.pi must be nonnegative for this formulation.")
if (ladder["lambda_bar"] < 0).any():
    raise ValueError("tariff_ladder.lambda_bar must be nonnegative.")

# =========================
# BUILD SETS / PARAMS
# =========================
T = sorted(prices["t"].astype(int).unique().tolist())
I = sorted(buses["bus_id"].astype(int).unique().tolist())
L = list(zip(lines["from_bus"].astype(int), lines["to_bus"].astype(int)))

# Fleet node (single depot assumed)
fleet_nodes = sorted(trucks["node"].astype(int).unique().tolist())
if len(fleet_nodes) != 1:
    raise ValueError(
        f"This model assumes ONE fleet node; found {len(fleet_nodes)}: {fleet_nodes}. "
        "Extend the model to split X per-depot or pre-aggregate your fleet to one node."
    )
FLEET_NODE = fleet_nodes[0]

# Tariff ladder indices
N = ladder["n"].astype(int).tolist()
lam_bar = dict(zip(ladder["n"].astype(int), ladder["lambda_bar"].astype(float)))

# Slack bus voltage reference
v0_rows = buses.loc[buses["v0bar_flag"] == 1]
if v0_rows.empty or len(v0_rows) != 1:
    raise ValueError("Exactly one slack bus must have v0bar_flag=1.")
v0bar = float(v0_rows["vmax"].iloc[0])  # or a dedicated column

# Parameters from time series & fleet
pi = dict(zip(prices["t"].astype(int), prices["pi"].astype(float)))

ell   = {(int(r.bus_id), int(r.t)): float(r.ell)   for r in inel.itertuples()}
qinel = {(int(r.bus_id), int(r.t)): float(r.qinel) for r in inel.itertuples()}
gPV   = {(int(r.bus_id), int(r.t)): float(r.gPV)   for r in pv.itertuples()}

# --- line params (renamed to avoid clash with deferral var) ---
r_dict    = {(int(f),int(t)): float(rv) for f,t,rv in zip(lines["from_bus"], lines["to_bus"], lines["r"])}
x_dict    = {(int(f),int(t)): float(xv) for f,t,xv in zip(lines["from_bus"], lines["to_bus"], lines["x"])}
Smax_dict = {(int(f),int(t)): float(s)  for f,t,s  in zip(lines["from_bus"], lines["to_bus"], lines["Smax"])}

vmin = dict(zip(buses["bus_id"].astype(int), buses["vmin"].astype(float)))
vmax = dict(zip(buses["bus_id"].astype(int), buses["vmax"].astype(float)))

# Aggregated hourly fleet capacity Cap[tau] = sum_b delta[b,tau]*Pmax[b]
Pmax  = dict(zip(trucks["truck_id"].astype(str), trucks["Pmax"].astype(float)))
delta = {(str(r.truck_id), int(r.t)): int(r.delta) for r in avail.itertuples()}
Cap   = {t: sum(delta.get((b,t),0) * Pmax[b] for b in Pmax.keys()) for t in T}

# Baseline energy by origin hour (kWh) — with Δ=1h, it's also kW
Bbase = dict(zip(basefx["t"].astype(int), basefx["Bbase"].astype(float)))

# Flex matrix dictionary Fflex[(t,k)]
K_all = []
for c in Fflex_df.columns:
    if c.startswith("k="):
        K_all.append(int(c.split("=")[1]))
K_all = sorted(set(K_all))
Fflex = {}
for t_idx, row in Fflex_df.iterrows():
    for k in K_all:
        col = f"k={k}"
        val = float(row[col]) if col in row.index else 0.0
        Fflex[(int(t_idx), int(k))] = val

# Optional: deferral penalty phi[t, tau]; if you have a sheet, load it; else zero
phi = {(t, tau): 0.0 for t in T for tau in T if tau >= t}

# ENS penalties
cENS = {(i,t): 3.0 for i in I for t in T}

# =========================
# FEASIBILITY DIAGNOSTICS (pre-model)
# =========================
# a) Ensure Fflex[t,0] >= Bbase[t]
ff0 = {}
for t in T:
    col = "k=0"
    ff0[t] = float(Fflex_df.loc[t, col]) if (col in Fflex_df.columns and t in Fflex_df.index) else 0.0
bad_k0 = [t for t in T if Bbase.get(t,0.0) > ff0.get(t,0.0) + 1e-8]
if bad_k0:
    print("\n[diagnostic] Fflex k=0 is smaller than baseline at hours:", bad_k0)

# b) Cumulative capacity must cover cumulative baseline
cum_bad = []
cumB = 0.0
cumC = 0.0
for tau in T:
    cumB += Bbase.get(tau, 0.0)
    cumC += Cap.get(tau, 0.0)
    if cumB > cumC + 1e-8:
        cum_bad.append((tau, cumB, cumC))
if cum_bad:
    print("\n[diagnostic] Cumulative baseline exceeds cumulative capacity at:")
    for tau, b, c in cum_bad:
        print(f"  τ={tau}: cumBaseline={b:.3f}  cumCapacity={c:.3f}")
else:
    print("\n[diagnostic] Cumulative capacity condition holds for all τ.")

# c) Monotonicity: Fflex[t,k] must be nonincreasing vs k
mono_bad = []
for t in T:
    prev = float("inf")
    for k in K_all:
        val = Fflex.get((t,k), 0.0)
        if val > prev + 1e-9:
            mono_bad.append((t, k-1, k, prev, val))
        prev = val
if mono_bad:
    print("\n[diagnostic] Non-monotone Fflex found (should decrease with k):")
    for t, k1, k2, v1, v2 in mono_bad[:10]:
        print(f"  t={t}, k={k1}->{k2}: {v1:.3f} -> {v2:.3f}")

# Quick table
print("\n[t]  Bbase[t]   Cap[t]   Fflex[t,0]")
for t in T:
    print(f"{t:2d}  {Bbase.get(t,0):9.3f}  {Cap.get(t,0):7.3f}  {ff0.get(t,0):10.3f}")

# =========================
# MODEL
# =========================
m = pyo.ConcreteModel()

# Sets
m.T = pyo.Set(initialize=T, ordered=True)
m.I = pyo.Set(initialize=I)
m.L = pyo.Set(initialize=L, dimen=2)
m.N = pyo.Set(initialize=N, ordered=True)

# Pairs (t, tau) with tau >= t
pairs = [(t, tau) for t in T for tau in T if tau >= t]
m.Pairs = pyo.Set(dimen=2, initialize=pairs, ordered=True)

# Flex K set (global)
m.K = pyo.Set(initialize=K_all, ordered=True)

# Params
m.Delta = pyo.Param(initialize=1.0)            # hour
m.pi    = pyo.Param(m.T, initialize=pi)

m.ell   = pyo.Param(m.I, m.T, initialize=ell, default=0.0)
m.qinel = pyo.Param(m.I, m.T, initialize=qinel, default=0.0)
m.gPV   = pyo.Param(m.I, m.T, initialize=gPV, default=0.0)

# line params (renamed)
m.rline = pyo.Param(m.L, initialize=r_dict)
m.xline = pyo.Param(m.L, initialize=x_dict)
m.Smax = pyo.Param(m.L, initialize=Smax_dict)

m.vmin = pyo.Param(m.I, initialize=vmin)
m.vmax = pyo.Param(m.I, initialize=vmax)
m.v0bar= pyo.Param(initialize=v0bar)

m.Bbase = pyo.Param(m.T, initialize=Bbase, default=0.0)
m.Cap   = pyo.Param(m.T, initialize=Cap, default=0.0)

def fflex_init(m, t, k):
    return Fflex.get((int(t), int(k)), 0.0)
m.Fflex = pyo.Param(m.T, m.K, initialize=fflex_init, default=0.0)

def phi_init(m, t, tau):
    return phi.get((int(t), int(tau)), 0.0)
m.phi = pyo.Param(m.Pairs, initialize=phi_init, default=0.0)

# Tariff ladder
m.lam_bar = pyo.Param(m.N, initialize=lam_bar)

# =========================
# VARIABLES
# =========================
# Ladder binaries and derived tariff τ_t
m.u = pyo.Var(m.T, m.N, within=pyo.Binary)
m.tau = pyo.Expression(m.T, rule=lambda m,t: sum(m.u[t,n] * m.lam_bar[n] for n in m.N))

# Grid states
m.Pslack = pyo.Var(m.T)
m.Qslack = pyo.Var(m.T)
m.P = pyo.Var(m.L, m.T)
m.Q = pyo.Var(m.L, m.T)
m.v = pyo.Var(m.I, m.T)

# ENS
m.kappa = pyo.Var(m.I, m.T, bounds=(0, None))

# ============ Follower (primal) ============
# Deferral accounting variables: X_{t, tau} ≥ 0 (energy from origin t delivered at tau)
m.X = pyo.Var(m.Pairs, bounds=(0, None))

# ============ Follower (dual variables) ============
# y1[t] unrestricted (per-origin equality)
m.y1 = pyo.Var(m.T)  # free by default in Pyomo
# y2[tau] ≥ 0 (per-delivery capacity)
m.y2 = pyo.Var(m.T, bounds=(0, None))
# y3[t,k] ≥ 0 (flex caps)
m.y3 = pyo.Var(m.T, m.K, bounds=(0, None))

# =========================
# OBJECTIVE (UL operating cost)
# =========================
def UL_cost(m):
    en_cost  = sum(m.pi[t]*m.Pslack[t]*m.Delta for t in m.T)
    ens_cost = sum(cENS[(i,t)]*m.kappa[i,t]*m.Delta for i in m.I for t in m.T) if USE_ENS else 0.0
    return en_cost + ens_cost
m.OBJ = pyo.Objective(rule=UL_cost, sense=pyo.minimize)

# =========================
# TARIFF LADDER CONSTRAINTS
# =========================
m.u_one = pyo.Constraint(m.T, rule=lambda m,t: sum(m.u[t,n] for n in m.N) == 1)

if USE_SWITCH_LIMIT:
    m.z = pyo.Var([t for t in T if t>min(T)], within=pyo.Binary)
    def switch_link_hi(m, t, n):
        if t == min(T): 
            return pyo.Constraint.Skip
        return  m.u[t,n] - m.u[t-1,n] <= m.z[t]
    def switch_link_lo(m, t, n):
        if t == min(T): 
            return pyo.Constraint.Skip
        return  m.u[t-1,n] - m.u[t,n] <= m.z[t]
    m.sw_hi = pyo.Constraint(m.T, m.N, rule=switch_link_hi)
    m.sw_lo = pyo.Constraint(m.T, m.N, rule=switch_link_lo)
    m.sw_budget = pyo.Constraint(expr = sum(m.z[t] for t in m.z.index_set()) <= ZMAX)

# =========================
# NETWORK: injections & physics
# =========================
# Aggregate fleet charging delivered at hour t equals sum of X[* , t]
m.Pagg = pyo.Expression(m.T, rule=lambda m,t: sum(m.X[(tt, t)] for tt in m.T if t >= tt))

if USE_NETWORK:
    # Substation
    m.subP = pyo.Constraint(m.T, rule=lambda m,t: m.Pslack[t] == sum(m.P[(0,k),t] for (j,k) in m.L if j==0))
    m.subQ = pyo.Constraint(m.T, rule=lambda m,t: m.Qslack[t] == sum(m.Q[(0,k),t] for (j,k) in m.L if j==0))

    def p_inj(m,i,t):
        fleet = m.Pagg[t] if i == FLEET_NODE else 0.0
        return (m.ell[i,t] - m.kappa[i,t]) + fleet - m.gPV[i,t]
    m.p_inj = pyo.Expression(m.I, m.T, rule=p_inj)
    m.q_inj = pyo.Expression(m.I, m.T, rule=lambda m,i,t: m.qinel[i,t])

    def flow_P_rule(m,i,t):
        if i == 0: 
            return pyo.Constraint.Skip
        out = sum(m.P[(i,k),t] for (j,k) in m.L if j==i)
        inc = sum(m.P[(j,i),t] for (j,k) in m.L if k==i)
        return out - inc == m.p_inj[i,t]
    m.flowP = pyo.Constraint(m.I, m.T, rule=flow_P_rule)

    def flow_Q_rule(m,i,t):
        if i == 0: 
            return pyo.Constraint.Skip
        out = sum(m.Q[(i,k),t] for (j,k) in m.L if j==i)
        inc = sum(m.Q[(j,i),t] for (j,k) in m.L if k==i)
        return out - inc == m.q_inj[i,t]
    m.flowQ = pyo.Constraint(m.I, m.T, rule=flow_Q_rule)

    # Voltage drops (Baran-Wu LinDistFlow)
    m.vdrop = pyo.Constraint(
        m.L, m.T,
        rule=lambda m,j,i,t: m.v[i,t] == m.v[j,t] - 2*(m.rline[(j,i)]*m.P[(j,i),t] + m.xline[(j,i)]*m.Q[(j,i),t])
    )
    m.v_slack = pyo.Constraint(m.T, rule=lambda m,t: m.v[0,t] == m.v0bar)
    m.v_lo = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.v[i,t] >= m.vmin[i])
    m.v_hi = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.v[i,t] <= m.vmax[i])

    # Thermal box limits (note (j,i,t) signature)
    m.P_box_lo = pyo.Constraint(m.L, m.T, rule=lambda m, j, i, t: m.P[(j, i), t] >= -m.Smax[(j, i)])
    m.P_box_hi = pyo.Constraint(m.L, m.T, rule=lambda m, j, i, t: m.P[(j, i), t] <=  m.Smax[(j, i)])
    m.Q_box_lo = pyo.Constraint(m.L, m.T, rule=lambda m, j, i, t: m.Q[(j, i), t] >= -m.Smax[(j, i)])
    m.Q_box_hi = pyo.Constraint(m.L, m.T, rule=lambda m, j, i, t: m.Q[(j, i), t] <=  m.Smax[(j, i)])
else:
    # No network: tie Pslack to delivered fleet power so objective is meaningful
    m.subP = pyo.Constraint(m.T, rule=lambda m,t: m.Pslack[t] == m.Pagg[t])
    # No vars/cons for Q or v are needed

# ENS bounds (keep if you want ENS active even when USE_NETWORK=False)
if USE_ENS:
    m.ens_bound = pyo.Constraint(m.I, m.T, rule=lambda m,i,t: m.kappa[i,t] <= m.ell[i,t])

# =========================
# FOLLOWER — PRIMAL FEASIBILITY
# =========================
# 1) Per-origin conservation: sum_{tau>=t} X[t,tau] = Bbase[t]
m.origin_cons = pyo.Constraint(m.T, rule=lambda m,t:
    sum(m.X[(t, tau)] for tau in m.T if tau >= t) == m.Bbase[t]
)

# 2) Per-delivery capacity: sum_{t<=tau} X[t,tau] ≤ Cap[tau]
m.delivery_cap = pyo.Constraint(m.T, rule=lambda m,tau:
    sum(m.X[(t, tau)] for t in m.T if t <= tau) <= m.Cap[tau] * m.Delta
)

# 3) Flex caps: sum_{tau>=t+k} X[t,tau] ≤ Fflex[t,k]
def flex_cap_rule(m, t, k):
    min_tau = t + k
    if min_tau > max(m.T):
        return pyo.Constraint.Feasible
    expr = sum(m.X[(t, tau)] for tau in m.T if tau >= min_tau)
    return expr <= m.Fflex[t, k]
m.flex_cap = pyo.Constraint(m.T, m.K, rule=flex_cap_rule)

# =========================
# FOLLOWER — DUAL FEASIBILITY
# =========================
def c_cost(m, t, tau):
    return (m.pi[tau] + m.tau[tau]) * m.Delta + m.phi[(t, tau)]
m.dual_feas = pyo.Constraint(m.Pairs, rule=lambda m,t,tau:
    m.y1[t] + m.y2[tau] + sum(m.y3[t, k] for k in m.K if tau >= t + k) <= c_cost(m, t, tau)
)

# =========================
# STRONG DUALITY
# =========================
def lhs_cost(m):
    return sum(c_cost(m, t, tau) * m.X[(t, tau)] for (t, tau) in m.Pairs)
def rhs_dual(m):
    return - sum(m.y1[t] * m.Bbase[t] for t in m.T) \
           - sum(m.y2[tau] * m.Cap[tau] * m.Delta for tau in m.T) \
           - sum(m.y3[t,k] * m.Fflex[t,k] for t in m.T for k in m.K)
m.strong_duality = pyo.Constraint(rule=lambda m: lhs_cost(m) == rhs_dual(m))

# =========================
# SOLVE (robust guard)
# =========================
opt = pyo.SolverFactory(SOLVER_NAME)
try:
    opt.options.update(SOLVER_OPTS)
except Exception:
    pass

results = opt.solve(m, tee=True)

tc = results.solver.termination_condition
if tc != TerminationCondition.optimally_solved and tc != TerminationCondition.optimal:
    # Write model for inspection (check magnitudes & constraints)
    m.write("model.lp", io_options={"symbolic_solver_labels": True})
    print(f"\n[warn] Solve did not succeed: {tc}. Wrote model.lp for inspection.")
    # Note: Gurobi 12 Pyomo plugin does not reliably export IIS via options.
    # Use model.lp in the Gurobi shell if you need an IIS:  gurobi_cl ResultFile=iis.ilp model.lp
    raise SystemExit()

# =========================
# EXTRACT & PRINT RESULTS
# =========================
tariffs = {t: pyo.value(m.tau[t]) for t in m.T}
u_levels = {(t,n): pyo.value(m.u[t,n]) for t in m.T for n in m.N}
pagg    = {t: pyo.value(m.Pagg[t]) for t in m.T}
slackP  = {t: pyo.value(m.Pslack[t]) for t in m.T}
if USE_NETWORK:
    voltmin = {t: min(pyo.value(m.v[i,t]) for i in m.I) for t in m.T}
else:
    voltmin = {t: float("nan") for t in m.T}

print("\n== DUoS (€/kWh) ==")
for t in T:
    print(f"t={t:02d}: {tariffs[t]:.4f}")

print("\n== Fleet charging delivered (kW) ==")
for t in T:
    print(f"t={t:02d}: {pagg[t]:.2f}")

print("\n== Substation P (kW) ==")
for t in T:
    print(f"t={t:02d}: {slackP[t]:.2f}")

if USE_NETWORK:
    print("\n== Min bus voltage (p.u.) ==")
    for t in T:
        print(f"t={t:02d}: {voltmin[t]:.4f}")
else:
    print("\n(Network disabled; voltages not computed.)")
