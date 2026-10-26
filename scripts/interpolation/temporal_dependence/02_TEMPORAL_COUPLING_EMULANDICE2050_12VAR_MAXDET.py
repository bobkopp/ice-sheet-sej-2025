#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final temporal coupling for the emulandice-2050 core chain.

Variables in the joint Gaussian copula are ordered:

    E2050, W2050, G2050,
    E2100, W2100, G2100,
    E2200, W2200, G2200,
    E2300, W2300, G2300

Hard constraints:
  * EAIS 2050/2100/2200/2300 temporal block from DP21 RCP2.6/4.5/8.5.
  * WAIS temporal block from the matching DP21 scenario.
  * GrIS temporal block from scenario-specific FittedISMIP correlations.
  * E-W dependence at 2050 from the paired emulandice workflow-1e ensemble.
  * Full E/W/G same-year dependence at 2100, 2200, and 2300 from the
    original SEJ year panels.
  * All unspecified cross-component/cross-time entries are maximum-logdet
    (maximum-entropy) completions.

2100 remains the fixed SEJ anchor. Complete 2200 and 2300 SEJ year panels are
matched as intact rows in three-dimensional E/W/G rank space. At 2050, paired
emulandice E/W draws remain paired; GrIS is rank-mapped independently because
workflow-1e does not provide a joint AIS-GrIS emulator sample pairing.

Created on: 2026-10-26
Author: Elizabeth H. Davis, Ph.D. (email: elizabeth.h.davis@rutgers.edu)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import norm, rankdata, spearmanr

import core_config as cfg
from core_imbie_helpers import load_imbie_common, reference_shifts_from_core_2005

MM_TO_CM = 0.1
WARMINGS = [2.0, 3.5, 5.0]
WARMING_CODES = {2.0: "2", 3.5: "35", 5.0: "5"}
WARMING_NAMES = {2.0: "LOW", 3.5: "INTERMEDIATE", 5.0: "HIGH"}
YEAR_CODES = {2100: "21", 2200: "22", 2300: "23"}
COMPONENTS = ("E", "W", "G")
YEARS = tuple(cfg.COUPLING_YEARS)

# Interleaved 12-variable positions: E,W,G at each time.
LABELS = [f"{c}{y}" for y in YEARS for c in COMPONENTS]
POS = {(c, y): LABELS.index(f"{c}{y}") for y in YEARS for c in COMPONENTS}
ANCHOR_POS = [POS[(c, 2100)] for c in COMPONENTS]
NONANCHOR_POS = [i for i in range(12) if i not in ANCHOR_POS]

OUTPUT_DIR = cfg.COUPLING_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COUPLED_FILE = cfg.COUPLED_SEJ_FILE
AUDIT_FILE = OUTPUT_DIR / "coupling_audit.csv"
CORR_FILE = OUTPUT_DIR / "target_vs_achieved_temporal_correlations.csv"
META_FILE = OUTPUT_DIR / "method_metadata.csv"

# ---------------------------------------------------------------------
# Metadata / ranking helpers
# ---------------------------------------------------------------------

WARMING_MAP = {"2": 2.0, "35": 3.5, "5": 5.0}
YEAR_MAP = {"21": 2100, "22": 2200, "23": 2300}


def parse_variable_name(column_name):
    m = re.fullmatch(r"(SLR|[EWG])([ADR]?)(2|35|5)_(21|22|23)", str(column_name))
    if m is None:
        return None
    ice, proc, warm, year = m.groups()
    return {
        "variable": column_name,
        "ice_sheet_code": ice,
        "process_code": proc if proc else "TOTAL",
        "warming": WARMING_MAP[warm],
        "year": YEAR_MAP[year],
    }


def build_metadata(df):
    rows = []
    for col in df.columns:
        p = parse_variable_name(col)
        if p is not None:
            rows.append(p)
    return pd.DataFrame(rows)


def find_total_column(meta, comp, warming, year):
    sub = meta.loc[
        (meta["ice_sheet_code"] == comp)
        & (meta["process_code"] == "TOTAL")
        & np.isclose(meta["warming"].astype(float), warming)
        & (meta["year"] == year)
    ]
    if len(sub) != 1:
        raise ValueError(f"Expected one TOTAL {comp} column for {warming}C/{year}; found {len(sub)}")
    return str(sub.iloc[0]["variable"])


def panel_columns(meta, warming, year):
    cols = meta.loc[
        np.isclose(meta["warming"].astype(float), warming)
        & meta["year"].eq(year),
        "variable",
    ].tolist()
    if not cols:
        raise ValueError(f"No SEJ panel columns for {warming}C/{year}")
    return cols


def rank01(x):
    x = np.asarray(x, float)
    return rankdata(x, method="average") / (len(x) + 1.0)


def zscore_rank(x):
    z = norm.ppf(rank01(x))
    z -= z.mean()
    sd = z.std(ddof=0)
    if sd <= 0:
        raise ValueError("Degenerate rank variable")
    return z / sd


def gaussian_rank_corr(x, y):
    return float(np.corrcoef(zscore_rank(x), zscore_rank(y))[0, 1])


def corr3(E, W, G):
    Z = np.column_stack([zscore_rank(E), zscore_rank(W), zscore_rank(G)])
    return np.corrcoef(Z, rowvar=False)


def deterministic_seed(*parts):
    text = "|".join(map(str, parts))
    h = hashlib.sha256(text.encode()).hexdigest()
    return (cfg.COUPLING_RNG_SEED + int(h[:8], 16)) % (2**32 - 1)

# ---------------------------------------------------------------------
# External temporal correlation inputs
# ---------------------------------------------------------------------


def load_dp21_component_matrix(component: str, warming: float):
    scenario = cfg.DP21_SCENARIOS[warming]
    path = cfg.DP21_ROOT / f"dp21_{component.lower()}_{scenario}.nc"
    if not path.is_file():
        raise FileNotFoundError(path)
    with Dataset(path, "r") as ds:
        years = np.asarray(ds.variables["years"][:]).astype(int)
        samps = np.ma.filled(ds.variables["samps"][:], np.nan).astype(float)
    idx = []
    for year in YEARS:
        hit = np.where(years == year)[0]
        if len(hit) != 1:
            raise ValueError(f"{path.name}: year {year} not found exactly once")
        idx.append(int(hit[0]))
    selected = samps[idx, :].T
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if len(selected) < 2:
        raise ValueError(f"{path.name}: fewer than two complete members")
    return np.corrcoef(selected, rowvar=False)


def load_dp21_total_ais_matrix(warming: float):
    """DP21 Total-AIS temporal matrix, derived member-by-member as EAIS+WAIS.

    This matrix is diagnostic only. It is NOT imposed as a fourth temporal
    copula constraint, because Total AIS is derived from the separately
    constrained EAIS and WAIS components.
    """
    scenario = cfg.DP21_SCENARIOS[warming]

    def read_component(component):
        path = cfg.DP21_ROOT / f"dp21_{component.lower()}_{scenario}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Dataset(path, "r") as ds:
            years = np.asarray(ds.variables["years"][:]).astype(int)
            samples = np.asarray(ds.variables["samples"][:])
            samps = np.ma.filled(ds.variables["samps"][:], np.nan).astype(float)
        return path, years, samples, samps

    epath, eyears, esamples, E = read_component("EAIS")
    wpath, wyears, wsamples, W = read_component("WAIS")

    if not np.array_equal(eyears, wyears):
        raise ValueError(f"DP21 EAIS/WAIS years do not align for {scenario}")
    if not np.array_equal(esamples, wsamples):
        raise ValueError(f"DP21 EAIS/WAIS sample IDs do not align for {scenario}")
    if E.shape != W.shape:
        raise ValueError(f"DP21 EAIS/WAIS shapes differ for {scenario}: {E.shape} vs {W.shape}")

    idx = []
    for year in YEARS:
        hit = np.where(eyears == year)[0]
        if len(hit) != 1:
            raise ValueError(f"{epath.name}/{wpath.name}: year {year} not found exactly once")
        idx.append(int(hit[0]))

    total = (E[idx, :] + W[idx, :]).T
    total = total[np.all(np.isfinite(total), axis=1)]
    if len(total) < 2:
        raise ValueError(f"DP21 Total AIS {scenario}: fewer than two complete members")
    return np.corrcoef(total, rowvar=False)


def load_corr_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    wanted = [str(y) for y in YEARS]
    df.columns = [str(c).strip() for c in df.columns]
    df.index = df.index.map(lambda x: str(x).strip())
    if all(c in df.columns for c in wanted) and all(r in df.index for r in wanted):
        arr = df.loc[wanted, wanted].to_numpy(float)
    elif all(c in df.columns for c in wanted):
        arr = df.loc[:, wanted].to_numpy(float)
    else:
        numeric = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=1, how="all")
        if numeric.shape[1] == len(YEARS) + 1:
            numeric = numeric.iloc[:, 1:]
        arr = numeric.to_numpy(float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}; got {arr.shape}")
    return arr


def regularize_corr(R):
    R = np.asarray(R, float)
    R = 0.5 * (R + R.T)
    vals, vecs = np.linalg.eigh(R)
    before = float(vals.min())
    vals = np.maximum(vals, cfg.CORRELATION_EIGEN_FLOOR)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    return out, before, float(np.linalg.eigvalsh(out).min()), float(np.max(np.abs(out - R)))

# ---------------------------------------------------------------------
# emulandice input
# ---------------------------------------------------------------------


def _read_emu(path: Path, year=2050):
    if not path.is_file():
        raise FileNotFoundError(path)
    with Dataset(path, "r") as ds:
        years = np.asarray(ds.variables["years"][:]).astype(int)
        samples = np.asarray(ds.variables["samples"][:])
        v = ds.variables["sea_level_change"]
        x = np.ma.filled(v[:], np.nan).astype(float)
        units = str(getattr(v, "units", "")).strip().lower()
        baseyear = int(getattr(ds, "baseyear", 2005))
    if x.ndim == 3:
        if x.shape[2] != 1:
            raise ValueError(f"Expected one global location in {path}")
        x = x[:, :, 0]
    hit = np.where(years == year)[0]
    if len(hit) != 1:
        raise ValueError(f"{path.name}: {year} not found exactly once")
    out = x[:, hit[0]]
    if units == "mm":
        out = out / 10.0
    elif units == "m":
        out = out * 100.0
    elif units not in ("cm", "centimeter", "centimeters"):
        raise ValueError(f"Unrecognized sea-level units {units!r} in {path}")
    if baseyear != 2005:
        raise ValueError(f"Expected emulandice baseyear=2005 in {path}; found {baseyear}")
    return samples, out


def find_gris_file(exp_dir: Path):
    d = exp_dir / "output" / "emulandice-gris"
    candidates = [
        d / "output-gslr-gris.nc",
        d / "output-gslr-gis.nc",
        d / "output-gslr-greenland.nc",
    ]
    for p in candidates:
        if p.is_file():
            return p
    hits = sorted(d.glob("output-gslr-*.nc")) if d.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"Could not uniquely identify GrIS file in {d}")


def load_emulandice(warming):
    exp = cfg.EMULANDICE_ROOT / cfg.EMULANDICE_EXPERIMENTS[warming]
    se, E = _read_emu(exp / "output" / "emulandice-ais" / "output-gslr-eais.nc")
    sw, W = _read_emu(exp / "output" / "emulandice-ais" / "output-gslr-wais.nc")
    sg, G = _read_emu(find_gris_file(exp))
    if not np.array_equal(se, sw):
        raise ValueError(f"{exp.name}: EAIS/WAIS sample IDs are not aligned")
    return {"E": E, "W": W, "G": G, "EW_samples": se, "G_samples": sg}

# ---------------------------------------------------------------------
# 12D maximum-determinant completion
# ---------------------------------------------------------------------


def build_fixed_target(Ecorr, Wcorr, Gcorr, future_sameyear, rho_EW_2050):
    R = np.full((12, 12), np.nan, float)
    np.fill_diagonal(R, 1.0)

    comp_corrs = {"E": Ecorr, "W": Wcorr, "G": Gcorr}
    for comp, C in comp_corrs.items():
        for i, y1 in enumerate(YEARS):
            for j, y2 in enumerate(YEARS):
                R[POS[(comp, y1)], POS[(comp, y2)]] = C[i, j]

    # emulandice contemporaneous Antarctic dependence at 2050.
    i, j = POS[("E", 2050)], POS[("W", 2050)]
    R[i, j] = R[j, i] = rho_EW_2050

    # Original SEJ contemporaneous dependence at elicited years.
    for year, C in future_sameyear.items():
        for a in range(3):
            for b in range(a + 1, 3):
                i = POS[(COMPONENTS[a], year)]
                j = POS[(COMPONENTS[b], year)]
                R[i, j] = R[j, i] = C[a, b]

    return R


def maxdet_complete(fixed_R, empirical_start=None):
    fixed = np.isfinite(fixed_R)
    unknown_pairs = [
        (i, j)
        for i in range(12)
        for j in range(i + 1, 12)
        if not fixed[i, j]
    ]

    def matrix(x):
        R = fixed_R.copy()
        for val, (i, j) in zip(x, unknown_pairs):
            R[i, j] = R[j, i] = val
        return R

    if empirical_start is None:
        x_emp = np.zeros(len(unknown_pairs), float)
    else:
        x_emp = np.array([empirical_start[i, j] for i, j in unknown_pairs], float)
        x_emp = np.nan_to_num(x_emp, nan=0.0)
        x_emp = np.clip(x_emp, -0.95, 0.95)

    def mineig(x):
        return float(np.linalg.eigvalsh(matrix(x)).min())

    starts = [x_emp, 0.75 * x_emp, 0.5 * x_emp, np.zeros_like(x_emp)]
    best_x = None
    best_eig = -np.inf
    for start in starts:
        res = minimize(
            lambda x: -mineig(x),
            start,
            method="SLSQP",
            bounds=[(-0.999, 0.999)] * len(start),
            options={"ftol": 1e-10, "maxiter": 5000, "disp": False},
        )
        e = mineig(res.x)
        if e > best_eig:
            best_eig, best_x = e, res.x.copy()

    if best_x is None or best_eig <= cfg.MAXDET_PD_EPS:
        raise RuntimeError(f"Could not find feasible 12D completion; best min eig={best_eig:.3e}")

    def objective_grad(x):
        R = matrix(x)
        eig = np.linalg.eigvalsh(R)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0 or eig[0] <= cfg.MAXDET_PD_EPS:
            return 1e12, np.zeros_like(x)
        P = np.linalg.inv(R)
        grad = np.array([-2.0 * P[i, j] for i, j in unknown_pairs])
        return -float(logdet), grad

    res = minimize(
        objective_grad,
        best_x,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-0.999, 0.999)] * len(best_x),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": cfg.MAXDET_MAXITER, "maxls": 100},
    )
    R = matrix(res.x)
    eig = np.linalg.eigvalsh(R)
    if eig[0] <= cfg.MAXDET_PD_EPS:
        raise RuntimeError(f"12D maxdet completion not PD; min eig={eig[0]:.3e}")
    return R, unknown_pairs, bool(res.success), str(res.message), float(eig[0])


def conditional_nonanchor(R, anchor_z, scenario_name):
    naa = np.array(ANCHOR_POS)
    nnn = np.array(NONANCHOR_POS)
    Raa = R[np.ix_(naa, naa)]
    Rna = R[np.ix_(nnn, naa)]
    Rnn = R[np.ix_(nnn, nnn)]
    inv = np.linalg.inv(Raa)
    cond = Rnn - Rna @ inv @ Rna.T
    cond = 0.5 * (cond + cond.T)
    vals, vecs = np.linalg.eigh(cond)
    if vals.min() <= 1e-10:
        vals = np.maximum(vals, 1e-10)
        cond = vecs @ np.diag(vals) @ vecs.T
        cond = 0.5 * (cond + cond.T)
    mu = anchor_z @ inv @ Rna.T

    rng = np.random.default_rng(deterministic_seed("12d_conditional", scenario_name))
    eps = rng.standard_normal((len(anchor_z), len(nnn)))
    # Orthogonalize finite-sample innovations against intercept + anchors.
    Q = np.column_stack([np.ones(len(anchor_z)), anchor_z])
    beta, *_ = np.linalg.lstsq(Q, eps, rcond=None)
    eps = eps - Q @ beta
    cov = eps.T @ eps / len(eps)
    L0 = np.linalg.cholesky(cov)
    white = eps @ np.linalg.inv(L0.T)
    L = np.linalg.cholesky(cond)
    return mu + white @ L.T

# ---------------------------------------------------------------------
# Matching intact source rows
# ---------------------------------------------------------------------


def normalized_rank_int(x, bits=14):
    maxv = (1 << bits) - 1
    return np.clip(np.rint(rank01(x) * maxv), 0, maxv).astype(np.uint64)


def morton_order_3d(x, y, z, bits=14):
    xi, yi, zi = [normalized_rank_int(v, bits) for v in (x, y, z)]
    code = np.zeros(len(xi), dtype=np.uint64)
    for b in range(bits):
        code |= ((xi >> b) & 1) << (3 * b)
        code |= ((yi >> b) & 1) << (3 * b + 1)
        code |= ((zi >> b) & 1) << (3 * b + 2)
    return np.lexsort((np.arange(len(code)), code))


def morton_order_2d(x, y, bits=16):
    xi, yi = [normalized_rank_int(v, bits) for v in (x, y)]
    code = np.zeros(len(xi), dtype=np.uint64)
    for b in range(bits):
        code |= ((xi >> b) & 1) << (2 * b)
        code |= ((yi >> b) & 1) << (2 * b + 1)
    return np.lexsort((np.arange(len(code)), code))


def local_optimize_ids(target_cols, source_cols, ids, order, offsets=None):
    ids = np.asarray(ids, int).copy()
    offsets = cfg.MATCH_LOCAL_OFFSETS if offsets is None else offsets
    Zt = np.column_stack([zscore_rank(v) for v in target_cols])
    Zs = np.column_stack([zscore_rank(v) for v in source_cols])
    for offset in offsets:
        starts = ([0] if offset > 0 else []) + list(range(offset, len(order), cfg.MATCH_BLOCK_SIZE))
        for start in starts:
            end = min(offset if (start == 0 and offset > 0) else start + cfg.MATCH_BLOCK_SIZE, len(order))
            if end - start <= 1:
                continue
            rows = order[start:end]
            src = ids[rows].copy()
            cost = np.sum((Zt[rows, None, :] - Zs[src][None, :, :]) ** 2, axis=2)
            ri, ci = linear_sum_assignment(cost)
            new = ids[rows].copy()
            new[ri] = src[ci]
            ids[rows] = new
    return ids


def match_sej_panel_3d(targetE, targetW, targetG, sourceE, sourceW, sourceG):
    to = morton_order_3d(targetE, targetW, targetG)
    so = morton_order_3d(sourceE, sourceW, sourceG)
    if len(to) != len(so):
        raise ValueError("SEJ target/source sizes differ")
    ids = np.empty(len(to), int)
    ids[to] = so
    return local_optimize_ids(
        [targetE, targetW, targetG], [sourceE, sourceW, sourceG], ids, to
    )


def expand_source_order(source_order, n_target):
    reps, rem = divmod(n_target, len(source_order))
    idx = np.repeat(source_order, reps)
    if rem:
        idx = np.concatenate([idx, source_order[np.linspace(0, len(source_order)-1, rem, dtype=int)]])
    return idx


def match_emulandice_EW(targetE, targetW, sourceE, sourceW):
    to = morton_order_2d(targetE, targetW)
    so = morton_order_2d(sourceE, sourceW)
    expanded = expand_source_order(so, len(to))
    ids = np.empty(len(to), int)
    ids[to] = expanded
    return local_optimize_ids([targetE, targetW], [sourceE, sourceW], ids, to)


def rank_map_empirical(target_order, source_values):
    """Assign the empirical source marginal exactly (up to unavoidable remainder)."""
    source = np.sort(np.asarray(source_values, float))
    target_order = np.asarray(target_order, float)
    n = len(target_order)
    reps, rem = divmod(n, len(source))
    expanded = np.repeat(source, reps)
    if rem:
        expanded = np.concatenate([expanded, source[np.linspace(0, len(source)-1, rem, dtype=int)]])
    expanded = np.sort(expanded)
    order = np.argsort(target_order, kind="mergesort")
    out = np.empty(n, float)
    out[order] = expanded
    return out

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Final selected 02I-B balanced 3D Hungarian matching and guarded refinement.
# Operates directly from the source SEJ and emulandice data. No legacy
# coupled CSV, precomputed permutations, or audit benchmark is required.
# ---------------------------------------------------------------------
PAIR_PRIORITY = {(2050,2100):4.,(2100,2200):4.,(2200,2300):4.,
                 (2100,2300):2.,(2050,2200):.35,(2050,2300):.15}
BLOCKS = {y: np.array([POS[(c,y)] for c in COMPONENTS]) for y in YEARS}

def final_3d_hungarian(targets, candidates):
    target_key = targets.sum(axis=1)
    source_key = candidates.sum(axis=1)
    torder = np.argsort(target_key, kind="stable")
    sorder = np.argsort(source_key, kind="stable")
    ids = np.full(len(targets),-1,dtype=int)
    for start in range(0,len(ids),cfg.STAGE02I_BLOCK_SIZE):
        t=torder[start:start+cfg.STAGE02I_BLOCK_SIZE]
        s=sorder[start:start+cfg.STAGE02I_BLOCK_SIZE]
        a=targets[t];b=candidates[s]
        cost=np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
        row,col=linear_sum_assignment(cost)
        ids[t[row]]=s[col]
    assert np.array_equal(np.sort(ids),np.arange(len(ids)))
    return ids

def target_nonanchor_given_2050_2100(R,z,rng):
    fixed=np.arange(6); future=np.arange(6,12)
    Rxx=R[np.ix_(fixed,fixed)]; Rfx=R[np.ix_(future,fixed)];Rff=R[np.ix_(future,future)]
    coeff=np.linalg.solve(Rxx,Rfx.T).T
    cov=Rff-coeff@Rfx.T;cov=(cov+cov.T)/2
    eig,v=np.linalg.eigh(cov)
    if eig.min() < -1e-7: raise ValueError(f"Non-PSD conditional matrix {eig.min()}")
    cov=(v*np.maximum(eig,0))@v.T
    noise=rng.standard_normal((len(z),6))@np.linalg.cholesky(cov+np.eye(6)*1e-11).T
    vals=z[:,:6]@coeff.T+noise
    return np.column_stack([zscore_rank(vals[:,i]) for i in range(6)])

def priority_weights():
    W=np.zeros((12,12))
    for i in range(12):
        for j in range(i+1,12):
            yi,yj=YEARS[i//3],YEARS[j//3]
            if yi!=yj:
                W[i,j]=W[j,i]=(PAIR_PRIORITY[(yi,yj)] if i%3==j%3 else .10)
    return W

def loss(C,R,W): return float(np.sum(np.triu(W*(C-R)**2,1)))

def adjacent_ceiling(C,R):
    ceiling={}
    for a,b in ((2050,2100),(2100,2200),(2200,2300)):
        for c in COMPONENTS:
            i,j=POS[(c,a)],POS[(c,b)]
            ceiling[i,j]=float(abs(C[i,j]-R[i,j])+cfg.STAGE02I_ADJACENT_ERROR_ALLOWANCE)
    return ceiling

def refine(z,p,R,rng):
    n=len(z);C=z.T@z/n; W=priority_weights(); hist=[]
    limits=adjacent_ceiling(C,R)
    for round_n in range(1,cfg.STAGE02I_REFINE_ROUNDS+1):
        accepted=0
        years=[2200,2300,2100]
        if round_n%2==0:years.reverse()
        for year in years:
            block=BLOCKS[year]
            others=np.array([i for i in range(12) if i not in block])
            wb=W[np.ix_(block,others)]
            residual=C[np.ix_(block,others)]-R[np.ix_(block,others)]
            total=cfg.STAGE02I_PROPOSALS_PER_YEAR_ROUND
            for offset in range(0,total,cfg.STAGE02I_SWAP_BATCH):
                m=min(cfg.STAGE02I_SWAP_BATCH,total-offset,n//2)
                ij=rng.choice(n,size=2*m,replace=False).reshape(m,2)
                a,b=ij[:,0],ij[:,1]
                db=z[b[:,None],block]-z[a[:,None],block]
                do=z[a[:,None],others]-z[b[:,None],others]
                delta=db[:,:,None]*do[:,None,:]/n
                gain=np.sum(wb[None,:,:]*(2*residual[None,:,:]*delta+delta**2),axis=(1,2))
                for k in np.argsort(gain)[:20]:
                    if gain[k]>=-1e-13: break
                    Cnew=C.copy(); Cnew[np.ix_(block,others)]+=delta[k]
                    Cnew[np.ix_(others,block)]+=delta[k].T
                    if any(abs(Cnew[i,j]-R[i,j])>lim+1e-12 for (i,j),lim in limits.items()):continue
                    i,j=map(int,ij[k]); z[np.ix_([i,j],block)]=z[np.ix_([j,i],block)].copy()
                    p[year][[i,j]]=p[year][[j,i]]
                    C=Cnew;residual+=delta[k];accepted+=1;break
        errors=[abs(C[POS[(c,a)],POS[(c,b)]]-R[POS[(c,a)],POS[(c,b)]])
                for (a,b) in ((2050,2100),(2100,2200),(2200,2300)) for c in COMPONENTS]
        hist.append(dict(round=round_n,accepted_swaps=accepted,
                         priority_adjacent_mae=float(np.mean(errors)),weighted_loss=loss(C,R,W)))
        print(f" round {round_n:02d}: accepted={accepted}, adjacent MAE={np.mean(errors):.5f}",flush=True)
        if accepted==0:break
    return hist

print("=" * 96)
print("FINAL 12-VARIABLE TEMPORAL COUPLING: EMULANDICE 2050 + DP21/FITTEDISMIP")
print("=" * 96)

mc = pd.read_csv(cfg.SEJ_DATA_FILE)
meta = build_metadata(mc)
n = len(mc)
trajectory_ids = np.arange(n, dtype=int)
raw = load_imbie_common(cfg.PROJECT_DIR)
shifts = reference_shifts_from_core_2005(raw)

coupled = mc.copy()
coupled["coupled_trajectory_id"] = trajectory_ids

audit_rows = []
corr_rows = []
convergence_rows = []
source_rows = []

for warming in WARMINGS:
    name = WARMING_NAMES[warming]
    code = WARMING_CODES[warming]
    print(f"\n--- {name} / {warming:g} C ---")

    # External component temporal targets.
    Ecorr_raw = load_dp21_component_matrix("EAIS", warming)
    Wcorr_raw = load_dp21_component_matrix("WAIS", warming)
    # Diagnostic only: derived DP21 Total AIS = paired EAIS + WAIS.
    AIScorr_raw = load_dp21_total_ais_matrix(warming)
    Gcorr_raw = load_corr_csv(cfg.GRIS_CORRELATION_FILES[warming])
    Ecorr, Emin0, Emin, Ediff = regularize_corr(Ecorr_raw)
    Wcorr, Wmin0, Wmin, Wdiff = regularize_corr(Wcorr_raw)
    Gcorr, Gmin0, Gmin, Gdiff = regularize_corr(Gcorr_raw)

    # Original SEJ total-component arrays at elicited years.
    panels = {}
    sameyear = {}
    for year in cfg.SEJ_YEARS:
        Ecol = find_total_column(meta, "E", warming, year)
        Wcol = find_total_column(meta, "W", warming, year)
        Gcol = find_total_column(meta, "G", warming, year)
        panels[year] = {
            "E": mc[Ecol].to_numpy(float),
            "W": mc[Wcol].to_numpy(float),
            "G": mc[Gcol].to_numpy(float),
            "columns": panel_columns(meta, warming, year),
        }
        sameyear[year] = corr3(panels[year]["E"], panels[year]["W"], panels[year]["G"])

    emu = load_emulandice(warming)
    rhoEW50 = gaussian_rank_corr(emu["E"], emu["W"])

    fixed = build_fixed_target(Ecorr, Wcorr, Gcorr, sameyear, rhoEW50)

    # Empirical start from original SEJ future joint ranks; unknown 2050 cross-lags start at zero.
    empirical = np.zeros((12, 12), float)
    np.fill_diagonal(empirical, 1.0)
    future_cols = []
    future_pos = []
    for year in cfg.SEJ_YEARS:
        for comp in COMPONENTS:
            future_cols.append(panels[year][comp])
            future_pos.append(POS[(comp, year)])
    Zfuture = np.column_stack([zscore_rank(x) for x in future_cols])
    Rfuture = np.corrcoef(Zfuture, rowvar=False)
    for a, ia in enumerate(future_pos):
        for b, ib in enumerate(future_pos):
            empirical[ia, ib] = Rfuture[a, b]

    R, unknown_pairs, opt_success, opt_message, min_eig = maxdet_complete(fixed, empirical)

    # Fixed 2100 anchor in original row order.
    anchor_z = np.column_stack([
        zscore_rank(panels[2100]["E"]),
        zscore_rank(panels[2100]["W"]),
        zscore_rank(panels[2100]["G"]),
    ])
    Znon = conditional_nonanchor(R, anchor_z, name)
    Zfull = np.empty((n, 12), float)
    Zfull[:, ANCHOR_POS] = anchor_z
    Zfull[:, NONANCHOR_POS] = Znon

    # Initial emulandice 2050: retain paired E/W, independently rank-map G.
    pair_ids = match_emulandice_EW(
        Zfull[:, POS[("E", 2050)]],Zfull[:, POS[("W", 2050)]],emu["E"],emu["W"])
    E50=emu["E"][pair_ids];W50=emu["W"][pair_ids]
    G50=rank_map_empirical(Zfull[:,POS[("G",2050)]],emu["G"])
    for c,a in (("E",E50),("W",W50),("G",G50)):
        coupled[f"EMU2050_{c}{code}_cm"]=a
    coupled[f"EMU2050_AIS{code}_cm"]=E50+W50
    coupled[f"EMU2050_SLR{code}_cm"]=E50+W50+G50
    coupled[f"EMU2050_PAIR_ID_{code}"]=pair_ids

    # Joint ranks use the fixed 2050 panel and original 2100 anchor.
    z=np.column_stack([zscore_rank(a) for y in YEARS for a in (
        (E50,W50,G50) if y==2050 else tuple(panels[y][c] for c in COMPONENTS))])
    desired=target_nonanchor_given_2050_2100(
        R,z,np.random.default_rng(cfg.STAGE02I_SEED+int(warming*100)))
    assignments={2100:trajectory_ids.copy()}
    for year,j in ((2200,0),(2300,3)):
        original_z=np.column_stack([zscore_rank(panels[year][c]) for c in COMPONENTS])
        assignments[year]=final_3d_hungarian(desired[:,j:j+3],original_z)
        z[:,BLOCKS[year]]=original_z[assignments[year]]

    convergence=refine(z,assignments,R,np.random.default_rng(cfg.STAGE02I_SEED+10000+int(warming*100)))
    for row in convergence:convergence_rows.append(dict(warming=warming,**row))
    for year in cfg.SEJ_YEARS:
        ix=assignments[year]
        if not np.array_equal(np.sort(ix),trajectory_ids):raise RuntimeError("Nonbijective SEJ panel")
        cols=panels[year]["columns"]
        coupled.loc[:,cols]=mc.iloc[ix][cols].to_numpy()
        for col in cols:
            if not np.array_equal(np.sort(coupled[col].to_numpy()),np.sort(mc[col].to_numpy())):
                raise RuntimeError("SEJ marginal changed: "+col)
    source_rows.append(pd.DataFrame(dict(warming=warming,trajectory_row=trajectory_ids,
        **{f"source_row_{y}":assignments[y] for y in cfg.SEJ_YEARS})))

    # Target-vs-achieved component temporal correlations.
    E2100 = panels[2100]["E"][assignments[2100]]
    W2100 = panels[2100]["W"][assignments[2100]]
    E2200 = panels[2200]["E"][assignments[2200]]
    W2200 = panels[2200]["W"][assignments[2200]]
    E2300 = panels[2300]["E"][assignments[2300]]
    W2300 = panels[2300]["W"][assignments[2300]]

    achieved_values = {
        "E": [E50, E2100, E2200, E2300],
        "W": [W50, W2100, W2200, W2300],
        "G": [G50, panels[2100]["G"][assignments[2100]], panels[2200]["G"][assignments[2200]], panels[2300]["G"][assignments[2300]]],
        # Derived diagnostic only — never a hard copula constraint.
        "AIS": [E50 + W50, E2100 + W2100, E2200 + W2200, E2300 + W2300],
    }
    targets = {"E": Ecorr_raw, "W": Wcorr_raw, "G": Gcorr_raw, "AIS": AIScorr_raw}
    errs = {}
    for comp in ("E", "W", "G", "AIS"):
        Z = np.column_stack([zscore_rank(x) for x in achieved_values[comp]])
        A = np.corrcoef(Z, rowvar=False)
        diff = []
        for i, j in combinations(range(4), 2):
            d = float(A[i, j] - targets[comp][i, j])
            diff.append(abs(d))
            corr_rows.append({
                "warming": warming, "warming_name": name, "component": comp,
                "year_1": YEARS[i], "year_2": YEARS[j],
                "target_correlation": float(targets[comp][i, j]),
                "achieved_gaussianized_rank_correlation": float(A[i, j]),
                "difference": d,
            })
        errs[comp] = (float(np.mean(diff)), float(np.max(diff)))

    # Exact preservation of original SEJ future marginals.
    max_marginal_diff = 0.0
    for year in (2100, 2200, 2300):
        for col in panels[year]["columns"]:
            d = float(np.max(np.abs(np.sort(mc[col].to_numpy(float)) - np.sort(coupled[col].to_numpy(float)))))
            max_marginal_diff = max(max_marginal_diff, d)

    source_ew = gaussian_rank_corr(emu["E"], emu["W"])
    achieved_ew = gaussian_rank_corr(E50, W50)
    audit_rows.append({
        "warming": warming,
        "warming_name": name,
        "maxdet_optimizer_success": opt_success,
        "maxdet_optimizer_message": opt_message,
        "joint_target_min_eigenvalue": min_eig,
        "DP21_E_regularization_max_abs_change": Ediff,
        "DP21_W_regularization_max_abs_change": Wdiff,
        "FittedISMIP_G_regularization_max_abs_change": Gdiff,
        "emulandice_EW_2050_target": source_ew,
        "emulandice_EW_2050_achieved": achieved_ew,
        "E_temporal_mean_abs_error": errs["E"][0],
        "E_temporal_max_abs_error": errs["E"][1],
        "W_temporal_mean_abs_error": errs["W"][0],
        "W_temporal_max_abs_error": errs["W"][1],
        "G_temporal_mean_abs_error": errs["G"][0],
        "G_temporal_max_abs_error": errs["G"][1],
        "derived_Total_AIS_temporal_mean_abs_error": errs["AIS"][0],
        "derived_Total_AIS_temporal_max_abs_error": errs["AIS"][1],
        "max_SEJ_future_marginal_difference": max_marginal_diff,
    })

coupled.to_csv(COUPLED_FILE, index=False)
pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False)
pd.DataFrame(corr_rows).to_csv(CORR_FILE, index=False)
pd.DataFrame([{
    "method": "02I-B balanced 3D Hungarian 5000 + guarded weighted swaps 30",
    "variables": ";".join(LABELS),
    "anchor": "2050 fixed; complete SEJ 2100/2200/2300 panels permutable",
    "EAIS_temporal_source": "DP21 raw scenario-specific component ensemble",
    "WAIS_temporal_source": "DP21 raw scenario-specific component ensemble",
    "GrIS_temporal_source": "scenario-specific FittedISMIP correlation matrix",
    "2050_EW_dependence_source": "paired emulandice workflow-1e E/W ensemble",
    "future_same_year_dependence": "original SEJ E/W/G year panels",
    "unspecified_dependence": "maximum log-determinant completion",
    "2050_marginals": "emulandice E/W/G at 2050",
    "2050_EW_assignment": "pair-preserving Morton initialization + local Hungarian refinement",
    "2050_GrIS_assignment": "empirical rank map",
    "SEJ_2200_2300_assignment": "intact 3-D E/W/G panel matching",
    "Total_AIS_temporal_role": "derived E+W; paired DP21 Total AIS used only as target-vs-achieved diagnostic",
}]).to_csv(META_FILE, index=False)

print("\nSaved:")
print(" ", COUPLED_FILE)
print(" ", AUDIT_FILE)
print(" ", CORR_FILE)
print("Done.")

pd.DataFrame(convergence_rows).to_csv(OUTPUT_DIR / "stage02I_B_convergence.csv", index=False)
pd.concat(source_rows,ignore_index=True).to_csv(OUTPUT_DIR / "stage02I_B_source_row_mapping.csv",index=False)
print("Final 02I-B Stage 02 completed from original input data; no legacy permutations used.")
