from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12

ALIASES: Dict[str, List[str]] = {
    "sample": [
        "sample", "sampleid", "sample_id", "patient", "patientid",
        "patient_id", "Patient_ID", "case", "caseid", "case_id",
        "participant", "participantid", "participant_id", "pair", "pair_id",
        "patient_pair", "sample_pair",
    ],
    "dx_branch": [
        "DX_branch_ge50", "dxbranch", "dx_branch", "diagnosisbranch",
        "diagnosis_branch", "branchdx", "branch_dx",
    ],
    "rel_branch": [
        "REL_branch_ge50", "relbranch", "rel_branch", "relapsebranch",
        "relapse_branch", "branchrel", "branch_rel",
    ],
    "total_disp": [
        "disp_total_6d", "disptotal6d", "totaldisplacement",
        "total_displacement", "dxreldisplacement", "dx_rel_displacement",
        "disp6dtotal", "disp_total", "delta_total", "deltatotal",
        "totaldisp", "total_disp",
    ],
    "malignant_disp": [
        "disp_malignant_3d", "disp_malignant_6d", "dispmalignant3d",
        "dispmalignant6d", "malignantdisplacement", "malignant_displacement",
        "dxrelmalignantdisplacement", "dx_rel_malignant_displacement",
        "disp_malignant", "malignantdisp", "delta_malignant",
        "deltamalignant",
    ],
    "tme_disp": [
        "disp_tme_3d", "disp_tme_6d", "disptme3d", "disptme6d",
        "tmedisplacement", "tme_displacement", "dxreltmedisplacement",
        "dx_rel_tme_displacement", "disp_tme", "tmedisp", "delta_tme",
        "deltatme",
    ],
    "stability": [
        "dx_to_rel_switch", "group", "branchstability", "branch_stability",
        "stability", "stable_switching", "switchingstatus",
        "switching_status", "is_switching", "switching",
    ],
    "jump_tier": [
        "jumpcandidatetier", "jump_candidate_tier", "candidate_tier", "tier",
    ],
}


def canonicalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_column(df: pd.DataFrame, key: str, required: bool = True) -> Optional[str]:
    canon_to_original = {canonicalize(col): col for col in df.columns}
    candidates = [canonicalize(x) for x in ALIASES.get(key, [])]
    for candidate in candidates:
        if candidate in canon_to_original:
            return canon_to_original[candidate]
    for candidate in candidates:
        for canon_col, original_col in canon_to_original.items():
            if candidate in canon_col:
                return original_col
    if required:
        raise KeyError(
            f"Could not resolve required column for {key!r}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def standardize_stability(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    if "stable" in text or "same" in text or "no_switch" in text:
        return "Stable"
    if "switch" in text or "chang" in text or "different" in text:
        return "Switching"
    if text in {"0", "false", "no"}:
        return "Stable"
    if text in {"1", "true", "yes"}:
        return "Switching"
    try:
        return "Switching" if float(text) != 0 else "Stable"
    except ValueError:
        return None


def prepare_non_gaussian_dataframe(
    df: pd.DataFrame,
    subset_query: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    resolved = {
        "sample": resolve_column(df, "sample"),
        "dx_branch": resolve_column(df, "dx_branch"),
        "rel_branch": resolve_column(df, "rel_branch"),
        "total_disp": resolve_column(df, "total_disp"),
        "malignant_disp": resolve_column(df, "malignant_disp"),
        "tme_disp": resolve_column(df, "tme_disp"),
        "stability": resolve_column(df, "stability", required=False),
        "jump_tier": resolve_column(df, "jump_tier", required=False),
    }

    out = df.copy()
    out["sample_std"] = out[resolved["sample"]].astype(str).str.strip()
    out["dx_branch_std"] = out[resolved["dx_branch"]].astype(str).str.strip()
    out["rel_branch_std"] = out[resolved["rel_branch"]].astype(str).str.strip()
    out["total_disp_std"] = pd.to_numeric(out[resolved["total_disp"]], errors="coerce")
    out["malignant_disp_std"] = pd.to_numeric(out[resolved["malignant_disp"]], errors="coerce")
    out["tme_disp_std"] = pd.to_numeric(out[resolved["tme_disp"]], errors="coerce")
    out["transition_std"] = out["dx_branch_std"] + "→" + out["rel_branch_std"]

    stability_col = resolved["stability"]
    if stability_col is not None:
        out["stability_std"] = out[stability_col].map(standardize_stability)
    else:
        out["stability_std"] = None

    fallback = np.where(
        out["dx_branch_std"] == out["rel_branch_std"], "Stable", "Switching"
    )
    out["stability_std"] = out["stability_std"].fillna(pd.Series(fallback, index=out.index))

    tier_col = resolved["jump_tier"]
    out["jump_tier_std"] = (
        out[tier_col].astype(str).fillna("").str.strip() if tier_col else ""
    )

    if subset_query:
        out = out.query(subset_query).copy()

    required_cols = [
        "sample_std", "dx_branch_std", "rel_branch_std", "total_disp_std",
        "malignant_disp_std", "tme_disp_std", "stability_std",
    ]
    out = out.dropna(subset=required_cols).copy()
    out["stability_std"] = out["stability_std"].astype(str).str.strip().str.title()
    out = out[out["stability_std"].isin(["Stable", "Switching"])].copy()

    if out.empty:
        raise ValueError("No rows remain after non-Gaussian input preparation.")
    n_stable = int((out["stability_std"] == "Stable").sum())
    n_switching = int((out["stability_std"] == "Switching").sum())
    if n_stable < 2 or n_switching < 2:
        raise ValueError(
            "Need at least two Stable and two Switching rows; "
            f"found Stable={n_stable}, Switching={n_switching}."
        )
    return out.reset_index(drop=True), resolved


def bootstrap_median_diff(
    switching: np.ndarray,
    stable: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    switching = np.asarray(switching, dtype=float)
    stable = np.asarray(stable, dtype=float)
    point = float(np.median(switching) - np.median(stable))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = np.median(rng.choice(switching, len(switching), replace=True)) - np.median(
            rng.choice(stable, len(stable), replace=True)
        )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) * len(y) == 0:
        return np.nan
    gt = sum(int(np.sum(value > y)) for value in x)
    lt = sum(int(np.sum(value < y)) for value in x)
    return float((gt - lt) / (len(x) * len(y)))


def robust_z_scores(values: np.ndarray, baseline: np.ndarray) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        sd = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 1.0
        scale = sd if sd > EPS else 1.0
    return (values - median) / scale, median, scale


def gaussian_quantiles(n: int) -> np.ndarray:
    nd = NormalDist()
    probs = (np.arange(1, n + 1) - 0.5) / n
    return np.array([nd.inv_cdf(float(prob)) for prob in probs], dtype=float)


def gaussian_surprisal_from_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    survival = 0.5 * np.array(
        [math.erfc(float(value) / math.sqrt(2.0)) for value in z], dtype=float
    )
    return -np.log10(np.maximum(survival, 1e-300))


def build_non_gaussian_result_tables(
    prepared: pd.DataFrame,
    *,
    n_boot: int = 10_000,
    seed: int = 1,
    top_jump_candidates: int = 12,
) -> dict[str, object]:
    out = prepared.copy()
    stable_total = out.loc[out["stability_std"] == "Stable", "total_disp_std"].to_numpy(float)
    z_total, baseline_median, baseline_scale = robust_z_scores(
        out["total_disp_std"].to_numpy(float), stable_total
    )
    out["z_total_std"] = z_total
    out["jump_score_std"] = gaussian_surprisal_from_z(z_total)

    ranked = out.sort_values("total_disp_std").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    stable_q95 = float(np.quantile(stable_total, 0.95))
    ranked["stable_total_q95"] = stable_q95

    qq = out.sort_values("z_total_std").reset_index(drop=True)
    qq["qq_rank"] = np.arange(1, len(qq) + 1)
    qq["theoretical_q"] = gaussian_quantiles(len(qq))

    rng = np.random.default_rng(seed)
    effects = []
    for label, column in [
        ("Total", "total_disp_std"),
        ("Malignant", "malignant_disp_std"),
        ("TME", "tme_disp_std"),
    ]:
        stable = out.loc[out["stability_std"] == "Stable", column].to_numpy(float)
        switching = out.loc[out["stability_std"] == "Switching", column].to_numpy(float)
        effect, lo, hi = bootstrap_median_diff(switching, stable, n_boot, rng)
        effects.append(
            {
                "metric": label,
                "column": column,
                "median_diff_switching_minus_stable": effect,
                "ci_lo": lo,
                "ci_hi": hi,
                "cliffs_delta": cliffs_delta(switching, stable),
                "n_stable": len(stable),
                "n_switching": len(switching),
                "n_boot": n_boot,
                "seed": seed,
            }
        )
    effect_df = pd.DataFrame(effects)

    jumps = out.sort_values("jump_score_std", ascending=False).head(top_jump_candidates).copy()
    jumps["jump_rank"] = np.arange(1, len(jumps) + 1)

    metadata = {
        "n_rows": int(len(out)),
        "n_stable": int((out["stability_std"] == "Stable").sum()),
        "n_switching": int((out["stability_std"] == "Switching").sum()),
        "baseline_median": baseline_median,
        "baseline_scale": baseline_scale,
        "stable_total_q95": stable_q95,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "top_jump_candidates": int(top_jump_candidates),
    }
    return {
        "prepared": out,
        "ranked": ranked,
        "qq": qq,
        "effects": effect_df,
        "jumps": jumps.reset_index(drop=True),
        "metadata": metadata,
    }
