from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd


def require_columns(path: Path, columns: list[str]) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pediatric_leukemia_stage4.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)
    output_dir = Path(config["paths"]["output_dir"]).expanduser()

    checks = [
        ("Figure 3 ranked table", output_dir / "figure3_ranked_displacement.csv",
         ["sample_std", "rank", "total_disp_std", "stable_total_q95"]),
        ("Figure 3 QQ table", output_dir / "figure3_qq_table.csv",
         ["sample_std", "theoretical_q", "z_total_std"]),
        ("Figure 3 effect table", output_dir / "figure3_effect_sizes.csv",
         ["metric", "median_diff_switching_minus_stable", "ci_lo", "ci_hi", "cliffs_delta"]),
        ("Figure 3 jump table", output_dir / "figure3_jump_candidates.csv",
         ["sample_std", "jump_rank", "jump_score_std"]),
    ]
    print("Stage 4 computation/graphics separation validation")
    print("=" * 72)
    for label, path, columns in checks:
        rows = require_columns(path, columns)
        print(f"[PASS] {label}: {rows} rows")

    metadata_path = output_dir / "figure3_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for key in ["baseline_median", "baseline_scale", "stable_total_q95", "n_boot", "seed"]:
        if key not in metadata:
            raise ValueError(f"Figure 3 metadata missing {key}")
    print("[PASS] Figure 3 metadata records baseline and bootstrap choices")

    figure3 = output_dir / f"{config['figure3_non_gaussian']['output_name']}.png"
    figure4 = output_dir / f"{config['figure4_model_comparison']['output_name']}.png"
    for path in [figure3, figure4]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        print(f"[PASS] Rendered figure: {path.name}")

    plot_scripts = [
        Path("application/pediatric_leukemia/plot_figure3_non_gaussian.py"),
        Path("application/pediatric_leukemia/plot_figure4_model_comparison.py"),
        Path("src/oulb/non_gaussian_plotting.py"),
        Path("src/oulb/model_comparison_plotting.py"),
    ]
    forbidden = ["scipy.optimize", "minimize(", "bootstrap_median_diff(", "fit_model("]
    for script in plot_scripts:
        text = script.read_text()
        hits = [token for token in forbidden if token in text]
        if hits:
            raise ValueError(f"{script} contains computation-only tokens: {hits}")
    print("[PASS] Plotting layer contains no model fitting, bootstrap, or optimizer calls")
    print("\nStage 4 completed: all figures are table-backed.")


if __name__ == "__main__":
    main()
