from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oulb.dynamics import (  # noqa: E402
    add_effective_dynamic_parameters,
    collapse_replicated_sample_scores,
    compute_reference_attractor,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "pediatric_leukemia_stage2.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Stage 2 TOML configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Stage 2 output files.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def require_input(path: Path) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_output(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\nRe-run with --overwrite to replace it."
        )
    return path


def load_main_patients(path: Path) -> set[str]:
    patients: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            patient = line.strip()
            if patient:
                patients.append(patient)
    return set(patients)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config["dynamic_parameters"]

    input_h5ad = require_input(Path(settings["input_h5ad"]))
    main_patients_path = require_input(Path(settings["main_patients_txt"]))
    output_csv = prepare_output(
        Path(settings["output_csv"]), overwrite=args.overwrite
    )
    output_attractor = prepare_output(
        Path(settings["output_attractor_tsv"]), overwrite=args.overwrite
    )

    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment check
        raise SystemExit(
            "This wrapper requires anndata. Install Stage 2 dependencies with:\n"
            "  python -m pip install -e '.[application,test]'"
        ) from exc

    adata = ad.read_h5ad(input_h5ad)
    main_patients = load_main_patients(main_patients_path)
    scaffold_columns = tuple(str(x) for x in settings["scaffold_columns"])
    reference_phase = str(settings.get("reference_timepoint", "DX"))
    main_state_count = int(settings.get("main_state_count", 4))
    if main_state_count < 2:
        raise ValueError("main_state_count must be at least 2.")

    coordinate_columns = tuple(
        str(value) for value in settings.get("coordinate_columns", ["PC1", "PC2"])
    )
    sample_table = collapse_replicated_sample_scores(
        adata.obs,
        scaffold_columns=scaffold_columns,
        coordinate_columns=coordinate_columns,
    )
    attractor = compute_reference_attractor(
        sample_table,
        scaffold_columns=scaffold_columns,
        phase_column="clinical_timepoint_coarse",
        reference_phase=reference_phase,
    )
    output = add_effective_dynamic_parameters(
        sample_table,
        attractor,
        scaffold_columns=scaffold_columns,
        max_branch_entropy=float(np.log(main_state_count)),
        main_patient_ids=main_patients,
        qc_flagged_patient_ids=tuple(
            str(value) for value in settings.get("exploratory_patients", [])
        ),
        minimum_main_cells=int(settings.get("minimum_main_cells", 50)),
        phase_order={
            str(key): int(value)
            for key, value in settings.get("phase_order", {}).items()
        },
    )

    output = output.sort_values(
        ["patient_id", "phase_order", "sample_id"]
    ).reset_index(drop=True)
    output.to_csv(output_csv, index=False)

    attractor_table = pd.DataFrame(
        {
            "feature": list(scaffold_columns),
            "dx_attractor_value": attractor[list(scaffold_columns)].to_numpy(
                dtype=float
            ),
        }
    )
    attractor_table.to_csv(output_attractor, sep="\t", index=False)

    print(f"[DONE] Saved {output_csv}")
    print(f"[DONE] Saved {output_attractor}")

    print("\n[SUMMARY: sample counts by phase]")
    print(output["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main-analysis sample counts by phase]")
    print(
        output.loc[
            output["is_main_analysis_sample"], "clinical_timepoint_coarse"
        ]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n[SUMMARY: theta_eff by phase]")
    print(
        output.groupby("clinical_timepoint_coarse")["theta_eff"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\n[SUMMARY: sigma_eff by phase]")
    print(
        output.groupby("clinical_timepoint_coarse")["sigma_eff"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\n[SUMMARY: mu_shift_from_dx by phase]")
    print(
        output.groupby("clinical_timepoint_coarse")["mu_shift_from_dx"]
        .describe()
        .round(4)
        .to_string()
    )

    if "AML21" in set(output["patient_id"].astype(str)):
        print("\n[INFO] AML21 sample dynamic parameters]")
        print(
            output[output["patient_id"] == "AML21"][
                [
                    "patient_id",
                    "sample_id",
                    "clinical_timepoint_coarse",
                    "n_cells",
                    "theta_eff",
                    "sigma_eff",
                    "mu_shift_from_dx",
                    "branch_id_dominant",
                    "ecotype_label",
                ]
            ].to_string(index=False)
        )

    if "AML1" in set(output["patient_id"].astype(str)):
        print("\n[INFO] AML1 exploratory sample dynamic parameters]")
        print(
            output[output["patient_id"] == "AML1"][
                [
                    "patient_id",
                    "sample_id",
                    "clinical_timepoint_coarse",
                    "n_cells",
                    "theta_eff",
                    "sigma_eff",
                    "mu_shift_from_dx",
                    "branch_id_dominant",
                    "ecotype_label",
                    "is_qc_flagged_patient",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
