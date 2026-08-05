from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from oulb.benchmarking import (
    adjusted_rand_index,
    binary_metrics,
    detect_jump_intervals,
    fit_brownian_mle,
    fit_ou_mle,
    fit_ou_mle_measurement_aware,
    fit_shifted_ou_mle,
    parameter_recovery_table,
    standardized_ou_innovations,
    summarize_recovery,
)
from oulb.calibration import calibrate_from_interval_table
from oulb.observation import (
    ObservationSpec,
    irregular_schedule,
    observe_latent,
    regular_schedule,
)
from oulb.simulation import (
    BranchSpec,
    JumpSpec,
    simulate_process,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pediatric_leukemia_stage5.toml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)

    output_dir = Path(config["paths"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        output_dir / "parameter_recovery.csv",
        output_dir / "recovery_summary.csv",
        output_dir / "jump_detection_metrics.csv",
        output_dir / "branch_recovery_metrics.csv",
        output_dir / "real_data_calibration.json",
        output_dir / "stress_test_results.csv",
        output_dir / "real_data_calibrated_simulations.csv",
    ]

    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            "Stage 5 outputs exist; use --overwrite"
        )

    seed = int(config["benchmark"]["seed"])
    replicates = int(config["benchmark"]["replicates"])
    n_observations = int(
        config["benchmark"]["n_observations"]
    )
    followup = float(config["benchmark"]["followup"])

    rng = np.random.default_rng(seed)

    recovery_records = []
    jump_rows = []
    branch_rows = []

    for replicate in range(replicates):
        replicate_seed = int(
            rng.integers(0, 2**31 - 1)
        )
        times = regular_schedule(
            0.0,
            followup,
            n_observations,
        )

        brownian = simulate_process(
            times,
            [0.0],
            model="brownian",
            drift=0.15,
            sigma=0.30,
            seed=replicate_seed,
        )
        brownian_fit = fit_brownian_mle(
            times,
            brownian.states[:, 0],
        )

        for parameter, truth in [
            ("drift", 0.15),
            ("sigma", 0.30),
        ]:
            recovery_records.append(
                {
                    "scenario": "brownian",
                    "replicate": replicate,
                    "parameter": parameter,
                    "truth": truth,
                    "estimate": brownian_fit[parameter],
                }
            )

        ou = simulate_process(
            times,
            [1.0],
            model="ou",
            theta=0.80,
            mu=0.20,
            sigma=0.25,
            seed=replicate_seed + 1,
        )
        ou_fit = fit_ou_mle(
            times,
            ou.states[:, 0],
        )

        for parameter, truth in [
            ("theta", 0.80),
            ("mu", 0.20),
            ("sigma", 0.25),
        ]:
            recovery_records.append(
                {
                    "scenario": "ou",
                    "replicate": replicate,
                    "parameter": parameter,
                    "truth": truth,
                    "estimate": ou_fit[parameter],
                }
            )

        def treatment_shift(time: float) -> float:
            return 0.60 if time >= followup / 2.0 else 0.0

        treatment = np.array(
            [
                1.0 if time >= followup / 2.0 else 0.0
                for time in times
            ]
        )

        shifted = simulate_process(
            times,
            [0.0],
            model="shifted_ou",
            theta=0.70,
            mu=0.10,
            sigma=0.20,
            treatment_shift=treatment_shift,
            seed=replicate_seed + 2,
        )
        shifted_fit = fit_shifted_ou_mle(
            times,
            shifted.states[:, 0],
            treatment,
        )

        for parameter, truth in [
            ("theta", 0.70),
            ("mu0", 0.10),
            ("delta", 0.60),
            ("sigma", 0.20),
        ]:
            recovery_records.append(
                {
                    "scenario": "shifted_ou",
                    "replicate": replicate,
                    "parameter": parameter,
                    "truth": truth,
                    "estimate": shifted_fit[parameter],
                }
            )

        jump_process = simulate_process(
            times,
            [0.0],
            model="ou_jump",
            theta=0.80,
            mu=0.0,
            sigma=0.20,
            jump=JumpSpec(rate=0.25, scale=0.80),
            seed=replicate_seed + 3,
        )

        innovations = standardized_ou_innovations(
            times,
            jump_process.states[:, 0],
            0.0,
            0.80,
            0.20,
        )

        predicted_jumps = detect_jump_intervals(
            innovations,
            float(
                config["benchmark"]["jump_z_threshold"]
            ),
        )

        true_jumps = np.zeros(
            len(times) - 1,
            dtype=bool,
        )

        for event in jump_process.event_ledger:
            if event["event"] == "jump":
                true_jumps[event["interval"]] = True

        jump_rows.append(
            {
                "replicate": replicate,
                **binary_metrics(
                    true_jumps,
                    predicted_jumps,
                ),
            }
        )

        rate_matrix = np.array(
            [
                [-0.40, 0.40],
                [0.25, -0.25],
            ]
        )

        branch_spec = BranchSpec(
            rate_matrix,
            np.array([0.80, 0.80]),
            np.array([[0.0], [1.20]]),
            np.array([0.20, 0.20]),
        )

        branching = simulate_process(
            times,
            [0.0],
            model="ou_branching",
            branch=branch_spec,
            seed=replicate_seed + 4,
        )

        predicted_branch = (
            branching.states[:, 0] > 0.60
        ).astype(int)

        branch_rows.append(
            {
                "replicate": replicate,
                "ari": adjusted_rand_index(
                    branching.branches,
                    predicted_branch,
                ),
                "state_accuracy": float(
                    np.mean(
                        branching.branches
                        == predicted_branch
                    )
                ),
            }
        )

    recovery = parameter_recovery_table(
        recovery_records
    )
    recovery.to_csv(targets[0], index=False)

    summarize_recovery(
        recovery
    ).to_csv(targets[1], index=False)

    pd.DataFrame(jump_rows).to_csv(
        targets[2],
        index=False,
    )

    pd.DataFrame(branch_rows).to_csv(
        targets[3],
        index=False,
    )

    interval_table = pd.read_csv(
        Path(
            config["paths"]["interval_table"]
        ).expanduser()
    )

    calibration = calibrate_from_interval_table(
        interval_table,
        config["calibration"]["displacement_column"],
        config["calibration"].get("dt_column") or None,
    )

    targets[4].write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )

    stress_rows = []

    for requested_n in config[
        "stress_test"
    ]["n_observations"]:
        for noise_sd in config[
            "stress_test"
        ]["noise_sd"]:
            for missing_probability in config[
                "stress_test"
            ]["missing_probability"]:
                for interval_pattern in config[
                    "stress_test"
                ]["interval_pattern"]:
                    stress_replicates = max(
                        5,
                        replicates // 10,
                    )

                    for replicate in range(
                        stress_replicates
                    ):
                        stress_seed = int(
                            rng.integers(
                                0,
                                2**31 - 1,
                            )
                        )
                        stress_rng = (
                            np.random.default_rng(
                                stress_seed
                            )
                        )

                        if interval_pattern == "regular":
                            latent_times = regular_schedule(
                                0.0,
                                followup,
                                int(requested_n),
                            )
                        else:
                            latent_times = irregular_schedule(
                                0.0,
                                followup,
                                int(requested_n),
                                stress_rng,
                            )

                        latent = simulate_process(
                            latent_times,
                            [0.50],
                            model="ou",
                            theta=0.80,
                            mu=0.20,
                            sigma=0.25,
                            seed=stress_seed,
                        )

                        observed_times, observed = (
                            observe_latent(
                                latent_times,
                                latent.states,
                                ObservationSpec(
                                    float(noise_sd),
                                    float(
                                        missing_probability
                                    ),
                                    True,
                                ),
                                stress_rng,
                            )
                        )

                        if len(observed_times) < 4:
                            continue

                        fit_naive = fit_ou_mle(
                            observed_times,
                            observed[:, 0],
                        )

                        fit_aware = (
                            fit_ou_mle_measurement_aware(
                                observed_times,
                                observed[:, 0],
                                measurement_sd=float(
                                    noise_sd
                                ),
                            )
                        )

                        stress_rows.append(
                            {
                                "n_observations":
                                    requested_n,
                                "noise_sd":
                                    noise_sd,
                                "missing_probability":
                                    missing_probability,
                                "interval_pattern":
                                    interval_pattern,
                                "replicate":
                                    replicate,
                                "observed_n":
                                    len(observed_times),
                                "theta_true":
                                    0.80,
                                "sigma_true":
                                    0.25,
                                "theta_naive":
                                    fit_naive["theta"],
                                "sigma_naive":
                                    fit_naive["sigma"],
                                "theta_abs_error_naive":
                                    abs(
                                        fit_naive["theta"]
                                        - 0.80
                                    ),
                                "sigma_abs_error_naive":
                                    abs(
                                        fit_naive["sigma"]
                                        - 0.25
                                    ),
                                "theta_bound_naive":
                                    fit_naive[
                                        "at_theta_bound"
                                    ],
                                "sigma_bound_naive":
                                    fit_naive[
                                        "at_sigma_bound"
                                    ],
                                "fit_success_naive":
                                    fit_naive["success"],
                                "theta_measurement_aware":
                                    fit_aware["theta"],
                                "sigma_measurement_aware":
                                    fit_aware["sigma"],
                                "theta_abs_error_measurement_aware":
                                    abs(
                                        fit_aware["theta"]
                                        - 0.80
                                    ),
                                "sigma_abs_error_measurement_aware":
                                    abs(
                                        fit_aware["sigma"]
                                        - 0.25
                                    ),
                                "theta_bound_measurement_aware":
                                    fit_aware[
                                        "at_theta_bound"
                                    ],
                                "sigma_bound_measurement_aware":
                                    fit_aware[
                                        "at_sigma_bound"
                                    ],
                                "fit_success_measurement_aware":
                                    fit_aware["success"],
                                "theta_estimate":
                                    fit_naive["theta"],
                                "theta_abs_error":
                                    abs(
                                        fit_naive["theta"]
                                        - 0.80
                                    ),
                                "sigma_estimate":
                                    fit_naive["sigma"],
                                "sigma_abs_error":
                                    abs(
                                        fit_naive["sigma"]
                                        - 0.25
                                    ),
                                "success":
                                    fit_naive["success"],
                            }
                        )

    pd.DataFrame(stress_rows).to_csv(
        targets[5],
        index=False,
    )

    real_rows = []
    real_n = max(
        2,
        int(calibration["n_intervals"]) + 1,
    )
    real_sigma = max(
        calibration["displacement_sd"],
        1e-3,
    )

    for replicate in range(replicates):
        replicate_seed = int(
            rng.integers(0, 2**31 - 1)
        )
        real_times = regular_schedule(
            0.0,
            float(real_n - 1),
            real_n,
        )
        simulated = simulate_process(
            real_times,
            [0.0],
            model="ou",
            theta=0.80,
            mu=0.0,
            sigma=real_sigma,
            seed=replicate_seed,
        )

        real_rows.append(
            {
                "replicate": replicate,
                "n_observations": real_n,
                "sigma_calibrated": real_sigma,
                "simulated_total_displacement":
                    float(
                        abs(
                            simulated.states[-1, 0]
                            - simulated.states[0, 0]
                        )
                    ),
                "simulated_max_interval_displacement":
                    float(
                        np.max(
                            np.abs(
                                np.diff(
                                    simulated.states[:, 0]
                                )
                            )
                        )
                    ),
            }
        )

    pd.DataFrame(real_rows).to_csv(
        targets[6],
        index=False,
    )

    print("[DONE] Stage 5 benchmark outputs")
    for target in targets:
        print(target)


if __name__ == "__main__":
    main()
