from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from oulb.evaluation import prepare_transition_metric_data, run_default_model_comparison
from oulb.plotting import create_model_comparison_figure


def test_model_comparison_figure_smoke() -> None:
    rows = []
    for index, value in enumerate([0.2, 0.22, 0.25, 0.27, 0.3, 0.33]):
        rows.append(
            {
                "sample": f"S{index}",
                "metric": value,
                "dx": "A",
                "rel": "A",
                "switch": 0,
            }
        )
    for index, value in enumerate([0.3, 0.35, 0.4, 0.55, 0.9, 1.3]):
        rows.append(
            {
                "sample": f"W{index}",
                "metric": value,
                "dx": "A",
                "rel": "B",
                "switch": 1,
            }
        )
    prepared = prepare_transition_metric_data(
        pd.DataFrame(rows),
        metric_column="metric",
        sample_column="sample",
        start_branch_column="dx",
        end_branch_column="rel",
        stability_column="switch",
    )
    artifacts = run_default_model_comparison(prepared)
    figure = create_model_comparison_figure(
        prepared, artifacts, figsize=(8.0, 5.0), top_annotate=2
    )
    assert len(figure.axes) == 4
    plt.close(figure)
