import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oulb.branch_plotting import (
    create_branch_count_sensitivity_figure,
    create_switch_rate_sensitivity_figure,
)


def test_switch_rate_figure_is_created_from_prepared_table():
    table = pd.DataFrame(
        {"threshold": [20, 50, 100], "switch_rate": [0.4, 0.5, 0.3]}
    )
    figure = create_switch_rate_sensitivity_figure(table)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_xlabel() == "Malignant-cell threshold"
    plt.close(figure)


def test_branch_count_figure_has_dx_and_rel_panels():
    table = pd.DataFrame(
        {
            "threshold": [20, 20, 50, 50],
            "timepoint": ["DX", "REL", "DX", "REL"],
            "B1": [2, 1, 1, 1],
            "B2": [1, 2, 1, 1],
            "B3": [1, 1, 1, 1],
        }
    )
    figure = create_branch_count_sensitivity_figure(table)
    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["DX", "REL"]
    plt.close(figure)
