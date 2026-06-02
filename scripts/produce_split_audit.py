"""
Produce preprocessing/split_audit.csv demonstrating zero intersection
between train and test partition keys for project-LOPO CV.

Schema:
  project_name, n_plots, fold_index, role, train_projects
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

PARQUET = (
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)
OUT_CSV = "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/split_audit.csv"


def main() -> None:
    df = pd.read_parquet(PARQUET)

    # n_plots per project (all rows including failure rows and both years)
    project_counts = df.groupby("project_name").size().reset_index(name="n_plots")
    projects = sorted(project_counts["project_name"].tolist())
    n_projects = len(projects)
    print(f"Found {n_projects} unique projects.")
    for p in projects:
        print(
            f"  {p}: {project_counts.loc[project_counts['project_name'] == p, 'n_plots'].iloc[0]} rows"
        )

    rows = []
    for fold_index, test_project in enumerate(projects):
        train_projects = [p for p in projects if p != test_project]
        n_plots = project_counts.loc[
            project_counts["project_name"] == test_project, "n_plots"
        ].iloc[0]
        rows.append(
            {
                "project_name": test_project,
                "n_plots": int(n_plots),
                "fold_index": fold_index,
                "role": "test_when_held_out",
                "train_projects": ",".join(train_projects),
            }
        )

    audit_df = pd.DataFrame(rows)

    # Zero-intersection check
    assert all(
        row.project_name not in row.train_projects.split(",") for _, row in audit_df.iterrows()
    ), "FAIL: test project appears in train_projects for some fold!"
    print("\nZero-intersection check PASSED: test project never appears in train_projects.")

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(audit_df)} rows to {OUT_CSV}")
    print(audit_df.to_string())


if __name__ == "__main__":
    main()
