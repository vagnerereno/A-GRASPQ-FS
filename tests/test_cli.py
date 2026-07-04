import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def test_cli_runs_with_auto_preprocessing(tmp_path):
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        random_state=11,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["category"] = ["a" if i % 2 else "b" for i in range(len(df))]
    df["target"] = y
    csv_path = tmp_path / "toy.csv"
    output_path = tmp_path / "out.json"
    df.to_csv(csv_path, index=False)

    cmd = [
        sys.executable,
        "-m",
        "agraspqfs",
        "--csv",
        str(csv_path),
        "--target",
        "target",
        "--preset",
        "fast",
        "--max-evaluations",
        "15",
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert payload["preprocessing_applied"] is True
    assert payload["n_selected_features"] >= 1
    assert payload["n_evaluations"] <= 15
