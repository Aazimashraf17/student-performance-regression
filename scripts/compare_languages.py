"""Fail if Python and R disagree on test rows, predictions, metrics or coefficients."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    pairs = {
        "metrics": (["model"], ["test_rows", "mae", "rmse", "r2"]),
        "predictions": (["model", "row_id"], ["actual", "predicted", "residual"]),
        "coefficients": (["model", "term"], ["estimate"]),
    }
    for name, (keys, values) in pairs.items():
        frames = [pd.read_csv(ROOT / f"reports/{language}/{name}.csv").sort_values(keys).reset_index(drop=True)
                  for language in ["python", "r"]]
        if not frames[0][keys].equals(frames[1][keys]):
            raise AssertionError(f"{name}: Python and R identifiers differ.")
        np.testing.assert_allclose(frames[0][values], frames[1][values], rtol=1e-9, atol=1e-9)
        print(f"PASS: {name} matches between Python and R.")


if __name__ == "__main__":
    main()
