"""Try the saved multiple-regression equation without loading a pickle."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def predict_grade(g1, g2, model_path=ROOT / "reports/python/model.json"):
    values = {"G1": float(g1), "G2": float(g2)}
    if not all(math.isfinite(v) and 0 <= v <= 20 for v in values.values()):
        raise ValueError("G1 and G2 must be finite numbers from 0 to 20.")
    model = json.loads(Path(model_path).read_text())
    return model["intercept"] + sum(model["coefficients"][feature] * values[feature] for feature in model["features"])


def main():
    parser = argparse.ArgumentParser(description="Estimate G3 using earlier grades, on a 0–20 scale.")
    parser.add_argument("--g1", type=float, required=True, help="First-period grade (0–20)")
    parser.add_argument("--g2", type=float, required=True, help="Second-period grade (0–20)")
    args = parser.parse_args()
    try:
        prediction = predict_grade(args.g1, args.g2)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Estimated final grade: {prediction:.2f} / 20 (raw linear-regression prediction)")
    if prediction < 0 or prediction > 20:
        print("The raw result is outside the grade scale; OLS does not constrain its predictions.")
    print("Learning example based on Portuguese school data; individual outcomes remain uncertain.")


if __name__ == "__main__":
    main()
