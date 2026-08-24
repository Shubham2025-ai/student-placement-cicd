"""
Validates the raw student dataset before it enters the training pipeline.

Checks performed:
  1. Required columns are present.
  2. No missing (NaN) values in any required column.
  3. Values fall inside a sane, realistic range for each column.

Run directly to validate data/students.csv and exit non-zero on failure,
so this can be used as its own CI step:

    python -m src.data_validation
"""

import sys

import pandas as pd

REQUIRED_COLUMNS = [
    "cgpa",
    "attendance",
    "coding_score",
    "projects",
    "internships",
    "communication_score",
    "placed",
]

VALID_RANGES = {
    "cgpa": (0, 10),
    "attendance": (0, 100),
    "coding_score": (0, 100),
    "projects": (0, 50),
    "internships": (0, 20),
    "communication_score": (0, 100),
    "placed": (0, 1),
}


class DataValidationError(Exception):
    """Raised when the dataset fails validation."""


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate a student dataframe. Raises DataValidationError on failure."""

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required column(s): {missing_cols}")

    if df.empty:
        raise DataValidationError("Dataset is empty")

    na_counts = df[REQUIRED_COLUMNS].isna().sum()
    bad_cols = na_counts[na_counts > 0]
    if not bad_cols.empty:
        raise DataValidationError(
            f"Missing/invalid (NaN) values detected in column(s): "
            f"{bad_cols.to_dict()}"
        )

    for col, (low, high) in VALID_RANGES.items():
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            raise DataValidationError(
                f"Column '{col}' has {len(out_of_range)} value(s) outside the "
                f"valid range [{low}, {high}]"
            )

    if not set(df["placed"].unique()).issubset({0, 1}):
        raise DataValidationError("Column 'placed' must only contain 0 or 1")


def validate_file(path: str = "data/students.csv") -> pd.DataFrame:
    """Load a CSV and validate it. Returns the dataframe if valid."""
    df = pd.read_csv(path)
    validate_dataframe(df)
    return df


if __name__ == "__main__":
    try:
        data = validate_file()
        print(f"Data validation PASSED: {len(data)} rows, all checks OK")
        sys.exit(0)
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"Data validation FAILED: {exc}")
        sys.exit(1)
