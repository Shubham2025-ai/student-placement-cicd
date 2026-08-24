import pandas as pd
import pytest

from src.data_validation import DataValidationError, validate_dataframe

VALID_ROW = {
    "cgpa": 8.0,
    "attendance": 85.0,
    "coding_score": 70.0,
    "projects": 3,
    "internships": 1,
    "communication_score": 75.0,
    "placed": 1,
}


def make_df(overrides=None, drop_col=None):
    row = dict(VALID_ROW)
    if overrides:
        row.update(overrides)
    df = pd.DataFrame([row, row])
    if drop_col:
        df = df.drop(columns=[drop_col])
    return df


def test_valid_data_passes():
    df = make_df()
    validate_dataframe(df)  # should not raise


def test_missing_column_fails():
    df = make_df(drop_col="cgpa")
    with pytest.raises(DataValidationError):
        validate_dataframe(df)


def test_missing_value_fails():
    df = make_df()
    df.loc[0, "cgpa"] = None
    with pytest.raises(DataValidationError):
        validate_dataframe(df)


def test_out_of_range_value_fails():
    df = make_df(overrides={"cgpa": 15.0})
    with pytest.raises(DataValidationError):
        validate_dataframe(df)


def test_invalid_placed_label_fails():
    df = make_df(overrides={"placed": 5})
    with pytest.raises(DataValidationError):
        validate_dataframe(df)


def test_empty_dataframe_fails():
    df = make_df().iloc[0:0]
    with pytest.raises(DataValidationError):
        validate_dataframe(df)
