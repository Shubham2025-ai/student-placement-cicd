import pandas as pd

from src.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    split_features_labels,
    train_test_split_data,
)


def make_sample_df(n=20):
    return pd.DataFrame(
        {
            "cgpa": [7.0 + (i % 3) for i in range(n)],
            "attendance": [80 + i for i in range(n)],
            "coding_score": [60 + i for i in range(n)],
            "projects": [i % 5 for i in range(n)],
            "internships": [i % 3 for i in range(n)],
            "communication_score": [50 + i for i in range(n)],
            "placed": [i % 2 for i in range(n)],
        }
    )


def test_split_features_labels_shapes():
    df = make_sample_df()
    X, y = split_features_labels(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(df)
    assert len(y) == len(df)


def test_train_test_split_data_proportions():
    df = make_sample_df(n=20)
    X_train, X_test, y_train, y_test = train_test_split_data(df, test_size=0.25)
    assert len(X_train) == 15
    assert len(X_test) == 5
    assert len(y_train) == 15
    assert len(y_test) == 5
    assert TARGET_COLUMN not in X_train.columns
