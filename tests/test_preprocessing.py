import pandas as pd

from credit_risk.preprocessing import (
    identify_high_null_columns,
    identify_single_value_columns,
    months_from_today,
    process_emp_length,
)


def test_process_emp_length():
    assert process_emp_length("10+ years") == 10
    assert process_emp_length("< 1 year") == 1
    assert process_emp_length(None) == 0
    assert process_emp_length("n/a") == 0


def test_months_from_today():
    today = pd.to_datetime("2020-01-01")
    assert months_from_today("Jan-19", today) == 12
    assert months_from_today("Dec-19", today) in (0, 1)  # integer division approx
    assert pd.isna(months_from_today("bad", today))


def test_identify_columns_helpers():
    df = pd.DataFrame(
        {
            "a": [1, None, None, None],  # 75% null
            "b": [1, 1, 1, 1],  # single value
            "c": [1, 2, 3, 4],
        }
    )
    high_null = identify_high_null_columns(df, threshold=0.70)
    single_val = identify_single_value_columns(df)
    assert "a" in high_null and "b" not in high_null
    assert single_val == ["b"]

