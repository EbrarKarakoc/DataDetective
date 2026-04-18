"""Core tools for automated EDA, hypothesis generation, and baseline modeling.

This module is intentionally self-contained so it can be exposed as tool
functions to an LLM agent (e.g., OpenAI tool calling or LM Studio-compatible
function schemas).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def run_automated_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Run a compact automated EDA routine and return structured findings.

    This function performs three core checks designed for quick diagnostics and
    LLM tool usage:
    1) infers column data types from pandas dtypes,
    2) computes missing-value counts and rates per column,
    3) computes pairwise correlation on numeric columns.

    The return value is a JSON-serializable dictionary so an agent can directly
    summarize the findings in natural language.

    Args:
        df: Input dataframe to analyze. Must be a non-empty ``pandas.DataFrame``.

    Returns:
        A dictionary with keys:
            - ``shape``: Dataset shape as ``{"rows": int, "columns": int}``.
            - ``dtypes``: Mapping of ``column -> dtype string``.
            - ``missing_values``: Mapping of ``column -> {"count": int, "ratio": float}``.
            - ``numeric_correlations``: Nested mapping from numeric correlation matrix
              (empty dict if fewer than two numeric columns).

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``df`` is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty. Provide a non-empty DataFrame.")

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    missing_counts = df.isna().sum()
    missing_ratio = (missing_counts / len(df)).fillna(0.0)
    missing_values = {
        col: {"count": int(missing_counts[col]), "ratio": float(missing_ratio[col])}
        for col in df.columns
    }

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr_df = numeric_df.corr(numeric_only=True)
        numeric_correlations: dict[str, Any] = corr_df.fillna(0.0).to_dict()
    else:
        numeric_correlations = {}

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": dtypes,
        "missing_values": missing_values,
        "numeric_correlations": numeric_correlations,
    }


def suggest_hypothesis(target_col: str) -> list[str]:
    """Generate practical, domain-agnostic hypotheses for a target variable.

    The output is intentionally concise (2-3 items) and phrased so an analyst
    can test the hypotheses with EDA or baseline modeling steps.

    Args:
        target_col: Name of the target column. Used only to personalize text.

    Returns:
        A list containing three hypothesis sentences centered on:
            - relationship between target and top numeric predictors,
            - impact of missing values/outliers on target behavior,
            - effect of categorical segments on target distribution.

    Raises:
        ValueError: If ``target_col`` is empty or only whitespace.
    """
    if not isinstance(target_col, str) or not target_col.strip():
        raise ValueError("target_col must be a non-empty string.")

    target = target_col.strip()
    return [
        f"'{target}' değişkeni, bazı sayısal özelliklerle güçlü doğrusal veya doğrusal olmayan ilişki gösterebilir.",
        f"'{target}' için gözlenen oynaklığın bir kısmı eksik değerler veya aykırı gözlemlerden kaynaklanıyor olabilir.",
        f"'{target}' dağılımı, belirli kategorik segmentler (ör. kullanıcı tipi, bölge, ürün grubu) arasında anlamlı biçimde farklılaşıyor olabilir.",
    ]


def recommend_and_train_model(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Train a baseline Random Forest model based on target type and report score.

    The function auto-detects whether the problem is classification or
    regression using target dtype/cardinality heuristics:
    - Non-numeric target -> classification
    - Numeric target with low unique values (<= max(20, 5% of rows)) -> classification
    - Otherwise -> regression

    It builds a minimal preprocessing + model pipeline:
    - Numeric features: median imputation
    - Categorical features: most-frequent imputation + one-hot encoding
    - Estimator: RandomForestClassifier or RandomForestRegressor

    Args:
        df: Full dataset including the target column.
        target_col: Name of the target column in ``df``.

    Returns:
        A dictionary with model summary and performance:
            - ``task_type``: ``"classification"`` or ``"regression"``
            - ``model``: Chosen model class name
            - ``metric``: ``"accuracy"`` or ``"mse"``
            - ``score``: Float metric value on test split
            - ``n_train``: Number of training rows
            - ``n_test``: Number of test rows
            - ``feature_count``: Number of input features used

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If dataframe is empty, target is missing, or data is
            insufficient for train/test split.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty. Provide a non-empty DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame.")

    work_df = df.copy()
    y = work_df[target_col]
    X = work_df.drop(columns=[target_col])

    if X.shape[1] == 0:
        raise ValueError("No feature columns found after removing target_col.")
    if len(work_df) < 10:
        raise ValueError("At least 10 rows are recommended for training/testing.")

    unique_count = int(y.nunique(dropna=True))
    threshold = max(20, int(0.05 * len(y)))
    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    task_type = "regression" if (is_numeric_target and unique_count > threshold) else "classification"

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        stratify = y if unique_count > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    if task_type == "classification":
        metric_name = "accuracy"
        score = float(accuracy_score(y_test, preds))
    else:
        metric_name = "mse"
        score = float(mean_squared_error(y_test, preds))

    return {
        "task_type": task_type,
        "model": model.__class__.__name__,
        "metric": metric_name,
        "score": score,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count": int(X.shape[1]),
    }
