"""
Analytics module for Toyota Canarias BI Dashboard.
Loads and analyses query_logs/query_log.csv.
"""

import os
import re
import ast
import pandas as pd
import numpy as np
from collections import Counter
from src.config import LOG_FILE


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_parse_list(val):
    """Parse a string that looks like a Python list into an actual list."""
    if pd.isna(val) or str(val).strip() in ("", "[]", "nan"):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [str(i).strip() for i in parsed if str(i).strip()]
        return [str(parsed).strip()]
    except Exception:
        # Fallback: strip brackets and split by comma
        cleaned = re.sub(r"[\[\]'\"]", "", str(val))
        return [s.strip() for s in cleaned.split(",") if s.strip()]


def _parse_budget(val):
    """
    Extract a representative numeric budget from free-text strings.
    Examples: '20000', '20k', '15000-20000', 'menos de 25000', 'around 30000'
    Returns the midpoint for ranges, or the single value.  NaN if unparseable.
    """
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return np.nan
    text = str(val).strip()
    # If it looks like a plain float (e.g. "25000.0"), parse directly
    try:
        candidate = float(text.replace(",", "").replace("€", ""))
        if candidate > 1000:
            return candidate
    except ValueError:
        pass
    text = text.lower().replace("€", "").replace(",", "")
    # Replace k/K shorthand before stripping dots
    text = re.sub(r"(\d+)\s*k", lambda m: str(int(m.group(1)) * 1000), text)
    # Remove non-numeric except digits and spaces/dashes (range separators)
    text = re.sub(r"[^\d\s\-]", " ", text)
    numbers = [int(n) for n in re.findall(r"\d+", text) if int(n) > 1000]
    if not numbers:
        return np.nan
    return float(np.mean(numbers))


# ── data loading ─────────────────────────────────────────────────────────────

def load_logs(path: str = LOG_FILE) -> pd.DataFrame:
    """Load query_log.csv and return a cleaned DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["timestamp"], on_bad_lines="skip")

    # Normalise string columns
    str_cols = [
        "intent", "fuel_preference", "body_type",
        "use_case", "model_mentioned", "sentiment",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "": np.nan, "none": np.nan})

    # Parse budget
    if "budget_mentioned" in df.columns:
        df["budget_numeric"] = df["budget_mentioned"].apply(_parse_budget)

    # Parse list columns
    for col in ("features_mentioned",):
        if col in df.columns:
            df[col + "_list"] = df[col].apply(_safe_parse_list)

    # Time columns
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["weekday"] = pd.to_datetime(df["timestamp"]).dt.day_name()

    return df


# ── KPIs ─────────────────────────────────────────────────────────────────────

def kpi_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    avg_budget = df["budget_numeric"].mean() if "budget_numeric" in df.columns else np.nan
    top_fuel = (
        df["fuel_preference"].dropna().mode().iloc[0]
        if "fuel_preference" in df.columns and df["fuel_preference"].notna().any()
        else "N/A"
    )
    return {
        "total_queries": len(df),
        "unique_sessions": df["session_id"].nunique() if "session_id" in df.columns else 0,
        "avg_budget": round(avg_budget, 0) if not np.isnan(avg_budget) else None,
        "top_fuel": top_fuel.title() if top_fuel != "N/A" else "N/A",
    }


# ── distributions ─────────────────────────────────────────────────────────────

def fuel_distribution(df: pd.DataFrame) -> pd.Series:
    if "fuel_preference" not in df.columns or df["fuel_preference"].dropna().empty:
        return pd.Series(dtype=int)
    return df["fuel_preference"].dropna().value_counts()


def body_type_distribution(df: pd.DataFrame) -> pd.Series:
    if "body_type" not in df.columns or df["body_type"].dropna().empty:
        return pd.Series(dtype=int)
    return df["body_type"].dropna().value_counts()


def intent_distribution(df: pd.DataFrame) -> pd.Series:
    if "intent" not in df.columns or df["intent"].dropna().empty:
        return pd.Series(dtype=int)
    return df["intent"].dropna().value_counts()


def sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    if "sentiment" not in df.columns or df["sentiment"].dropna().empty:
        return pd.Series(dtype=int)
    return df["sentiment"].dropna().value_counts()


def model_distribution(df: pd.DataFrame) -> pd.Series:
    if "model_mentioned" not in df.columns or df["model_mentioned"].dropna().empty:
        return pd.Series(dtype=int)
    return df["model_mentioned"].dropna().value_counts()


def use_case_distribution(df: pd.DataFrame) -> pd.Series:
    if "use_case" not in df.columns or df["use_case"].dropna().empty:
        return pd.Series(dtype=int)
    return df["use_case"].dropna().value_counts()


def budget_bins(df: pd.DataFrame) -> pd.Series:
    """Return counts per budget bracket."""
    if "budget_numeric" not in df.columns:
        return pd.Series(dtype=int)
    valid = df["budget_numeric"].dropna()
    if valid.empty:
        return pd.Series(dtype=int)
    bins   = [0, 15000, 20000, 25000, 30000, 40000, 60000, float("inf")]
    labels = ["<15k", "15-20k", "20-25k", "25-30k", "30-40k", "40-60k", ">60k"]
    cut = pd.cut(valid, bins=bins, labels=labels, right=False)
    return cut.value_counts().sort_index()


# ── co-occurrence ─────────────────────────────────────────────────────────────

def fuel_body_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: fuel type (rows) × body type (cols), values = count."""
    if "fuel_preference" not in df.columns or "body_type" not in df.columns:
        return pd.DataFrame()
    sub = df[["fuel_preference", "body_type"]].dropna()
    if sub.empty:
        return pd.DataFrame()
    return pd.crosstab(sub["fuel_preference"], sub["body_type"])


# ── session analysis ──────────────────────────────────────────────────────────

def session_stats(df: pd.DataFrame) -> dict:
    if "session_id" not in df.columns or df.empty:
        return {}
    counts = df.groupby("session_id").size()
    top = counts.nlargest(5).reset_index()
    top.columns = ["session_id", "queries"]
    return {
        "avg_queries_per_session": round(counts.mean(), 2),
        "max_queries": int(counts.max()),
        "top_sessions": top,
    }


# ── time series ───────────────────────────────────────────────────────────────

def queries_by_day(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns or df.empty:
        return pd.Series(dtype=int)
    return df.groupby("date").size().rename("count")


def queries_by_hour(df: pd.DataFrame) -> pd.Series:
    if "hour" not in df.columns or df.empty:
        return pd.Series(dtype=int)
    return df.groupby("hour").size().rename("count")


# ── unmet demand ──────────────────────────────────────────────────────────────

def unmet_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows where the user expressed preferences (fuel/body/budget) but no
    Toyota model appears in model_mentioned — potential stock/info gap.
    """
    if df.empty:
        return pd.DataFrame()

    has_prefs = (
        df["fuel_preference"].notna() |
        df["body_type"].notna() |
        (df.get("budget_numeric", pd.Series(dtype=float)).notna())
    )
    no_model = df["model_mentioned"].isna() | (df["model_mentioned"] == "")
    result = df[has_prefs & no_model].copy()

    cols = [
        c for c in [
            "timestamp", "session_id", "user_message",
            "intent", "fuel_preference", "body_type",
            "budget_mentioned", "use_case",
        ]
        if c in result.columns
    ]
    return result[cols].reset_index(drop=True)
