# -*- coding: utf-8 -*-
"""Financial-statement utilities for Finance Bro Stock Research.

The module keeps data acquisition and financial calculations separate from
Streamlit presentation. Missing values remain missing; they are never replaced
with zero merely for display purposes.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yfinance as yf


INCOME_STATEMENT_LINES: dict[str, list[str]] = {
    "Revenue": ["Total Revenue", "Operating Revenue"],
    "Cost of Revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "Gross Profit": ["Gross Profit"],
    "Operating Income": ["Operating Income"],
    "EBIT": ["EBIT", "Operating Income"],
    "EBITDA": ["EBITDA", "Normalized EBITDA"],
    "Interest Expense": ["Interest Expense", "Interest Expense Non Operating"],
    "Pretax Income": ["Pretax Income", "Income Before Tax"],
    "Tax Provision": ["Tax Provision", "Income Tax Expense"],
    "Net Income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Continuous Operations",
    ],
    "Basic EPS": ["Basic EPS"],
    "Diluted EPS": ["Diluted EPS"],
    "Diluted Average Shares": ["Diluted Average Shares"],
}

BALANCE_SHEET_LINES: dict[str, list[str]] = {
    "Cash and Equivalents": [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash Financial",
    ],
    "Accounts Receivable": ["Receivables", "Accounts Receivable"],
    "Inventory": ["Inventory"],
    "Current Assets": ["Current Assets", "Total Current Assets"],
    "Property, Plant and Equipment": [
        "Net PPE",
        "Property Plant Equipment Net",
    ],
    "Goodwill": ["Goodwill"],
    "Total Assets": ["Total Assets"],
    "Accounts Payable": ["Payables", "Accounts Payable"],
    "Current Liabilities": [
        "Current Liabilities",
        "Total Current Liabilities",
    ],
    "Short-Term Debt": [
        "Current Debt",
        "Current Debt And Capital Lease Obligation",
    ],
    "Long-Term Debt": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    ],
    "Total Debt": ["Total Debt"],
    "Total Liabilities": [
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
    ],
    "Shareholders' Equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
    ],
    "Working Capital": ["Working Capital"],
}

CASH_FLOW_LINES: dict[str, list[str]] = {
    "Operating Cash Flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
    ],
    "Capital Expenditure": ["Capital Expenditure", "Capital Expenditures"],
    "Free Cash Flow": ["Free Cash Flow"],
    "Investing Cash Flow": [
        "Investing Cash Flow",
        "Total Cashflows From Investing Activities",
    ],
    "Financing Cash Flow": [
        "Financing Cash Flow",
        "Total Cash From Financing Activities",
    ],
    "Acquisitions": ["Net Business Purchases", "Acquisitions Net"],
    "Repurchase of Capital Stock": [
        "Repurchase Of Capital Stock",
        "Repurchase Of Stock",
    ],
    "Cash Dividends Paid": [
        "Cash Dividends Paid",
        "Common Stock Dividend Paid",
    ],
    "Issuance of Debt": ["Issuance Of Debt"],
    "Repayment of Debt": ["Repayment Of Debt"],
    "Change in Cash": ["Changes In Cash", "Change In Cash"],
}


def safe_float(value: Any) -> float:
    """Convert a scalar to float while preserving unavailable values as NaN."""

    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vector division that returns NaN for zero denominators and infinities."""

    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    numerator = pd.to_numeric(numerator, errors="coerce")
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def clean_statement(statement: Any) -> pd.DataFrame:
    """Normalize a Yahoo statement while retaining the original accounting rows."""

    if statement is None:
        return pd.DataFrame()

    if not isinstance(statement, pd.DataFrame):
        try:
            statement = pd.DataFrame(statement)
        except Exception:
            return pd.DataFrame()

    statement = statement.copy()
    statement = statement.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if statement.empty:
        return pd.DataFrame()

    normalized_columns: list[pd.Timestamp | str] = []
    for column in statement.columns:
        try:
            normalized_columns.append(pd.Timestamp(column).tz_localize(None))
        except Exception:
            normalized_columns.append(str(column))

    statement.columns = normalized_columns

    dated_columns = [column for column in statement.columns if isinstance(column, pd.Timestamp)]
    other_columns = [column for column in statement.columns if not isinstance(column, pd.Timestamp)]
    statement = statement.reindex(columns=sorted(dated_columns) + other_columns)

    return statement.apply(pd.to_numeric, errors="coerce")


def _call_statement_method(
    ticker_object: yf.Ticker,
    method_names: Iterable[str],
    frequency: str,
) -> pd.DataFrame:
    """Try supported yfinance getter names and compatible frequency arguments."""

    for method_name in method_names:
        method = getattr(ticker_object, method_name, None)
        if not callable(method):
            continue

        for kwargs in (
            {"freq": frequency},
            {"frequency": frequency},
            {},
        ):
            try:
                cleaned = clean_statement(method(**kwargs))
                if not cleaned.empty:
                    return cleaned
            except TypeError:
                continue
            except Exception:
                break

    return pd.DataFrame()


def fetch_statement(
    ticker_object: yf.Ticker,
    statement_type: str,
    frequency: str,
) -> pd.DataFrame:
    """Retrieve one statement with getter and property fallbacks."""

    getter_map = {
        "income": ("get_income_stmt",),
        "balance": ("get_balance_sheet",),
        "cashflow": ("get_cashflow", "get_cash_flow"),
    }

    property_map = {
        ("income", "yearly"): ("income_stmt", "financials"),
        ("income", "quarterly"): (
            "quarterly_income_stmt",
            "quarterly_financials",
        ),
        ("balance", "yearly"): ("balance_sheet", "balancesheet"),
        ("balance", "quarterly"): (
            "quarterly_balance_sheet",
            "quarterly_balancesheet",
        ),
        ("cashflow", "yearly"): ("cashflow", "cash_flow"),
        ("cashflow", "quarterly"): (
            "quarterly_cashflow",
            "quarterly_cash_flow",
        ),
    }

    cleaned = _call_statement_method(
        ticker_object=ticker_object,
        method_names=getter_map[statement_type],
        frequency=frequency,
    )
    if not cleaned.empty:
        return cleaned

    for property_name in property_map[(statement_type, frequency)]:
        try:
            cleaned = clean_statement(getattr(ticker_object, property_name, None))
            if not cleaned.empty:
                return cleaned
        except Exception:
            continue

    return pd.DataFrame()


def extract_line(statement: pd.DataFrame, alternatives: list[str]) -> pd.Series:
    """Extract an accounting line using exact matches before cautious fuzzy matches."""

    if statement.empty:
        return pd.Series(dtype=float)

    normalized_index = {
        " ".join(str(index).strip().lower().split()): index
        for index in statement.index
    }

    for alternative in alternatives:
        key = " ".join(alternative.strip().lower().split())
        if key in normalized_index:
            return pd.to_numeric(
                statement.loc[normalized_index[key]],
                errors="coerce",
            )

    for alternative in alternatives:
        words = [word for word in alternative.lower().split() if len(word) > 2]
        for normalized_name, original_index in normalized_index.items():
            if words and all(word in normalized_name for word in words):
                return pd.to_numeric(statement.loc[original_index], errors="coerce")

    return pd.Series(np.nan, index=statement.columns, dtype=float)


def create_selected_statement(
    statement: pd.DataFrame,
    line_dictionary: dict[str, list[str]],
) -> pd.DataFrame:
    """Create a compact, comparable statement from selected accounting lines."""

    if statement.empty:
        return pd.DataFrame()

    selected = pd.DataFrame(
        {
            display_name: extract_line(statement, alternatives)
            for display_name, alternatives in line_dictionary.items()
        }
    ).T

    return selected.dropna(axis=0, how="all")


def add_calculated_lines(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add transparent derived lines only when Yahoo did not supply them."""

    income = income_statement.copy()
    balance = balance_sheet.copy()
    cash = cash_flow.copy()

    if not cash.empty:
        if "Free Cash Flow" not in cash.index or cash.loc["Free Cash Flow"].isna().all():
            operating = (
                cash.loc["Operating Cash Flow"]
                if "Operating Cash Flow" in cash.index
                else pd.Series(np.nan, index=cash.columns)
            )
            capex = (
                cash.loc["Capital Expenditure"]
                if "Capital Expenditure" in cash.index
                else pd.Series(np.nan, index=cash.columns)
            )
            cash.loc["Free Cash Flow"] = operating + capex

    if not balance.empty:
        if "Total Debt" not in balance.index or balance.loc["Total Debt"].isna().all():
            short_debt = (
                balance.loc["Short-Term Debt"]
                if "Short-Term Debt" in balance.index
                else pd.Series(0.0, index=balance.columns)
            )
            long_debt = (
                balance.loc["Long-Term Debt"]
                if "Long-Term Debt" in balance.index
                else pd.Series(0.0, index=balance.columns)
            )
            calculated = short_debt.fillna(0) + long_debt.fillna(0)
            calculated[(short_debt.isna()) & (long_debt.isna())] = np.nan
            balance.loc["Total Debt"] = calculated

    return income, balance, cash


def _aligned_row(dataframe: pd.DataFrame, row_name: str, columns: list[Any]) -> pd.Series:
    if dataframe.empty or row_name not in dataframe.index:
        return pd.Series(np.nan, index=columns, dtype=float)
    return pd.to_numeric(dataframe.loc[row_name], errors="coerce").reindex(columns)


def calculate_growth_rate(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").pct_change(fill_method=None) * 100


def calculate_financial_ratios(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate ratios from statement data with no zero-filling of missing inputs."""

    columns = sorted(
        set(income_statement.columns)
        | set(balance_sheet.columns)
        | set(cash_flow.columns)
    )
    if not columns:
        return pd.DataFrame()

    revenue = _aligned_row(income_statement, "Revenue", columns)
    gross_profit = _aligned_row(income_statement, "Gross Profit", columns)
    operating_income = _aligned_row(income_statement, "Operating Income", columns)
    ebitda = _aligned_row(income_statement, "EBITDA", columns)
    net_income = _aligned_row(income_statement, "Net Income", columns)

    current_assets = _aligned_row(balance_sheet, "Current Assets", columns)
    current_liabilities = _aligned_row(balance_sheet, "Current Liabilities", columns)
    total_assets = _aligned_row(balance_sheet, "Total Assets", columns)
    total_debt = _aligned_row(balance_sheet, "Total Debt", columns)
    equity = _aligned_row(balance_sheet, "Shareholders' Equity", columns)
    cash = _aligned_row(balance_sheet, "Cash and Equivalents", columns)

    operating_cash_flow = _aligned_row(cash_flow, "Operating Cash Flow", columns)
    free_cash_flow = _aligned_row(cash_flow, "Free Cash Flow", columns)

    average_assets = (total_assets + total_assets.shift(1)) / 2
    average_equity = (equity + equity.shift(1)) / 2

    ratios = pd.DataFrame(
        {
            "Revenue Growth (%)": calculate_growth_rate(revenue),
            "Gross Margin (%)": safe_divide(gross_profit, revenue) * 100,
            "Operating Margin (%)": safe_divide(operating_income, revenue) * 100,
            "EBITDA Margin (%)": safe_divide(ebitda, revenue) * 100,
            "Net Margin (%)": safe_divide(net_income, revenue) * 100,
            "Return on Assets (%)": safe_divide(net_income, average_assets) * 100,
            "Return on Equity (%)": safe_divide(net_income, average_equity) * 100,
            "Current Ratio": safe_divide(current_assets, current_liabilities),
            "Debt-to-Equity": safe_divide(total_debt, equity),
            "Debt-to-Assets (%)": safe_divide(total_debt, total_assets) * 100,
            "Cash-to-Debt": safe_divide(cash, total_debt),
            "Operating Cash Flow Margin (%)": safe_divide(operating_cash_flow, revenue) * 100,
            "Free Cash Flow Margin (%)": safe_divide(free_cash_flow, revenue) * 100,
        }
    ).T

    return ratios.dropna(axis=0, how="all")


def create_financial_insights(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    ratios: pd.DataFrame,
) -> list[str]:
    """Generate restrained descriptive observations, not investment recommendations."""

    insights: list[str] = []

    if not income_statement.empty and "Revenue" in income_statement.index:
        growth = calculate_growth_rate(income_statement.loc["Revenue"]).dropna()
        if not growth.empty:
            value = float(growth.iloc[-1])
            direction = "increased" if value >= 0 else "decreased"
            insights.append(
                f"Revenue {direction} by {abs(value):.1f}% in the latest comparable period."
            )

    for ratio_name, description in (
        ("Net Margin (%)", "Latest net margin"),
        ("Return on Equity (%)", "Latest return on equity"),
        ("Current Ratio", "Latest current ratio"),
        ("Debt-to-Equity", "Latest debt-to-equity"),
    ):
        if not ratios.empty and ratio_name in ratios.index:
            series = ratios.loc[ratio_name].dropna()
            if not series.empty:
                value = float(series.iloc[-1])
                suffix = "%" if "(%)" in ratio_name else ""
                insights.append(f"{description}: {value:.2f}{suffix}.")

    if not cash_flow.empty and "Free Cash Flow" in cash_flow.index:
        series = cash_flow.loc["Free Cash Flow"].dropna()
        if not series.empty:
            description = "positive" if float(series.iloc[-1]) > 0 else "negative"
            insights.append(f"Latest reported free cash flow was {description}.")

    if not insights:
        insights.append(
            "There was not enough consistent statement data to generate automatic observations."
        )

    return insights



def _first_finite(*values: Any) -> float:
    """Return the first finite numeric value, otherwise NaN."""

    for value in values:
        numeric = safe_float(value)
        if np.isfinite(numeric):
            return numeric
    return np.nan


def _latest_statement_value(
    statement: pd.DataFrame,
    alternatives: list[str],
) -> float:
    """Return the latest available value for one accounting line."""

    if statement.empty:
        return np.nan

    series = extract_line(statement, alternatives)
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.iloc[-1]) if not series.empty else np.nan


def _average_latest_statement_value(
    statement: pd.DataFrame,
    alternatives: list[str],
) -> float:
    """Return the average of the latest two available balance-sheet values."""

    if statement.empty:
        return np.nan

    series = extract_line(statement, alternatives)
    series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        return np.nan
    if len(series) == 1:
        return float(series.iloc[-1])
    return float(series.iloc[-2:].mean())


def _safe_history(
    ticker_object: yf.Ticker,
    **kwargs: Any,
) -> pd.DataFrame:
    """Call yfinance history with a compatibility fallback."""

    try:
        data = ticker_object.history(repair=True, **kwargs)
    except TypeError:
        data = ticker_object.history(**kwargs)
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()
    return data.copy()


def _latest_share_count(ticker_object: yf.Ticker) -> float:
    """Retrieve the latest available share count when Yahoo provides it."""

    try:
        shares = ticker_object.get_shares_full(
            start=date.today() - timedelta(days=730)
        )
        if isinstance(shares, pd.Series):
            shares = pd.to_numeric(shares, errors="coerce").dropna()
            if not shares.empty:
                return float(shares.iloc[-1])
    except Exception:
        pass

    return np.nan


def _trailing_dividend_yield(
    ticker_object: yf.Ticker,
    current_price: float,
) -> float:
    """Estimate trailing-twelve-month dividend yield from paid dividends."""

    if not np.isfinite(current_price) or current_price <= 0:
        return np.nan

    try:
        dividends = ticker_object.dividends
        if not isinstance(dividends, pd.Series) or dividends.empty:
            return np.nan

        dividends = pd.to_numeric(dividends, errors="coerce").dropna()
        index = pd.DatetimeIndex(pd.to_datetime(dividends.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        dividends.index = index

        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        trailing_amount = float(dividends.loc[dividends.index >= cutoff].sum())
        return trailing_amount / current_price if trailing_amount > 0 else np.nan
    except Exception:
        return np.nan


def _estimate_market_beta(ticker: str) -> float:
    """Estimate five-year weekly beta against the S&P 500 as a fallback."""

    try:
        stock_history = _safe_history(
            yf.Ticker(ticker),
            period="5y",
            interval="1wk",
            auto_adjust=True,
            actions=False,
        )
        market_history = _safe_history(
            yf.Ticker("^GSPC"),
            period="5y",
            interval="1wk",
            auto_adjust=True,
            actions=False,
        )

        if stock_history.empty or market_history.empty:
            return np.nan
        if "Close" not in stock_history or "Close" not in market_history:
            return np.nan

        prices = pd.concat(
            [
                pd.to_numeric(stock_history["Close"], errors="coerce").rename("stock"),
                pd.to_numeric(market_history["Close"], errors="coerce").rename("market"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) < 52:
            return np.nan

        market_variance = float(returns["market"].var(ddof=1))
        if not np.isfinite(market_variance) or market_variance <= 0:
            return np.nan

        covariance = float(returns["stock"].cov(returns["market"]))
        return covariance / market_variance
    except Exception:
        return np.nan



def _normalise_timestamp(value: Any) -> pd.Timestamp | None:
    """Convert a date-like value to a timezone-naive pandas Timestamp."""

    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None

    if pd.isna(timestamp):
        return None

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)

    return timestamp


def _timestamp_from_epoch(value: Any) -> pd.Timestamp | None:
    """Convert a Unix timestamp returned by Yahoo Finance."""

    numeric = safe_float(value)
    if not np.isfinite(numeric):
        return None

    try:
        return pd.Timestamp(numeric, unit="s", tz="UTC").tz_localize(None)
    except Exception:
        return None


def _statement_series(
    statement: pd.DataFrame,
    alternatives: list[str],
) -> pd.Series:
    """Return a cleaned accounting series ordered by reporting date."""

    if statement is None or statement.empty:
        return pd.Series(dtype=float)

    series = pd.to_numeric(
        extract_line(statement, alternatives),
        errors="coerce",
    ).dropna()

    if series.empty:
        return pd.Series(dtype=float)

    dated_values: list[tuple[pd.Timestamp, float]] = []

    for index, value in series.items():
        timestamp = _normalise_timestamp(index)
        if timestamp is not None and np.isfinite(safe_float(value)):
            dated_values.append((timestamp, float(value)))

    if not dated_values:
        return pd.Series(dtype=float)

    dated_values.sort(key=lambda item: item[0])
    return pd.Series(
        [value for _, value in dated_values],
        index=[timestamp for timestamp, _ in dated_values],
        dtype=float,
    )


def _ttm_or_latest_annual_value(
    quarterly_statement: pd.DataFrame,
    annual_statement: pd.DataFrame,
    alternatives: list[str],
) -> tuple[float, pd.Timestamp | None, str]:
    """Prefer the latest four quarterly observations; otherwise use latest annual."""

    quarterly = _statement_series(quarterly_statement, alternatives)

    if len(quarterly) >= 4:
        latest_four = quarterly.iloc[-4:]
        return (
            float(latest_four.sum()),
            _normalise_timestamp(latest_four.index[-1]),
            "TTM from the latest four reported quarters",
        )

    annual = _statement_series(annual_statement, alternatives)

    if not annual.empty:
        return (
            float(annual.iloc[-1]),
            _normalise_timestamp(annual.index[-1]),
            "Latest available annual period",
        )

    return np.nan, None, "Unavailable"


def _latest_balance_value_with_period(
    quarterly_statement: pd.DataFrame,
    annual_statement: pd.DataFrame,
    alternatives: list[str],
) -> tuple[float, pd.Timestamp | None, str]:
    """Prefer the latest quarterly balance; otherwise use the latest annual balance."""

    quarterly = _statement_series(quarterly_statement, alternatives)

    if not quarterly.empty:
        return (
            float(quarterly.iloc[-1]),
            _normalise_timestamp(quarterly.index[-1]),
            "Latest available quarterly balance",
        )

    annual = _statement_series(annual_statement, alternatives)

    if not annual.empty:
        return (
            float(annual.iloc[-1]),
            _normalise_timestamp(annual.index[-1]),
            "Latest available annual balance",
        )

    return np.nan, None, "Unavailable"


def _average_balance_value_with_period(
    quarterly_statement: pd.DataFrame,
    annual_statement: pd.DataFrame,
    alternatives: list[str],
) -> tuple[float, pd.Timestamp | None, str]:
    """Use the latest two quarterly balances where possible."""

    quarterly = _statement_series(quarterly_statement, alternatives)

    if len(quarterly) >= 2:
        latest_two = quarterly.iloc[-2:]
        return (
            float(latest_two.mean()),
            _normalise_timestamp(latest_two.index[-1]),
            "Average of the latest two quarterly balances",
        )

    if len(quarterly) == 1:
        return (
            float(quarterly.iloc[-1]),
            _normalise_timestamp(quarterly.index[-1]),
            "Latest available quarterly balance",
        )

    annual = _statement_series(annual_statement, alternatives)

    if len(annual) >= 2:
        latest_two = annual.iloc[-2:]
        return (
            float(latest_two.mean()),
            _normalise_timestamp(latest_two.index[-1]),
            "Average of the latest two annual balances",
        )

    if len(annual) == 1:
        return (
            float(annual.iloc[-1]),
            _normalise_timestamp(annual.index[-1]),
            "Latest available annual balance",
        )

    return np.nan, None, "Unavailable"


def _latest_share_count_with_date(
    ticker_object: yf.Ticker,
) -> tuple[float, pd.Timestamp | None]:
    """Return the latest Yahoo share count and its observation date."""

    try:
        shares = ticker_object.get_shares_full(
            start=date.today() - timedelta(days=730)
        )
        if isinstance(shares, pd.Series):
            shares = pd.to_numeric(shares, errors="coerce").dropna()
            if not shares.empty:
                timestamp = _normalise_timestamp(shares.index[-1])
                return float(shares.iloc[-1]), timestamp
    except Exception:
        pass

    return np.nan, None


def _profile_metric_metadata(
    source: str,
    retrieved_at: str,
    *,
    as_of: pd.Timestamp | str | None = None,
    period: pd.Timestamp | str | None = None,
    method: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Create one consistent provenance record for a snapshot metric."""

    def serialise(value: pd.Timestamp | str | None) -> str | None:
        if value is None:
            return None
        timestamp = _normalise_timestamp(value)
        if timestamp is not None:
            return timestamp.isoformat()
        text = str(value).strip()
        return text or None

    return {
        "source": source,
        "retrieved_at": retrieved_at,
        "as_of": serialise(as_of),
        "period": serialise(period),
        "method": method,
        "note": note,
    }


def _unavailable_metric_metadata(
    retrieved_at: str,
    note: str,
) -> dict[str, Any]:
    return _profile_metric_metadata(
        "Unavailable",
        retrieved_at,
        note=note,
    )


def fetch_company_profile(
    ticker: str,
    ticker_object: yf.Ticker | None = None,
    annual_income: pd.DataFrame | None = None,
    annual_balance: pd.DataFrame | None = None,
    quarterly_income: pd.DataFrame | None = None,
    quarterly_balance: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the latest available company data with metric-level provenance."""

    ticker = ticker.strip().upper()
    ticker_object = ticker_object or yf.Ticker(ticker)

    retrieved_timestamp = datetime.now(timezone.utc)
    retrieved_at = retrieved_timestamp.isoformat()
    retrieved_date = pd.Timestamp(retrieved_timestamp).tz_convert("UTC").tz_localize(None)

    info: dict[str, Any] = {}
    rich_info_available = False
    detailed_info_error: str | None = None

    try:
        result = ticker_object.get_info()
        if isinstance(result, dict):
            info.update(result)
            rich_info_available = any(
                result.get(key) is not None
                for key in (
                    "longName",
                    "sector",
                    "industry",
                    "trailingPE",
                    "beta",
                )
            )
    except Exception as error:
        detailed_info_error = str(error) or error.__class__.__name__

    try:
        metadata = ticker_object.get_history_metadata()
        if isinstance(metadata, dict):
            for source_key, target_key in (
                ("currency", "currency"),
                ("regularMarketPrice", "regularMarketPrice"),
                ("regularMarketTime", "regularMarketTime"),
                ("fiftyTwoWeekHigh", "fiftyTwoWeekHigh"),
                ("fiftyTwoWeekLow", "fiftyTwoWeekLow"),
                ("exchangeName", "exchange"),
                ("fullExchangeName", "fullExchangeName"),
                ("instrumentType", "quoteType"),
                ("longName", "longName"),
                ("shortName", "shortName"),
            ):
                value = metadata.get(source_key)
                if value is not None:
                    info.setdefault(target_key, value)
    except Exception:
        pass

    try:
        fast_info = ticker_object.fast_info
        aliases = {
            "currency": ("currency",),
            "last_price": ("last_price", "lastPrice"),
            "market_cap": ("market_cap", "marketCap"),
            "year_high": ("year_high", "yearHigh"),
            "year_low": ("year_low", "yearLow"),
            "exchange": ("exchange",),
        }

        for target_key, possible_keys in aliases.items():
            value = None
            for key in possible_keys:
                value = getattr(fast_info, key, None)
                if value is None and hasattr(fast_info, "get"):
                    value = fast_info.get(key)
                if value is not None:
                    break
            if value is not None:
                info.setdefault(target_key, value)
    except Exception:
        pass

    # Yahoo price history is also used to timestamp the latest market observation.
    one_year_history = _safe_history(
        ticker_object,
        period="1y",
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if not one_year_history.empty:
        one_year_history.index = pd.DatetimeIndex(
            pd.to_datetime(one_year_history.index)
        )
        if one_year_history.index.tz is not None:
            one_year_history.index = one_year_history.index.tz_localize(None)
        one_year_history = one_year_history.sort_index()

    latest_market_date: pd.Timestamp | None = None
    historical_close = np.nan

    if "Close" in one_year_history:
        close_series = pd.to_numeric(
            one_year_history["Close"],
            errors="coerce",
        ).dropna()
        if not close_series.empty:
            historical_close = float(close_series.iloc[-1])
            latest_market_date = _normalise_timestamp(close_series.index[-1])

    quote_timestamp = _timestamp_from_epoch(info.get("regularMarketTime"))
    market_as_of = quote_timestamp or latest_market_date or retrieved_date

    direct_current_price = _first_finite(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("last_price"),
        info.get("lastPrice"),
    )
    current_price = (
        direct_current_price
        if np.isfinite(direct_current_price)
        else historical_close
    )

    metric_metadata: dict[str, dict[str, Any]] = {}

    if np.isfinite(direct_current_price):
        metric_metadata["current_price"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=market_as_of,
            method="Latest available Yahoo market quote.",
        )
    elif np.isfinite(historical_close):
        metric_metadata["current_price"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=latest_market_date,
            method="Latest available closing price from Yahoo Finance price history.",
        )
    else:
        metric_metadata["current_price"] = _unavailable_metric_metadata(
            retrieved_at,
            "Yahoo Finance did not return a valid current or historical closing price.",
        )

    direct_shares = _first_finite(info.get("sharesOutstanding"))
    shares_outstanding = direct_shares
    shares_date: pd.Timestamp | None = None

    direct_market_cap = _first_finite(
        info.get("marketCap"),
        info.get("market_cap"),
    )
    market_cap = direct_market_cap

    if not np.isfinite(shares_outstanding) and not np.isfinite(market_cap):
        shares_outstanding, shares_date = _latest_share_count_with_date(ticker_object)

    if np.isfinite(direct_market_cap):
        metric_metadata["market_cap"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=market_as_of,
            method="Market capitalization returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(current_price)
        and np.isfinite(shares_outstanding)
    ):
        market_cap = current_price * shares_outstanding
        metric_metadata["market_cap"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            as_of=market_as_of,
            period=shares_date,
            method="Latest available price × latest available shares outstanding.",
        )
    else:
        metric_metadata["market_cap"] = _unavailable_metric_metadata(
            retrieved_at,
            "A valid market price and share count were not both available.",
        )

    if annual_income is None:
        annual_income = fetch_statement(ticker_object, "income", "yearly")
    if annual_balance is None:
        annual_balance = fetch_statement(ticker_object, "balance", "yearly")
    if quarterly_income is None:
        quarterly_income = fetch_statement(ticker_object, "income", "quarterly")
    if quarterly_balance is None:
        quarterly_balance = fetch_statement(ticker_object, "balance", "quarterly")

    net_income, net_income_period, net_income_basis = _ttm_or_latest_annual_value(
        quarterly_income,
        annual_income,
        INCOME_STATEMENT_LINES["Net Income"],
    )
    ebitda, ebitda_period, ebitda_basis = _ttm_or_latest_annual_value(
        quarterly_income,
        annual_income,
        INCOME_STATEMENT_LINES["EBITDA"],
    )

    total_equity, equity_period, equity_basis = _latest_balance_value_with_period(
        quarterly_balance,
        annual_balance,
        BALANCE_SHEET_LINES["Shareholders' Equity"],
    )
    average_equity, average_equity_period, average_equity_basis = (
        _average_balance_value_with_period(
            quarterly_balance,
            annual_balance,
            BALANCE_SHEET_LINES["Shareholders' Equity"],
        )
    )
    average_assets, average_assets_period, average_assets_basis = (
        _average_balance_value_with_period(
            quarterly_balance,
            annual_balance,
            BALANCE_SHEET_LINES["Total Assets"],
        )
    )
    total_debt, debt_period, debt_basis = _latest_balance_value_with_period(
        quarterly_balance,
        annual_balance,
        BALANCE_SHEET_LINES["Total Debt"],
    )
    cash, cash_period, cash_basis = _latest_balance_value_with_period(
        quarterly_balance,
        annual_balance,
        BALANCE_SHEET_LINES["Cash and Equivalents"],
    )

    direct_enterprise_value = _first_finite(info.get("enterpriseValue"))
    enterprise_value = direct_enterprise_value

    if np.isfinite(direct_enterprise_value):
        metric_metadata["enterprise_value"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Enterprise value returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(market_cap)
        and np.isfinite(total_debt)
        and np.isfinite(cash)
    ):
        enterprise_value = market_cap + total_debt - cash
        balance_period = debt_period or cash_period
        metric_metadata["enterprise_value"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            as_of=market_as_of,
            period=balance_period,
            method=(
                "Market capitalization + total debt − cash and equivalents. "
                f"Balance data basis: {debt_basis if debt_basis != 'Unavailable' else cash_basis}."
            ),
        )
    else:
        metric_metadata["enterprise_value"] = _unavailable_metric_metadata(
            retrieved_at,
            "Yahoo Finance did not provide enterprise value and its components were incomplete.",
        )

    direct_trailing_pe = _first_finite(info.get("trailingPE"))
    trailing_pe = direct_trailing_pe

    if np.isfinite(direct_trailing_pe):
        metric_metadata["trailing_pe"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Trailing P/E returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(market_cap)
        and np.isfinite(net_income)
        and net_income > 0
    ):
        trailing_pe = market_cap / net_income
        metric_metadata["trailing_pe"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            as_of=market_as_of,
            period=net_income_period,
            method=f"Market capitalization ÷ net income. Income basis: {net_income_basis}.",
        )
    else:
        metric_metadata["trailing_pe"] = _unavailable_metric_metadata(
            retrieved_at,
            "A positive trailing net-income figure was not available.",
        )

    forward_pe = safe_float(info.get("forwardPE"))
    if np.isfinite(forward_pe):
        metric_metadata["forward_pe"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Forward P/E returned directly by Yahoo Finance.",
            note="Forward estimates may change when analyst forecasts are revised.",
        )
    else:
        metric_metadata["forward_pe"] = _unavailable_metric_metadata(
            retrieved_at,
            "Yahoo Finance did not return a current forward P/E estimate.",
        )

    direct_price_to_book = _first_finite(info.get("priceToBook"))
    price_to_book = direct_price_to_book

    if np.isfinite(direct_price_to_book):
        metric_metadata["price_to_book"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Price-to-book returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(market_cap)
        and np.isfinite(total_equity)
        and total_equity > 0
    ):
        price_to_book = market_cap / total_equity
        metric_metadata["price_to_book"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            as_of=market_as_of,
            period=equity_period,
            method=f"Market capitalization ÷ shareholders' equity. Equity basis: {equity_basis}.",
        )
    else:
        metric_metadata["price_to_book"] = _unavailable_metric_metadata(
            retrieved_at,
            "Positive shareholders' equity was not available.",
        )

    direct_enterprise_to_ebitda = _first_finite(info.get("enterpriseToEbitda"))
    enterprise_to_ebitda = direct_enterprise_to_ebitda

    if np.isfinite(direct_enterprise_to_ebitda):
        metric_metadata["enterprise_to_ebitda"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="EV/EBITDA returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(enterprise_value)
        and np.isfinite(ebitda)
        and ebitda > 0
    ):
        enterprise_to_ebitda = enterprise_value / ebitda
        metric_metadata["enterprise_to_ebitda"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            as_of=market_as_of,
            period=ebitda_period,
            method=f"Enterprise value ÷ EBITDA. EBITDA basis: {ebitda_basis}.",
        )
    else:
        metric_metadata["enterprise_to_ebitda"] = _unavailable_metric_metadata(
            retrieved_at,
            "A positive EBITDA figure or enterprise value was not available.",
        )

    direct_dividend_yield = _first_finite(info.get("dividendYield"))
    dividend_yield = direct_dividend_yield

    if np.isfinite(direct_dividend_yield):
        metric_metadata["dividend_yield"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Dividend yield returned directly by Yahoo Finance.",
        )
    else:
        dividend_yield = _trailing_dividend_yield(ticker_object, current_price)
        if np.isfinite(dividend_yield):
            trailing_end = latest_market_date or retrieved_date
            trailing_start = trailing_end - pd.Timedelta(days=365)
            metric_metadata["dividend_yield"] = _profile_metric_metadata(
                "Finance Bro — Calculated",
                retrieved_at,
                as_of=market_as_of,
                period=f"{trailing_start.date().isoformat()} to {trailing_end.date().isoformat()}",
                method="Cash dividends paid over the trailing 12 months ÷ latest available price.",
            )
        else:
            metric_metadata["dividend_yield"] = _unavailable_metric_metadata(
                retrieved_at,
                "Yahoo Finance did not return a yield and sufficient dividend history was unavailable.",
            )

    direct_return_on_equity = _first_finite(info.get("returnOnEquity"))
    return_on_equity = direct_return_on_equity

    if np.isfinite(direct_return_on_equity):
        metric_metadata["return_on_equity"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Return on equity returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(net_income)
        and np.isfinite(average_equity)
        and average_equity != 0
    ):
        return_on_equity = net_income / average_equity
        period = net_income_period or average_equity_period
        metric_metadata["return_on_equity"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            period=period,
            method=(
                "Net income ÷ average shareholders' equity. "
                f"Income basis: {net_income_basis}; equity basis: {average_equity_basis}."
            ),
        )
    else:
        metric_metadata["return_on_equity"] = _unavailable_metric_metadata(
            retrieved_at,
            "Net income and average shareholders' equity were not both available.",
        )

    direct_return_on_assets = _first_finite(info.get("returnOnAssets"))
    return_on_assets = direct_return_on_assets

    if np.isfinite(direct_return_on_assets):
        metric_metadata["return_on_assets"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Return on assets returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(net_income)
        and np.isfinite(average_assets)
        and average_assets != 0
    ):
        return_on_assets = net_income / average_assets
        period = net_income_period or average_assets_period
        metric_metadata["return_on_assets"] = _profile_metric_metadata(
            "Finance Bro — Calculated",
            retrieved_at,
            period=period,
            method=(
                "Net income ÷ average total assets. "
                f"Income basis: {net_income_basis}; asset basis: {average_assets_basis}."
            ),
        )
    else:
        metric_metadata["return_on_assets"] = _unavailable_metric_metadata(
            retrieved_at,
            "Net income and average total assets were not both available.",
        )

    direct_beta = _first_finite(info.get("beta"))
    beta = direct_beta
    beta_source = "Yahoo"

    if np.isfinite(direct_beta):
        metric_metadata["beta"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=retrieved_date,
            method="Beta returned directly by Yahoo Finance.",
        )
    else:
        beta = _estimate_market_beta(ticker)
        if np.isfinite(beta):
            beta_source = "Estimated"
            metric_metadata["beta"] = _profile_metric_metadata(
                "Finance Bro — Estimated",
                retrieved_at,
                as_of=latest_market_date or retrieved_date,
                period="Latest five years of weekly returns",
                method="Covariance with S&P 500 weekly returns ÷ S&P 500 return variance.",
                note="This estimate may differ from Yahoo's beta because methodology and sampling can differ.",
            )
        else:
            beta_source = "Unavailable"
            metric_metadata["beta"] = _unavailable_metric_metadata(
                retrieved_at,
                "Yahoo Finance did not return beta and there was insufficient history to estimate it.",
            )

    direct_fifty_two_week_high = _first_finite(
        info.get("fiftyTwoWeekHigh"),
        info.get("year_high"),
        info.get("yearHigh"),
    )
    direct_fifty_two_week_low = _first_finite(
        info.get("fiftyTwoWeekLow"),
        info.get("year_low"),
        info.get("yearLow"),
    )

    fifty_two_week_high = direct_fifty_two_week_high
    fifty_two_week_low = direct_fifty_two_week_low

    if not np.isfinite(fifty_two_week_high) and "High" in one_year_history:
        high = pd.to_numeric(one_year_history["High"], errors="coerce").dropna()
        if not high.empty:
            fifty_two_week_high = float(high.max())

    if not np.isfinite(fifty_two_week_low) and "Low" in one_year_history:
        low = pd.to_numeric(one_year_history["Low"], errors="coerce").dropna()
        if not low.empty:
            fifty_two_week_low = float(low.min())

    if (
        np.isfinite(direct_fifty_two_week_high)
        and np.isfinite(direct_fifty_two_week_low)
    ):
        metric_metadata["fifty_two_week_range"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=market_as_of,
            period="Trailing 52 weeks",
            method="52-week high and low returned directly by Yahoo Finance.",
        )
    elif (
        np.isfinite(fifty_two_week_high)
        and np.isfinite(fifty_two_week_low)
    ):
        metric_metadata["fifty_two_week_range"] = _profile_metric_metadata(
            "Yahoo Finance — Direct",
            retrieved_at,
            as_of=latest_market_date or retrieved_date,
            period="Latest available one-year price history",
            method="Highest daily high and lowest daily low in Yahoo Finance price history.",
        )
    else:
        metric_metadata["fifty_two_week_range"] = _unavailable_metric_metadata(
            retrieved_at,
            "A complete one-year high/low range was not available.",
        )

    trading_currency = (
        info.get("currency")
        or info.get("financialCurrency")
        or "N/A"
    )

    latest_financial_periods = [
        value
        for value in (
            net_income_period,
            ebitda_period,
            equity_period,
            debt_period,
            cash_period,
        )
        if value is not None
    ]
    latest_financial_period = (
        max(latest_financial_periods)
        if latest_financial_periods
        else None
    )

    fallback_metrics = [
        key
        for key, metadata_record in metric_metadata.items()
        if metadata_record.get("source") in {
            "Finance Bro — Calculated",
            "Finance Bro — Estimated",
        }
    ]

    unavailable_metrics = [
        key
        for key, metadata_record in metric_metadata.items()
        if metadata_record.get("source") == "Unavailable"
    ]

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "quote_type": info.get("quoteType") or info.get("typeDisp") or "N/A",
        "sector": info.get("sector") or "N/A",
        "industry": info.get("industry") or "N/A",
        "country": info.get("country") or "N/A",
        "exchange": info.get("fullExchangeName") or info.get("exchange") or "N/A",
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "trading_currency": trading_currency,
        "financial_currency": info.get("financialCurrency") or trading_currency,
        "current_price": current_price,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_book": price_to_book,
        "enterprise_to_ebitda": enterprise_to_ebitda,
        "dividend_yield": dividend_yield,
        "return_on_equity": return_on_equity,
        "return_on_assets": return_on_assets,
        "beta": beta,
        "beta_source": beta_source,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "shares_outstanding": shares_outstanding,
        "metadata_limited": not rich_info_available,
        "metric_metadata": metric_metadata,
        "retrieved_at": retrieved_at,
        "latest_market_date": (
            latest_market_date.isoformat()
            if latest_market_date is not None
            else None
        ),
        "latest_financial_period": (
            latest_financial_period.isoformat()
            if latest_financial_period is not None
            else None
        ),
        "fallback_metrics": fallback_metrics,
        "unavailable_metrics": unavailable_metrics,
        "detailed_info_error": detailed_info_error,
    }


def fetch_financial_package(ticker: str) -> dict[str, Any]:
    """Retrieve annual and quarterly statements, selected lines and ratios."""

    ticker = ticker.strip().upper()
    ticker_object = yf.Ticker(ticker)

    raw: dict[str, pd.DataFrame] = {}
    selected: dict[str, pd.DataFrame] = {}
    ratios: dict[str, pd.DataFrame] = {}
    insights: dict[str, list[str]] = {}

    for frequency in ("yearly", "quarterly"):
        income_raw = fetch_statement(ticker_object, "income", frequency)
        balance_raw = fetch_statement(ticker_object, "balance", frequency)
        cash_raw = fetch_statement(ticker_object, "cashflow", frequency)

        income = create_selected_statement(income_raw, INCOME_STATEMENT_LINES)
        balance = create_selected_statement(balance_raw, BALANCE_SHEET_LINES)
        cash = create_selected_statement(cash_raw, CASH_FLOW_LINES)
        income, balance, cash = add_calculated_lines(income, balance, cash)

        raw[f"{frequency}_income"] = income_raw
        raw[f"{frequency}_balance"] = balance_raw
        raw[f"{frequency}_cashflow"] = cash_raw
        selected[f"{frequency}_income"] = income
        selected[f"{frequency}_balance"] = balance
        selected[f"{frequency}_cashflow"] = cash
        ratios[frequency] = calculate_financial_ratios(income, balance, cash)
        insights[frequency] = create_financial_insights(income, balance, cash, ratios[frequency])

    return {
        "ticker": ticker,
        "profile": fetch_company_profile(
            ticker,
            ticker_object=ticker_object,
            annual_income=raw.get("yearly_income"),
            annual_balance=raw.get("yearly_balance"),
            quarterly_income=raw.get("quarterly_income"),
            quarterly_balance=raw.get("quarterly_balance"),
        ),
        "raw": raw,
        "selected": selected,
        "ratios": ratios,
        "insights": insights,
    }

