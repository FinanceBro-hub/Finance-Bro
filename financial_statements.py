# -*- coding: utf-8 -*-
"""Financial-statement utilities for Finance Bro Stock Research.

The module keeps data acquisition and financial calculations separate from
Streamlit presentation. Missing values remain missing; they are never replaced
with zero merely for display purposes.
"""

from __future__ import annotations

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


def fetch_company_profile(ticker: str) -> dict[str, Any]:
    """Return a safe company profile dictionary from Yahoo metadata."""

    ticker_object = yf.Ticker(ticker)
    info: dict[str, Any] = {}

    try:
        result = ticker_object.get_info()
        if isinstance(result, dict):
            info.update(result)
    except Exception:
        pass

    try:
        fast_info = ticker_object.fast_info
        for key in (
            "currency",
            "last_price",
            "market_cap",
            "year_high",
            "year_low",
            "exchange",
        ):
            value = getattr(fast_info, key, None)
            if value is None and hasattr(fast_info, "get"):
                value = fast_info.get(key)
            if value is not None:
                info.setdefault(key, value)
    except Exception:
        pass

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "quote_type": info.get("quoteType") or info.get("typeDisp") or "N/A",
        "sector": info.get("sector") or "N/A",
        "industry": info.get("industry") or "N/A",
        "country": info.get("country") or "N/A",
        "exchange": info.get("fullExchangeName") or info.get("exchange") or "N/A",
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "trading_currency": info.get("currency") or info.get("financialCurrency") or "N/A",
        "financial_currency": info.get("financialCurrency") or info.get("currency") or "N/A",
        "current_price": safe_float(
            info.get("currentPrice", info.get("regularMarketPrice", info.get("last_price")))
        ),
        "market_cap": safe_float(info.get("marketCap", info.get("market_cap"))),
        "enterprise_value": safe_float(info.get("enterpriseValue")),
        "trailing_pe": safe_float(info.get("trailingPE")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "price_to_book": safe_float(info.get("priceToBook")),
        "enterprise_to_ebitda": safe_float(info.get("enterpriseToEbitda")),
        "dividend_yield": safe_float(info.get("dividendYield")),
        "return_on_equity": safe_float(info.get("returnOnEquity")),
        "return_on_assets": safe_float(info.get("returnOnAssets")),
        "beta": safe_float(info.get("beta")),
        "fifty_two_week_high": safe_float(info.get("fiftyTwoWeekHigh", info.get("year_high"))),
        "fifty_two_week_low": safe_float(info.get("fiftyTwoWeekLow", info.get("year_low"))),
        "shares_outstanding": safe_float(info.get("sharesOutstanding")),
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
        "profile": fetch_company_profile(ticker),
        "raw": raw,
        "selected": selected,
        "ratios": ratios,
        "insights": insights,
    }
