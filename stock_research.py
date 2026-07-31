# -*- coding: utf-8 -*-
"""Finance Bro — pre-portfolio stock research module.

This Streamlit module lets users inspect historical prices, compare selected
stocks with a benchmark, review company information, and examine annual or
quarterly financial statements before transferring tickers to Portfolio
Analysis.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
import html
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from financial_statements import (
    fetch_company_profile,
    fetch_financial_package,
)


BRAND_NAVY = "#0A1F44"
BRAND_BLUE = "#1663F0"
BRAND_GREEN = "#2DB24A"
BRAND_MUTED = "#64748B"
BRAND_BORDER = "#DDE7F5"

MAX_RESEARCH_TICKERS = 15

NEWS_PERIOD_OPTIONS = {
    "24 hours": pd.Timedelta(hours=24),
    "7 days": pd.Timedelta(days=7),
    "30 days": pd.Timedelta(days=30),
}

NEWS_TYPE_OPTIONS = {
    "All": "all",
    "News": "news",
    "Press Releases": "press releases",
}


BENCHMARK_OPTIONS = {
    "None": None,
    "S&P 500 (^GSPC)": "^GSPC",
    "EURO STOXX 50 (^STOXX50E)": "^STOXX50E",
    "Nasdaq Composite (^IXIC)": "^IXIC",
    "Dow Jones (^DJI)": "^DJI",
}

STATEMENT_LABELS = {
    "Income Statement": "income",
    "Balance Sheet": "balance",
    "Cash Flow Statement": "cashflow",
}

UNIT_OPTIONS = {
    "Raw units": 1.0,
    "Thousands": 1_000.0,
    "Millions": 1_000_000.0,
    "Billions": 1_000_000_000.0,
}


@st.cache_data(ttl=900, show_spinner=False)
def load_price_history(
    ticker: str,
    start_date_iso: str,
    end_date_iso: str | None,
) -> pd.DataFrame:
    """Download daily unadjusted and adjusted data for one ticker."""

    ticker = ticker.strip().upper()
    end_exclusive = None

    if end_date_iso:
        end_exclusive = (
            pd.Timestamp(end_date_iso)
            + pd.Timedelta(days=1)
        )

    base_kwargs = {
        "tickers": ticker,
        "start": pd.Timestamp(start_date_iso),
        "end": end_exclusive,
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "threads": False,
    }

    attempts = [
        {**base_kwargs, "repair": True, "multi_level_index": False},
        {**base_kwargs, "repair": True},
        base_kwargs,
    ]

    data = pd.DataFrame()
    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            data = yf.download(**kwargs)
            if isinstance(data, pd.DataFrame) and not data.empty:
                break
        except TypeError as error:
            last_error = error
            continue
        except Exception as error:
            last_error = error
            continue

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        message = f"No historical prices were returned for {ticker}."
        if last_error is not None:
            message += f" Data-provider message: {last_error}"
        raise ValueError(message)

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        selected = None
        for level in range(data.columns.nlevels):
            values = data.columns.get_level_values(level).astype(str)
            if ticker in set(values):
                try:
                    selected = data.xs(ticker, axis=1, level=level, drop_level=True)
                    break
                except Exception:
                    continue

        if selected is not None:
            data = selected
        else:
            data.columns = [
                " ".join(str(part) for part in column if str(part) != ticker).strip()
                for column in data.columns
            ]

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(column[0]) for column in data.columns]

    data.columns = [str(column).strip() for column in data.columns]
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index))

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data = (
        data.sort_index()
        .loc[~data.index.duplicated(keep="last")]
        .dropna(axis=0, how="all")
    )

    if "Close" not in data.columns:
        raise ValueError(f"Yahoo Finance did not return a closing-price series for {ticker}.")

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["Adj Close"].dropna().shape[0] < 2:
        raise ValueError(f"There are not enough valid price observations for {ticker}.")

    data.index.name = "Date"
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_profile(ticker: str) -> dict[str, Any]:
    return fetch_company_profile(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_package(ticker: str) -> dict[str, Any]:
    return fetch_financial_package(ticker)



@st.cache_data(ttl=600, show_spinner=False)
def load_company_news(
    ticker: str,
    news_tab: str,
    count: int = 50,
) -> dict[str, Any]:
    """
    Retrieve recent Yahoo Finance news for one ticker.

    Primary method:
        yfinance.Ticker.get_news()

    Transparent fallback:
        yfinance.Search().news

    The fallback is broader than the ticker-specific feed, so the
    interface identifies it whenever it is used.
    """

    ticker = ticker.strip().upper()
    retrieved_at = pd.Timestamp.now(tz="UTC")

    primary_error = ""
    fallback_error = ""
    articles: list[dict[str, Any]] = []
    retrieval_method = "Yahoo Finance ticker feed"
    used_search_fallback = False

    try:
        ticker_object = yf.Ticker(ticker)

        try:
            raw_news = ticker_object.get_news(
                count=count,
                tab=news_tab,
            )
        except TypeError:
            # Compatibility with older yfinance releases.
            raw_news = ticker_object.get_news(
                count=count
            )

        if isinstance(raw_news, list):
            articles = [
                article
                for article in raw_news
                if isinstance(article, dict)
            ]

    except Exception as error:
        primary_error = str(error)

    if not articles:
        try:
            try:
                search_object = yf.Search(
                    ticker,
                    max_results=1,
                    news_count=count,
                    enable_fuzzy_query=False,
                )
            except TypeError:
                search_object = yf.Search(
                    ticker,
                    max_results=1,
                    news_count=count,
                )

            raw_search_news = getattr(
                search_object,
                "news",
                [],
            )

            if isinstance(raw_search_news, list):
                articles = [
                    article
                    for article in raw_search_news
                    if isinstance(article, dict)
                ]

            if articles:
                retrieval_method = (
                    "Yahoo Finance search fallback"
                )
                used_search_fallback = True

        except Exception as error:
            fallback_error = str(error)

    return {
        "ticker": ticker,
        "requested_tab": news_tab,
        "articles": articles,
        "retrieved_at": retrieved_at,
        "retrieval_method": retrieval_method,
        "used_search_fallback": used_search_fallback,
        "primary_error": primary_error,
        "fallback_error": fallback_error,
    }


def clean_news_text(value: Any) -> str:
    """Normalize publisher-supplied text without changing its meaning."""

    if value is None:
        return ""

    cleaned = html.unescape(
        re.sub(
            r"<[^>]+>",
            " ",
            str(value),
        )
    )

    return re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()


def escape_markdown_text(value: str) -> str:
    """Escape external text before inserting it into Markdown."""

    escaped = value.replace("\\", "\\\\")

    for character in (
        "*",
        "_",
        "[",
        "]",
        "(",
        ")",
        "#",
        "`",
        ">",
    ):
        escaped = escaped.replace(
            character,
            f"\\{character}",
        )

    return escaped


def parse_news_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse Unix or ISO publication timestamps into UTC."""

    if value in (None, ""):
        return None

    try:
        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            numeric_value = float(value)
            unit = (
                "ms"
                if abs(numeric_value) >= 1_000_000_000_000
                else "s"
            )

            timestamp = pd.to_datetime(
                numeric_value,
                unit=unit,
                utc=True,
                errors="coerce",
            )
        else:
            timestamp = pd.to_datetime(
                value,
                utc=True,
                errors="coerce",
            )

    except Exception:
        return None

    if pd.isna(timestamp):
        return None

    return pd.Timestamp(timestamp)


def extract_news_url(value: Any) -> str:
    """Extract and validate an external article URL."""

    if isinstance(value, dict):
        value = (
            value.get("url")
            or value.get("href")
            or ""
        )

    url = clean_news_text(value)

    if url.startswith(
        (
            "https://",
            "http://",
        )
    ):
        return url

    return ""


def normalize_news_article(
    article: dict[str, Any],
    requested_type: str,
) -> dict[str, Any] | None:
    """
    Normalize both current and legacy yfinance news response formats.
    """

    nested_content = article.get("content")

    if isinstance(nested_content, dict):
        payload = nested_content
    else:
        payload = article

    title = clean_news_text(
        payload.get("title")
        or article.get("title")
    )

    if not title:
        return None

    summary = clean_news_text(
        payload.get("summary")
        or payload.get("description")
        or article.get("summary")
        or article.get("description")
    )

    provider_value = (
        payload.get("provider")
        or article.get("provider")
    )

    if isinstance(provider_value, dict):
        publisher = clean_news_text(
            provider_value.get("displayName")
            or provider_value.get("name")
            or provider_value.get("title")
        )
    else:
        publisher = clean_news_text(
            provider_value
        )

    if not publisher:
        publisher = clean_news_text(
            article.get("publisher")
            or payload.get("publisher")
        )

    if not publisher:
        publisher = "Publisher not supplied"

    published_at = parse_news_timestamp(
        payload.get("pubDate")
        or payload.get("providerPublishTime")
        or payload.get("publishedAt")
        or article.get("pubDate")
        or article.get("providerPublishTime")
        or article.get("publishedAt")
    )

    url_candidates = [
        payload.get("clickThroughUrl"),
        payload.get("canonicalUrl"),
        payload.get("previewUrl"),
        payload.get("link"),
        payload.get("url"),
        article.get("clickThroughUrl"),
        article.get("canonicalUrl"),
        article.get("link"),
        article.get("url"),
    ]

    article_url = ""

    for candidate in url_candidates:
        article_url = extract_news_url(
            candidate
        )

        if article_url:
            break

    content_type = clean_news_text(
        payload.get("contentType")
        or payload.get("type")
        or article.get("contentType")
        or article.get("type")
    )

    type_text = content_type.lower()

    is_press_release = (
        "press" in type_text
        or requested_type == "Press Releases"
    )

    article_type = (
        "Press Release"
        if is_press_release
        else "News"
    )

    identifier = clean_news_text(
        payload.get("id")
        or payload.get("uuid")
        or article.get("id")
        or article.get("uuid")
    )

    return {
        "id": identifier,
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "published_at": published_at,
        "url": article_url,
        "article_type": article_type,
        "is_press_release": is_press_release,
    }


def prepare_news_articles(
    raw_articles: list[dict[str, Any]],
    period_label: str,
    type_label: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Filter, deduplicate and order news articles."""

    now_utc = pd.Timestamp.now(tz="UTC")
    period_delta = NEWS_PERIOD_OPTIONS[
        period_label
    ]
    earliest_allowed = (
        now_utc
        - period_delta
    )

    prepared: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for raw_article in raw_articles:
        normalized = normalize_news_article(
            raw_article,
            type_label,
        )

        if normalized is None:
            continue

        published_at = normalized[
            "published_at"
        ]

        # A publication date is required because the user selected
        # a time window and Finance Bro should not guess article age.
        if published_at is None:
            continue

        if published_at < earliest_allowed:
            continue

        if (
            type_label == "News"
            and normalized["is_press_release"]
        ):
            continue

        if (
            type_label == "Press Releases"
            and not normalized["is_press_release"]
        ):
            continue

        deduplication_key = (
            normalized["url"].strip().lower()
            or normalized["id"].strip().lower()
            or re.sub(
                r"[^a-z0-9]+",
                "",
                normalized["title"].lower(),
            )
        )

        if not deduplication_key:
            continue

        if deduplication_key in seen_keys:
            continue

        seen_keys.add(
            deduplication_key
        )
        prepared.append(
            normalized
        )

    prepared.sort(
        key=lambda item: item["published_at"],
        reverse=True,
    )

    return prepared[:limit]


def render_latest_news(
    valid_tickers: list[str],
) -> None:
    """Render the transparent Latest News workspace."""

    st.subheader("Latest Company News")
    st.caption(
        "Review recent external coverage before completing the "
        "company analysis. Headlines and summaries are publisher-"
        "supplied metadata and are not investment recommendations."
    )

    control_1, control_2, control_3 = st.columns(
        [1.2, 1.0, 1.0]
    )

    with control_1:
        news_ticker = st.selectbox(
            "Latest News Company",
            options=valid_tickers,
            key="stock_research_news_ticker",
        )

    with control_2:
        news_period = st.selectbox(
            "Publication Period",
            options=list(
                NEWS_PERIOD_OPTIONS.keys()
            ),
            index=1,
            key="stock_research_news_period",
        )

    with control_3:
        news_type_label = st.selectbox(
            "Content Type",
            options=list(
                NEWS_TYPE_OPTIONS.keys()
            ),
            index=0,
            key="stock_research_news_type",
        )

    requested_tab = NEWS_TYPE_OPTIONS[
        news_type_label
    ]

    with st.spinner(
        f"Loading recent news for {news_ticker}..."
    ):
        news_package = load_company_news(
            news_ticker,
            requested_tab,
            count=50,
        )

    retrieved_at = pd.Timestamp(
        news_package["retrieved_at"]
    )

    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.tz_localize(
            "UTC"
        )
    else:
        retrieved_at = retrieved_at.tz_convert(
            "UTC"
        )

    if news_package[
        "used_search_fallback"
    ]:
        st.warning(
            "The ticker-specific Yahoo Finance feed did not return "
            "usable articles. Finance Bro is showing the Yahoo Finance "
            "search fallback, which may include broader company-related "
            "coverage."
        )

    raw_articles = news_package[
        "articles"
    ]

    articles = prepare_news_articles(
        raw_articles,
        news_period,
        news_type_label,
        limit=8,
    )

    st.caption(
        f"Retrieved through {news_package['retrieval_method']} on "
        f"{retrieved_at.strftime('%d %b %Y, %H:%M UTC')}. "
        "Duplicate links are removed and articles without a verifiable "
        "publication timestamp are excluded."
    )

    if not articles:
        st.info(
            "No dated articles matched the selected company, period "
            "and content type. Try 30 days or choose All."
        )

        errors = [
            news_package.get(
                "primary_error",
                "",
            ),
            news_package.get(
                "fallback_error",
                "",
            ),
        ]

        errors = [
            error
            for error in errors
            if error
        ]

        if errors:
            with st.expander(
                "Data-provider details",
                expanded=False,
            ):
                for error in errors:
                    st.code(
                        error
                    )

        return

    for article in articles:
        with st.container(
            border=True
        ):
            content_column, action_column = st.columns(
                [5.0, 1.25],
                vertical_alignment="top",
            )

            with content_column:
                st.markdown(
                    "#### "
                    + escape_markdown_text(
                        article["title"]
                    )
                )

                published_text = article[
                    "published_at"
                ].tz_convert(
                    "UTC"
                ).strftime(
                    "%d %b %Y, %H:%M UTC"
                )

                st.caption(
                    f"{article['publisher']} · "
                    f"{published_text} · "
                    f"{article['article_type']}"
                )

                if article[
                    "summary"
                ]:
                    st.write(
                        article["summary"]
                    )
                else:
                    st.caption(
                        "The publisher did not supply a summary in "
                        "the available news metadata."
                    )

                st.caption(
                    f"Source: {article['publisher']} · "
                    f"Published: {published_text} · "
                    f"Retrieved by Finance Bro: "
                    f"{retrieved_at.strftime('%d %b %Y, %H:%M UTC')}"
                )

            with action_column:
                if article[
                    "url"
                ]:
                    st.link_button(
                        "Read full article",
                        article["url"],
                        use_container_width=True,
                    )
                else:
                    st.caption(
                        "Original link unavailable"
                    )

    with st.expander(
        "News data and interpretation",
        expanded=False,
    ):
        st.write(
            "Finance Bro does not rewrite headlines or infer whether "
            "an article is bullish or bearish. Summaries shown here are "
            "the descriptions supplied in Yahoo Finance's news metadata. "
            "Open the original publisher's article for full context."
        )
        st.write(
            "News availability varies by company, market, publisher and "
            "region. Publication times are displayed in UTC. A recent "
            "article can explain market attention, but it does not prove "
            "that the news caused a price movement."
        )


def parse_tickers(raw_input: str) -> list[str]:
    """Normalize and deduplicate ticker input while preserving order."""

    tickers: list[str] = []

    for raw_ticker in raw_input.replace(";", ",").split(","):
        ticker = raw_ticker.strip().upper()

        if ticker and ticker not in tickers:
            tickers.append(ticker)

    return tickers


def format_number(value: Any, decimals: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not np.isfinite(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def format_percent(value: Any, input_is_decimal: bool = False, decimals: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not np.isfinite(value):
        return "N/A"

    if input_is_decimal:
        value *= 100

    return f"{value:,.{decimals}f}%"


def format_large_value(value: Any, currency: str = "") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not np.isfinite(value):
        return "N/A"

    absolute = abs(value)
    prefix = f"{currency} " if currency and currency != "N/A" else ""

    if absolute >= 1_000_000_000_000:
        return f"{prefix}{value / 1_000_000_000_000:,.2f}T"
    if absolute >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{prefix}{value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"{prefix}{value / 1_000:,.2f}K"

    return f"{prefix}{value:,.2f}"


def get_price_series(history: pd.DataFrame, price_basis: str = "Adjusted Close") -> pd.Series:
    column = "Adj Close" if price_basis == "Adjusted Close" else "Close"
    if column not in history.columns or history[column].dropna().empty:
        column = "Close"
    return pd.to_numeric(history[column], errors="coerce").dropna()


def calculate_price_metrics(price_series: pd.Series) -> dict[str, Any]:
    price_series = pd.to_numeric(price_series, errors="coerce").dropna()

    if len(price_series) < 2:
        return {
            "start_price": np.nan,
            "end_price": np.nan,
            "cumulative_return": np.nan,
            "annualized_volatility": np.nan,
            "maximum_drawdown": np.nan,
            "observations": len(price_series),
            "start_date": pd.NaT,
            "end_date": pd.NaT,
        }

    returns = price_series.pct_change(fill_method=None).dropna()
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    return {
        "start_price": float(price_series.iloc[0]),
        "end_price": float(price_series.iloc[-1]),
        "cumulative_return": float((price_series.iloc[-1] / price_series.iloc[0] - 1) * 100),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252) * 100),
        "maximum_drawdown": float(drawdown.min() * 100),
        "observations": int(len(price_series)),
        "start_date": price_series.index.min(),
        "end_date": price_series.index.max(),
    }


def build_price_summary(
    histories: dict[str, pd.DataFrame],
    price_basis: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ticker, history in histories.items():
        metrics = calculate_price_metrics(get_price_series(history, price_basis))
        rows.append(
            {
                "Ticker": ticker,
                "Start Date": metrics["start_date"],
                "End Date": metrics["end_date"],
                "Start Price": metrics["start_price"],
                "End Price": metrics["end_price"],
                "Cumulative Return (%)": metrics["cumulative_return"],
                "Annualized Volatility (%)": metrics["annualized_volatility"],
                "Maximum Drawdown (%)": metrics["maximum_drawdown"],
                "Price Observations": metrics["observations"],
            }
        )

    return pd.DataFrame(rows)


def build_comparison_prices(
    histories: dict[str, pd.DataFrame],
    selected_tickers: list[str],
    price_basis: str,
    strict_common_dates: bool,
) -> pd.DataFrame:
    series_list: list[pd.Series] = []

    for ticker in selected_tickers:
        if ticker not in histories:
            continue
        series = get_price_series(histories[ticker], price_basis).rename(ticker)
        series_list.append(series)

    if not series_list:
        return pd.DataFrame()

    combined = pd.concat(series_list, axis=1).sort_index()

    if strict_common_dates:
        combined = combined.dropna(axis=0, how="any")
    else:
        combined = combined.dropna(axis=0, how="all")

    return combined


def transform_price_view(prices: pd.DataFrame, view: str) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    if view == "Actual Price":
        return prices.copy()

    if view == "Normalized to 100":
        first_valid = prices.apply(lambda column: column.dropna().iloc[0] if not column.dropna().empty else np.nan)
        return prices.divide(first_valid, axis=1) * 100

    returns = prices.pct_change(fill_method=None)

    if view == "Cumulative Return":
        return ((1 + returns).cumprod() - 1) * 100

    wealth = (1 + returns).cumprod()
    return (wealth / wealth.cummax() - 1) * 100


def calculate_benchmark_comparison(
    asset_prices: pd.Series,
    benchmark_prices: pd.Series,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [asset_prices.rename("Asset"), benchmark_prices.rename("Benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 3:
        return {
            "asset_return": np.nan,
            "benchmark_return": np.nan,
            "active_return": np.nan,
            "beta": np.nan,
            "correlation": np.nan,
            "r_squared": np.nan,
            "observations": len(aligned),
        }

    cumulative_asset = aligned["Asset"].iloc[-1] / aligned["Asset"].iloc[0] - 1
    cumulative_benchmark = aligned["Benchmark"].iloc[-1] / aligned["Benchmark"].iloc[0] - 1

    returns = aligned.pct_change(fill_method=None).dropna()
    benchmark_variance = returns["Benchmark"].var(ddof=1)

    beta = (
        returns["Asset"].cov(returns["Benchmark"]) / benchmark_variance
        if benchmark_variance and np.isfinite(benchmark_variance)
        else np.nan
    )
    correlation = returns["Asset"].corr(returns["Benchmark"])

    return {
        "asset_return": float(cumulative_asset * 100),
        "benchmark_return": float(cumulative_benchmark * 100),
        "active_return": float((cumulative_asset - cumulative_benchmark) * 100),
        "beta": float(beta) if np.isfinite(beta) else np.nan,
        "correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        "r_squared": float(correlation**2) if np.isfinite(correlation) else np.nan,
        "observations": int(len(returns)),
    }


def make_line_chart(
    dataframe: pd.DataFrame,
    title: str,
    y_axis_title: str,
    hover_suffix: str = "",
) -> go.Figure:
    figure = go.Figure()

    for column in dataframe.columns:
        figure.add_trace(
            go.Scatter(
                x=dataframe.index,
                y=dataframe[column],
                mode="lines",
                name=str(column),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>"
                    + str(column)
                    + ": %{y:,.2f}"
                    + hover_suffix
                    + "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=title,
        height=560,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        legend_title_text="Series",
        margin=dict(l=45, r=25, t=80, b=45),
    )

    return figure


def make_volume_chart(
    history: pd.DataFrame,
    ticker: str,
    period_label: str = "1 year",
) -> tuple[go.Figure | None, str]:
    """Create a readable volume chart with period filtering and aggregation."""

    if "Volume" not in history.columns or history["Volume"].dropna().empty:
        return None, "Trading-volume data is unavailable for this company."

    volume = history[["Volume"]].copy()
    volume.index = pd.to_datetime(volume.index)
    volume["Volume"] = pd.to_numeric(volume["Volume"], errors="coerce")
    volume = volume.dropna(subset=["Volume"]).sort_index()

    if volume.empty:
        return None, "Trading-volume data is unavailable for this company."

    period_days = {
        "3 months": 93,
        "6 months": 186,
        "1 year": 366,
        "3 years": 1_096,
        "All available": None,
    }

    selected_days = period_days.get(period_label, 366)
    latest_date = volume.index.max()

    if selected_days is not None:
        start_date = latest_date - pd.Timedelta(days=selected_days)
        volume = volume.loc[volume.index >= start_date]

    if volume.empty:
        return None, "There are no volume observations in the selected period."

    span_days = max((volume.index.max() - volume.index.min()).days, 1)

    if span_days <= 550:
        chart_data = volume.copy()
        frequency_label = "Daily"
        average_window = 20
        average_label = "20-session average"
        aggregation_note = "Each bar represents one trading session."
    elif span_days <= 1_500:
        chart_data = volume.resample("W-FRI").sum(min_count=1).dropna()
        frequency_label = "Weekly"
        average_window = 4
        average_label = "4-week average"
        aggregation_note = "Daily volumes are aggregated into weekly totals."
    else:
        chart_data = volume.resample("MS").sum(min_count=1).dropna()
        frequency_label = "Monthly"
        average_window = 3
        average_label = "3-month average"
        aggregation_note = "Daily volumes are aggregated into monthly totals."

    chart_data["Average Volume"] = (
        chart_data["Volume"]
        .rolling(
            window=average_window,
            min_periods=max(2, average_window // 3),
        )
        .mean()
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            name=f"{frequency_label} volume",
            marker_color="#2563EB",
            marker_line_color="#1D4ED8",
            marker_line_width=0.35,
            opacity=0.90,
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>Volume: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Average Volume"],
            mode="lines",
            name=average_label,
            line=dict(
                color="#0A1F44",
                width=3.0,
            ),
            hovertemplate=(
                "%{x|%d %b %Y}"
                "<br>Average volume: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=f"{ticker} — {frequency_label} Trading Volume",
        height=430,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Volume",
        legend_title_text="Series",
        bargap=0.08,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0A1F44"),
        margin=dict(l=45, r=25, t=75, b=45),
    )

    figure.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
    )
    figure.update_yaxes(
        tickformat=".3s",
        rangemode="tozero",
        gridcolor="#D9E2F2",
        zerolinecolor="#AFC0D8",
    )

    first_date = chart_data.index.min().strftime("%d %b %Y")
    last_date = chart_data.index.max().strftime("%d %b %Y")
    note = (
        f"{aggregation_note} Showing {first_date} to {last_date}. "
        f"The line shows the {average_label.lower()}."
    )

    return figure, note


def scale_statement_for_display(
    statement: pd.DataFrame,
    scale: float,
) -> pd.DataFrame:
    display = statement.copy().astype(float)

    for row in display.index:
        row_name = str(row).lower()
        if "eps" in row_name or "ratio" in row_name or "margin" in row_name:
            continue
        display.loc[row] = display.loc[row] / scale

    display.columns = [
        column.strftime("%d/%m/%Y") if isinstance(column, pd.Timestamp) else str(column)
        for column in display.columns
    ]

    return display


def statement_styler(dataframe: pd.DataFrame):
    return (
        dataframe.style
        .format(lambda value: "N/A" if pd.isna(value) else f"{value:,.2f}")
        .set_properties(**{"text-align": "center", "vertical-align": "middle"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
    )


def render_research_welcome() -> None:
    st.markdown("## Stock Research")
    st.write(
        "Investigate price history, benchmark behaviour and the latest annual or "
        "quarterly financial statements before deciding which stocks belong in "
        "the portfolio."
    )

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        st.info(
            "**1 — Explore prices**\n\nCompare actual prices, normalized performance, "
            "cumulative returns and drawdowns."
        )
    with column_2:
        st.info(
            "**2 — Review the business**\n\nInspect company information, valuation data, "
            "financial statements, ratios and trends."
        )
    with column_3:
        st.info(
            "**3 — Build the portfolio**\n\nTransfer the researched tickers directly to "
            "Portfolio Analysis."
        )

    st.caption(
        "Enter up to 15 tickers in the sidebar and press Run Stock Research."
    )



def _format_snapshot_date(value: Any, include_time: bool = False) -> str:
    """Format ISO dates used by Company Snapshot provenance labels."""

    if value is None or str(value).strip() in ("", "None", "NaT"):
        return ""

    text = str(value).strip()

    # Preserve descriptive periods such as "Latest five years of weekly returns".
    if not any(character.isdigit() for character in text):
        return text

    try:
        timestamp = pd.Timestamp(text)
        if pd.isna(timestamp):
            return text
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        if include_time:
            return timestamp.strftime("%d %b %Y, %H:%M UTC")
        return timestamp.strftime("%d %b %Y")
    except Exception:
        # Date ranges and other transparent free-text periods stay unchanged.
        return text


def _snapshot_metadata_line(metadata: dict[str, Any]) -> str:
    """Create the compact source-and-date label shown below each metric."""

    source = str(metadata.get("source") or "Unavailable")
    parts = [source]

    as_of = _format_snapshot_date(metadata.get("as_of"))
    period = _format_snapshot_date(metadata.get("period"))
    retrieved = _format_snapshot_date(
        metadata.get("retrieved_at"),
        include_time=True,
    )

    if as_of:
        parts.append(f"As of {as_of}")
    if period:
        parts.append(f"Period: {period}")
    if not as_of and not period and retrieved:
        parts.append(f"Checked {retrieved}")

    return " · ".join(parts)


def _render_snapshot_metric(
    label: str,
    value: str,
    metadata: dict[str, Any],
) -> None:
    """Render one metric with its provenance directly underneath."""

    st.metric(label=label, value=value)
    st.caption(_snapshot_metadata_line(metadata))



def render_company_snapshot(ticker: str, profile: dict[str, Any]) -> None:
    st.subheader(f"{profile['name']} ({ticker})")

    caption_parts = [
        str(profile.get(key))
        for key in ("quote_type", "exchange", "sector", "industry", "country")
        if str(profile.get(key) or "").strip() not in ("", "N/A", "None")
    ]
    st.caption(" · ".join(caption_parts) if caption_parts else "Company metadata unavailable")

    st.info(
        "**Data provenance:** each metric shows whether it came directly from Yahoo Finance, "
        "was calculated or estimated by Finance Bro, or was unavailable. Market dates and "
        "financial reporting periods are shown separately because they may not be the same."
    )

    currency = str(profile.get("trading_currency") or "")
    metadata = profile.get("metric_metadata") or {}

    metric_columns = st.columns(4)
    metrics = [
        (
            "Current Price",
            format_large_value(profile.get("current_price"), currency),
            "current_price",
        ),
        (
            "Market Capitalization",
            format_large_value(profile.get("market_cap"), currency),
            "market_cap",
        ),
        (
            "Enterprise Value",
            format_large_value(profile.get("enterprise_value"), currency),
            "enterprise_value",
        ),
        (
            "Beta",
            format_number(profile.get("beta"), 3),
            "beta",
        ),
    ]

    for column, (label, value, metadata_key) in zip(metric_columns, metrics):
        with column:
            _render_snapshot_metric(
                label,
                value,
                metadata.get(metadata_key, {}),
            )

    valuation_columns = st.columns(4)
    valuation_metrics = [
        (
            "Trailing P/E",
            format_number(profile.get("trailing_pe"), 2),
            "trailing_pe",
        ),
        (
            "Forward P/E",
            format_number(profile.get("forward_pe"), 2),
            "forward_pe",
        ),
        (
            "Price-to-Book",
            format_number(profile.get("price_to_book"), 2),
            "price_to_book",
        ),
        (
            "EV / EBITDA",
            format_number(profile.get("enterprise_to_ebitda"), 2),
            "enterprise_to_ebitda",
        ),
    ]

    for column, (label, value, metadata_key) in zip(
        valuation_columns,
        valuation_metrics,
    ):
        with column:
            _render_snapshot_metric(
                label,
                value,
                metadata.get(metadata_key, {}),
            )

    quality_columns = st.columns(4)
    quality_metrics = [
        (
            "Dividend Yield",
            format_percent(
                profile.get("dividend_yield"),
                input_is_decimal=True,
            ),
            "dividend_yield",
        ),
        (
            "Return on Equity",
            format_percent(
                profile.get("return_on_equity"),
                input_is_decimal=True,
            ),
            "return_on_equity",
        ),
        (
            "Return on Assets",
            format_percent(
                profile.get("return_on_assets"),
                input_is_decimal=True,
            ),
            "return_on_assets",
        ),
        (
            "52-Week Range",
            f"{format_number(profile.get('fifty_two_week_low'))} — "
            f"{format_number(profile.get('fifty_two_week_high'))}",
            "fifty_two_week_range",
        ),
    ]

    for column, (label, value, metadata_key) in zip(
        quality_columns,
        quality_metrics,
    ):
        with column:
            _render_snapshot_metric(
                label,
                value,
                metadata.get(metadata_key, {}),
            )

    transparent_methods = {
        "current_price": "Current Price",
        "market_cap": "Market Capitalization",
        "enterprise_value": "Enterprise Value",
        "beta": "Beta",
        "trailing_pe": "Trailing P/E",
        "forward_pe": "Forward P/E",
        "price_to_book": "Price-to-Book",
        "enterprise_to_ebitda": "EV / EBITDA",
        "dividend_yield": "Dividend Yield",
        "return_on_equity": "Return on Equity",
        "return_on_assets": "Return on Assets",
        "fifty_two_week_range": "52-Week Range",
    }

    with st.expander("Sources, dates and calculation methods", expanded=False):
        for metadata_key, display_label in transparent_methods.items():
            record = metadata.get(metadata_key, {})
            source_line = _snapshot_metadata_line(record)
            method = str(record.get("method") or "").strip()
            note = str(record.get("note") or "").strip()

            st.markdown(f"**{display_label}**")
            st.caption(source_line)

            if method:
                st.write(method)
            if note:
                st.caption(note)

    fallback_count = len(profile.get("fallback_metrics") or [])
    unavailable_count = len(profile.get("unavailable_metrics") or [])

    if fallback_count:
        st.warning(
            f"{fallback_count} metric(s) could not be obtained directly from Yahoo Finance. "
            "Finance Bro used transparent calculations or estimates and labelled each one above."
        )

    if unavailable_count:
        st.caption(
            f"{unavailable_count} metric(s) remain unavailable because Finance Bro did not have "
            "enough reliable information to calculate them without making unsupported assumptions."
        )

    if profile.get("summary"):
        with st.expander("Company Description", expanded=False):
            st.write(profile["summary"])

    retrieved_text = _format_snapshot_date(
        profile.get("retrieved_at"),
        include_time=True,
    )
    latest_market_text = _format_snapshot_date(profile.get("latest_market_date"))
    latest_financial_text = _format_snapshot_date(
        profile.get("latest_financial_period")
    )

    timing_lines = [
        f"**Trading currency:** {profile['trading_currency']}",
        (
            "**Financial-statement reporting currency:** "
            f"{profile['financial_currency']}"
        ),
    ]

    if latest_market_text:
        timing_lines.append(f"**Latest Yahoo market date:** {latest_market_text}")
    if latest_financial_text:
        timing_lines.append(
            f"**Latest financial period used by available calculations:** "
            f"{latest_financial_text}"
        )
    if retrieved_text:
        timing_lines.append(f"**Data retrieved:** {retrieved_text}")

    st.info("  \n".join(timing_lines))


def render_financial_statement_table(
    package: dict[str, Any],
    frequency: str,
    statement_key: str,
    unit_label: str,
) -> None:
    statement = package["selected"].get(f"{frequency}_{statement_key}", pd.DataFrame())

    if statement.empty:
        st.warning(
            "No consistent statement data was returned for this company and frequency. "
            "This is common for ETFs, indices, some foreign listings and incomplete Yahoo records."
        )
        return

    scale = UNIT_OPTIONS[unit_label]
    display = scale_statement_for_display(statement, scale)
    financial_currency = package["profile"].get("financial_currency", "N/A")

    latest_period = statement.columns[-1]
    latest_period_text = (
        latest_period.strftime("%d/%m/%Y")
        if isinstance(latest_period, pd.Timestamp)
        else str(latest_period)
    )

    st.caption(
        f"Reporting currency: {financial_currency} · Display units: {unit_label} · "
        f"Latest available period in this table: {latest_period_text}. "
        "EPS rows remain per-share values. Missing observations are shown as N/A, not zero."
    )

    st.dataframe(
        statement_styler(display),
        use_container_width=True,
        height=min(760, 80 + 36 * len(display.index)),
    )


def render_financial_trends(
    package: dict[str, Any],
    frequency: str,
    unit_label: str,
) -> None:
    combined_rows: dict[str, pd.Series] = {}

    for statement_key in ("income", "balance", "cashflow"):
        statement = package["selected"].get(f"{frequency}_{statement_key}", pd.DataFrame())
        if statement.empty:
            continue
        for row_name in statement.index:
            combined_rows.setdefault(str(row_name), statement.loc[row_name])

    if not combined_rows:
        st.warning("No financial trends are available for this selection.")
        return

    preferred = [
        metric
        for metric in (
            "Revenue",
            "Gross Profit",
            "Operating Income",
            "EBITDA",
            "Net Income",
            "Operating Cash Flow",
            "Free Cash Flow",
            "Cash and Equivalents",
            "Total Debt",
        )
        if metric in combined_rows
    ][:5]

    selected_metrics = st.multiselect(
        "Financial Metrics to Plot",
        options=list(combined_rows.keys()),
        default=preferred,
        key=f"financial_trend_metrics_{package['ticker']}_{frequency}",
    )

    if selected_metrics:
        scale = UNIT_OPTIONS[unit_label]
        rows: list[dict[str, Any]] = []

        for metric in selected_metrics:
            series = pd.to_numeric(combined_rows[metric], errors="coerce").dropna()
            metric_scale = 1.0 if "eps" in metric.lower() else scale
            for period, value in series.items():
                rows.append(
                    {
                        "Period": period,
                        "Metric": metric,
                        "Value": value / metric_scale,
                    }
                )

        chart_data = pd.DataFrame(rows)
        if not chart_data.empty:
            chart_data["Period"] = pd.to_datetime(chart_data["Period"], errors="coerce")
            chart_data = chart_data.sort_values("Period")
            figure = px.line(
                chart_data,
                x="Period",
                y="Value",
                color="Metric",
                markers=True,
                title=f"{package['ticker']} — Financial Statement Trends",
            )
            figure.update_layout(
                height=540,
                xaxis_title="Reporting Period",
                yaxis_title=f"Value ({unit_label})",
                hovermode="x unified",
            )
            figure.update_traces(hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>")
            st.plotly_chart(figure, use_container_width=True)

    ratios = package["ratios"].get(frequency, pd.DataFrame())

    if ratios.empty:
        st.info("There are not enough consistent statement lines to calculate financial ratios.")
    else:
        st.markdown("### Calculated Financial Ratios")
        ratio_display = ratios.copy()
        ratio_display.columns = [
            column.strftime("%d/%m/%Y") if isinstance(column, pd.Timestamp) else str(column)
            for column in ratio_display.columns
        ]
        st.dataframe(
            statement_styler(ratio_display),
            use_container_width=True,
            height=min(620, 80 + 36 * len(ratio_display.index)),
        )

        ratio_options = list(ratios.index)
        default_ratios = [
            ratio
            for ratio in (
                "Revenue Growth (%)",
                "Operating Margin (%)",
                "Net Margin (%)",
                "Return on Equity (%)",
                "Current Ratio",
                "Debt-to-Equity",
            )
            if ratio in ratio_options
        ][:4]

        selected_ratios = st.multiselect(
            "Ratios to Plot",
            options=ratio_options,
            default=default_ratios,
            key=f"financial_ratio_metrics_{package['ticker']}_{frequency}",
        )

        if selected_ratios:
            ratio_long = (
                ratios.loc[selected_ratios]
                .T
                .reset_index(names="Period")
                .melt(id_vars="Period", var_name="Ratio", value_name="Value")
                .dropna()
            )
            ratio_long["Period"] = pd.to_datetime(ratio_long["Period"], errors="coerce")
            ratio_long = ratio_long.sort_values("Period")

            figure = px.line(
                ratio_long,
                x="Period",
                y="Value",
                color="Ratio",
                markers=True,
                title=f"{package['ticker']} — Financial Ratio Trends",
            )
            figure.update_layout(
                height=520,
                xaxis_title="Reporting Period",
                yaxis_title="Ratio Value",
                hovermode="x unified",
            )
            st.plotly_chart(figure, use_container_width=True)

    st.markdown("### Automatic Financial Observations")
    for insight in package["insights"].get(frequency, []):
        st.write(f"• {insight}")

    st.caption(
        "These observations describe reported accounting data; they are not investment "
        "recommendations and do not replace analysis of company filings and footnotes."
    )


def render_stock_research() -> None:
    """Render Stock Research controls and the complete research workspace."""

    with st.sidebar:
        st.markdown("### Research Stocks")
        st.caption("Study the companies before choosing portfolio weights.")

        research_ticker_input = st.text_input(
            "Research Tickers",
            value="",
            placeholder="Example: AAPL, MSFT, GOOGL",
            help=(
                f"Enter up to {MAX_RESEARCH_TICKERS} Yahoo Finance "
                "ticker symbols separated by commas."
            ),
            key="stock_research_ticker_input",
        )

        research_start_date = st.date_input(
            "Research Start Date",
            value=pd.Timestamp("2021-01-01").date(),
            max_value=pd.Timestamp.today().date(),
            key="stock_research_start_date",
        )

        research_use_latest = st.checkbox(
            "Use latest available date",
            value=True,
            key="stock_research_use_latest",
        )

        if research_use_latest:
            research_end_date: date | None = None
            st.caption("End date: latest available market observation.")
        else:
            research_end_date = st.date_input(
                "Research End Date",
                value=pd.Timestamp.today().date(),
                min_value=research_start_date,
                max_value=pd.Timestamp.today().date(),
                key="stock_research_end_date",
            )

        benchmark_label = st.selectbox(
            "Research Benchmark",
            options=list(BENCHMARK_OPTIONS.keys()),
            index=1,
            help=(
                "Used for normalized comparison and exploratory return sensitivity. "
                "Stock Research does not convert local prices into one common currency."
            ),
            key="stock_research_benchmark_label",
        )

        run_research = st.button(
            "Run Stock Research",
            type="primary",
            use_container_width=True,
            key="run_stock_research_button",
        )

    current_tickers = parse_tickers(research_ticker_input)
    ticker_count = len(current_tickers)

    st.sidebar.caption(
        f"{ticker_count}/{MAX_RESEARCH_TICKERS} unique tickers entered."
    )

    too_many_tickers = ticker_count > MAX_RESEARCH_TICKERS

    if too_many_tickers:
        excess_tickers = ticker_count - MAX_RESEARCH_TICKERS

        st.sidebar.error(
            f"Maximum allowed: {MAX_RESEARCH_TICKERS} tickers. "
            f"Remove {excess_tickers} ticker"
            f"{'s' if excess_tickers != 1 else ''} before running "
            "Stock Research."
        )

    if run_research:
        if not current_tickers:
            st.error("Enter at least one valid ticker.")
            return

        if too_many_tickers:
            st.error(
                f"Stock Research accepts a maximum of "
                f"{MAX_RESEARCH_TICKERS} unique tickers."
            )
            return

        if not research_use_latest and research_end_date <= research_start_date:
            st.error("Research End Date must be later than Research Start Date.")
            return

        st.session_state["stock_research_config"] = {
            "tickers": current_tickers,
            "start_date": research_start_date.isoformat(),
            "end_date": None if research_use_latest else research_end_date.isoformat(),
            "benchmark": BENCHMARK_OPTIONS[benchmark_label],
            "benchmark_label": benchmark_label,
        }

    config = st.session_state.get("stock_research_config")

    if not config:
        render_research_welcome()
        return

    requested_tickers: list[str] = config["tickers"]
    benchmark_ticker: str | None = config["benchmark"]

    histories: dict[str, pd.DataFrame] = {}
    download_errors: dict[str, str] = {}

    with st.spinner("Downloading price history and preparing Stock Research..."):
        for ticker in requested_tickers:
            try:
                histories[ticker] = load_price_history(
                    ticker,
                    config["start_date"],
                    config["end_date"],
                )
            except Exception as error:
                download_errors[ticker] = str(error)

        benchmark_history = None
        if benchmark_ticker:
            try:
                benchmark_history = load_price_history(
                    benchmark_ticker,
                    config["start_date"],
                    config["end_date"],
                )
            except Exception as error:
                download_errors[benchmark_ticker] = str(error)

    valid_tickers = list(histories.keys())

    if not valid_tickers:
        st.error("No valid stock-price data was downloaded. Check the tickers and dates.")
        for ticker, message in download_errors.items():
            st.caption(f"{ticker}: {message}")
        return

    if current_tickers != requested_tickers:
        st.info(
            "The sidebar inputs have changed since the last completed research run. "
            "Press Run Stock Research to apply the new tickers or dates."
        )

    if download_errors:
        with st.expander("Data-Provider Warnings", expanded=False):
            for ticker, message in download_errors.items():
                st.warning(f"{ticker}: {message}")

    st.markdown("## Stock Research Workspace")
    st.caption(
        f"Price period: {config['start_date']} to "
        f"{config['end_date'] or 'latest available'} · Valid tickers: "
        f"{', '.join(valid_tickers)}"
    )

    (
        price_tab,
        snapshot_tab,
        news_tab,
        statements_tab,
        trends_tab,
        portfolio_tab,
    ) = st.tabs(
        [
            "Price Explorer",
            "Company Snapshot",
            "Latest News",
            "Financial Statements",
            "Trends & Ratios",
            "Build Portfolio",
        ]
    )

    with price_tab:
        control_column_1, control_column_2, control_column_3 = st.columns(3)

        with control_column_1:
            default_chart_tickers = valid_tickers[
                : min(5, len(valid_tickers))
            ]

            chart_tickers = st.multiselect(
                "Stocks in Price Chart",
                options=valid_tickers,
                default=default_chart_tickers,
                help=(
                    "Choose which researched companies appear in the chart. "
                    "The comparison tables still use every valid ticker."
                ),
                key="stock_research_chart_tickers",
            )

        with control_column_2:
            chart_view = st.selectbox(
                "Chart View",
                options=[
                    "Actual Price",
                    "Normalized to 100",
                    "Cumulative Return",
                    "Drawdown",
                ],
                index=1,
                key="stock_research_chart_view",
            )

        with control_column_3:
            price_basis = st.selectbox(
                "Price Basis",
                options=["Adjusted Close", "Close"],
                index=0,
                help=(
                    "Adjusted Close incorporates Yahoo's historical adjustment for "
                    "corporate actions. Close is the unadjusted closing-price series."
                ),
                key="stock_research_price_basis",
            )

        strict_common_dates = chart_view != "Actual Price"
        chart_histories = dict(histories)
        chart_labels = list(chart_tickers)

        if benchmark_ticker and benchmark_history is not None and chart_view != "Actual Price":
            include_benchmark = st.checkbox(
                f"Include benchmark: {config['benchmark_label']}",
                value=True,
                key="stock_research_include_benchmark",
            )
            if include_benchmark:
                chart_histories[benchmark_ticker] = benchmark_history
                chart_labels.append(benchmark_ticker)

        comparison_prices = build_comparison_prices(
            chart_histories,
            chart_labels,
            price_basis,
            strict_common_dates=strict_common_dates,
        )

        if not chart_tickers:
            st.info(
                "Select at least one company in Stocks in Price Chart."
            )
        elif comparison_prices.empty or len(comparison_prices) < 2:
            st.warning(
                "There are not enough common observations for the "
                "selected chart."
            )
        else:
            transformed = transform_price_view(comparison_prices, chart_view)

            axis_labels = {
                "Actual Price": "Local Trading-Currency Price",
                "Normalized to 100": "Index (First Common Observation = 100)",
                "Cumulative Return": "Cumulative Return (%)",
                "Drawdown": "Drawdown (%)",
            }
            suffixes = {
                "Actual Price": "",
                "Normalized to 100": "",
                "Cumulative Return": "%",
                "Drawdown": "%",
            }

            figure = make_line_chart(
                transformed,
                title=f"Stock Price Research — {chart_view}",
                y_axis_title=axis_labels[chart_view],
                hover_suffix=suffixes[chart_view],
            )
            st.plotly_chart(figure, use_container_width=True)

            if chart_view == "Actual Price" and len(chart_tickers) > 1:
                st.warning(
                    "Actual prices may be quoted in different currencies and are not directly "
                    "comparable. Use Normalized to 100 or Cumulative Return for relative "
                    "performance comparison."
                )
            elif strict_common_dates:
                st.caption(
                    "Comparison views use the strict intersection of available dates so every "
                    "line starts from the same common observation."
                )

        # ============================================================
        # STOCK COMPARISON TABLE
        # ============================================================

        st.markdown("### Stock Comparison")
        st.caption(
            "Compare the historical performance and risk of the "
            "selected companies."
        )

        # Annual risk-free rate used in the Sharpe Ratio
        risk_free_rate = 0.0385

        comparison_rows = []

        # Prepare benchmark prices, when available
        benchmark_prices_for_table = None

        if (
            benchmark_history is not None
            and isinstance(benchmark_history, pd.DataFrame)
            and not benchmark_history.empty
        ):
            benchmark_prices_for_table = get_price_series(
                benchmark_history,
                price_basis,
            ).dropna()

        # Calculate the metrics for every valid stock
        for ticker in valid_tickers:
            history = histories.get(ticker)

            if history is None or history.empty:
                continue

            prices = get_price_series(
                history,
                price_basis,
            ).dropna()

            if len(prices) < 2:
                continue

            returns = prices.pct_change(
                fill_method=None,
            ).dropna()

            if returns.empty:
                continue

            # Cumulative return
            cumulative_return = (
                prices.iloc[-1] / prices.iloc[0] - 1
            ) * 100

            # Annualized return (CAGR)
            elapsed_days = (
                prices.index[-1] - prices.index[0]
            ).days
            elapsed_years = elapsed_days / 365.25
            total_growth = prices.iloc[-1] / prices.iloc[0]

            if elapsed_years > 0 and total_growth > 0:
                annualized_return = (
                    total_growth ** (1 / elapsed_years) - 1
                ) * 100
            else:
                annualized_return = np.nan

            # Annualized volatility
            daily_volatility = returns.std(ddof=1)
            annualized_volatility = (
                daily_volatility * np.sqrt(252) * 100
            )

            # Sharpe Ratio
            daily_risk_free_rate = (
                (1 + risk_free_rate) ** (1 / 252) - 1
            )

            if pd.notna(daily_volatility) and daily_volatility > 0:
                sharpe_ratio = (
                    (returns.mean() - daily_risk_free_rate)
                    / daily_volatility
                    * np.sqrt(252)
                )
            else:
                sharpe_ratio = np.nan

            # Maximum drawdown
            running_maximum = prices.cummax()
            drawdown = prices / running_maximum - 1
            maximum_drawdown = drawdown.min() * 100

            # Beta and correlation with the benchmark
            beta = np.nan
            benchmark_correlation = np.nan

            if (
                benchmark_prices_for_table is not None
                and not benchmark_prices_for_table.empty
            ):
                benchmark_returns = (
                    benchmark_prices_for_table
                    .pct_change(fill_method=None)
                    .dropna()
                    .rename("Benchmark")
                )
                stock_returns = returns.rename("Stock")

                aligned_returns = pd.concat(
                    [stock_returns, benchmark_returns],
                    axis=1,
                    join="inner",
                ).dropna()

                if len(aligned_returns) >= 2:
                    benchmark_variance = aligned_returns[
                        "Benchmark"
                    ].var(ddof=1)

                    if (
                        pd.notna(benchmark_variance)
                        and benchmark_variance > 0
                    ):
                        beta = (
                            aligned_returns["Stock"].cov(
                                aligned_returns["Benchmark"]
                            )
                            / benchmark_variance
                        )

                    benchmark_correlation = aligned_returns[
                        "Stock"
                    ].corr(aligned_returns["Benchmark"])

            comparison_rows.append(
                {
                    "Ticker": ticker,
                    "Cumulative Return (%)": cumulative_return,
                    "Annualized Return (%)": annualized_return,
                    "Annualized Volatility (%)": annualized_volatility,
                    "Sharpe Ratio": sharpe_ratio,
                    "Maximum Drawdown (%)": maximum_drawdown,
                    "Beta": beta,
                    "Benchmark Correlation": benchmark_correlation,
                }
            )

        stock_comparison_table = pd.DataFrame(comparison_rows)

        if stock_comparison_table.empty:
            st.warning(
                "There is not enough historical data to create "
                "the stock comparison table."
            )
        else:
            stock_comparison_table = (
                stock_comparison_table
                .sort_values(
                    by="Sharpe Ratio",
                    ascending=False,
                    na_position="last",
                )
                .reset_index(drop=True)
            )

            stock_comparison_styler = (
                stock_comparison_table.style
                .format(
                    {
                        "Cumulative Return (%)": "{:,.2f}%",
                        "Annualized Return (%)": "{:,.2f}%",
                        "Annualized Volatility (%)": "{:,.2f}%",
                        "Sharpe Ratio": "{:,.2f}",
                        "Maximum Drawdown (%)": "{:,.2f}%",
                        "Beta": "{:,.2f}",
                        "Benchmark Correlation": "{:,.2f}",
                    },
                    na_rep="N/A",
                )
                .set_properties(
                    **{
                        "text-align": "center",
                        "vertical-align": "middle",
                    }
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("text-align", "center"),
                                ("vertical-align", "middle"),
                            ],
                        }
                    ]
                )
            )

            st.dataframe(
                stock_comparison_styler,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Sharpe Ratio calculated using an annual risk-free "
            f"rate of {risk_free_rate * 100:.2f}%. "
            "Results are historical and do not represent forecasts."
        )

        volume_company_column, volume_period_column = st.columns([1.25, 1.0])

        with volume_company_column:
            focus_ticker = st.selectbox(
                "Volume Chart Company",
                options=valid_tickers,
                key="stock_research_volume_ticker",
            )

        with volume_period_column:
            volume_period = st.selectbox(
                "Volume Chart Period",
                options=[
                    "3 months",
                    "6 months",
                    "1 year",
                    "3 years",
                    "All available",
                ],
                index=2,
                key="stock_research_volume_period",
                help=(
                    "Long periods are automatically aggregated into weekly "
                    "or monthly totals so the chart remains readable."
                ),
            )

        volume_figure, volume_note = make_volume_chart(
            histories[focus_ticker],
            focus_ticker,
            volume_period,
        )

        if volume_figure is not None:
            st.plotly_chart(volume_figure, use_container_width=True)
            st.caption(volume_note)
        else:
            st.info(volume_note)

        if benchmark_ticker and benchmark_history is not None:
            st.markdown("### Benchmark Comparison")
            benchmark_prices = get_price_series(benchmark_history, price_basis)
            comparison_rows = []

            for ticker in valid_tickers:
                values = calculate_benchmark_comparison(
                    get_price_series(histories[ticker], price_basis),
                    benchmark_prices,
                )
                comparison_rows.append(
                    {
                        "Ticker": ticker,
                        "Stock Cumulative Return (%)": values["asset_return"],
                        "Benchmark Cumulative Return (%)": values["benchmark_return"],
                        "Cumulative Active Return (p.p.)": values["active_return"],
                        "Exploratory Beta": values["beta"],
                        "Correlation": values["correlation"],
                        "R-Squared": values["r_squared"],
                        "Return Observations": values["observations"],
                    }
                )

            comparison_table = pd.DataFrame(comparison_rows)
            st.dataframe(
                comparison_table.style
                .format(
                    {
                        "Stock Cumulative Return (%)": "{:,.2f}%",
                        "Benchmark Cumulative Return (%)": "{:,.2f}%",
                        "Cumulative Active Return (p.p.)": "{:,.2f}",
                        "Exploratory Beta": "{:,.3f}",
                        "Correlation": "{:,.3f}",
                        "R-Squared": "{:,.3f}",
                        "Return Observations": "{:,.0f}",
                    },
                    na_rep="N/A",
                )
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]),
                use_container_width=True,
                hide_index=True,
            )
            st.info(
                "The Stock Research benchmark table is an exploratory local-price comparison. "
                "It does not apply the portfolio module's daily FX conversion or excess-return "
                "regression. Final portfolio beta, alpha and active performance are calculated "
                "more rigorously inside Portfolio Analysis."
            )

    with snapshot_tab:
        snapshot_ticker = st.selectbox(
            "Company",
            options=valid_tickers,
            key="stock_research_snapshot_ticker",
        )
        with st.spinner(f"Loading company information for {snapshot_ticker}..."):
            snapshot_package = load_financial_package(snapshot_ticker)
            profile = snapshot_package["profile"]
        render_company_snapshot(snapshot_ticker, profile)

    with news_tab:
        render_latest_news(
            valid_tickers
        )

    with statements_tab:
        statement_control_1, statement_control_2, statement_control_3 = st.columns(3)

        with statement_control_1:
            statement_ticker = st.selectbox(
                "Company",
                options=valid_tickers,
                key="stock_research_statement_ticker",
            )

        with statement_control_2:
            frequency_label = st.radio(
                "Reporting Frequency",
                options=["Annual", "Quarterly"],
                horizontal=True,
                key="stock_research_statement_frequency",
            )

        with statement_control_3:
            unit_label = st.selectbox(
                "Display Units",
                options=list(UNIT_OPTIONS.keys()),
                index=2,
                key="stock_research_statement_units",
            )

        frequency = "yearly" if frequency_label == "Annual" else "quarterly"

        with st.spinner(f"Loading financial statements for {statement_ticker}..."):
            package = load_financial_package(statement_ticker)

        profile = package["profile"]
        st.info(
            f"**{profile['name']} ({statement_ticker})**  \n"
            f"Statement reporting currency: **{profile['financial_currency']}**. "
            "The values are not converted into the portfolio currency."
        )

        income_tab, balance_tab, cashflow_tab = st.tabs(
            ["Income Statement", "Balance Sheet", "Cash Flow Statement"]
        )

        with income_tab:
            render_financial_statement_table(package, frequency, "income", unit_label)
        with balance_tab:
            render_financial_statement_table(package, frequency, "balance", unit_label)
        with cashflow_tab:
            render_financial_statement_table(package, frequency, "cashflow", unit_label)

        with st.expander("Data and Accounting Limitations", expanded=False):
            st.write(
                "Statement labels differ across companies and accounting standards. Finance Bro "
                "uses exact label matching first and cautious fallback matching second. A missing "
                "value remains N/A rather than being replaced with zero. Calculated Free Cash Flow "
                "equals Operating Cash Flow plus Capital Expenditure when Yahoo does not provide "
                "a direct Free Cash Flow line and CapEx is reported as a negative cash outflow."
            )
            st.write(
                "Yahoo data may be delayed, restated or incomplete. Important investment decisions "
                "should be checked against the company's official annual or quarterly filing and "
                "the accompanying notes."
            )

    with trends_tab:
        trend_control_1, trend_control_2, trend_control_3 = st.columns(3)

        with trend_control_1:
            trend_ticker = st.selectbox(
                "Company",
                options=valid_tickers,
                key="stock_research_trend_ticker",
            )
        with trend_control_2:
            trend_frequency_label = st.radio(
                "Trend Frequency",
                options=["Annual", "Quarterly"],
                horizontal=True,
                key="stock_research_trend_frequency",
            )
        with trend_control_3:
            trend_unit_label = st.selectbox(
                "Trend Units",
                options=list(UNIT_OPTIONS.keys()),
                index=2,
                key="stock_research_trend_units",
            )

        trend_frequency = "yearly" if trend_frequency_label == "Annual" else "quarterly"
        with st.spinner(f"Preparing financial trends for {trend_ticker}..."):
            trend_package = load_financial_package(trend_ticker)

        st.info(
            f"Financial trends are shown in {trend_package['profile']['financial_currency']} "
            f"and {trend_unit_label.lower()}. Ratios are not currency-converted."
        )
        render_financial_trends(trend_package, trend_frequency, trend_unit_label)

    with portfolio_tab:
        st.subheader("Use Researched Stocks in Portfolio Analysis")
        st.write(
            "Select the companies that passed your research process. Their ticker symbols will be "
            "copied to the Portfolio Analysis sidebar, where you can choose weights, investment, "
            "currency, benchmark and risk settings."
        )

        portfolio_selection = st.multiselect(
            "Stocks to Transfer",
            options=valid_tickers,
            default=valid_tickers,
            key="stock_research_portfolio_selection",
        )

        if st.button(
            "Use Selected Stocks in Portfolio",
            type="primary",
            use_container_width=True,
            key="stock_research_transfer_button",
        ):
            if not portfolio_selection:
                st.error("Select at least one stock to transfer.")
            else:
                st.session_state["portfolio_ticker_input_v2"] = ", ".join(portfolio_selection)
                st.session_state["pending_analysis_mode"] = "Portfolio Analysis"
                st.rerun()

        st.caption(
            "Only ticker symbols are transferred. Research results do not automatically determine "
            "portfolio weights or constitute an investment recommendation."
        )

    st.caption(
        "Market prices, company metadata and statements are retrieved through yfinance from Yahoo "
        "Finance. Availability and completeness vary by security and market."
    )

