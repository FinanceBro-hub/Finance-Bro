# -*- coding: utf-8 -*-
"""Finance Bro — Live Investment workspace."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from live_investment_database import (
    create_portfolio,
    delete_portfolio,
    delete_position,
    initialize_database,
    load_portfolios,
    load_positions,
    rename_portfolio,
    save_position,
)


LIVE_INVESTMENT_VERSION = "2.5"


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_market_data(ticker: str) -> dict[str, Any]:
    """Load the latest available daily market data."""

    clean_ticker = str(ticker).strip().upper()

    if not clean_ticker:
        raise ValueError("Ticker cannot be empty.")

    security = yf.Ticker(clean_ticker)

    try:
        history = security.history(
            period="7d",
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
        )
    except TypeError:
        history = security.history(
            period="7d",
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

    if history is None or history.empty:
        raise ValueError(
            f"No market data was returned for {clean_ticker}."
        )

    price_column = (
        "Adj Close"
        if "Adj Close" in history.columns
        and not history["Adj Close"].dropna().empty
        else "Close"
    )

    prices = pd.to_numeric(
        history[price_column],
        errors="coerce",
    ).dropna()

    if prices.empty:
        raise ValueError(
            f"No valid closing price was returned for {clean_ticker}."
        )

    current_price = float(prices.iloc[-1])

    previous_close = (
        float(prices.iloc[-2])
        if len(prices) >= 2
        else np.nan
    )

    daily_change = (
        current_price / previous_close - 1
        if np.isfinite(previous_close)
        and previous_close != 0
        else np.nan
    )

    name = clean_ticker
    currency = "N/A"

    try:
        info = security.get_info()

        name = (
            info.get("longName")
            or info.get("shortName")
            or clean_ticker
        )

        currency = (
            info.get("currency")
            or info.get("financialCurrency")
            or "N/A"
        )

    except Exception:
        pass

    last_market_date = pd.Timestamp(prices.index[-1])

    if last_market_date.tzinfo is not None:
        last_market_date = last_market_date.tz_localize(None)

    return {
        "ticker": clean_ticker,
        "name": str(name),
        "currency": str(currency),
        "current_price": current_price,
        "previous_close": previous_close,
        "daily_change": daily_change,
        "last_market_date": last_market_date,
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_price_history(
    ticker: str,
    start_date_text: str,
) -> pd.Series:
    """Load daily historical prices from a saved purchase date."""

    clean_ticker = str(ticker).strip().upper()
    start_value = pd.Timestamp(start_date_text).date()
    end_value = date.today() + timedelta(days=1)

    security = yf.Ticker(clean_ticker)

    try:
        history = security.history(
            start=start_value.isoformat(),
            end=end_value.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
        )
    except TypeError:
        history = security.history(
            start=start_value.isoformat(),
            end=end_value.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

    if history is None or history.empty:
        raise ValueError(
            f"No historical data was returned for {clean_ticker}."
        )

    price_column = (
        "Adj Close"
        if "Adj Close" in history.columns
        and not history["Adj Close"].dropna().empty
        else "Close"
    )

    prices = pd.to_numeric(
        history[price_column],
        errors="coerce",
    ).dropna()

    if prices.empty:
        raise ValueError(
            f"No valid historical prices were returned for {clean_ticker}."
        )

    prices.index = pd.to_datetime(prices.index)

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    prices.name = clean_ticker
    return prices


def format_money(
    value: float,
    currency: str,
) -> str:
    """Format a monetary value."""

    if value is None or not np.isfinite(value):
        return "N/A"

    prefix = (
        ""
        if currency in ("", "N/A")
        else f"{currency} "
    )

    return f"{prefix}{value:,.2f}"


def format_percent(value: float) -> str:
    """Format a decimal return."""

    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value * 100:,.2f}%"


def signed_colour(value: Any) -> str:
    """Colour positive and negative numerical results."""

    if pd.isna(value):
        return ""

    value = float(value)

    if value > 0:
        return "color: #2DB24A; font-weight: 650;"

    if value < 0:
        return "color: #D92D20; font-weight: 650;"

    return ""


def build_live_holdings(
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Combine saved positions with current market observations."""

    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for _, position in positions.iterrows():
        ticker = str(position["ticker"]).upper()

        try:
            market = load_latest_market_data(ticker)
        except Exception as error:
            errors[ticker] = str(error)
            continue

        quantity = float(position["quantity"])
        average_cost = float(position["average_cost"])

        invested_value = quantity * average_cost
        current_value = quantity * market["current_price"]
        gain_loss = current_value - invested_value

        return_decimal = (
            gain_loss / invested_value
            if invested_value != 0
            else np.nan
        )

        today_change_value = (
            quantity
            * (
                market["current_price"]
                - market["previous_close"]
            )
            if np.isfinite(market["previous_close"])
            else np.nan
        )

        rows.append(
            {
                "Company": market["name"],
                "Ticker": ticker,
                "Quantity": quantity,
                "Average Cost": average_cost,
                "Current Price": market["current_price"],
                "Invested Value": invested_value,
                "Current Value": current_value,
                "Gain / Loss": gain_loss,
                "Return (%)": return_decimal * 100,
                "Today Change": today_change_value,
                "Today Change (%)": (
                    market["daily_change"] * 100
                    if np.isfinite(market["daily_change"])
                    else np.nan
                ),
                "Currency": market["currency"],
                "Purchase Date": position["purchase_date"],
                "Last Market Date": market["last_market_date"],
            }
        )

    return pd.DataFrame(rows), errors


def build_portfolio_history(
    holdings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Create a simple portfolio-value history."""

    position_values: list[pd.Series] = []
    invested_values: list[pd.Series] = []
    errors: dict[str, str] = {}

    for _, holding in holdings.iterrows():
        ticker = str(holding["Ticker"]).upper()
        purchase_date = pd.Timestamp(
            holding["Purchase Date"]
        ).normalize()

        try:
            prices = load_price_history(
                ticker,
                purchase_date.date().isoformat(),
            )
        except Exception as error:
            errors[ticker] = str(error)
            continue

        prices = prices[
            prices.index.normalize() >= purchase_date
        ]

        if prices.empty:
            errors[ticker] = (
                "No market observation exists on or after "
                "the saved purchase date."
            )
            continue

        value_series = (
            prices * float(holding["Quantity"])
        )
        value_series.name = ticker

        invested_series = pd.Series(
            float(holding["Invested Value"]),
            index=prices.index,
            name=ticker,
            dtype=float,
        )

        position_values.append(value_series)
        invested_values.append(invested_series)

    if not position_values:
        return pd.DataFrame(), errors

    value_frame = (
        pd.concat(
            position_values,
            axis=1,
            join="outer",
        )
        .sort_index()
        .ffill()
        .fillna(0.0)
    )

    invested_frame = (
        pd.concat(
            invested_values,
            axis=1,
            join="outer",
        )
        .sort_index()
        .ffill()
        .fillna(0.0)
    )

    history = pd.DataFrame(
        {
            "Date": value_frame.index,
            "Portfolio Value": value_frame.sum(axis=1),
            "Total Invested": invested_frame.sum(axis=1),
        }
    )

    return history, errors


def render_refresh_controls(portfolio_id: int) -> None:
    """Show the manual refresh button and refresh time."""

    button_column, time_column = st.columns([1.2, 2.8])
    refresh_key = f"live_last_refresh_{portfolio_id}"

    with button_column:
        refresh_clicked = st.button(
            "🔄 Refresh Market Prices",
            type="primary",
            use_container_width=True,
            key=f"live_refresh_{portfolio_id}",
        )

    if refresh_clicked:
        load_latest_market_data.clear()
        load_price_history.clear()

        st.session_state[refresh_key] = datetime.now()
        st.rerun()

    with time_column:
        refresh_time = st.session_state.get(refresh_key)

        if refresh_time is None:
            st.caption(
                "Prices update when the dashboard loads. "
                "The cache lasts five minutes."
            )
        else:
            st.caption(
                "Last refreshed: "
                f"{refresh_time:%d/%m/%Y at %H:%M:%S}"
            )


def render_dashboard(
    portfolio_id: int,
    positions: pd.DataFrame,
    portfolio_name: str,
) -> None:
    """Render the selected portfolio dashboard."""

    st.subheader(portfolio_name)
    render_refresh_controls(portfolio_id)

    if positions.empty:
        st.info(
            "This portfolio has no positions yet. "
            "Open Manage Positions and add the first investment."
        )
        return

    with st.spinner(
        "Updating the latest available market prices..."
    ):
        holdings, market_errors = build_live_holdings(positions)

    refresh_key = f"live_last_refresh_{portfolio_id}"

    if refresh_key not in st.session_state:
        st.session_state[refresh_key] = datetime.now()

    if market_errors:
        with st.expander(
            "Market-Data Warnings",
            expanded=False,
        ):
            for ticker, message in market_errors.items():
                st.warning(f"{ticker}: {message}")

    if holdings.empty:
        st.error(
            "No saved position could be updated. "
            "Check the saved tickers."
        )
        return

    currencies = sorted(
        holdings["Currency"]
        .fillna("N/A")
        .astype(str)
        .unique()
        .tolist()
    )

    if len(currencies) > 1:
        st.warning(
            "The portfolio contains several trading currencies. "
            "Totals are shown separately instead of adding unlike currencies."
        )

    for currency in currencies:
        currency_holdings = holdings[
            holdings["Currency"].astype(str) == currency
        ].copy()

        total_invested = float(
            currency_holdings["Invested Value"].sum()
        )
        current_value = float(
            currency_holdings["Current Value"].sum()
        )
        gain_loss = current_value - total_invested

        total_return = (
            gain_loss / total_invested
            if total_invested != 0
            else np.nan
        )

        today_change = currency_holdings[
            "Today Change"
        ].sum(min_count=1)

        st.markdown(f"### Portfolio Summary — {currency}")

        columns = st.columns(5)

        metrics = [
            (
                "Current Value",
                format_money(current_value, currency),
            ),
            (
                "Total Invested",
                format_money(total_invested, currency),
            ),
            (
                "Unrealized Gain / Loss",
                format_money(gain_loss, currency),
            ),
            (
                "Total Return",
                format_percent(total_return),
            ),
            (
                "Today's Change",
                format_money(today_change, currency),
            ),
        ]

        for column, (label, value) in zip(columns, metrics):
            with column:
                st.metric(label=label, value=value)

        best = currency_holdings.loc[
            currency_holdings["Return (%)"].idxmax()
        ]
        worst = currency_holdings.loc[
            currency_holdings["Return (%)"].idxmin()
        ]

        best_column, worst_column = st.columns(2)

        with best_column:
            st.metric(
                "Best Performing Position",
                str(best["Ticker"]),
                delta=f"{best['Return (%)']:.2f}%",
            )

        with worst_column:
            st.metric(
                "Worst Performing Position",
                str(worst["Ticker"]),
                delta=f"{worst['Return (%)']:.2f}%",
            )

        display_columns = [
            "Company",
            "Ticker",
            "Quantity",
            "Average Cost",
            "Current Price",
            "Invested Value",
            "Current Value",
            "Gain / Loss",
            "Return (%)",
            "Today Change (%)",
            "Purchase Date",
            "Last Market Date",
        ]

        display = currency_holdings[
            display_columns
        ].copy()

        display["Purchase Date"] = (
            pd.to_datetime(
                display["Purchase Date"],
                errors="coerce",
            )
            .dt.strftime("%d/%m/%Y")
        )

        display["Last Market Date"] = (
            pd.to_datetime(
                display["Last Market Date"],
                errors="coerce",
            )
            .dt.strftime("%d/%m/%Y")
        )

        styled = (
            display.style
            .format(
                {
                    "Quantity": "{:,.4f}",
                    "Average Cost": "{:,.2f}",
                    "Current Price": "{:,.2f}",
                    "Invested Value": "{:,.2f}",
                    "Current Value": "{:,.2f}",
                    "Gain / Loss": "{:,.2f}",
                    "Return (%)": "{:,.2f}%",
                    "Today Change (%)": "{:,.2f}%",
                },
                na_rep="N/A",
            )
            .map(
                signed_colour,
                subset=[
                    "Gain / Loss",
                    "Return (%)",
                    "Today Change (%)",
                ],
            )
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )

        allocation_column, history_column = st.columns(2)

        with allocation_column:
            allocation_figure = px.pie(
                currency_holdings,
                names="Ticker",
                values="Current Value",
                hole=0.55,
                title=f"Current Allocation — {currency}",
            )

            allocation_figure.update_traces(
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Current Value: %{value:,.2f}<br>"
                    "Weight: %{percent}"
                    "<extra></extra>"
                ),
            )

            allocation_figure.update_layout(height=460)

            st.plotly_chart(
                allocation_figure,
                use_container_width=True,
                key=f"allocation_{portfolio_id}_{currency}",
            )

        with history_column:
            history, history_errors = build_portfolio_history(
                currency_holdings
            )

            if history_errors:
                with st.expander(
                    "Historical-Chart Warnings",
                    expanded=False,
                ):
                    for ticker, message in history_errors.items():
                        st.warning(f"{ticker}: {message}")

            if history.empty:
                st.info(
                    "Historical portfolio evolution "
                    "could not be calculated."
                )
            else:
                history_long = (
                    history.melt(
                        id_vars="Date",
                        var_name="Series",
                        value_name="Value",
                    )
                )

                history_figure = px.line(
                    history_long,
                    x="Date",
                    y="Value",
                    color="Series",
                    title=f"Portfolio Evolution — {currency}",
                )

                history_figure.update_traces(
                    line=dict(width=3),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}"
                        "<br>Value: %{y:,.2f}"
                        "<extra></extra>"
                    ),
                )

                history_figure.update_layout(
                    height=460,
                    yaxis_title=f"Value ({currency})",
                )

                st.plotly_chart(
                    history_figure,
                    use_container_width=True,
                    key=f"history_{portfolio_id}_{currency}",
                )

                st.caption(
                    "This is a simple reconstruction from saved "
                    "purchase dates, quantities and average costs."
                )


def render_position_manager(
    portfolio_id: int,
    positions: pd.DataFrame,
) -> None:
    """Add, update or delete positions."""

    saved_tickers = (
        positions["ticker"].astype(str).tolist()
        if not positions.empty
        else []
    )

    choice = st.selectbox(
        "Position",
        options=["Add new position"] + saved_tickers,
        key=f"live_position_choice_{portfolio_id}",
    )

    is_new = choice == "Add new position"

    if is_new:
        ticker = st.text_input(
            "Ticker — add one stock at a time",
            placeholder="Example: AAPL",
            help=(
                "Save one ticker, then repeat the process "
                "to add another stock."
            ),
            key=f"live_new_ticker_{portfolio_id}",
        ).strip().upper()

        st.caption(
            "Add one ticker, save the position, "
            "and then repeat for the next stock."
        )

        default_quantity = 1.0
        default_cost = 0.0
        default_date = date.today()

        use_latest_price = st.checkbox(
            "Use latest available market price",
            value=True,
            key=f"live_use_latest_{portfolio_id}",
        )

    else:
        ticker = str(choice).upper()

        current = positions[
            positions["ticker"].astype(str) == ticker
        ].iloc[0]

        default_quantity = float(current["quantity"])
        default_cost = float(current["average_cost"])
        default_date = pd.to_datetime(
            current["purchase_date"]
        ).date()

        use_latest_price = False

        st.info(
            f"Editing {ticker}. Saving replaces the current saved values."
        )

    quantity = st.number_input(
        "Quantity",
        min_value=0.0001,
        value=float(default_quantity),
        step=1.0,
        format="%.4f",
        key=f"live_quantity_{portfolio_id}_{choice}",
    )

    average_cost = st.number_input(
        "Average Purchase Price",
        min_value=0.0,
        value=float(default_cost),
        step=1.0,
        format="%.4f",
        disabled=is_new and use_latest_price,
        key=f"live_cost_{portfolio_id}_{choice}",
    )

    purchase_date = st.date_input(
        "Purchase Date",
        value=default_date,
        max_value=date.today(),
        disabled=is_new and use_latest_price,
        key=f"live_date_{portfolio_id}_{choice}",
    )

    save_column, delete_column = st.columns(2)

    with save_column:
        save_clicked = st.button(
            "Save Position",
            type="primary",
            use_container_width=True,
            key=f"live_save_{portfolio_id}",
        )

    with delete_column:
        delete_clicked = st.button(
            "Delete Position",
            use_container_width=True,
            disabled=is_new,
            key=f"live_delete_{portfolio_id}",
        )

    if save_clicked:
        try:
            if not ticker:
                raise ValueError("Enter a valid ticker.")

            market = load_latest_market_data(ticker)

            final_cost = float(average_cost)
            final_date = purchase_date

            if is_new and use_latest_price:
                final_cost = float(market["current_price"])
                final_date = date.today()

            save_position(
                portfolio_id=portfolio_id,
                ticker=ticker,
                quantity=float(quantity),
                average_cost=final_cost,
                purchase_date=final_date,
            )

            st.success(
                f"{ticker} was validated and saved successfully."
            )
            st.rerun()

        except Exception as error:
            st.error(
                "The position could not be saved. "
                f"{error}"
            )

    if delete_clicked and not is_new:
        delete_position(
            portfolio_id,
            ticker,
        )
        st.success(f"{ticker} was deleted.")
        st.rerun()

    st.markdown("### Saved Positions")

    if positions.empty:
        st.caption("No positions saved in this portfolio.")
    else:
        saved_display = positions[
            [
                "ticker",
                "quantity",
                "average_cost",
                "purchase_date",
                "updated_at",
            ]
        ].copy()

        saved_display.columns = [
            "Ticker",
            "Quantity",
            "Average Cost",
            "Purchase Date",
            "Last Updated",
        ]

        st.dataframe(
            saved_display,
            use_container_width=True,
            hide_index=True,
        )


def render_portfolio_management(
    portfolios: pd.DataFrame,
    selected_id: int,
    selected_name: str,
) -> None:
    """Create, rename and delete portfolios."""

    with st.expander(
        "Manage Portfolios",
        expanded=False,
    ):
        st.markdown("#### Create a Portfolio")

        new_name = st.text_input(
            "New Portfolio Name",
            placeholder="Example: Long Term",
            key="live_new_portfolio_name",
        )

        if st.button(
            "Create Portfolio",
            type="primary",
            use_container_width=True,
            key="live_create_portfolio",
        ):
            try:
                new_id = create_portfolio(new_name)

                st.session_state[
                    "pending_live_portfolio_id"
                ] = new_id

                st.rerun()

            except Exception as error:
                st.error(str(error))

        st.divider()
        st.markdown("#### Rename Current Portfolio")

        rename_value = st.text_input(
            "New Name",
            value=selected_name,
            key=f"live_rename_{selected_id}",
        )

        if st.button(
            "Rename Portfolio",
            use_container_width=True,
            key=f"live_rename_button_{selected_id}",
        ):
            try:
                rename_portfolio(
                    selected_id,
                    rename_value,
                )

                st.session_state[
                    "pending_live_portfolio_id"
                ] = selected_id

                st.rerun()

            except Exception as error:
                st.error(str(error))

        st.divider()
        st.markdown("#### Delete Current Portfolio")

        confirm_delete = st.checkbox(
            f"Delete {selected_name} and all its positions",
            key=f"live_confirm_delete_{selected_id}",
        )

        if st.button(
            "Delete Current Portfolio",
            use_container_width=True,
            disabled=not confirm_delete,
            key=f"live_delete_portfolio_{selected_id}",
        ):
            try:
                deleted_portfolio_id = int(selected_id)
                deleted_portfolio_name = str(selected_name)

                delete_portfolio(deleted_portfolio_id)

                remaining = load_portfolios()

                # Remove widget state linked to the deleted portfolio.
                deleted_id_text = str(deleted_portfolio_id)

                for state_key in list(st.session_state.keys()):
                    if state_key == "live_portfolio_selector":
                        continue

                    if (
                        state_key.endswith(f"_{deleted_id_text}")
                        or f"_{deleted_id_text}_" in state_key
                    ):
                        del st.session_state[state_key]

                st.session_state.pop(
                    "live_portfolio_selector",
                    None,
                )

                if remaining.empty:
                    st.session_state.pop(
                        "pending_live_portfolio_id",
                        None,
                    )

                    st.session_state[
                        "live_portfolio_flash_message"
                    ] = (
                        f"{deleted_portfolio_name} was deleted. "
                        "No portfolios remain."
                    )
                else:
                    next_portfolio_id = int(
                        remaining.iloc[0]["id"]
                    )
                    next_portfolio_name = str(
                        remaining.iloc[0]["name"]
                    )

                    st.session_state[
                        "pending_live_portfolio_id"
                    ] = next_portfolio_id

                    st.session_state[
                        "live_portfolio_flash_message"
                    ] = (
                        f"{deleted_portfolio_name} was deleted. "
                        f"Now viewing {next_portfolio_name}."
                    )

                load_latest_market_data.clear()
                load_price_history.clear()

                st.rerun()

            except Exception as error:
                st.error(str(error))


def render_live_investment() -> None:
    """Render the complete Live Investment workspace."""

    initialize_database()

    st.markdown("## Live Investment")
    st.caption(
        f"Live Investment version {LIVE_INVESTMENT_VERSION}"
    )

    st.write(
        "Create simulated portfolios and follow them using the latest "
        "available daily market prices."
    )

    st.caption(
        "This paper-investing tool does not connect to a broker, "
        "place orders or provide investment advice."
    )

    portfolios = load_portfolios()

    flash_message = st.session_state.pop(
        "live_portfolio_flash_message",
        None,
    )

    if flash_message:
        st.success(flash_message)

    if portfolios.empty:
        st.info(
            "No portfolios exist yet. Create the first portfolio "
            "to start using Live Investment."
        )

        st.markdown("### Create Your First Portfolio")

        first_portfolio_name = st.text_input(
            "Portfolio Name",
            placeholder="Example: My Portfolio",
            key="live_first_portfolio_name",
        )

        if st.button(
            "Create First Portfolio",
            type="primary",
            use_container_width=True,
            key="live_create_first_portfolio",
        ):
            try:
                new_portfolio_id = create_portfolio(
                    first_portfolio_name
                )

                st.session_state[
                    "pending_live_portfolio_id"
                ] = new_portfolio_id

                st.session_state[
                    "live_portfolio_flash_message"
                ] = "Portfolio created successfully."

                st.rerun()

            except Exception as error:
                st.error(str(error))

        return

    name_map = {
        int(row["id"]): str(row["name"])
        for _, row in portfolios.iterrows()
    }

    portfolio_ids = list(name_map.keys())

    pending_id = st.session_state.pop(
        "pending_live_portfolio_id",
        None,
    )

    if pending_id in portfolio_ids:
        st.session_state[
            "live_portfolio_selector"
        ] = pending_id

    # Never allow the selector to keep a deleted portfolio ID.
    current_selector_value = st.session_state.get(
        "live_portfolio_selector"
    )

    if current_selector_value not in portfolio_ids:
        st.session_state[
            "live_portfolio_selector"
        ] = portfolio_ids[0]


    selector_column, info_column = st.columns(
        [1.25, 2.75]
    )

    with selector_column:
        selected_id = st.selectbox(
            "Portfolio",
            options=portfolio_ids,
            format_func=lambda portfolio_id: name_map[
                portfolio_id
            ],
            key="live_portfolio_selector",
        )

    selected_name = name_map[int(selected_id)]

    with info_column:
        st.info(
            f"Currently viewing: **{selected_name}**"
        )

    render_portfolio_management(
        portfolios=portfolios,
        selected_id=int(selected_id),
        selected_name=selected_name,
    )

    positions = load_positions(int(selected_id))

    dashboard_tab, manage_tab = st.tabs(
        [
            "Dashboard",
            "Manage Positions",
        ]
    )

    with dashboard_tab:
        render_dashboard(
            portfolio_id=int(selected_id),
            positions=positions,
            portfolio_name=selected_name,
        )

    with manage_tab:
        render_position_manager(
            portfolio_id=int(selected_id),
            positions=positions,
        )
