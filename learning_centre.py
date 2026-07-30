
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def render_learning_centre(
    results: dict[str, Any],
    portfolio_currency: str,
    currency_symbol: str,
    rolling_window: int,
) -> None:
    """
    Render the Finance Bro educational centre.

    Every topic starts closed and connects the current portfolio result with
    the corresponding financial definition, formula, interpretation, common
    mistake and limitation.
    """

    st.subheader("Finance Bro Learning Centre")

    st.write(
        "Explore the concepts used throughout the analysis. Every topic "
        "connects financial theory with the result calculated for the "
        "selected portfolio."
    )

    st.info(
        "Choose a category and open only the concept you want to study. "
        "All topics start closed. Interpretations are educational, "
        "sample-dependent and not investment recommendations."
    )

    summary_1, summary_2, summary_3 = st.columns(3)

    with summary_1:
        st.metric("Learning Topics", "34")

    with summary_2:
        st.metric(
            "Portfolio Currency",
            f"{portfolio_currency} ({currency_symbol})",
        )

    with summary_3:
        st.metric(
            "Selected Benchmark",
            str(results["benchmark_ticker"]),
        )

    def fmt_number(
        value: Any,
        suffix: str = "",
        prefix: str = "",
        decimals: int = 2,
    ) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{prefix}{float(value):,.{decimals}f}{suffix}"

    def fmt_money(
        value: Any,
        decimals: int = 2,
    ) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        value = float(value)
        if value < 0:
            return f"-{currency_symbol}{abs(value):,.{decimals}f}"
        return f"{currency_symbol}{value:,.{decimals}f}"

    def fmt_date(value: Any) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return pd.Timestamp(value).strftime("%d/%m/%Y")

    def write_section(
        heading: str,
        content: str | list[str] | None,
        alert: str | None = None,
    ) -> None:
        if not content:
            return
        st.markdown(f"**{heading}**")
        paragraphs = content if isinstance(content, list) else [content]
        for paragraph in paragraphs:
            if alert == "warning":
                st.warning(paragraph)
            elif alert == "info":
                st.info(paragraph)
            else:
                st.write(paragraph)

    def render_topic(
        title: str,
        result_cards: list[tuple[str, str]] | None = None,
        meaning: str | list[str] | None = None,
        simple_formula: str | None = None,
        professional_formulas: list[str] | None = None,
        symbols: list[tuple[str, str]] | None = None,
        interpretation: str | list[str] | None = None,
        example: str | None = None,
        common_mistake: str | None = None,
        limitation: str | None = None,
        methodology: str | None = None,
    ) -> None:
        with st.expander(title, expanded=False):
            if result_cards:
                columns = st.columns(len(result_cards))
                for column, (label, value) in zip(columns, result_cards):
                    with column:
                        st.metric(label=label, value=value)

            write_section("What it means", meaning)

            if simple_formula:
                st.markdown("**Simple formula**")
                st.code(simple_formula, language=None)

            if professional_formulas:
                st.markdown("**Professional formula**")
                for formula in professional_formulas:
                    st.latex(formula)

            if symbols:
                st.markdown("**What each symbol means**")
                st.markdown(
                    "\n".join(
                        f"- ${symbol}$: {description}"
                        for symbol, description in symbols
                    )
                )

            write_section("How to interpret it", interpretation)
            write_section("Example", example)
            write_section("Common mistake", common_mistake, alert="warning")
            write_section("Limitation", limitation)
            write_section("Finance Bro methodology", methodology, alert="info")

    portfolio_returns = results["portfolio_returns"].dropna()
    best_return = (
        float(portfolio_returns.max() * 100)
        if not portfolio_returns.empty
        else np.nan
    )
    worst_return = (
        float(portfolio_returns.min() * 100)
        if not portfolio_returns.empty
        else np.nan
    )
    best_date = (
        portfolio_returns.idxmax()
        if not portfolio_returns.empty
        else pd.NaT
    )
    worst_date = (
        portfolio_returns.idxmin()
        if not portfolio_returns.empty
        else pd.NaT
    )

    latest_rolling_volatility = (
        float(results["rolling_volatility"].iloc[-1])
        if not results["rolling_volatility"].empty
        else np.nan
    )

    quality = results["data_quality_headline"]
    asset_quality = results["data_quality_asset_table"].copy()
    total_missing_prices = int(asset_quality["Missing Prices"].sum())
    lowest_coverage_index = asset_quality["Coverage (%)"].idxmin()
    lowest_coverage_asset = str(
        asset_quality.loc[lowest_coverage_index, "Asset"]
    )
    lowest_coverage_value = float(
        asset_quality.loc[lowest_coverage_index, "Coverage (%)"]
    )

    allocation = results["allocation_table"].copy()
    weights = allocation["Weight (%)"]
    equal_weighted = bool(np.allclose(weights, weights.iloc[0]))

    construction = results["initial_portfolio_construction"].copy()
    converted_asset_count = int(
        (
            construction["Trading Currency"]
            != portfolio_currency
        ).sum()
    )

    contributions = results["contribution_table"].copy()
    return_contributor = contributions.loc[
        contributions["Return Contribution (p.p.)"].idxmax()
    ]
    risk_contributor = contributions.loc[
        contributions["Risk Contribution (%)"].idxmax()
    ]

    benchmark_ticker = str(results["benchmark_ticker"])
    benchmark_name = {
        "^GSPC": "S&P 500",
        "^STOXX50E": "EURO STOXX 50",
        "^IXIC": "Nasdaq Composite",
        "^DJI": "Dow Jones",
    }.get(benchmark_ticker, benchmark_ticker)

    market_stress = results["market_stress_summary"]
    custom_stress = results["custom_stress_summary"]

    diversification_summary = results[
        "diversification_summary"
    ].copy()

    average_correlation_column = (
        "Average Correlation with Other Assets"
    )

    absolute_correlation_table = (
        diversification_summary[
            [
                "Asset",
                average_correlation_column,
            ]
        ]
        .dropna()
        .copy()
    )

    if absolute_correlation_table.empty:
        lowest_absolute_correlation_asset = "N/A"
        lowest_average_correlation_value = np.nan
        lowest_absolute_correlation_value = np.nan
    else:
        absolute_correlation_table[
            "Absolute Average Correlation"
        ] = (
            absolute_correlation_table[
                average_correlation_column
            ].abs()
        )

        lowest_absolute_index = (
            absolute_correlation_table[
                "Absolute Average Correlation"
            ].idxmin()
        )

        lowest_absolute_correlation_asset = str(
            absolute_correlation_table.loc[
                lowest_absolute_index,
                "Asset",
            ]
        )

        lowest_average_correlation_value = float(
            absolute_correlation_table.loc[
                lowest_absolute_index,
                average_correlation_column,
            ]
        )

        lowest_absolute_correlation_value = float(
            absolute_correlation_table.loc[
                lowest_absolute_index,
                "Absolute Average Correlation",
            ]
        )

    (
        performance_tab,
        risk_tab,
        regression_tab,
        construction_tab,
        diversification_tab,
        quality_tab,
        stress_tab,
    ) = st.tabs(
        [
            "Performance",
            "Risk",
            "Benchmark & Regression",
            "Portfolio Construction",
            "Diversification",
            "Data Quality",
            "Stress Testing",
        ]
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    with performance_tab:
        render_topic(
            title="Portfolio Value and Profit / Loss",
            result_cards=[
                ("Initial Value", fmt_money(results["initial_investment"])),
                ("Final Value", fmt_money(results["final_value"])),
                ("Profit / Loss", fmt_money(results["profit_loss"])),
            ],
            meaning=(
                "Portfolio value is the monetary value of the investment "
                "after applying the portfolio return path. Profit or loss is "
                "the final value minus the initial investment."
            ),
            simple_formula=(
                "Profit / Loss = Final Portfolio Value − Initial Investment"
            ),
            professional_formulas=[
                r"V_T=V_0\prod_{t=1}^{T}(1+r_{p,t})",
                r"\Pi_T=V_T-V_0",
            ],
            symbols=[
                (r"V_0", "initial portfolio value"),
                (r"V_T", "portfolio value at the end of the period"),
                (r"r_{p,t}", "portfolio return on trading day t"),
                (r"\Pi_T", "monetary profit or loss"),
            ],
            interpretation=(
                "A positive value means the portfolio finished above its "
                "starting value. Monetary profit should be considered "
                "together with percentage return when comparing portfolios "
                "of different sizes."
            ),
            example=(
                f"An investment of {currency_symbol}10,000 that finishes at "
                f"{currency_symbol}11,500 produces a profit of "
                f"{currency_symbol}1,500."
            ),
            common_mistake=(
                "Comparing monetary profits across portfolios with different "
                "initial values without also comparing percentage returns."
            ),
            limitation=(
                "The current model excludes taxes, transaction costs and "
                "execution spreads."
            ),
        )

        render_topic(
            title="Daily Returns, Best Day and Worst Day",
            result_cards=[
                ("Best Day", fmt_number(best_return, suffix="%")),
                ("Best Date", fmt_date(best_date)),
                ("Worst Day", fmt_number(worst_return, suffix="%")),
                ("Worst Date", fmt_date(worst_date)),
            ],
            meaning=(
                "A simple daily return measures the proportional change in "
                "portfolio value from one trading day to the next. The best "
                "and worst day are the largest observed daily gain and loss "
                "inside the selected sample."
            ),
            simple_formula=(
                "Daily Return = Current Value ÷ Previous Value − 1"
            ),
            professional_formulas=[
                r"r_{p,t}=\frac{V_t}{V_{t-1}}-1",
            ],
            symbols=[
                (r"r_{p,t}", "portfolio simple return on day t"),
                (r"V_t", "portfolio value on day t"),
                (r"V_{t-1}", "portfolio value on the previous trading day"),
            ],
            interpretation=(
                f"The analysis contains {len(portfolio_returns):,} daily "
                "return observations. Extreme days can materially influence "
                "volatility, VaR, Expected Shortfall and regression tests."
            ),
            example=(
                "A rise from 100 to 103 is a 3% gain. A subsequent fall from "
                "103 to 100 is approximately −2.91%, not −3%."
            ),
            common_mistake=(
                "Assuming gains and losses are symmetric. A 50% loss requires "
                "a 100% gain to return to the starting value."
            ),
            limitation=(
                "Historical best and worst days are sample extrema, not "
                "estimates of the largest possible future move."
            ),
        )

        render_topic(
            title="Cumulative Return",
            result_cards=[
                (
                    "Portfolio Result",
                    fmt_number(
                        results["cumulative_return"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Cumulative return measures the total compounded change in "
                "portfolio value from the beginning to the end of the "
                "selected period."
            ),
            simple_formula=(
                "Cumulative Return = Final Value ÷ Initial Value − 1"
            ),
            professional_formulas=[
                r"R_{p,0,T}^{\mathrm{cum}}"
                r"=\prod_{t=1}^{T}(1+r_{p,t})-1",
            ],
            symbols=[
                (
                    r"R_{p,0,T}^{\mathrm{cum}}",
                    "portfolio cumulative return over the full period",
                ),
                (r"r_{p,t}", "portfolio daily return"),
                (r"T", "number of return observations"),
            ],
            interpretation=(
                "A positive value means the portfolio finished above its "
                "starting value. Its magnitude must be interpreted together "
                "with sample length and risk."
            ),
            example=(
                "A portfolio that grows from 10,000 to 12,000 has a "
                "cumulative return of 20%."
            ),
            common_mistake=(
                "Treating cumulative return as an annual return. A 20% gain "
                "over five years is not 20% per year."
            ),
            limitation=(
                "Cumulative return does not reveal volatility, drawdowns or "
                "the path followed between the start and end dates."
            ),
        )

        render_topic(
            title="Arithmetic Annualized Return",
            result_cards=[
                (
                    "Annualized Estimate",
                    fmt_number(
                        results["annualized_portfolio_return"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Finance Bro reports an arithmetic annualized return by "
                "multiplying the mean daily portfolio return by 252. It is "
                "not the realized compound annual growth rate."
            ),
            simple_formula=(
                "Arithmetic Annualized Return = Average Daily Return × 252"
            ),
            professional_formulas=[
                r"\widehat{\mu}_{p,\mathrm{ann}}=252\,\bar r_p",
                r"\bar r_p=\frac{1}{T}\sum_{t=1}^{T}r_{p,t}",
            ],
            symbols=[
                (
                    r"\widehat{\mu}_{p,\mathrm{ann}}",
                    "estimated arithmetic annual return",
                ),
                (r"\bar r_p", "sample mean daily portfolio return"),
            ],
            interpretation=(
                "It places average daily return on a common annual scale. "
                "Because it does not compound the actual wealth path, it can "
                "exceed realized compound growth when volatility is high."
            ),
            example=(
                "An average daily return of 0.04% produces an arithmetic "
                "annualized estimate of approximately 10.08%."
            ),
            common_mistake=(
                "Calling this measure CAGR. CAGR uses total compounded wealth "
                "growth and elapsed time."
            ),
            limitation=(
                "The 252-day convention and historical mean may not describe "
                "future returns."
            ),
        )

        render_topic(
            title="Benchmark Cumulative Return",
            result_cards=[
                (
                    "Benchmark",
                    f"{benchmark_name} ({benchmark_ticker})",
                ),
                (
                    "Benchmark Return",
                    fmt_number(
                        results["benchmark_cumulative_return"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Benchmark cumulative return is the selected index's "
                "compounded return over the same aligned period and in the "
                "same portfolio currency."
            ),
            professional_formulas=[
                r"R_{b,0,T}^{\mathrm{cum}}"
                r"=\prod_{t=1}^{T}(1+r_{b,t})-1",
            ],
            interpretation=(
                "The benchmark represents a reference opportunity set. It "
                "should resemble the portfolio's geography, asset class and "
                "systematic exposures."
            ),
            example=(
                "A broad U.S. equity portfolio can be compared with the "
                "S&P 500, while a broad euro-area equity portfolio may be "
                "better matched with a European index."
            ),
            common_mistake=(
                "Selecting a benchmark only because it makes the portfolio "
                "look favourable."
            ),
            limitation=(
                "One index may not capture sector, size, style, geographic "
                "and currency exposures simultaneously."
            ),
        )

        render_topic(
            title="Cumulative Active Return",
            result_cards=[
                (
                    "Active Return",
                    fmt_number(
                        results["active_return"],
                        suffix=" p.p.",
                    ),
                ),
            ],
            meaning=(
                "Cumulative active return is the difference between portfolio "
                "and benchmark cumulative returns over the common period."
            ),
            simple_formula=(
                "Cumulative Active Return = Portfolio Cumulative Return − "
                "Benchmark Cumulative Return"
            ),
            professional_formulas=[
                r"AR_{0,T}"
                r"=R_{p,0,T}^{\mathrm{cum}}"
                r"-R_{b,0,T}^{\mathrm{cum}}",
            ],
            symbols=[
                (r"AR_{0,T}", "cumulative active return"),
                (
                    r"R_{p,0,T}^{\mathrm{cum}}",
                    "portfolio cumulative return",
                ),
                (
                    r"R_{b,0,T}^{\mathrm{cum}}",
                    "benchmark cumulative return",
                ),
            ],
            interpretation=(
                "A positive value means cumulative outperformance; a "
                "negative value means cumulative underperformance."
            ),
            example=(
                "Portfolio return of 15% minus benchmark return of 11% "
                "equals +4 percentage points."
            ),
            common_mistake=(
                "Calling the difference 4% instead of 4 percentage points."
            ),
            limitation=(
                "It does not adjust for beta, volatility, drawdown or any "
                "other difference in risk."
            ),
        )

    # ========================================================
    # RISK
    # ========================================================

    with risk_tab:
        render_topic(
            title="Annualized Volatility",
            result_cards=[
                (
                    "Portfolio Volatility",
                    fmt_number(
                        results["annualized_volatility"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Volatility measures the dispersion of daily returns around "
                "their sample mean. Finance Bro annualizes sample daily "
                "volatility using square-root-of-time scaling."
            ),
            simple_formula=(
                "Annualized Volatility = Daily Standard Deviation × √252"
            ),
            professional_formulas=[
                r"\widehat{\sigma}_{p,\mathrm{ann}}"
                r"=\widehat{\sigma}_{p,d}\sqrt{252}",
                r"\widehat{\sigma}_{p,d}"
                r"=\sqrt{\frac{1}{T-1}"
                r"\sum_{t=1}^{T}(r_{p,t}-\bar r_p)^2}",
            ],
            interpretation=(
                "Higher volatility indicates wider historical fluctuations. "
                "It includes both upside and downside movements."
            ),
            example=(
                "Daily volatility of 1% corresponds to approximately 15.87% "
                "annualized volatility."
            ),
            common_mistake=(
                "Interpreting volatility as the expected loss."
            ),
            limitation=(
                "Volatility changes over time and square-root scaling can be "
                "imperfect when returns are dependent or volatility clusters."
            ),
        )

        render_topic(
            title="Rolling Window and Rolling Volatility",
            result_cards=[
                (
                    "Selected Window",
                    f"{int(rolling_window)} trading days",
                ),
                (
                    "Latest Rolling Volatility",
                    fmt_number(
                        latest_rolling_volatility,
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "A rolling statistic is recalculated at each date using only "
                "the most recent observations. A 20-day window uses the "
                "current return and the previous 19 returns."
            ),
            simple_formula=(
                "Rolling Volatility at t = Standard Deviation of the Most "
                "Recent W Returns × √252"
            ),
            professional_formulas=[
                r"\widehat{\sigma}_{p,t}^{(W)}"
                r"=\sqrt{252}\;"
                r"s(r_{p,t-W+1},\ldots,r_{p,t})",
            ],
            symbols=[
                (r"W", "rolling-window length"),
                (r"t", "current date"),
                (r"s(\cdot)", "sample standard deviation"),
            ],
            interpretation=[
                "A shorter window reacts faster but is noisier.",
                "A longer window is smoother but reacts more slowly.",
            ],
            example=(
                "A 60-day estimate uses the current return and the preceding "
                "59 daily returns."
            ),
            common_mistake=(
                "Thinking the app downloads W extra observations before the "
                "selected start date. It does not."
            ),
            limitation=(
                "The result is backward-looking and sensitive to the chosen "
                "window."
            ),
        )

        render_topic(
            title="Maximum Drawdown",
            result_cards=[
                (
                    "Maximum Drawdown",
                    fmt_number(
                        results["maximum_drawdown"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Drawdown measures the decline from the previous running "
                "peak. Maximum drawdown is the deepest peak-to-trough decline "
                "observed in the sample."
            ),
            simple_formula=(
                "Drawdown = Current Value ÷ Previous Running Peak − 1"
            ),
            professional_formulas=[
                r"M_t=\max_{0\leq u\leq t}V_u",
                r"DD_t=\frac{V_t}{M_t}-1",
                r"MDD=\min_t DD_t",
            ],
            interpretation=(
                "A drawdown of −30% means the portfolio was 30% below a "
                "previous peak at its deepest point."
            ),
            example=(
                "A fall from 12,000 to 9,000 after reaching the 12,000 peak "
                "is a −25% drawdown."
            ),
            common_mistake=(
                "Confusing maximum drawdown with the worst daily return."
            ),
            limitation=(
                "It does not show how long the decline lasted or how quickly "
                "the portfolio recovered."
            ),
        )

        render_topic(
            title="Sharpe Ratio",
            result_cards=[
                (
                    "Sharpe Ratio",
                    fmt_number(
                        results["sharpe_ratio"],
                        decimals=3,
                    ),
                ),
                (
                    "Average Annual Risk-Free Rate",
                    fmt_number(
                        results[
                            "risk_free_average_annual_rate_percent"
                        ],
                        suffix="%",
                        decimals=3,
                    ),
                ),
            ],
            meaning=(
                "The Sharpe Ratio measures average excess return per unit of "
                "total volatility."
            ),
            simple_formula=(
                "Sharpe Ratio = Average Daily Excess Return ÷ Daily "
                "Excess-Return Volatility × √252"
            ),
            professional_formulas=[
                r"ER_t=r_{p,t}-r_{f,t}",
                r"SR_{\mathrm{ann}}"
                r"=\sqrt{252}\frac{\overline{ER}}{s(ER)}",
            ],
            interpretation=(
                "A higher value means more historical excess return per unit "
                "of volatility. A negative value means average performance "
                "was below the risk-free reference."
            ),
            example=(
                "A Sharpe Ratio of 1 indicates roughly one unit of "
                "annualized excess return per unit of annualized volatility."
            ),
            common_mistake=(
                "Applying universal cut-offs without considering asset class, "
                "sample length and return distribution."
            ),
            limitation=(
                "It treats upside and downside volatility equally and can be "
                "misleading for skewed or fat-tailed returns."
            ),
        )

        render_topic(
            title="Confidence Level",
            result_cards=[
                (
                    "Selected Confidence",
                    fmt_number(
                        results["confidence_level"],
                        suffix="%",
                    ),
                ),
                (
                    "Tail Probability",
                    fmt_number(
                        100 - results["confidence_level"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "The confidence level determines the lower-tail probability "
                "used by VaR and Expected Shortfall."
            ),
            professional_formulas=[
                r"\alpha=1-c",
            ],
            symbols=[
                (r"c", "confidence level"),
                (r"\alpha", "tail probability"),
            ],
            interpretation=(
                "At 95% confidence the model focuses on the worst 5% of "
                "daily outcomes. At 99% it focuses on the worst 1%."
            ),
            example=(
                "With 1,000 observations, a 95% historical tail contains "
                "about 50 observations, while a 99% tail contains about 10."
            ),
            common_mistake=(
                "Assuming a higher confidence level automatically produces a "
                "more accurate estimate."
            ),
            limitation=(
                "Moving further into the tail leaves fewer observations and "
                "increases sampling uncertainty."
            ),
        )

        render_topic(
            title="Historical Value at Risk",
            result_cards=[
                (
                    "Historical VaR (%)",
                    fmt_number(
                        results["historical_var_return"],
                        suffix="%",
                    ),
                ),
                (
                    "Historical VaR",
                    fmt_money(results["historical_var_money"]),
                ),
            ],
            meaning=(
                "Historical VaR estimates a daily loss threshold directly "
                "from the empirical return distribution."
            ),
            simple_formula=(
                "Historical VaR = Negative of the Lower-Tail Historical "
                "Return Quantile"
            ),
            professional_formulas=[
                r"VaR_c^{\mathrm{hist}}=-q_{1-c}",
            ],
            interpretation=(
                "At 95% confidence, approximately 5% of historical daily "
                "returns were worse than the reported threshold."
            ),
            example=(
                f"A 2% VaR on {currency_symbol}10,000 corresponds to "
                f"{currency_symbol}200."
            ),
            common_mistake=(
                "Calling VaR the maximum possible loss."
            ),
            limitation=(
                "Historical VaR is sensitive to sample choice and may be "
                "unstable with few tail observations."
            ),
            methodology=(
                "Finance Bro uses a linear empirical quantile and reports the "
                "result as a positive loss magnitude."
            ),
        )

        render_topic(
            title="Parametric Value at Risk",
            result_cards=[
                (
                    "Parametric VaR (%)",
                    fmt_number(
                        results["parametric_var_return"],
                        suffix="%",
                    ),
                ),
                (
                    "Parametric VaR",
                    fmt_money(results["parametric_var_money"]),
                ),
            ],
            meaning=(
                "Parametric VaR estimates the loss threshold from sample "
                "mean and volatility under a normal-return assumption."
            ),
            professional_formulas=[
                r"VaR_c^{N}"
                r"=-\left(\widehat{\mu}_d"
                r"+z_{1-c}\widehat{\sigma}_d\right)",
            ],
            symbols=[
                (r"\widehat{\mu}_d", "sample mean daily return"),
                (r"\widehat{\sigma}_d", "sample daily volatility"),
                (r"z_{1-c}", "standard-normal lower-tail quantile"),
            ],
            interpretation=(
                "Differences from historical VaR reveal how the normal "
                "approximation differs from the empirical tail."
            ),
            example=(
                "At 95% confidence, the lower normal quantile is about −1.645."
            ),
            common_mistake=(
                "Assuming returns are normal because the formula is convenient."
            ),
            limitation=(
                "Normal VaR can understate risk under skewness, fat tails and "
                "volatility clustering."
            ),
        )

        render_topic(
            title="Expected Shortfall",
            result_cards=[
                (
                    "Historical ES (%)",
                    fmt_number(
                        results["historical_es_return"],
                        suffix="%",
                    ),
                ),
                (
                    "Historical ES",
                    fmt_money(results["historical_es_money"]),
                ),
                (
                    "Parametric ES (%)",
                    fmt_number(
                        results["parametric_es_return"],
                        suffix="%",
                    ),
                ),
                (
                    "Parametric ES",
                    fmt_money(results["parametric_es_money"]),
                ),
            ],
            meaning=(
                "Expected Shortfall is the average loss conditional on being "
                "inside the tail beyond the VaR threshold."
            ),
            professional_formulas=[
                r"ES_c^{\mathrm{hist}}"
                r"=-\mathbb{E}[r_p\mid r_p\leq q_{1-c}]",
                r"ES_c^{N}"
                r"=-\left(\widehat{\mu}_d"
                r"-\widehat{\sigma}_d"
                r"\frac{\phi(z_{1-c})}{1-c}\right)",
            ],
            interpretation=(
                "ES is generally larger than VaR because it measures average "
                "severity after entering the tail."
            ),
            example=(
                "A 95% VaR of 2% and ES of 3.2% mean that the average loss "
                "among the worst 5% of days was approximately 3.2%."
            ),
            common_mistake=(
                "Interpreting ES as the average loss across all days."
            ),
            limitation=(
                "Historical ES can be dominated by a few extreme events; "
                "parametric ES inherits the normality assumption."
            ),
        )

    # ========================================================
    # BENCHMARK AND REGRESSION
    # ========================================================

    with regression_tab:
        render_topic(
            title="Benchmark Selection",
            result_cards=[
                (
                    "Selected Benchmark",
                    f"{benchmark_name} ({benchmark_ticker})",
                ),
                (
                    "Benchmark Currency",
                    str(results["benchmark_currency"]),
                ),
            ],
            meaning=(
                "The benchmark is used for comparison, regression, beta, "
                "correlation and the beta-based market stress scenario."
            ),
            interpretation=[
                (
                    "A relevant benchmark should resemble the portfolio's "
                    "investment universe and systematic risk."
                ),
                (
                    "Changing the benchmark can materially change beta, "
                    "alpha, R² and stress results."
                ),
            ],
            example=(
                "A broad euro-area equity portfolio may be compared with a "
                "European equity index rather than an unrelated U.S. index."
            ),
            common_mistake=(
                "Treating benchmark choice as cosmetic."
            ),
            limitation=(
                "One benchmark cannot capture every systematic factor."
            ),
        )

        render_topic(
            title="Risk-Free Rate",
            result_cards=[
                (
                    "Source",
                    str(results["risk_free_source"]),
                ),
                (
                    "Average Annual Rate",
                    fmt_number(
                        results[
                            "risk_free_average_annual_rate_percent"
                        ],
                        suffix="%",
                        decimals=3,
                    ),
                ),
                (
                    "Latest Annual Rate",
                    fmt_number(
                        results[
                            "risk_free_latest_annual_rate_percent"
                        ],
                        suffix="%",
                        decimals=3,
                    ),
                ),
            ],
            meaning=(
                "The risk-free rate is subtracted from portfolio and "
                "benchmark returns to obtain excess returns."
            ),
            simple_formula=(
                "Daily Risk-Free Return = "
                "(1 + Annual Risk-Free Rate)^(1/252) − 1"
            ),
            professional_formulas=[
                r"r_{f,t}^{d}=(1+y_{f,t}^{a})^{1/252}-1",
            ],
            interpretation=(
                "EUR portfolios automatically use the ECB three-month "
                "compounded €STR average rate. USD portfolios use the "
                "three-month U.S. Treasury constant-maturity yield. Manual "
                "mode uses the rate entered by the user."
            ),
            example=(
                "An annual rate of 3% is approximately 0.0117% per trading "
                "day under equivalent compounding."
            ),
            common_mistake=(
                "Subtracting an annual yield directly from a daily return."
            ),
            limitation=(
                "Reference yields differ from investor-specific executable "
                "returns after taxes, fees and spreads."
            ),
            methodology=(
                "Official observations are aligned using the latest value "
                "available on or before each market date; no linear "
                "interpolation is used."
            ),
        )

        render_topic(
            title="Excess-Return Regression Model",
            result_cards=[
                (
                    "Observations",
                    f"{results['regression_observation_count']:,}",
                ),
                (
                    "Newey–West Lags",
                    str(results["regression_hac_lags"]),
                ),
            ],
            meaning=(
                "The model relates portfolio excess returns to benchmark "
                "excess returns. Alpha is the intercept, beta is the slope "
                "and the residual is the unexplained component."
            ),
            professional_formulas=[
                r"R_{p,t}-R_{f,t}"
                r"=\alpha+\beta(R_{b,t}-R_{f,t})+\varepsilon_t",
                r"\widehat{\boldsymbol{\theta}}"
                r"=(X^{\mathsf T}X)^{-1}X^{\mathsf T}y",
            ],
            interpretation=(
                "The fitted line describes a historical conditional mean "
                "association, not causality."
            ),
            example=(
                "Beta 1.2 means a 1% benchmark excess-return movement is "
                "associated with approximately 1.2% portfolio excess-return "
                "movement before alpha."
            ),
            common_mistake=(
                "Using the fitted equation as a guaranteed forecast."
            ),
            limitation=(
                "A one-factor linear model can omit relevant factors and "
                "nonlinear relationships."
            ),
        )

        render_topic(
            title="Beta",
            result_cards=[
                ("Beta", fmt_number(results["beta"], decimals=3)),
                (
                    "Robust p-value",
                    fmt_number(
                        results["beta_p_value_hac"],
                        decimals=4,
                    ),
                ),
                (
                    "95% Confidence Interval",
                    (
                        f"[{results['beta_confidence_interval'][0]:.3f}, "
                        f"{results['beta_confidence_interval'][1]:.3f}]"
                    ),
                ),
            ],
            meaning=(
                "Beta is the slope of the excess-return regression and "
                "measures historical sensitivity to the selected benchmark."
            ),
            professional_formulas=[
                r"\widehat{\beta}"
                r"=\frac{\operatorname{Cov}(R_p-R_f,R_b-R_f)}"
                r"{\operatorname{Var}(R_b-R_f)}",
            ],
            interpretation=(
                "Beta above 1 indicates greater historical sensitivity than "
                "the benchmark; beta between 0 and 1 indicates lower positive "
                "sensitivity; negative beta indicates an inverse relationship."
            ),
            example=(
                "Beta 1.24 suggests that a 1% benchmark excess-return move "
                "was associated with about a 1.24% portfolio move."
            ),
            common_mistake=(
                "Calling beta total portfolio risk."
            ),
            limitation=(
                "Beta changes with benchmark, sample, frequency, currency and "
                "market regime."
            ),
        )

        render_topic(
            title="Alpha",
            result_cards=[
                (
                    "Annualized Alpha",
                    fmt_number(
                        results["alpha_annualized"],
                        suffix="%",
                    ),
                ),
                (
                    "Robust p-value",
                    fmt_number(
                        results["alpha_p_value_hac"],
                        decimals=4,
                    ),
                ),
            ],
            meaning=(
                "Alpha is the regression intercept: the average excess return "
                "not attributed to benchmark exposure by the fitted model."
            ),
            professional_formulas=[
                r"\widehat{\alpha}_{\mathrm{ann}}"
                r"=252\,\widehat{\alpha}_{d}",
            ],
            interpretation=(
                "Positive alpha means the fitted intercept is above zero. "
                "Its p-value and confidence interval indicate estimation "
                "uncertainty."
            ),
            example=(
                "Annualized alpha of 2% summarizes the historical fitted "
                "intercept; it is not a promise of 2% future outperformance."
            ),
            common_mistake=(
                "Treating positive alpha as skill without checking benchmark "
                "choice, p-value and model specification."
            ),
            limitation=(
                "Alpha can absorb omitted factor exposures and model error."
            ),
            methodology=(
                "Finance Bro annualizes daily alpha arithmetically because "
                "the OLS intercept is an additive daily-return coefficient."
            ),
        )

        render_topic(
            title="R-Squared",
            result_cards=[
                (
                    "R-Squared",
                    fmt_number(
                        results["r_squared"] * 100,
                        suffix="%",
                    ),
                ),
                (
                    "Adjusted R-Squared",
                    fmt_number(
                        results["adjusted_r_squared"] * 100,
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "R² is the fraction of sample variation in portfolio excess "
                "returns explained by the benchmark excess-return model."
            ),
            professional_formulas=[
                r"R^2"
                r"=1-\frac{\sum_t\widehat{\varepsilon}_t^2}"
                r"{\sum_t(y_t-\bar y)^2}",
            ],
            interpretation=(
                "R² of 62% means the fitted benchmark factor explains 62% "
                "of historical excess-return variation in the sample."
            ),
            example=(
                "Low R² can reflect idiosyncratic risk or omitted systematic "
                "factors."
            ),
            common_mistake=(
                "Interpreting high R² as high investment quality."
            ),
            limitation=(
                "R² is descriptive, sample-specific and non-causal."
            ),
        )

        render_topic(
            title="Statistical Inference and Advanced Diagnostics",
            result_cards=[
                (
                    "Durbin–Watson",
                    fmt_number(
                        results["durbin_watson"],
                        decimals=3,
                    ),
                ),
                (
                    "Ljung–Box p-value",
                    fmt_number(
                        results["ljung_box_p_value"],
                        decimals=4,
                    ),
                ),
                (
                    "Breusch–Pagan p-value",
                    fmt_number(
                        results["breusch_pagan_p_value"],
                        decimals=4,
                    ),
                ),
                (
                    "Jarque–Bera p-value",
                    fmt_number(
                        results["jarque_bera_p_value"],
                        decimals=4,
                    ),
                ),
            ],
            meaning=(
                "Diagnostics evaluate uncertainty and possible violations of "
                "regression assumptions. Finance Bro uses Newey–West HAC "
                "standard errors for coefficient inference."
            ),
            professional_formulas=[
                r"\widehat{\operatorname{Var}}_{\mathrm{HAC}}"
                r"(\widehat{\theta})"
                r"=(X^{\mathsf T}X)^{-1}"
                r"\widehat{S}_{\mathrm{HAC}}"
                r"(X^{\mathsf T}X)^{-1}",
                r"CI_{1-\alpha}"
                r"=\widehat{\theta}"
                r"\pm t_{1-\alpha/2}"
                r"SE_{\mathrm{HAC}}(\widehat{\theta})",
            ],
            interpretation=[
                (
                    "Ljung–Box assesses residual autocorrelation; "
                    "Breusch–Pagan assesses heteroscedasticity; Jarque–Bera "
                    "assesses normality; Ramsey RESET assesses functional form."
                ),
                (
                    "Cook's distance flags potentially influential "
                    "observations."
                ),
            ],
            example=(
                "A p-value below 0.05 is commonly described as evidence "
                "against the tested null hypothesis at the 5% level."
            ),
            common_mistake=(
                "Believing a p-value is the probability that the null "
                "hypothesis is true."
            ),
            limitation=(
                "Robust standard errors improve inference but do not repair a "
                "misspecified economic model."
            ),
        )

    # ========================================================
    # PORTFOLIO CONSTRUCTION
    # ========================================================

    with construction_tab:
        render_topic(
            title="Portfolio Weights",
            result_cards=[
                (
                    "Weighting Method",
                    "Equal Weights" if equal_weighted else "Custom Weights",
                ),
                ("Number of Assets", str(len(allocation))),
                (
                    "Weight Sum",
                    fmt_number(weights.sum(), suffix="%"),
                ),
            ],
            meaning=(
                "A weight is the proportion of portfolio capital allocated "
                "to one asset. Long-only weights are non-negative and sum to "
                "100%."
            ),
            professional_formulas=[
                r"w_i=\frac{V_i}{V_p}",
                r"\sum_{i=1}^{N}w_i=1,\qquad w_i\geq0",
            ],
            interpretation=(
                "Higher weights give an asset more influence on portfolio "
                "return and risk."
            ),
            example=(
                "Four equal-weighted assets each receive 25%."
            ),
            common_mistake=(
                "Assuming equal weights always minimize risk."
            ),
            limitation=(
                "The current return model keeps weights constant through "
                "time, which represents periodic rebalancing."
            ),
        )

        render_topic(
            title="Initial Portfolio Construction and Fractional Shares",
            result_cards=[
                (
                    "Initial Investment",
                    fmt_money(results["initial_investment"]),
                ),
                ("Assets", str(len(construction))),
                ("Assets Requiring FX", str(converted_asset_count)),
                ("Purchase Method", "Fractional Shares"),
            ],
            meaning=(
                "The initial-construction table converts weights into "
                "monetary allocations, base-currency entry prices and "
                "theoretical fractional quantities."
            ),
            professional_formulas=[
                r"A_i=V_0w_i",
                r"N_{i,0}=\frac{A_i}{P_{i,0}^{B}}",
            ],
            interpretation=(
                "The first available unadjusted closing price on or after the "
                "requested start date is used as the theoretical entry price."
            ),
            example=(
                f"A 30% weight in a {currency_symbol}10,000 portfolio "
                f"allocates {currency_symbol}3,000."
            ),
            common_mistake=(
                "Assuming every security trades on the requested calendar date."
            ),
            limitation=(
                "Real brokers may impose minimum orders, fractional-share "
                "limits and transaction costs."
            ),
        )

        render_topic(
            title="Portfolio Currency and Daily FX Conversion",
            result_cards=[
                (
                    "Portfolio Currency",
                    f"{portfolio_currency} ({currency_symbol})",
                ),
                (
                    "Foreign-Currency Assets",
                    str(converted_asset_count),
                ),
            ],
            meaning=(
                "The portfolio currency is the common monetary unit for "
                "prices, values and risk metrics. Foreign prices are converted "
                "daily, so FX movements enter returns."
            ),
            professional_formulas=[
                r"X_{B/L,t}"
                r"=\frac{X_{B/EUR,t}}{X_{L/EUR,t}}",
                r"P_{i,t}^{B}=P_{i,t}^{L}X_{B/L,t}",
                r"1+R_{i,t}^{B}"
                r"=(1+R_{i,t}^{L})(1+R_{FX,t})",
            ],
            interpretation=(
                "The same foreign asset can have different measured returns "
                "in EUR and USD because local asset performance and currency "
                "movement interact."
            ),
            example=(
                "A U.S. stock can rise in USD but produce a smaller EUR gain "
                "when the dollar weakens."
            ),
            common_mistake=(
                "Changing only the displayed currency symbol."
            ),
            limitation=(
                "ECB reference rates can differ from broker execution rates "
                "and intraday FX prices."
            ),
        )

        render_topic(
            title="Contribution to Portfolio Return",
            result_cards=[
                (
                    "Largest Contributor",
                    str(return_contributor["Asset"]),
                ),
                (
                    "Contribution",
                    fmt_number(
                        return_contributor[
                            "Return Contribution (p.p.)"
                        ],
                        suffix=" p.p.",
                    ),
                ),
            ],
            meaning=(
                "Return contribution decomposes arithmetic annualized "
                "portfolio return into weighted asset contributions."
            ),
            professional_formulas=[
                r"\widehat{\mu}_{p,\mathrm{ann}}"
                r"=\sum_{i=1}^{N}w_i"
                r"\widehat{\mu}_{i,\mathrm{ann}}",
                r"RC_i=w_i\widehat{\mu}_{i,\mathrm{ann}}",
            ],
            interpretation=(
                "A high-return asset can contribute little when its weight "
                "is small."
            ),
            example=(
                "A 20% weight multiplied by a 15% annualized return "
                "contributes 3 percentage points."
            ),
            common_mistake=(
                "Confusing asset return with portfolio contribution."
            ),
            limitation=(
                "The decomposition follows the constant-weight arithmetic "
                "return model."
            ),
        )

        render_topic(
            title="Risk Contribution and Marginal Risk",
            result_cards=[
                (
                    "Largest Risk Contributor",
                    str(risk_contributor["Asset"]),
                ),
                (
                    "Risk Contribution",
                    fmt_number(
                        risk_contributor["Risk Contribution (%)"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Risk contribution decomposes portfolio volatility and "
                "depends on weights, standalone risk and covariance."
            ),
            professional_formulas=[
                r"\sigma_p=\sqrt{w^{\mathsf T}\Sigma w}",
                r"MRC_i=\frac{(\Sigma w)_i}{\sigma_p}",
                r"CRC_i=w_iMRC_i",
                r"CRC_i^{\%}=\frac{CRC_i}{\sigma_p}",
            ],
            interpretation=(
                "Marginal risk approximates the volatility change from a very "
                "small increase in an asset weight."
            ),
            example=(
                "A volatile asset can contribute modest risk if it has a low "
                "weight or strong diversification benefits."
            ),
            common_mistake=(
                "Ranking contributions using standalone volatility alone."
            ),
            limitation=(
                "The covariance matrix is historical and can change sharply "
                "in market stress."
            ),
        )

    # ========================================================
    # DIVERSIFICATION
    # ========================================================

    with diversification_tab:
        render_topic(
            title="Correlation",
            result_cards=[
                (
                    "Portfolio–Benchmark Correlation",
                    fmt_number(
                        results["portfolio_benchmark_correlation"],
                        decimals=3,
                    ),
                ),
            ],
            meaning=(
                "Correlation measures the direction and strength of a linear "
                "relationship between two return series."
            ),
            professional_formulas=[
                r"\rho_{XY}"
                r"=\frac{\operatorname{Cov}(X,Y)}"
                r"{\sigma_X\sigma_Y}",
            ],
            interpretation=(
                "Correlation ranges from −1 to +1. Values near zero indicate "
                "weak linear association, not necessarily independence."
            ),
            example=(
                "Correlation 0.80 indicates a strong positive linear "
                "relationship."
            ),
            common_mistake=(
                "Interpreting correlation as causality."
            ),
            limitation=(
                "Correlation captures only linear dependence and often rises "
                "during crises."
            ),
        )

        render_topic(
            title="Correlation Matrix",
            result_cards=[
                (
                    "Assets in Matrix",
                    str(
                        len(
                            results[
                                "asset_correlation_matrix"
                            ].columns
                        )
                    ),
                ),
            ],
            meaning=(
                "The correlation matrix contains all pairwise asset-return "
                "correlations."
            ),
            professional_formulas=[
                r"\mathbf{R}=[\rho_{ij}]_{i,j=1}^{N}",
                r"\rho_{ii}=1,\qquad \rho_{ij}=\rho_{ji}",
            ],
            interpretation=(
                "The diagonal is one. Off-diagonal entries show historical "
                "relationships between different assets."
            ),
            example=(
                "A pairwise correlation of 0.20 generally offers more linear "
                "diversification than 0.90, all else equal."
            ),
            common_mistake=(
                "Comparing correlations calculated from mismatched dates or "
                "different currency bases."
            ),
            limitation=(
                "A full-sample matrix hides time variation."
            ),
        )

        render_topic(
            title="Average Portfolio Correlation",
            result_cards=[
                (
                    "Average Correlation",
                    fmt_number(
                        results["average_portfolio_correlation"],
                        decimals=3,
                    ),
                ),
            ],
            meaning=(
                "Average portfolio correlation is the mean of all unique "
                "off-diagonal pairwise correlations."
            ),
            professional_formulas=[
                r"\bar{\rho}"
                r"=\frac{2}{N(N-1)}"
                r"\sum_{i<j}\rho_{ij}",
            ],
            interpretation=(
                "Lower values generally indicate greater diversification "
                "potential, but total risk also depends on weights and "
                "individual volatilities."
            ),
            example=(
                "Two portfolios can have the same average correlation but "
                "different volatility."
            ),
            common_mistake=(
                "Treating average correlation as a complete diversification "
                "score."
            ),
            limitation=(
                "An average can hide highly correlated clusters."
            ),
        )

        render_topic(
            title="Lowest Absolute Average Correlation Asset",
            result_cards=[
                (
                    "Lowest Absolute Average Correlation Asset",
                    lowest_absolute_correlation_asset,
                ),
                (
                    "Average Correlation",
                    fmt_number(
                        lowest_average_correlation_value,
                        decimals=3,
                    ),
                ),
                (
                    "Absolute Average Correlation",
                    fmt_number(
                        lowest_absolute_correlation_value,
                        decimals=3,
                    ),
                ),
            ],
            meaning=(
                "This indicator identifies the asset whose average historical "
                "correlation with all the other portfolio assets is closest "
                "to zero. The ranking is based on the absolute value, so the "
                "positive or negative sign does not determine which asset is "
                "selected."
            ),
            professional_formulas=[
                r"\bar{\rho}_i"
                r"=\frac{1}{N-1}\sum_{j\neq i}\rho_{ij}",
                r"i^*=\arg\min_i\left|\bar{\rho}_i\right|",
            ],
            symbols=[
                (
                    r"\bar{\rho}_i",
                    "average correlation of asset i with the other assets",
                ),
                (
                    r"\left|\bar{\rho}_i\right|",
                    "absolute magnitude of that average correlation",
                ),
                (
                    r"i^*",
                    "asset with the smallest absolute average correlation",
                ),
            ],
            interpretation=[
                (
                    "A value close to zero indicates weak average linear "
                    "co-movement with the rest of the portfolio."
                ),
                (
                    "A negative value indicates average movement in the "
                    "opposite direction, while a positive value indicates "
                    "average movement in the same direction. The sign is "
                    "shown for context but is ignored in the ranking."
                ),
            ],
            example=(
                "If one asset has an average correlation of -0.14 and another "
                "has 0.02, their absolute values are 0.14 and 0.02. The asset "
                "with 0.02 is selected because its average relationship is "
                "closer to zero."
            ),
            common_mistake=(
                "Choosing the most negative correlation. A more negative "
                "number is numerically lower, but it is not necessarily "
                "closer to zero."
            ),
            limitation=(
                "This is not a complete measure of diversification benefit. "
                "It ignores portfolio weights, asset volatility, covariance "
                "contribution, liquidity and tail dependence."
            ),
            methodology=(
                "Finance Bro calculates each asset's average correlation with "
                "the other portfolio assets and selects the smallest absolute "
                "average correlation."
            ),
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    with quality_tab:
        render_topic(
            title="Raw Market Dates and Common Price Dates",
            result_cards=[
                (
                    "Raw Market Dates",
                    f"{quality['raw_market_dates']:,}",
                ),
                (
                    "Common Price Dates",
                    f"{quality['common_price_dates']:,}",
                ),
                (
                    "Dates Removed",
                    f"{quality['dates_removed']:,}",
                ),
            ],
            meaning=(
                "Raw dates are initially downloaded rows. Common price dates "
                "remain after requiring every asset to have a valid converted "
                "price."
            ),
            professional_formulas=[
                r"\mathcal{T}_{\mathrm{common}}"
                r"=\bigcap_{i=1}^{N}\mathcal{T}_i",
            ],
            interpretation=(
                "Strict common-date alignment ensures synchronous observations "
                "for returns, covariance and correlation."
            ),
            example=(
                "A date can be removed when one exchange is closed and "
                "another is open."
            ),
            common_mistake=(
                "Treating returns from different dates as synchronous."
            ),
            limitation=(
                "Strict alignment can shorten multi-country samples."
            ),
        )

        render_topic(
            title="Common-Date Retention",
            result_cards=[
                (
                    "Common-Date Retention",
                    fmt_number(
                        quality["data_retention_percent"],
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Common-date retention is the percentage of raw market-date "
                "rows preserved after strict alignment."
            ),
            professional_formulas=[
                r"Retention"
                r"=\frac{|\mathcal{T}_{\mathrm{common}}|}"
                r"{|\mathcal{T}_{\mathrm{raw}}|}\times100",
            ],
            interpretation=(
                "100% means no raw date row was lost during alignment. A "
                "lower value means at least one asset lacked a valid price on "
                "some dates."
            ),
            example=(
                "480 common dates from 500 raw dates equals 96% retention."
            ),
            common_mistake=(
                "Interpreting retention as portfolio performance."
            ),
            limitation=(
                "High retention does not guarantee error-free prices."
            ),
        )

        render_topic(
            title="Price Coverage and Missing Prices",
            result_cards=[
                (
                    "Overall Coverage",
                    fmt_number(
                        quality[
                            "overall_price_cell_coverage_percent"
                        ],
                        suffix="%",
                    ),
                ),
                (
                    "Missing Prices",
                    f"{total_missing_prices:,}",
                ),
                (
                    "Lowest-Coverage Asset",
                    lowest_coverage_asset,
                ),
                (
                    "Lowest Coverage",
                    fmt_number(
                        lowest_coverage_value,
                        suffix="%",
                    ),
                ),
            ],
            meaning=(
                "Price-cell coverage is the proportion of possible asset-date "
                "cells containing valid observations before strict filtering."
            ),
            professional_formulas=[
                r"Coverage"
                r"=\frac{\sum_{i,t}"
                r"\mathbf{1}\{P_{i,t}\ \mathrm{valid}\}}"
                r"{N|\mathcal{T}_{\mathrm{raw}}|}\times100",
            ],
            interpretation=(
                "Coverage identifies incomplete histories and exchange-calendar "
                "differences."
            ),
            example=(
                "297 valid cells out of 300 possible cells equals 99% coverage."
            ),
            common_mistake=(
                "Confusing price-cell coverage with common-date retention."
            ),
            limitation=(
                "A non-missing price can still be stale or incorrect."
            ),
        )

        render_topic(
            title="Why Finance Bro Does Not Interpolate Stock Prices",
            meaning=(
                "Linear interpolation creates synthetic prices between real "
                "observations. Finance Bro does not interpolate, forward-fill "
                "or backward-fill missing stock prices."
            ),
            professional_formulas=[
                r"\widetilde{P}_t"
                r"=P_{t_0}"
                r"+\frac{t-t_0}{t_1-t_0}"
                r"(P_{t_1}-P_{t_0})",
            ],
            interpretation=(
                "Synthetic prices can create artificial returns and distort "
                "volatility, covariance, beta and tail-risk estimates."
            ),
            example=(
                "Inserting 105 between observed prices of 100 and 110 assumes "
                "a smooth path that may never have traded."
            ),
            common_mistake=(
                "Treating an interpolated price as an actual market close."
            ),
            limitation=(
                "Strict deletion reduces the sample, but the app prioritizes "
                "transparent observed prices."
            ),
            methodology=(
                "FX and risk-free series may use the latest official value "
                "available on or before the date. This is an as-of alignment "
                "rule, not linear price interpolation."
            ),
        )

        render_topic(
            title="Potential Return Anomalies",
            result_cards=[
                (
                    "Flagged Observations",
                    str(quality["potential_anomaly_count"]),
                ),
            ],
            meaning=(
                "Potential anomalies are unusually large returns flagged for "
                "review. They remain in the analysis because an extreme "
                "observation can be genuine."
            ),
            professional_formulas=[
                r"M_t"
                r"=0.67448975"
                r"\frac{r_t-\operatorname{median}(r)}"
                r"{\operatorname{median}"
                r"(|r_t-\operatorname{median}(r)|)}",
            ],
            interpretation=(
                "Finance Bro flags |M| > 3.5, absolute returns above 50%, or "
                "simple returns at or below −100%."
            ),
            example=(
                "Earnings surprises, takeovers and crashes can generate real "
                "statistical outliers."
            ),
            common_mistake=(
                "Automatically deleting every flagged observation."
            ),
            limitation=(
                "The rule can flag genuine events and miss subtle data errors."
            ),
        )

        render_topic(
            title="Actual Analysis Period and Alignment",
            result_cards=[
                (
                    "Actual Start",
                    fmt_date(quality["common_first_date"]),
                ),
                (
                    "Actual End",
                    fmt_date(quality["common_last_date"]),
                ),
                (
                    "Portfolio Returns",
                    f"{quality['portfolio_return_observations']:,}",
                ),
                (
                    "Regression Observations",
                    f"{quality['regression_observations']:,}",
                ),
            ],
            meaning=(
                "The actual period is determined by valid common asset prices, "
                "benchmark returns and risk-free observations."
            ),
            interpretation=(
                "The first return requires two prices, so the return sample "
                "contains one fewer observation than the common-price sample."
            ),
            example=(
                "A weekend start date moves to the first valid trading day; a "
                "holiday end date uses the final available observation on or "
                "before it."
            ),
            common_mistake=(
                "Assuming every metric uses the raw requested calendar dates."
            ),
            limitation=(
                "Daily alignment does not guarantee identical intraday "
                "information timestamps across global markets."
            ),
        )

    # ========================================================
    # STRESS TESTING
    # ========================================================

    with stress_tab:
        render_topic(
            title="Beta-Based Market Correction",
            result_cards=[
                (
                    "Benchmark Shock",
                    fmt_number(
                        results["benchmark_stress_shock"],
                        suffix="%",
                    ),
                ),
                (
                    "Portfolio Impact",
                    fmt_number(
                        market_stress["Portfolio Change (%)"],
                        suffix="%",
                    ),
                ),
                (
                    "Stressed Value",
                    fmt_money(
                        market_stress["Stressed Portfolio Value"]
                    ),
                ),
            ],
            meaning=(
                "The market-correction scenario applies a benchmark shock and "
                "maps it to each asset using estimated beta."
            ),
            professional_formulas=[
                r"s_i=\widehat{\beta}_i s_b",
                r"\Delta V_i=V_i s_i",
                r"\Delta V_p=\sum_{i=1}^{N}\Delta V_i",
            ],
            interpretation=(
                "It translates historical market sensitivity into a "
                "hypothetical immediate portfolio impact."
            ),
            example=(
                "Beta 1.3 and benchmark shock −10% imply an asset shock of "
                "−13% in the linear scenario."
            ),
            common_mistake=(
                "Interpreting the scenario as a forecast or probability."
            ),
            limitation=(
                "Betas can change in crises and relationships can become "
                "nonlinear."
            ),
        )

        render_topic(
            title="Custom Stress Scenario",
            result_cards=[
                ("Scenario", str(custom_stress["Scenario"])),
                (
                    "Portfolio Impact",
                    fmt_number(
                        custom_stress["Portfolio Change (%)"],
                        suffix="%",
                    ),
                ),
                (
                    "Stressed Value",
                    fmt_money(
                        custom_stress["Stressed Portfolio Value"]
                    ),
                ),
            ],
            meaning=(
                "A custom scenario applies user-defined percentage shocks "
                "directly to each asset position."
            ),
            professional_formulas=[
                r"\Delta V_p"
                r"=\sum_{i=1}^{N}V_i s_i^{\mathrm{custom}}",
                r"V_p^{\mathrm{stress}}=V_p+\Delta V_p",
            ],
            interpretation=(
                "It supports asset-specific assumptions instead of one common "
                "market-factor shock."
            ),
            example=(
                "A technology sell-off can assign larger negative shocks to "
                "technology positions."
            ),
            common_mistake=(
                "Using arbitrary shocks without an economic narrative."
            ),
            limitation=(
                "The scenario is deterministic and has no assigned probability."
            ),
        )

        render_topic(
            title="Stress Loss Contribution",
            result_cards=[
                (
                    "Market Largest Loss Contributor",
                    str(
                        market_stress[
                            "Largest Loss Contributor"
                        ]
                    ),
                ),
                (
                    "Market Most Resilient Asset",
                    str(
                        market_stress[
                            "Most Resilient Asset"
                        ]
                    ),
                ),
                (
                    "Custom Largest Loss Contributor",
                    str(
                        custom_stress[
                            "Largest Loss Contributor"
                        ]
                    ),
                ),
            ],
            meaning=(
                "Loss contribution identifies which positions create the "
                "largest monetary losses under a selected scenario."
            ),
            professional_formulas=[
                r"LC_i"
                r"=\frac{-\min(\Delta V_i,0)}"
                r"{\sum_j-\min(\Delta V_j,0)}",
            ],
            interpretation=(
                "A position can dominate losses because of a large weight, a "
                "severe shock or both."
            ),
            example=(
                "A moderate percentage shock can create the largest loss when "
                "the position is very large."
            ),
            common_mistake=(
                "Looking only at shock percentages and ignoring position size."
            ),
            limitation=(
                "Contributions are conditional on the chosen static scenario."
            ),
        )

        render_topic(
            title="What Stress Tests Do — and Do Not — Tell You",
            meaning=(
                "Stress testing answers: what would happen to the current "
                "portfolio if the stated shocks occurred immediately?"
            ),
            interpretation=[
                (
                    "Stress tests reveal concentration, sensitivity and "
                    "potential loss channels."
                ),
                (
                    "They do not estimate the probability of the scenario."
                ),
                (
                    "They do not automatically capture liquidity stress, "
                    "trading halts, feedback effects, defaults, nonlinear "
                    "derivatives or changing correlations."
                ),
            ],
            example=(
                "A −20% scenario can be useful even when its probability is "
                "unknown because it tests portfolio resilience."
            ),
            common_mistake=(
                "Calling a stress scenario a prediction, expected loss or "
                "confidence interval."
            ),
            limitation=(
                "Scenario quality depends on economic plausibility and "
                "coverage of relevant risk factors."
            ),
        )

