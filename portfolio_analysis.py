from io import StringIO
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    linear_reset,
)
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson, jarque_bera


CURRENCY_SYMBOLS = {
    "EUR": "€",
    "USD": "$",
}


def validate_portfolio_currency(
    portfolio_currency: str,
) -> str:
    """
    Valida e normaliza a moeda-base da carteira.
    """

    portfolio_currency = (
        portfolio_currency
        .strip()
        .upper()
    )

    if portfolio_currency not in CURRENCY_SYMBOLS:
        raise ValueError(
            "Portfolio currency must be EUR or USD."
        )

    return portfolio_currency


def get_currency_symbol(
    currency: str,
) -> str:
    """
    Devolve o símbolo da moeda-base da carteira.
    """

    currency = validate_portfolio_currency(
        currency
    )

    return CURRENCY_SYMBOLS[
        currency
    ]


def normalize_currency_code_and_multiplier(
    currency: str,
) -> tuple[str, float]:
    """
    Converte moedas reportadas em unidades menores para a unidade principal.

    Exemplos:
    GBp/GBX -> GBP
    ZAc -> ZAR
    ILA -> ILS
    """

    currency = str(
        currency
    ).strip()

    minor_unit_currencies = {
        "GBP": ("GBP", 1.0),
        "GBp": ("GBP", 0.01),
        "GBX": ("GBP", 0.01),
        "ZAc": ("ZAR", 0.01),
        "ILA": ("ILS", 0.01),
    }

    if currency in minor_unit_currencies:
        return minor_unit_currencies[
            currency
        ]

    return (
        currency.upper(),
        1.0,
    )


def normalize_currency_and_price(
    currency: str,
    price: float,
) -> tuple[str, float]:
    """
    Normaliza o código da moeda e o preço para a respetiva unidade principal.
    """

    (
        normalized_currency,
        multiplier,
    ) = normalize_currency_code_and_multiplier(
        currency
    )

    return (
        normalized_currency,
        float(price) * multiplier,
    )


def download_ecb_raw_reference_series(
    currency: str,
    start_date,
    end_date,
) -> pd.Series:
    """
    Descarrega observações oficiais do BCE.

    Cada observação representa unidades da moeda por 1 EUR.
    """

    currency = (
        str(currency)
        .strip()
        .upper()
    )

    start_timestamp = pd.Timestamp(
        start_date
    ).normalize()

    end_timestamp = pd.Timestamp(
        end_date
    ).normalize()

    if end_timestamp < start_timestamp:
        raise ValueError(
            "The ECB end date cannot be earlier than the start date."
        )

    if currency == "EUR":
        date_index = pd.date_range(
            start=start_timestamp,
            end=end_timestamp,
            freq="D",
        )

        return pd.Series(
            1.0,
            index=date_index,
            name="EUR per EUR",
            dtype=float,
        )

    series_key = (
        f"D.{currency}.EUR.SP00.A"
    )

    query_parameters = urlencode({
        "startPeriod":
            start_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "endPeriod":
            end_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "format":
            "csvdata",
    })

    request_url = (
        "https://data-api.ecb.europa.eu/"
        f"service/data/EXR/{series_key}"
        f"?{query_parameters}"
    )

    request = Request(
        request_url,
        headers={
            "User-Agent":
                "Finance Bro educational portfolio app"
        },
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            response_text = (
                response
                .read()
                .decode("utf-8")
            )

    except Exception as error:
        raise ValueError(
            f"The ECB exchange-rate series for {currency} "
            f"could not be downloaded: {error}"
        ) from error

    exchange_rate_data = pd.read_csv(
        StringIO(
            response_text
        )
    )

    required_columns = {
        "TIME_PERIOD",
        "OBS_VALUE",
    }

    if not required_columns.issubset(
        exchange_rate_data.columns
    ):
        raise ValueError(
            f"The ECB response for {currency} did not contain "
            "the expected exchange-rate fields."
        )

    exchange_rate_data[
        "TIME_PERIOD"
    ] = pd.to_datetime(
        exchange_rate_data[
            "TIME_PERIOD"
        ],
        errors="coerce",
    )

    exchange_rate_data[
        "OBS_VALUE"
    ] = pd.to_numeric(
        exchange_rate_data[
            "OBS_VALUE"
        ],
        errors="coerce",
    )

    exchange_rate_data = (
        exchange_rate_data
        .dropna(
            subset=[
                "TIME_PERIOD",
                "OBS_VALUE",
            ]
        )
        .sort_values(
            "TIME_PERIOD"
        )
        .drop_duplicates(
            subset="TIME_PERIOD",
            keep="last",
        )
    )

    if exchange_rate_data.empty:
        raise ValueError(
            f"No ECB reference-rate observations were available "
            f"for {currency} in the requested period."
        )

    exchange_rate_series = pd.Series(
        exchange_rate_data[
            "OBS_VALUE"
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            exchange_rate_data[
                "TIME_PERIOD"
            ]
        ).normalize(),
        name=f"{currency} per EUR",
    )

    exchange_rate_series = (
        exchange_rate_series[
            exchange_rate_series > 0
        ]
    )

    if exchange_rate_series.empty:
        raise ValueError(
            f"The ECB exchange-rate observations for {currency} "
            "were invalid."
        )

    return exchange_rate_series


def get_ecb_reference_rates_for_dates(
    currency: str,
    dates,
) -> pd.Series:
    """
    Alinha taxas diárias do BCE com as datas de mercado.

    Em fins de semana e feriados utiliza a última taxa oficial disponível.
    """

    original_index = pd.DatetimeIndex(
        pd.to_datetime(
            dates
        )
    )

    if original_index.tz is not None:
        original_index = (
            original_index
            .tz_localize(None)
        )

    normalized_dates = (
        original_index
        .normalize()
    )

    if len(normalized_dates) == 0:
        return pd.Series(
            dtype=float,
            index=original_index,
        )

    currency = (
        str(currency)
        .strip()
        .upper()
    )

    if currency == "EUR":
        return pd.Series(
            1.0,
            index=original_index,
            name="EUR per EUR",
            dtype=float,
        )

    lookback_start = (
        normalized_dates.min()
        - pd.Timedelta(days=15)
    )

    end_date = (
        normalized_dates.max()
    )

    raw_rates = (
        download_ecb_raw_reference_series(
            currency=currency,
            start_date=lookback_start,
            end_date=end_date,
        )
    )

    calendar_index = pd.date_range(
        start=lookback_start,
        end=end_date,
        freq="D",
    )

    filled_rates = (
        raw_rates
        .reindex(
            calendar_index
        )
        .ffill()
    )

    aligned_values = (
        filled_rates
        .reindex(
            normalized_dates
        )
        .to_numpy(
            dtype=float
        )
    )

    aligned_rates = pd.Series(
        aligned_values,
        index=original_index,
        name=f"{currency} per EUR",
    )

    if aligned_rates.isna().any():
        raise ValueError(
            f"Some ECB reference rates for {currency} "
            "could not be aligned with the market dates."
        )

    return aligned_rates


def get_ecb_cross_rates_for_dates(
    from_currency: str,
    to_currency: str,
    dates,
) -> pd.Series:
    """
    Calcula unidades da moeda de destino por unidade da moeda de origem.
    """

    from_currency = (
        str(from_currency)
        .strip()
        .upper()
    )

    to_currency = (
        str(to_currency)
        .strip()
        .upper()
    )

    original_index = pd.DatetimeIndex(
        pd.to_datetime(
            dates
        )
    )

    if original_index.tz is not None:
        original_index = (
            original_index
            .tz_localize(None)
        )

    if from_currency == to_currency:
        return pd.Series(
            1.0,
            index=original_index,
            name=(
                f"{to_currency} per "
                f"{from_currency}"
            ),
            dtype=float,
        )

    from_rates_per_eur = (
        get_ecb_reference_rates_for_dates(
            currency=from_currency,
            dates=original_index,
        )
    )

    to_rates_per_eur = (
        get_ecb_reference_rates_for_dates(
            currency=to_currency,
            dates=original_index,
        )
    )

    cross_rates = (
        to_rates_per_eur
        / from_rates_per_eur
    )

    cross_rates.name = (
        f"{to_currency} per "
        f"{from_currency}"
    )

    return cross_rates


def download_ecb_cross_rate(
    from_currency: str,
    to_currency: str,
    reference_date,
) -> tuple[float, pd.Timestamp]:
    """
    Obtém uma taxa cruzada oficial do BCE para uma data específica.

    A taxa representa unidades da moeda de destino por unidade da moeda
    de origem.
    """

    from_currency = (
        str(from_currency)
        .strip()
        .upper()
    )

    to_currency = (
        str(to_currency)
        .strip()
        .upper()
    )

    reference_timestamp = pd.Timestamp(
        reference_date
    ).normalize()

    if from_currency == to_currency:
        return (
            1.0,
            reference_timestamp,
        )

    lookback_start = (
        reference_timestamp
        - pd.Timedelta(days=15)
    )

    if from_currency == "EUR":
        to_series = (
            download_ecb_raw_reference_series(
                currency=to_currency,
                start_date=lookback_start,
                end_date=reference_timestamp,
            )
        )

        valid_dates = (
            to_series.index[
                to_series.index
                <= reference_timestamp
            ]
        )

        if len(valid_dates) == 0:
            raise ValueError(
                f"No ECB rate was available for {to_currency} "
                f"on or before {reference_timestamp.date()}."
            )

        fx_date = valid_dates.max()

        return (
            float(
                to_series.loc[
                    fx_date
                ]
            ),
            pd.Timestamp(
                fx_date
            ).normalize(),
        )

    if to_currency == "EUR":
        from_series = (
            download_ecb_raw_reference_series(
                currency=from_currency,
                start_date=lookback_start,
                end_date=reference_timestamp,
            )
        )

        valid_dates = (
            from_series.index[
                from_series.index
                <= reference_timestamp
            ]
        )

        if len(valid_dates) == 0:
            raise ValueError(
                f"No ECB rate was available for {from_currency} "
                f"on or before {reference_timestamp.date()}."
            )

        fx_date = valid_dates.max()

        return (
            1.0
            / float(
                from_series.loc[
                    fx_date
                ]
            ),
            pd.Timestamp(
                fx_date
            ).normalize(),
        )

    from_series = (
        download_ecb_raw_reference_series(
            currency=from_currency,
            start_date=lookback_start,
            end_date=reference_timestamp,
        )
    )

    to_series = (
        download_ecb_raw_reference_series(
            currency=to_currency,
            start_date=lookback_start,
            end_date=reference_timestamp,
        )
    )

    common_dates = (
        from_series.index
        .intersection(
            to_series.index
        )
    )

    common_dates = (
        common_dates[
            common_dates
            <= reference_timestamp
        ]
    )

    if len(common_dates) == 0:
        raise ValueError(
            f"No common ECB reference date was available for "
            f"{from_currency}/{to_currency} on or before "
            f"{reference_timestamp.date()}."
        )

    fx_date = common_dates.max()

    cross_rate = (
        float(
            to_series.loc[
                fx_date
            ]
        )
        / float(
            from_series.loc[
                fx_date
            ]
        )
    )

    return (
        cross_rate,
        pd.Timestamp(
            fx_date
        ).normalize(),
    )


def convert_price_series_to_currency(
    price_series: pd.Series,
    from_currency: str,
    to_currency: str,
) -> pd.Series:
    """
    Converte uma série histórica de preços para a moeda-base da carteira.
    """

    if price_series.empty:
        return price_series.copy()

    cross_rates = (
        get_ecb_cross_rates_for_dates(
            from_currency=from_currency,
            to_currency=to_currency,
            dates=price_series.index,
        )
    )

    converted_prices = (
        price_series.astype(float)
        * cross_rates.to_numpy(
            dtype=float
        )
    )

    converted_prices.name = (
        price_series.name
    )

    return converted_prices



def prepare_yfinance_end_date(
    end_date: Optional[object],
) -> Optional[pd.Timestamp]:
    """
    Converte uma data final inclusiva para o limite exclusivo exigido
    pelo Yahoo Finance.

    Exemplo:
        End Date escolhida: 31/12/2025
        Limite enviado ao Yahoo: 01/01/2026

    Desta forma, a observação de 31/12/2025 pode ser incluída.
    """

    if end_date is None:
        return None

    return (
        pd.Timestamp(
            end_date
        ).normalize()
        + pd.Timedelta(
            days=1
        )
    )


def download_portfolio_prices(
    tickers: list[str],
    start_date,
    end_date: Optional[object] = None,
    portfolio_currency: str = "EUR",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    dict,
]:
    """
    Downloads adjusted prices, converts them to the portfolio currency and
    documents every cleaning step.

    No price interpolation, backward filling or forward filling is performed.
    A date is retained only when every portfolio asset has a valid price.
    """

    portfolio_currency = validate_portfolio_currency(
        portfolio_currency
    )

    yahoo_end_date = prepare_yfinance_end_date(
        end_date
    )

    data = yf.download(
        tickers,
        start=start_date,
        end=yahoo_end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No market data was downloaded. "
            "Check the tickers and selected dates."
        )

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(
            name=tickers[0]
        )

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = (
            prices.columns
            .get_level_values(-1)
        )

    prices = prices.copy()
    prices.index = pd.DatetimeIndex(
        pd.to_datetime(prices.index)
    )

    if prices.index.tz is not None:
        prices.index = (
            prices.index
            .tz_localize(None)
        )

    prices = (
        prices
        .sort_index()
        .loc[
            ~prices.index.duplicated(
                keep="last"
            )
        ]
    )

    missing_tickers = [
        ticker
        for ticker in tickers
        if (
            ticker not in prices.columns
            or prices[ticker].dropna().empty
        )
    ]

    if missing_tickers:
        raise ValueError(
            "No valid data was found for: "
            + ", ".join(missing_tickers)
        )

    raw_prices = prices.reindex(
        columns=tickers
    )

    asset_currencies: dict[str, str] = {}
    local_price_series: dict[str, pd.Series] = {}
    converted_price_series: dict[str, pd.Series] = {}
    asset_quality_rows: list[dict] = []
    anomaly_frames: list[pd.DataFrame] = []

    raw_date_count = len(
        raw_prices.index
    )

    for ticker in tickers:

        yahoo_currency = (
            get_asset_trading_currency(
                ticker
            )
        )

        (
            trading_currency,
            price_multiplier,
        ) = normalize_currency_code_and_multiplier(
            yahoo_currency
        )

        asset_currencies[
            ticker
        ] = trading_currency

        normalized_local_prices = (
            raw_prices[
                ticker
            ].astype(float)
            * price_multiplier
        )

        normalized_local_prices.name = ticker

        local_price_series[
            ticker
        ] = normalized_local_prices

        converted_price_series[
            ticker
        ] = convert_price_series_to_currency(
            price_series=normalized_local_prices,
            from_currency=trading_currency,
            to_currency=portfolio_currency,
        )

    local_prices_union = pd.DataFrame(
        local_price_series
    )

    converted_prices_union = pd.DataFrame(
        converted_price_series
    )

    converted_prices = (
        converted_prices_union
        .dropna(
            how="any"
        )
    )

    local_prices = (
        local_prices_union
        .reindex(
            converted_prices.index
        )
    )

    if len(converted_prices) < 2:
        raise ValueError(
            "There are not enough common price observations "
            "after currency conversion and strict date alignment."
        )

    common_index = converted_prices.index
    common_date_count = len(
        common_index
    )

    for ticker in tickers:

        raw_asset_prices = (
            local_prices_union[
                ticker
            ]
        )

        valid_mask = (
            raw_asset_prices
            .notna()
        )

        valid_count = int(
            valid_mask.sum()
        )

        missing_count = int(
            raw_date_count
            - valid_count
        )

        valid_prices = (
            raw_asset_prices[
                valid_mask
            ]
        )

        first_valid_date = (
            valid_prices.index.min()
            if not valid_prices.empty
            else pd.NaT
        )

        last_valid_date = (
            valid_prices.index.max()
            if not valid_prices.empty
            else pd.NaT
        )

        removed_by_alignment = int(
            (
                valid_mask
                & ~raw_asset_prices.index.isin(
                    common_index
                )
            ).sum()
        )

        local_returns = (
            valid_prices
            .pct_change(
                fill_method=None
            )
            .dropna()
        )

        asset_anomalies = (
            detect_return_anomalies(
                return_series=local_returns,
                series_name=ticker,
                return_basis="Local adjusted price return",
            )
        )

        anomaly_frames.append(
            asset_anomalies
        )

        largest_absolute_return = (
            float(
                local_returns.abs().max()
                * 100
            )
            if not local_returns.empty
            else np.nan
        )

        asset_quality_rows.append({
            "Asset":
                ticker,
            "Trading Currency":
                trading_currency,
            "Portfolio Currency":
                portfolio_currency,
            "First Valid Date":
                first_valid_date,
            "Last Valid Date":
                last_valid_date,
            "Raw Date Rows":
                raw_date_count,
            "Valid Prices":
                valid_count,
            "Missing Prices":
                missing_count,
            "Coverage (%)":
                (
                    valid_count
                    / raw_date_count
                    * 100
                    if raw_date_count > 0
                    else np.nan
                ),
            "Dates Removed by Common Alignment":
                removed_by_alignment,
            "Potential Return Anomalies":
                int(
                    len(
                        asset_anomalies
                    )
                ),
            "Largest Absolute Daily Return (%)":
                largest_absolute_return,
        })

    anomalies = (
        pd.concat(
            anomaly_frames,
            ignore_index=True,
        )
        if anomaly_frames
        else pd.DataFrame()
    )

    total_possible_price_cells = (
        raw_date_count
        * len(tickers)
    )

    total_valid_price_cells = int(
        raw_prices
        .notna()
        .sum()
        .sum()
    )

    portfolio_quality = {
        "requested_start_date":
            pd.Timestamp(
                start_date
            ).normalize(),
        "requested_end_date":
            (
                pd.Timestamp(
                    end_date
                ).normalize()
                if end_date is not None
                else None
            ),
        "raw_first_date":
            raw_prices.index.min(),
        "raw_last_date":
            raw_prices.index.max(),
        "common_first_date":
            converted_prices.index.min(),
        "common_last_date":
            converted_prices.index.max(),
        "raw_market_dates":
            raw_date_count,
        "common_price_dates":
            common_date_count,
        "dates_removed":
            int(
                raw_date_count
                - common_date_count
            ),
        "common_date_retention_percent":
            (
                common_date_count
                / raw_date_count
                * 100
                if raw_date_count > 0
                else np.nan
            ),
        "total_possible_price_cells":
            total_possible_price_cells,
        "total_valid_price_cells":
            total_valid_price_cells,
        "overall_price_cell_coverage_percent":
            (
                total_valid_price_cells
                / total_possible_price_cells
                * 100
                if total_possible_price_cells > 0
                else np.nan
            ),
        "asset_quality_table":
            pd.DataFrame(
                asset_quality_rows
            ),
        "anomalies":
            anomalies,
        "price_cleaning_method":
            (
                "Strict common-date alignment; no price interpolation, "
                "forward filling or backward filling."
            ),
        "fx_alignment_method":
            (
                "ECB exchange rates use the last official observation "
                "available on or before each market date; no linear "
                "interpolation is used."
            ),
    }

    converted_prices.index.name = "Date"
    local_prices.index.name = "Date"

    return (
        converted_prices,
        local_prices,
        asset_currencies,
        portfolio_quality,
    )




def download_benchmark_prices(
    benchmark_ticker: str,
    start_date,
    end_date: Optional[object] = None,
    portfolio_currency: str = "EUR",
) -> tuple[
    pd.Series,
    pd.Series,
    str,
    dict,
]:
    """
    Downloads and converts benchmark prices without price interpolation.
    """

    benchmark_ticker = (
        benchmark_ticker
        .strip()
        .upper()
    )

    portfolio_currency = (
        validate_portfolio_currency(
            portfolio_currency
        )
    )

    yahoo_end_date = prepare_yfinance_end_date(
        end_date
    )

    data = yf.download(
        benchmark_ticker,
        start=start_date,
        end=yahoo_end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No benchmark data was found for {benchmark_ticker}."
        )

    benchmark_prices = data[
        "Close"
    ]

    if isinstance(
        benchmark_prices,
        pd.DataFrame,
    ):
        benchmark_prices = (
            benchmark_prices
            .iloc[:, 0]
        )

    benchmark_prices = benchmark_prices.copy()
    benchmark_prices.index = pd.DatetimeIndex(
        pd.to_datetime(
            benchmark_prices.index
        )
    )

    if benchmark_prices.index.tz is not None:
        benchmark_prices.index = (
            benchmark_prices.index
            .tz_localize(None)
        )

    benchmark_prices = (
        benchmark_prices
        .sort_index()
        .loc[
            ~benchmark_prices.index.duplicated(
                keep="last"
            )
        ]
    )

    raw_date_rows = len(
        benchmark_prices
    )

    valid_benchmark_prices = (
        benchmark_prices
        .dropna()
        .astype(float)
    )

    if len(valid_benchmark_prices) < 2:
        raise ValueError(
            "There are not enough benchmark price observations."
        )

    yahoo_currency = (
        get_asset_trading_currency(
            benchmark_ticker
        )
    )

    (
        benchmark_currency,
        price_multiplier,
    ) = normalize_currency_code_and_multiplier(
        yahoo_currency
    )

    benchmark_prices_local = (
        valid_benchmark_prices
        * price_multiplier
    )

    benchmark_prices_local.name = (
        benchmark_ticker
    )

    benchmark_prices_converted = (
        convert_price_series_to_currency(
            price_series=benchmark_prices_local,
            from_currency=benchmark_currency,
            to_currency=portfolio_currency,
        )
    )

    benchmark_prices_converted.name = (
        benchmark_ticker
    )

    benchmark_returns_local = (
        benchmark_prices_local
        .pct_change(
            fill_method=None
        )
        .dropna()
    )

    benchmark_anomalies = (
        detect_return_anomalies(
            return_series=benchmark_returns_local,
            series_name=benchmark_ticker,
            return_basis="Benchmark local adjusted price return",
        )
    )

    benchmark_quality = {
        "benchmark":
            benchmark_ticker,
        "benchmark_currency":
            benchmark_currency,
        "raw_date_rows":
            raw_date_rows,
        "valid_price_observations":
            int(
                valid_benchmark_prices
                .notna()
                .sum()
            ),
        "missing_price_observations":
            int(
                raw_date_rows
                - valid_benchmark_prices
                .notna()
                .sum()
            ),
        "first_valid_date":
            valid_benchmark_prices.index.min(),
        "last_valid_date":
            valid_benchmark_prices.index.max(),
        "potential_return_anomalies":
            int(
                len(
                    benchmark_anomalies
                )
            ),
        "largest_absolute_daily_return_percent":
            float(
                benchmark_returns_local
                .abs()
                .max()
                * 100
            ),
        "anomalies":
            benchmark_anomalies,
    }

    benchmark_prices_converted.index.name = "Date"
    benchmark_prices_local.index.name = "Date"

    return (
        benchmark_prices_converted,
        benchmark_prices_local,
        benchmark_currency,
        benchmark_quality,
    )



def calculate_stress_scenario(
    weights: pd.Series,
    current_portfolio_value: float,
    asset_shocks: pd.Series,
    scenario_name: str,
    scenario_type: str,
    asset_betas: Optional[pd.Series] = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Calcula o impacto de um cenário de stress por ativo.

    Os montantes são expressos na moeda-base escolhida para a carteira.
    """

    if current_portfolio_value <= 0:
        raise ValueError(
            "Current portfolio value must be positive."
        )

    weights = (
        weights
        .astype(float)
    )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):
        raise ValueError(
            "Portfolio weights must sum to 100%."
        )

    asset_shocks = (
        asset_shocks
        .astype(float)
        .reindex(weights.index)
    )

    if asset_shocks.isna().any():
        missing_assets = asset_shocks[
            asset_shocks.isna()
        ].index.tolist()

        raise ValueError(
            "Stress shocks are missing for: "
            + ", ".join(missing_assets)
        )

    if (asset_shocks < -1.0).any():
        raise ValueError(
            "An asset shock cannot be lower than -100%."
        )

    if asset_betas is None:
        asset_betas = pd.Series(
            np.nan,
            index=weights.index,
            dtype=float,
        )
    else:
        asset_betas = (
            asset_betas
            .astype(float)
            .reindex(weights.index)
        )

    current_position_values = (
        current_portfolio_value
        * weights
    )

    asset_value_changes = (
        current_position_values
        * asset_shocks
    )

    stressed_position_values = (
        current_position_values
        + asset_value_changes
    )

    portfolio_change_money = float(
        asset_value_changes.sum()
    )

    portfolio_change_percent = float(
        portfolio_change_money
        / current_portfolio_value
        * 100
    )

    stressed_portfolio_value = float(
        current_portfolio_value
        + portfolio_change_money
    )

    portfolio_impact_percentage_points = (
        weights
        * asset_shocks
        * 100
    )

    asset_losses = (
        -asset_value_changes.clip(
            upper=0
        )
    )

    total_asset_losses = float(
        asset_losses.sum()
    )

    if np.isclose(
        total_asset_losses,
        0,
    ):
        loss_contribution_percentage = pd.Series(
            0.0,
            index=weights.index,
        )
    else:
        loss_contribution_percentage = (
            asset_losses
            / total_asset_losses
            * 100
        )

    negative_changes = asset_value_changes[
        asset_value_changes < 0
    ]

    if negative_changes.empty:
        largest_loss_contributor = "None"
    else:
        largest_loss_contributor = (
            negative_changes.idxmin()
        )

    most_resilient_asset = (
        asset_value_changes.idxmax()
    )

    scenario_summary = {
        "Scenario":
            scenario_name,
        "Scenario Type":
            scenario_type,
        "Portfolio Change (%)":
            portfolio_change_percent,
        "Portfolio Change":
            portfolio_change_money,
        "Stressed Portfolio Value":
            stressed_portfolio_value,
        "Largest Loss Contributor":
            largest_loss_contributor,
        "Most Resilient Asset":
            most_resilient_asset,
    }

    scenario_detail = pd.DataFrame({
        "Asset":
            weights.index,
        "Weight (%)":
            weights.values * 100,
        "Beta":
            asset_betas.values,
        "Shock (%)":
            asset_shocks.values * 100,
        "Portfolio Impact (p.p.)":
            portfolio_impact_percentage_points.values,
        "Current Position Value":
            current_position_values.values,
        "Value Change":
            asset_value_changes.values,
        "Loss Contribution (%)":
            loss_contribution_percentage.values,
        "Stressed Position Value":
            stressed_position_values.values,
    })

    return (
        scenario_summary,
        scenario_detail,
    )


def get_asset_trading_currency(
    ticker: str,
) -> str:
    """
    Obtém a moeda de negociação reportada pelo Yahoo Finance.
    """

    ticker_object = yf.Ticker(
        ticker
    )

    currency = None

    try:
        fast_info = (
            ticker_object
            .fast_info
        )

        currency = getattr(
            fast_info,
            "currency",
            None,
        )

        if currency is None and hasattr(
            fast_info,
            "get",
        ):
            currency = fast_info.get(
                "currency"
            )

    except Exception:
        currency = None

    if not currency:

        try:
            ticker_information = (
                ticker_object
                .get_info()
            )

            currency = ticker_information.get(
                "currency"
            )

        except Exception:
            currency = None

    if not currency:
        raise ValueError(
            f"The trading currency could not be identified for {ticker}."
        )

    return str(
        currency
    ).strip()


def download_entry_price(
    ticker: str,
    requested_start_date,
    known_trading_currency: Optional[str] = None,
) -> tuple[pd.Timestamp, float, str]:
    """
    Obtém o primeiro fecho não ajustado disponível na data ou depois dela.
    """

    requested_date = pd.Timestamp(
        requested_start_date
    ).normalize()

    search_end_date = (
        requested_date
        + pd.Timedelta(days=31)
    )

    ticker_object = yf.Ticker(
        ticker
    )

    price_history = ticker_object.history(
        start=requested_date,
        end=search_end_date,
        auto_adjust=False,
        actions=False,
    )

    if price_history.empty:
        raise ValueError(
            f"No entry-price data was found for {ticker} "
            f"on or shortly after {requested_date.date()}."
        )

    if "Close" not in price_history.columns:
        raise ValueError(
            f"The unadjusted closing price is unavailable for {ticker}."
        )

    valid_closing_prices = (
        price_history[
            "Close"
        ]
        .dropna()
    )

    if valid_closing_prices.empty:
        raise ValueError(
            f"No valid entry price was found for {ticker}."
        )

    entry_timestamp = pd.Timestamp(
        valid_closing_prices.index[
            0
        ]
    )

    if entry_timestamp.tzinfo is not None:
        entry_timestamp = (
            entry_timestamp
            .tz_localize(None)
        )

    entry_date = (
        entry_timestamp
        .normalize()
    )

    raw_entry_price = float(
        valid_closing_prices.iloc[
            0
        ]
    )

    if known_trading_currency is None:
        yahoo_currency = (
            get_asset_trading_currency(
                ticker
            )
        )
    else:
        yahoo_currency = (
            known_trading_currency
        )

    (
        trading_currency,
        entry_price_local,
    ) = normalize_currency_and_price(
        currency=yahoo_currency,
        price=raw_entry_price,
    )

    return (
        entry_date,
        entry_price_local,
        trading_currency,
    )


def calculate_initial_portfolio_construction(
    tickers: list[str],
    weights: pd.Series,
    initial_investment: float,
    requested_start_date,
    portfolio_currency: str,
    asset_currencies: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Reconstrói a compra teórica inicial com ações fracionadas e câmbio BCE.
    """

    portfolio_currency = (
        validate_portfolio_currency(
            portfolio_currency
        )
    )

    requested_date = pd.Timestamp(
        requested_start_date
    ).normalize()

    construction_rows = []
    fx_cache = {}

    for ticker in tickers:

        known_currency = None

        if asset_currencies is not None:
            known_currency = (
                asset_currencies.get(
                    ticker
                )
            )

        (
            entry_date,
            entry_price_local,
            trading_currency,
        ) = download_entry_price(
            ticker=ticker,
            requested_start_date=requested_date,
            known_trading_currency=known_currency,
        )

        fx_cache_key = (
            trading_currency,
            portfolio_currency,
            entry_date,
        )

        if fx_cache_key not in fx_cache:

            fx_cache[
                fx_cache_key
            ] = download_ecb_cross_rate(
                from_currency=trading_currency,
                to_currency=portfolio_currency,
                reference_date=entry_date,
            )

        (
            cross_rate,
            exchange_rate_date,
        ) = fx_cache[
            fx_cache_key
        ]

        entry_price_portfolio_currency = (
            entry_price_local
            * cross_rate
        )

        allocated_amount = (
            initial_investment
            * float(
                weights.loc[
                    ticker
                ]
            )
        )

        fractional_shares = (
            allocated_amount
            / entry_price_portfolio_currency
        )

        construction_rows.append({
            "Asset":
                ticker,
            "Requested Start Date":
                requested_date.date(),
            "Entry Date Used":
                entry_date.date(),
            "Trading Currency":
                trading_currency,
            "Portfolio Currency":
                portfolio_currency,
            "Entry Price (Local)":
                entry_price_local,
            "ECB Cross Rate (Portfolio per Local)":
                cross_rate,
            "FX Date Used":
                exchange_rate_date.date(),
            "Entry Price (Portfolio Currency)":
                entry_price_portfolio_currency,
            "Target Weight (%)":
                float(
                    weights.loc[
                        ticker
                    ]
                ) * 100,
            "Amount Invested (Portfolio Currency)":
                allocated_amount,
            "Fractional Shares Purchased":
                fractional_shares,
        })

    return pd.DataFrame(
        construction_rows
    )



def detect_return_anomalies(
    return_series: pd.Series,
    series_name: str,
    return_basis: str,
    modified_z_threshold: float = 3.5,
    absolute_return_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Flags potential anomalies without deleting or modifying observations.

    The robust modified z-score is:
        M_t = 0.67448975 * (r_t - median(r)) / MAD

    An observation is flagged when |M_t| > 3.5, when the absolute return
    exceeds 50%, or when the simple return is less than or equal to -100%.
    """

    returns = (
        pd.Series(
            return_series,
            dtype=float,
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    columns = [
        "Date",
        "Series",
        "Return Basis",
        "Return (%)",
        "Modified Z-Score",
        "Flag Rule",
    ]

    if returns.empty:
        return pd.DataFrame(
            columns=columns
        )

    median_return = float(
        returns.median()
    )

    median_absolute_deviation = float(
        np.median(
            np.abs(
                returns.to_numpy()
                - median_return
            )
        )
    )

    if np.isclose(
        median_absolute_deviation,
        0.0,
    ):
        modified_z_scores = pd.Series(
            np.nan,
            index=returns.index,
            dtype=float,
        )
    else:
        modified_z_scores = (
            0.6744897501960817
            * (
                returns
                - median_return
            )
            / median_absolute_deviation
        )

    robust_flag = (
        modified_z_scores
        .abs()
        > modified_z_threshold
    )

    absolute_flag = (
        returns.abs()
        > absolute_return_threshold
    )

    impossible_loss_flag = (
        returns
        <= -1.0
    )

    combined_flag = (
        robust_flag.fillna(False)
        | absolute_flag
        | impossible_loss_flag
    )

    flagged_returns = (
        returns[
            combined_flag
        ]
    )

    if flagged_returns.empty:
        return pd.DataFrame(
            columns=columns
        )

    anomaly_rows = []

    for date, value in flagged_returns.items():

        rules = []

        if bool(
            robust_flag.loc[
                date
            ]
        ):
            rules.append(
                f"|Modified z| > {modified_z_threshold:.1f}"
            )

        if bool(
            absolute_flag.loc[
                date
            ]
        ):
            rules.append(
                f"|Return| > {absolute_return_threshold * 100:.0f}%"
            )

        if bool(
            impossible_loss_flag.loc[
                date
            ]
        ):
            rules.append(
                "Simple return <= -100%"
            )

        anomaly_rows.append({
            "Date":
                pd.Timestamp(
                    date
                ),
            "Series":
                series_name,
            "Return Basis":
                return_basis,
            "Return (%)":
                float(
                    value
                    * 100
                ),
            "Modified Z-Score":
                float(
                    modified_z_scores.loc[
                        date
                    ]
                )
                if pd.notna(
                    modified_z_scores.loc[
                        date
                    ]
                )
                else np.nan,
            "Flag Rule":
                "; ".join(
                    rules
                ),
        })

    return (
        pd.DataFrame(
            anomaly_rows
        )
        .sort_values(
            by="Date"
        )
        .reset_index(
            drop=True
        )
    )


def download_ecb_estr_3m_series(
    start_date,
    end_date,
) -> pd.Series:
    """
    Downloads the ECB compounded €STR average rate for the 3-month tenor.

    Series key:
        EST.B.EU000A2QQF32.CR

    Values are annualized percentages.
    """

    start_timestamp = pd.Timestamp(
        start_date
    ).normalize()

    end_timestamp = pd.Timestamp(
        end_date
    ).normalize()

    query_parameters = urlencode({
        "startPeriod":
            start_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "endPeriod":
            end_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "format":
            "csvdata",
    })

    request_url = (
        "https://data-api.ecb.europa.eu/"
        "service/data/EST/B.EU000A2QQF32.CR"
        f"?{query_parameters}"
    )

    request = Request(
        request_url,
        headers={
            "User-Agent":
                "Finance Bro educational portfolio app"
        },
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            response_text = (
                response
                .read()
                .decode("utf-8")
            )

    except Exception as error:
        raise ValueError(
            "The ECB 3-month compounded €STR series could not "
            f"be downloaded: {error}"
        ) from error

    data = pd.read_csv(
        StringIO(
            response_text
        )
    )

    required_columns = {
        "TIME_PERIOD",
        "OBS_VALUE",
    }

    if not required_columns.issubset(
        data.columns
    ):
        raise ValueError(
            "The ECB €STR response did not contain the expected fields."
        )

    data[
        "TIME_PERIOD"
    ] = pd.to_datetime(
        data[
            "TIME_PERIOD"
        ],
        errors="coerce",
    )

    data[
        "OBS_VALUE"
    ] = pd.to_numeric(
        data[
            "OBS_VALUE"
        ],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                "TIME_PERIOD",
                "OBS_VALUE",
            ]
        )
        .sort_values(
            "TIME_PERIOD"
        )
        .drop_duplicates(
            subset="TIME_PERIOD",
            keep="last",
        )
    )

    if data.empty:
        raise ValueError(
            "No ECB 3-month compounded €STR observations were "
            "available for the requested period. Automatic EUR "
            "risk-free data begins in April 2021; use the manual "
            "risk-free option for an earlier period."
        )

    series = pd.Series(
        data[
            "OBS_VALUE"
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            data[
                "TIME_PERIOD"
            ]
        ).normalize(),
        name="3-Month Compounded €STR (%)",
    )

    return series



def download_ecb_deposit_facility_series(
    start_date,
    end_date,
) -> pd.Series:
    """
    Downloads the official ECB deposit-facility rate.

    Series key:
        FM.D.U2.EUR.4F.KR.DFR.LEV

    The source is event-driven: an observation is published when the
    policy rate changes. Values are annualized percentages.
    """

    start_timestamp = pd.Timestamp(
        start_date
    ).normalize()

    end_timestamp = pd.Timestamp(
        end_date
    ).normalize()

    query_parameters = urlencode({
        "startPeriod":
            start_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "endPeriod":
            end_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "format":
            "csvdata",
    })

    request_url = (
        "https://data-api.ecb.europa.eu/"
        "service/data/FM/D.U2.EUR.4F.KR.DFR.LEV"
        f"?{query_parameters}"
    )

    request = Request(
        request_url,
        headers={
            "User-Agent":
                "Finance Bro educational portfolio app"
        },
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            response_text = (
                response
                .read()
                .decode("utf-8")
            )

    except Exception as error:
        raise ValueError(
            "The ECB deposit-facility-rate series could not "
            f"be downloaded: {error}"
        ) from error

    data = pd.read_csv(
        StringIO(
            response_text
        )
    )

    required_columns = {
        "TIME_PERIOD",
        "OBS_VALUE",
    }

    if not required_columns.issubset(
        data.columns
    ):
        raise ValueError(
            "The ECB deposit-facility response did not contain "
            "the expected fields."
        )

    data[
        "TIME_PERIOD"
    ] = pd.to_datetime(
        data[
            "TIME_PERIOD"
        ],
        errors="coerce",
    )

    data[
        "OBS_VALUE"
    ] = pd.to_numeric(
        data[
            "OBS_VALUE"
        ],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                "TIME_PERIOD",
                "OBS_VALUE",
            ]
        )
        .sort_values(
            "TIME_PERIOD"
        )
        .drop_duplicates(
            subset="TIME_PERIOD",
            keep="last",
        )
    )

    if data.empty:
        raise ValueError(
            "No ECB deposit-facility-rate observations were "
            "available for the requested period."
        )

    return pd.Series(
        data[
            "OBS_VALUE"
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            data[
                "TIME_PERIOD"
            ]
        ).normalize(),
        name="ECB Deposit Facility Rate (%)",
    )

def download_fred_dgs3mo_series(
    start_date,
    end_date,
) -> pd.Series:
    """
    Downloads the daily 3-month U.S. Treasury constant-maturity yield
    from FRED (series DGS3MO). Values are annualized percentages.
    """

    start_timestamp = pd.Timestamp(
        start_date
    ).normalize()

    end_timestamp = pd.Timestamp(
        end_date
    ).normalize()

    query_parameters = urlencode({
        "id":
            "DGS3MO",
        "cosd":
            start_timestamp.strftime(
                "%Y-%m-%d"
            ),
        "coed":
            end_timestamp.strftime(
                "%Y-%m-%d"
            ),
    })

    request_url = (
        "https://fred.stlouisfed.org/"
        f"graph/fredgraph.csv?{query_parameters}"
    )

    request = Request(
        request_url,
        headers={
            "User-Agent":
                "Finance Bro educational portfolio app"
        },
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            response_text = (
                response
                .read()
                .decode("utf-8")
            )

    except Exception as error:
        raise ValueError(
            "The U.S. 3-month Treasury rate could not be "
            f"downloaded from FRED: {error}"
        ) from error

    data = pd.read_csv(
        StringIO(
            response_text
        )
    )

    date_column = (
        "DATE"
        if "DATE" in data.columns
        else "observation_date"
        if "observation_date" in data.columns
        else None
    )

    if (
        date_column is None
        or "DGS3MO" not in data.columns
    ):
        raise ValueError(
            "The FRED DGS3MO response did not contain the expected fields."
        )

    data[
        date_column
    ] = pd.to_datetime(
        data[
            date_column
        ],
        errors="coerce",
    )

    data[
        "DGS3MO"
    ] = pd.to_numeric(
        data[
            "DGS3MO"
        ],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                date_column,
                "DGS3MO",
            ]
        )
        .sort_values(
            date_column
        )
        .drop_duplicates(
            subset=date_column,
            keep="last",
        )
    )

    if data.empty:
        raise ValueError(
            "No U.S. 3-month Treasury observations were "
            "available for the requested period."
        )

    return pd.Series(
        data[
            "DGS3MO"
        ].to_numpy(
            dtype=float
        ),
        index=pd.DatetimeIndex(
            data[
                date_column
            ]
        ).normalize(),
        name="3-Month U.S. Treasury Yield (%)",
    )


def build_risk_free_return_table(
    portfolio_currency: str,
    market_dates,
    risk_free_mode: str = "Automatic",
    manual_annual_risk_free_rate: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Aligns an official risk-free proxy with market dates.

    EUR automatic hierarchy:
        1. ECB 3-month compounded €STR, whenever available.
        2. ECB deposit-facility rate for earlier dates or when the
           compounded €STR endpoint is temporarily unavailable.

    USD automatic hierarchy:
        1. FRED DGS3MO 3-month U.S. Treasury yield.

    The latest official observation available on or before each market
    date is used. No linear interpolation is performed.
    """

    portfolio_currency = validate_portfolio_currency(
        portfolio_currency
    )

    target_dates = pd.DatetimeIndex(
        pd.to_datetime(
            market_dates
        )
    )

    if target_dates.tz is not None:
        target_dates = target_dates.tz_localize(
            None
        )

    target_dates = (
        target_dates
        .normalize()
        .sort_values()
        .unique()
    )

    target_dates = pd.DatetimeIndex(
        target_dates
    )

    if len(target_dates) == 0:
        raise ValueError(
            "No market dates were supplied for risk-free alignment."
        )

    normalized_mode = (
        str(
            risk_free_mode
        )
        .strip()
        .lower()
    )

    lookback_start = (
        target_dates.min()
        - pd.Timedelta(days=45)
    )

    end_date = target_dates.max()

    target_frame = pd.DataFrame({
        "Date":
            pd.DatetimeIndex(
                target_dates
            ).astype(
                "datetime64[ns]"
            ),
    })

    def prepare_source_frame(
        series: pd.Series,
        source_label: str,
        series_id: str,
    ) -> pd.DataFrame:
        frame = (
            series
            .dropna()
            .sort_index()
            .rename(
                "Annual Risk-Free Rate (%)"
            )
            .reset_index()
        )

        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "Source Observation Date",
                    "Annual Risk-Free Rate (%)",
                    "Risk-Free Source",
                    "Risk-Free Series ID",
                ]
            )

        frame = frame.rename(
            columns={
                frame.columns[0]:
                    "Source Observation Date"
            }
        )

        frame[
            "Source Observation Date"
        ] = pd.to_datetime(
            frame[
                "Source Observation Date"
            ],
            errors="coerce",
        ).dt.normalize()

        frame[
            "Annual Risk-Free Rate (%)"
        ] = pd.to_numeric(
            frame[
                "Annual Risk-Free Rate (%)"
            ],
            errors="coerce",
        )

        frame = (
            frame
            .dropna(
                subset=[
                    "Source Observation Date",
                    "Annual Risk-Free Rate (%)",
                ]
            )
            .sort_values(
                "Source Observation Date"
            )
            .drop_duplicates(
                subset="Source Observation Date",
                keep="last",
            )
        )

        frame[
            "Source Observation Date"
        ] = frame[
            "Source Observation Date"
        ].astype(
            "datetime64[ns]"
        )

        frame[
            "Risk-Free Source"
        ] = source_label

        frame[
            "Risk-Free Series ID"
        ] = series_id

        return frame

    primary_download_error = ""
    fallback_download_error = ""
    fallback_observation_count = 0

    if normalized_mode == "manual":

        if manual_annual_risk_free_rate is None:
            raise ValueError(
                "Enter a manual annual risk-free rate."
            )

        annual_rate_percent = float(
            manual_annual_risk_free_rate
        )

        source_series = pd.Series(
            annual_rate_percent,
            index=pd.DatetimeIndex(
                [lookback_start]
            ),
            name="Manual Annual Risk-Free Rate (%)",
        )

        source_name = "Manual user input"
        source_series_id = "MANUAL"
        source_url = ""
        rate_description = (
            f"Constant manual annual rate of "
            f"{annual_rate_percent:.4f}%."
        )

        source_frame = prepare_source_frame(
            source_series,
            source_name,
            source_series_id,
        )

        aligned = pd.merge_asof(
            target_frame.sort_values(
                "Date"
            ),
            source_frame.sort_values(
                "Source Observation Date"
            ),
            left_on="Date",
            right_on="Source Observation Date",
            direction="backward",
            allow_exact_matches=True,
        )

        alignment_method = (
            "The constant manual annual rate is applied to every "
            "market date."
        )

    elif normalized_mode == "automatic":

        if portfolio_currency == "EUR":

            estr_series = pd.Series(
                dtype=float
            )

            try:
                estr_series = (
                    download_ecb_estr_3m_series(
                        start_date=lookback_start,
                        end_date=end_date,
                    )
                )
            except Exception as error:
                primary_download_error = str(
                    error
                )

            deposit_series = pd.Series(
                dtype=float
            )

            try:
                deposit_series = (
                    download_ecb_deposit_facility_series(
                        start_date=pd.Timestamp(
                            "1999-01-01"
                        ),
                        end_date=end_date,
                    )
                )
            except Exception as error:
                fallback_download_error = str(
                    error
                )

            estr_frame = prepare_source_frame(
                estr_series,
                (
                    "European Central Bank — "
                    "3-Month Compounded €STR"
                ),
                "EST.B.EU000A2QQF32.CR",
            )

            deposit_frame = prepare_source_frame(
                deposit_series,
                (
                    "European Central Bank — "
                    "Deposit Facility Rate "
                    "(historical fallback)"
                ),
                "FM.D.U2.EUR.4F.KR.DFR.LEV",
            )

            estr_aligned = pd.merge_asof(
                target_frame.sort_values(
                    "Date"
                ),
                estr_frame.sort_values(
                    "Source Observation Date"
                ),
                left_on="Date",
                right_on="Source Observation Date",
                direction="backward",
                allow_exact_matches=True,
            ).rename(
                columns={
                    "Source Observation Date":
                        "€STR Observation Date",
                    "Annual Risk-Free Rate (%)":
                        "€STR Annual Rate (%)",
                    "Risk-Free Source":
                        "€STR Source",
                    "Risk-Free Series ID":
                        "€STR Series ID",
                }
            )

            deposit_aligned = pd.merge_asof(
                target_frame.sort_values(
                    "Date"
                ),
                deposit_frame.sort_values(
                    "Source Observation Date"
                ),
                left_on="Date",
                right_on="Source Observation Date",
                direction="backward",
                allow_exact_matches=True,
            ).rename(
                columns={
                    "Source Observation Date":
                        "Fallback Observation Date",
                    "Annual Risk-Free Rate (%)":
                        "Fallback Annual Rate (%)",
                    "Risk-Free Source":
                        "Fallback Source",
                    "Risk-Free Series ID":
                        "Fallback Series ID",
                }
            )

            aligned = target_frame.copy()

            use_primary = estr_aligned[
                "€STR Annual Rate (%)"
            ].notna()

            aligned[
                "Annual Risk-Free Rate (%)"
            ] = estr_aligned[
                "€STR Annual Rate (%)"
            ].where(
                use_primary,
                deposit_aligned[
                    "Fallback Annual Rate (%)"
                ],
            )

            aligned[
                "Source Observation Date"
            ] = estr_aligned[
                "€STR Observation Date"
            ].where(
                use_primary,
                deposit_aligned[
                    "Fallback Observation Date"
                ],
            )

            aligned[
                "Risk-Free Source"
            ] = estr_aligned[
                "€STR Source"
            ].where(
                use_primary,
                deposit_aligned[
                    "Fallback Source"
                ],
            )

            aligned[
                "Risk-Free Series ID"
            ] = estr_aligned[
                "€STR Series ID"
            ].where(
                use_primary,
                deposit_aligned[
                    "Fallback Series ID"
                ],
            )

            fallback_observation_count = int(
                (
                    ~use_primary
                    & aligned[
                        "Annual Risk-Free Rate (%)"
                    ].notna()
                ).sum()
            )

            used_primary = bool(
                use_primary.any()
            )

            used_fallback = bool(
                (
                    aligned[
                        "Risk-Free Series ID"
                    ]
                    == "FM.D.U2.EUR.4F.KR.DFR.LEV"
                ).any()
            )

            if used_primary and used_fallback:
                source_name = (
                    "ECB 3-Month Compounded €STR, with the "
                    "ECB Deposit Facility Rate as a historical fallback"
                )
                source_series_id = (
                    "EST.B.EU000A2QQF32.CR | "
                    "FM.D.U2.EUR.4F.KR.DFR.LEV"
                )
                source_url = (
                    "https://data.ecb.europa.eu/data/datasets/EST/"
                    "EST.B.EU000A2QQF32.CR | "
                    "https://data.ecb.europa.eu/data/datasets/FM/"
                    "FM.D.U2.EUR.4F.KR.DFR.LEV"
                )
                rate_description = (
                    "The ECB 3-month compounded €STR is used from "
                    "its first available observation. Earlier market "
                    "dates use the official ECB deposit-facility rate "
                    "and are explicitly identified as fallback dates."
                )

            elif used_primary:
                source_name = (
                    "European Central Bank — "
                    "3-Month Compounded €STR"
                )
                source_series_id = (
                    "EST.B.EU000A2QQF32.CR"
                )
                source_url = (
                    "https://data.ecb.europa.eu/data/datasets/EST/"
                    "EST.B.EU000A2QQF32.CR"
                )
                rate_description = (
                    "Backward-looking compounded €STR average rate "
                    "for a standardized 3-month tenor."
                )

            elif used_fallback:
                source_name = (
                    "European Central Bank — Deposit Facility Rate "
                    "(automatic fallback)"
                )
                source_series_id = (
                    "FM.D.U2.EUR.4F.KR.DFR.LEV"
                )
                source_url = (
                    "https://data.ecb.europa.eu/data/datasets/FM/"
                    "FM.D.U2.EUR.4F.KR.DFR.LEV"
                )
                rate_description = (
                    "The primary compounded €STR series was unavailable "
                    "for the requested dates, so the official ECB deposit-"
                    "facility rate was used and identified as a fallback."
                )

            else:
                error_details = []

                if primary_download_error:
                    error_details.append(
                        "€STR: "
                        + primary_download_error
                    )

                if fallback_download_error:
                    error_details.append(
                        "deposit facility: "
                        + fallback_download_error
                    )

                detail_text = (
                    " ".join(
                        error_details
                    )
                    if error_details
                    else (
                        "No official observations covered the "
                        "requested dates."
                    )
                )

                raise ValueError(
                    "Automatic EUR risk-free data could not be "
                    "prepared. "
                    + detail_text
                    + " Select Manual risk-free mode to continue."
                )

            alignment_method = (
                "For EUR, the ECB 3-month compounded €STR is the "
                "primary source. Dates before its first available "
                "observation use the ECB deposit-facility rate. "
                "For each source, the last official observation "
                "available on or before the market date is used; "
                "no linear interpolation is performed."
            )

        else:

            source_series = (
                download_fred_dgs3mo_series(
                    start_date=lookback_start,
                    end_date=end_date,
                )
            )

            source_name = (
                "Federal Reserve / FRED — "
                "3-Month U.S. Treasury Constant Maturity"
            )

            source_series_id = "DGS3MO"

            source_url = (
                "https://fred.stlouisfed.org/series/DGS3MO"
            )

            rate_description = (
                "Daily 3-month U.S. Treasury constant-maturity "
                "yield quoted on an investment basis."
            )

            source_frame = prepare_source_frame(
                source_series,
                source_name,
                source_series_id,
            )

            aligned = pd.merge_asof(
                target_frame.sort_values(
                    "Date"
                ),
                source_frame.sort_values(
                    "Source Observation Date"
                ),
                left_on="Date",
                right_on="Source Observation Date",
                direction="backward",
                allow_exact_matches=True,
            )

            alignment_method = (
                "The latest official U.S. Treasury observation "
                "available on or before each market date is used; "
                "no linear interpolation is performed."
            )

    else:
        raise ValueError(
            "Risk-free mode must be Automatic or Manual."
        )

    if aligned[
        "Annual Risk-Free Rate (%)"
    ].isna().any():

        first_missing_date = aligned.loc[
            aligned[
                "Annual Risk-Free Rate (%)"
            ].isna(),
            "Date",
        ].iloc[0]

        if (
            normalized_mode == "automatic"
            and portfolio_currency == "EUR"
        ):
            raise ValueError(
                "No official automatic EUR risk-free observation "
                "could be aligned with the market date "
                f"{first_missing_date.date()}. Select Manual "
                "risk-free mode for this period."
            )

        raise ValueError(
            "Risk-free data could not be aligned with the "
            f"market date {first_missing_date.date()}."
        )

    annual_rates_decimal = (
        aligned[
            "Annual Risk-Free Rate (%)"
        ]
        / 100
    )

    if (
        annual_rates_decimal
        <= -1
    ).any():
        raise ValueError(
            "A risk-free annual rate was less than or equal to -100%."
        )

    aligned[
        "Daily Risk-Free Return"
    ] = (
        np.power(
            1
            + annual_rates_decimal,
            1 / 252,
        )
        - 1
    )

    aligned[
        "Staleness (Calendar Days)"
    ] = (
        aligned[
            "Date"
        ]
        - aligned[
            "Source Observation Date"
        ]
    ).dt.days

    aligned = aligned.set_index(
        "Date"
    )

    source_breakdown = (
        aligned[
            "Risk-Free Source"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    metadata = {
        "risk_free_mode":
            (
                "Manual"
                if normalized_mode == "manual"
                else "Automatic"
            ),
        "risk_free_source":
            source_name,
        "risk_free_series_id":
            source_series_id,
        "risk_free_source_url":
            source_url,
        "risk_free_description":
            rate_description,
        "risk_free_alignment_method":
            alignment_method,
        "risk_free_daily_conversion":
            (
                "r_f,daily = (1 + y_annual)^(1/252) - 1"
            ),
        "risk_free_average_annual_rate_percent":
            float(
                aligned[
                    "Annual Risk-Free Rate (%)"
                ].mean()
            ),
        "risk_free_latest_annual_rate_percent":
            float(
                aligned[
                    "Annual Risk-Free Rate (%)"
                ].iloc[-1]
            ),
        "risk_free_latest_source_date":
            pd.Timestamp(
                aligned[
                    "Source Observation Date"
                ].iloc[-1]
            ),
        "risk_free_carried_forward_observations":
            int(
                (
                    aligned[
                        "Staleness (Calendar Days)"
                    ] > 0
                ).sum()
            ),
        "risk_free_max_staleness_days":
            int(
                aligned[
                    "Staleness (Calendar Days)"
                ].max()
            ),
        "risk_free_fallback_observations":
            int(
                fallback_observation_count
            ),
        "risk_free_source_breakdown":
            source_breakdown,
        "risk_free_primary_download_error":
            primary_download_error,
        "risk_free_fallback_download_error":
            fallback_download_error,
    }

    return (
        aligned,
        metadata,
    )


def calculate_benchmark_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    daily_risk_free_returns: pd.Series,
    confidence_level: float = 0.95,
) -> dict:
    """
    Estimates the excess-return market model with OLS and Newey-West HAC
    covariance estimates:

        R_p,t - R_f,t = alpha + beta (R_m,t - R_f,t) + epsilon_t
    """

    if not (
        0 < confidence_level < 1
    ):
        raise ValueError(
            "Regression confidence level must be between 0 and 1."
        )

    regression_frame = pd.concat(
        [
            portfolio_returns.rename(
                "Portfolio Return"
            ),
            benchmark_returns.rename(
                "Benchmark Return"
            ),
            daily_risk_free_returns.rename(
                "Daily Risk-Free Return"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(regression_frame) < 30:
        raise ValueError(
            "At least 30 common return observations are required "
            "for the benchmark regression and diagnostics."
        )

    regression_frame[
        "Portfolio Excess Return"
    ] = (
        regression_frame[
            "Portfolio Return"
        ]
        - regression_frame[
            "Daily Risk-Free Return"
        ]
    )

    regression_frame[
        "Benchmark Excess Return"
    ] = (
        regression_frame[
            "Benchmark Return"
        ]
        - regression_frame[
            "Daily Risk-Free Return"
        ]
    )

    y = regression_frame[
        "Portfolio Excess Return"
    ]

    x = regression_frame[
        "Benchmark Excess Return"
    ]

    design_matrix = sm.add_constant(
        x,
        has_constant="add",
    )

    ols_results = sm.OLS(
        y,
        design_matrix,
        missing="drop",
    ).fit()

    observation_count = int(
        ols_results.nobs
    )

    automatic_hac_lags = int(
        np.floor(
            4
            * (
                observation_count
                / 100
            )
            ** (
                2 / 9
            )
        )
    )

    hac_lags = max(
        1,
        min(
            automatic_hac_lags,
            observation_count - 2,
        ),
    )

    hac_results = (
        ols_results
        .get_robustcov_results(
            cov_type="HAC",
            maxlags=hac_lags,
            use_correction=True,
        )
    )

    parameter_names = list(
        design_matrix.columns
    )

    robust_parameters = pd.Series(
        hac_results.params,
        index=parameter_names,
        dtype=float,
    )

    robust_standard_errors = pd.Series(
        hac_results.bse,
        index=parameter_names,
        dtype=float,
    )

    robust_p_values = pd.Series(
        hac_results.pvalues,
        index=parameter_names,
        dtype=float,
    )

    confidence_interval_alpha = (
        1
        - confidence_level
    )

    robust_confidence_intervals = pd.DataFrame(
        hac_results.conf_int(
            alpha=confidence_interval_alpha
        ),
        index=parameter_names,
        columns=[
            "Lower",
            "Upper",
        ],
    )

    alpha_daily = float(
        robust_parameters[
            "const"
        ]
    )

    beta = float(
        robust_parameters[
            "Benchmark Excess Return"
        ]
    )

    alpha_annualized_percent = (
        alpha_daily
        * 252
        * 100
    )

    residuals = pd.Series(
        ols_results.resid,
        index=regression_frame.index,
        name="Regression Residual",
        dtype=float,
    )

    fitted_values = pd.Series(
        ols_results.fittedvalues,
        index=regression_frame.index,
        name="Fitted Portfolio Excess Return",
        dtype=float,
    )

    regression_frame[
        "Fitted Portfolio Excess Return"
    ] = fitted_values

    regression_frame[
        "Residual"
    ] = residuals

    regression_plot_data = (
        regression_frame[
            [
                "Portfolio Excess Return",
                "Benchmark Excess Return",
                "Fitted Portfolio Excess Return",
                "Residual",
            ]
        ]
        .copy()
        * 100
    )

    regression_plot_data = (
        regression_plot_data
        .reset_index()
        .rename(
            columns={
                regression_plot_data.index.name
                or "index":
                    "Date"
            }
        )
    )

    durbin_watson_statistic = float(
        durbin_watson(
            residuals
        )
    )

    (
        breusch_pagan_lm_statistic,
        breusch_pagan_lm_p_value,
        breusch_pagan_f_statistic,
        breusch_pagan_f_p_value,
    ) = het_breuschpagan(
        residuals,
        design_matrix,
    )

    (
        jarque_bera_statistic,
        jarque_bera_p_value,
        residual_skewness,
        residual_kurtosis,
    ) = jarque_bera(
        residuals
    )

    ljung_box_lag = int(
        min(
            10,
            max(
                1,
                observation_count // 5,
            ),
        )
    )

    ljung_box_result = acorr_ljungbox(
        residuals,
        lags=[
            ljung_box_lag
        ],
        return_df=True,
    )

    ljung_box_statistic = float(
        ljung_box_result[
            "lb_stat"
        ].iloc[0]
    )

    ljung_box_p_value = float(
        ljung_box_result[
            "lb_pvalue"
        ].iloc[0]
    )

    try:
        reset_result = linear_reset(
            ols_results,
            power=2,
            use_f=True,
        )

        reset_statistic = float(
            reset_result.fvalue
        )

        reset_p_value = float(
            reset_result.pvalue
        )

    except Exception:
        reset_statistic = np.nan
        reset_p_value = np.nan

    influence = OLSInfluence(
        ols_results
    )

    cooks_distance = pd.Series(
        influence.cooks_distance[0],
        index=regression_frame.index,
        dtype=float,
    )

    cooks_threshold = (
        4
        / observation_count
    )

    influential_observation_count = int(
        (
            cooks_distance
            > cooks_threshold
        ).sum()
    )

    residual_volatility_annualized_percent = (
        float(
            residuals.std(
                ddof=1
            )
            * np.sqrt(
                252
            )
            * 100
        )
    )

    coefficient_table = pd.DataFrame({
        "Coefficient": [
            "Alpha (Daily)",
            "Alpha (Annualized)",
            "Beta",
        ],
        "Units": [
            "Decimal return per trading day",
            "Percentage points per year",
            "Unitless slope coefficient",
        ],
        "Estimate": [
            alpha_daily,
            alpha_annualized_percent,
            beta,
        ],
        "Robust Standard Error": [
            float(
                robust_standard_errors[
                    "const"
                ]
            ),
            float(
                robust_standard_errors[
                    "const"
                ]
                * 252
                * 100
            ),
            float(
                robust_standard_errors[
                    "Benchmark Excess Return"
                ]
            ),
        ],
        "Robust p-value": [
            float(
                robust_p_values[
                    "const"
                ]
            ),
            float(
                robust_p_values[
                    "const"
                ]
            ),
            float(
                robust_p_values[
                    "Benchmark Excess Return"
                ]
            ),
        ],
        f"{confidence_level * 100:.0f}% CI Lower": [
            float(
                robust_confidence_intervals.loc[
                    "const",
                    "Lower",
                ]
            ),
            float(
                robust_confidence_intervals.loc[
                    "const",
                    "Lower",
                ]
                * 252
                * 100
            ),
            float(
                robust_confidence_intervals.loc[
                    "Benchmark Excess Return",
                    "Lower",
                ]
            ),
        ],
        f"{confidence_level * 100:.0f}% CI Upper": [
            float(
                robust_confidence_intervals.loc[
                    "const",
                    "Upper",
                ]
            ),
            float(
                robust_confidence_intervals.loc[
                    "const",
                    "Upper",
                ]
                * 252
                * 100
            ),
            float(
                robust_confidence_intervals.loc[
                    "Benchmark Excess Return",
                    "Upper",
                ]
            ),
        ],
    })

    diagnostics_table = pd.DataFrame({
        "Diagnostic": [
            "Number of observations",
            "R-squared",
            "Adjusted R-squared",
            "HAC / Newey-West lags",
            "Annualized residual volatility (%)",
            "Durbin-Watson statistic",
            f"Ljung-Box statistic (lag {ljung_box_lag})",
            f"Ljung-Box p-value (lag {ljung_box_lag})",
            "Breusch-Pagan LM statistic",
            "Breusch-Pagan LM p-value",
            "Breusch-Pagan F statistic",
            "Breusch-Pagan F p-value",
            "Jarque-Bera statistic",
            "Jarque-Bera p-value",
            "Residual skewness",
            "Residual kurtosis",
            "Ramsey RESET F statistic",
            "Ramsey RESET p-value",
            "Maximum Cook's distance",
            "Cook's distance threshold (4/n)",
            "Influential observations above 4/n",
            "Regression condition number",
        ],
        "Value": [
            observation_count,
            float(
                ols_results.rsquared
            ),
            float(
                ols_results.rsquared_adj
            ),
            hac_lags,
            residual_volatility_annualized_percent,
            durbin_watson_statistic,
            ljung_box_statistic,
            ljung_box_p_value,
            float(
                breusch_pagan_lm_statistic
            ),
            float(
                breusch_pagan_lm_p_value
            ),
            float(
                breusch_pagan_f_statistic
            ),
            float(
                breusch_pagan_f_p_value
            ),
            float(
                jarque_bera_statistic
            ),
            float(
                jarque_bera_p_value
            ),
            float(
                residual_skewness
            ),
            float(
                residual_kurtosis
            ),
            reset_statistic,
            reset_p_value,
            float(
                cooks_distance.max()
            ),
            float(
                cooks_threshold
            ),
            influential_observation_count,
            float(
                ols_results.condition_number
            ),
        ],
    })

    return {
        "regression_frame":
            regression_frame,
        "regression_plot_data":
            regression_plot_data,
        "regression_coefficients":
            coefficient_table,
        "regression_diagnostics":
            diagnostics_table,
        "alpha_daily":
            alpha_daily,
        "alpha_annualized_percent":
            float(
                alpha_annualized_percent
            ),
        "alpha_standard_error_hac":
            float(
                robust_standard_errors[
                    "const"
                ]
            ),
        "alpha_p_value_hac":
            float(
                robust_p_values[
                    "const"
                ]
            ),
        "alpha_confidence_interval_daily":
            (
                float(
                    robust_confidence_intervals.loc[
                        "const",
                        "Lower",
                    ]
                ),
                float(
                    robust_confidence_intervals.loc[
                        "const",
                        "Upper",
                    ]
                ),
            ),
        "beta":
            beta,
        "beta_standard_error_hac":
            float(
                robust_standard_errors[
                    "Benchmark Excess Return"
                ]
            ),
        "beta_p_value_hac":
            float(
                robust_p_values[
                    "Benchmark Excess Return"
                ]
            ),
        "beta_confidence_interval":
            (
                float(
                    robust_confidence_intervals.loc[
                        "Benchmark Excess Return",
                        "Lower",
                    ]
                ),
                float(
                    robust_confidence_intervals.loc[
                        "Benchmark Excess Return",
                        "Upper",
                    ]
                ),
            ),
        "r_squared":
            float(
                ols_results.rsquared
            ),
        "adjusted_r_squared":
            float(
                ols_results.rsquared_adj
            ),
        "observation_count":
            observation_count,
        "hac_lags":
            hac_lags,
        "residuals":
            residuals,
        "fitted_values":
            fitted_values,
        "residual_volatility_annualized_percent":
            residual_volatility_annualized_percent,
        "durbin_watson":
            durbin_watson_statistic,
        "ljung_box_lag":
            ljung_box_lag,
        "ljung_box_p_value":
            ljung_box_p_value,
        "breusch_pagan_p_value":
            float(
                breusch_pagan_lm_p_value
            ),
        "jarque_bera_p_value":
            float(
                jarque_bera_p_value
            ),
        "reset_p_value":
            reset_p_value,
        "influential_observation_count":
            influential_observation_count,
        "cooks_distance":
            cooks_distance,
    }


def calculate_asset_betas_from_excess_returns(
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    daily_risk_free_returns: pd.Series,
) -> pd.Series:
    """
    Estimates each asset beta using the same excess-return specification
    as the portfolio regression.
    """

    asset_betas = pd.Series(
        index=daily_returns.columns,
        dtype=float,
    )

    for asset in daily_returns.columns:

        frame = pd.concat(
            [
                daily_returns[
                    asset
                ].rename(
                    "Asset Return"
                ),
                benchmark_returns.rename(
                    "Benchmark Return"
                ),
                daily_risk_free_returns.rename(
                    "Daily Risk-Free Return"
                ),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(frame) < 2:
            asset_betas.loc[
                asset
            ] = np.nan
            continue

        asset_excess = (
            frame[
                "Asset Return"
            ]
            - frame[
                "Daily Risk-Free Return"
            ]
        )

        benchmark_excess = (
            frame[
                "Benchmark Return"
            ]
            - frame[
                "Daily Risk-Free Return"
            ]
        )

        benchmark_excess_variance = (
            benchmark_excess.var(
                ddof=1
            )
        )

        if np.isclose(
            benchmark_excess_variance,
            0,
        ):
            asset_betas.loc[
                asset
            ] = np.nan
        else:
            asset_betas.loc[
                asset
            ] = float(
                asset_excess.cov(
                    benchmark_excess
                )
                / benchmark_excess_variance
            )

    return asset_betas


def build_data_quality_results(
    portfolio_quality: dict,
    benchmark_quality: dict,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    regression_result: dict,
    risk_free_table: pd.DataFrame,
    risk_free_metadata: dict,
) -> dict:
    """
    Builds transparent data-quality and alignment diagnostics.
    """

    raw_market_dates = int(
        portfolio_quality[
            "raw_market_dates"
        ]
    )

    common_price_dates = int(
        portfolio_quality[
            "common_price_dates"
        ]
    )

    portfolio_return_observations = int(
        len(
            portfolio_returns
        )
    )

    portfolio_benchmark_common = int(
        len(
            pd.concat(
                [
                    portfolio_returns,
                    benchmark_returns,
                ],
                axis=1,
                join="inner",
            ).dropna()
        )
    )

    regression_observations = int(
        regression_result[
            "observation_count"
        ]
    )

    alignment_rows = [
        {
            "Stage":
                "Raw portfolio market-date rows",
            "Observations":
                raw_market_dates,
            "Removed from Previous Stage":
                0,
            "Retention from Previous Stage (%)":
                100.0,
        },
        {
            "Stage":
                "Common asset price dates",
            "Observations":
                common_price_dates,
            "Removed from Previous Stage":
                raw_market_dates
                - common_price_dates,
            "Retention from Previous Stage (%)":
                (
                    common_price_dates
                    / raw_market_dates
                    * 100
                    if raw_market_dates > 0
                    else np.nan
                ),
        },
        {
            "Stage":
                "Portfolio return observations",
            "Observations":
                portfolio_return_observations,
            "Removed from Previous Stage":
                common_price_dates
                - portfolio_return_observations,
            "Retention from Previous Stage (%)":
                (
                    portfolio_return_observations
                    / common_price_dates
                    * 100
                    if common_price_dates > 0
                    else np.nan
                ),
        },
        {
            "Stage":
                "Portfolio–benchmark common returns",
            "Observations":
                portfolio_benchmark_common,
            "Removed from Previous Stage":
                portfolio_return_observations
                - portfolio_benchmark_common,
            "Retention from Previous Stage (%)":
                (
                    portfolio_benchmark_common
                    / portfolio_return_observations
                    * 100
                    if portfolio_return_observations > 0
                    else np.nan
                ),
        },
        {
            "Stage":
                "Regression observations after risk-free alignment",
            "Observations":
                regression_observations,
            "Removed from Previous Stage":
                portfolio_benchmark_common
                - regression_observations,
            "Retention from Previous Stage (%)":
                (
                    regression_observations
                    / portfolio_benchmark_common
                    * 100
                    if portfolio_benchmark_common > 0
                    else np.nan
                ),
        },
    ]

    anomaly_frames = [
        portfolio_quality.get(
            "anomalies",
            pd.DataFrame(),
        ),
        benchmark_quality.get(
            "anomalies",
            pd.DataFrame(),
        ),
        detect_return_anomalies(
            return_series=portfolio_returns,
            series_name="PORTFOLIO",
            return_basis="Portfolio-currency return",
        ),
    ]

    anomalies = pd.concat(
        [
            frame
            for frame in anomaly_frames
            if (
                isinstance(
                    frame,
                    pd.DataFrame,
                )
                and not frame.empty
            )
        ],
        ignore_index=True,
    ) if any(
        isinstance(frame, pd.DataFrame)
        and not frame.empty
        for frame in anomaly_frames
    ) else pd.DataFrame(
        columns=[
            "Date",
            "Series",
            "Return Basis",
            "Return (%)",
            "Modified Z-Score",
            "Flag Rule",
        ]
    )

    headline = {
        "raw_market_dates":
            raw_market_dates,
        "common_price_dates":
            common_price_dates,
        "portfolio_return_observations":
            portfolio_return_observations,
        "dates_removed":
            int(
                portfolio_quality[
                    "dates_removed"
                ]
            ),
        "data_retention_percent":
            float(
                portfolio_quality[
                    "common_date_retention_percent"
                ]
            ),
        "overall_price_cell_coverage_percent":
            float(
                portfolio_quality[
                    "overall_price_cell_coverage_percent"
                ]
            ),
        "potential_anomaly_count":
            int(
                len(
                    anomalies
                )
            ),
        "requested_start_date":
            portfolio_quality[
                "requested_start_date"
            ],
        "requested_end_date":
            portfolio_quality[
                "requested_end_date"
            ],
        "common_first_date":
            portfolio_quality[
                "common_first_date"
            ],
        "common_last_date":
            portfolio_quality[
                "common_last_date"
            ],
        "benchmark_raw_date_rows":
            int(
                benchmark_quality[
                    "raw_date_rows"
                ]
            ),
        "benchmark_valid_price_observations":
            int(
                benchmark_quality[
                    "valid_price_observations"
                ]
            ),
        "risk_free_aligned_observations":
            int(
                len(
                    risk_free_table
                )
            ),
        "risk_free_carried_forward_observations":
            int(
                risk_free_metadata[
                    "risk_free_carried_forward_observations"
                ]
            ),
        "risk_free_max_staleness_days":
            int(
                risk_free_metadata[
                    "risk_free_max_staleness_days"
                ]
            ),
        "regression_observations":
            regression_observations,
    }

    methodology_table = pd.DataFrame({
        "Process": [
            "Price missing values",
            "Price-date alignment",
            "Potential outliers",
            "Foreign-exchange rates",
            "Risk-free rates",
        ],
        "Method": [
            "No interpolation, forward fill or backward fill.",
            (
                "Only dates with valid prices for every portfolio asset "
                "are retained."
            ),
            (
                "Flagged with robust modified z-scores and an absolute-"
                "return rule; observations are not automatically removed."
            ),
            portfolio_quality[
                "fx_alignment_method"
            ],
            risk_free_metadata[
                "risk_free_alignment_method"
            ],
        ],
    })

    return {
        "data_quality_headline":
            headline,
        "data_quality_asset_table":
            portfolio_quality[
                "asset_quality_table"
            ],
        "data_quality_alignment_table":
            pd.DataFrame(
                alignment_rows
            ),
        "data_quality_anomalies":
            anomalies.sort_values(
                by=[
                    "Date",
                    "Series",
                ]
            ).reset_index(
                drop=True
            ),
        "data_quality_methodology_table":
            methodology_table,
        "benchmark_quality":
            benchmark_quality,
    }


def analyze_portfolio(
    tickers: list[str],
    weights: pd.Series,
    initial_investment: float,
    start_date,
    end_date=None,
    portfolio_currency: str = "EUR",
    rolling_window: int = 20,
    annual_risk_free_rate: Optional[float] = None,
    risk_free_mode: str = "Automatic",
    confidence_level: float = 0.95,
    regression_confidence_level: float = 0.95,
    benchmark_ticker: str = "^GSPC",
    benchmark_stress_shock: float = -0.10,
    custom_stress_shocks: Optional[pd.Series] = None,
    stress_scenario_name: str = "Custom Scenario",
) -> dict:
    """
    Executes the complete portfolio analysis.

    Mathematical conventions
    ------------------------
    * Prices are never interpolated.
    * Portfolio dates are the strict intersection of valid asset prices.
    * Foreign prices and benchmark prices are converted daily into the
      selected portfolio currency.
    * EUR automatic risk-free proxy: 3-month compounded €STR.
    * USD automatic risk-free proxy: 3-month U.S. Treasury constant maturity.
    * Beta and alpha are estimated from an excess-return OLS regression.
    * Statistical inference uses Newey-West HAC standard errors.
    """

    tickers = [
        ticker.strip().upper()
        for ticker in tickers
    ]

    if not tickers:
        raise ValueError(
            "Enter at least one ticker."
        )

    benchmark_ticker = (
        benchmark_ticker
        .strip()
        .upper()
    )

    portfolio_currency = (
        validate_portfolio_currency(
            portfolio_currency
        )
    )

    currency_symbol = (
        get_currency_symbol(
            portfolio_currency
        )
    )

    weights = (
        weights
        .astype(float)
        .reindex(
            tickers
        )
    )

    if weights.isna().any():
        raise ValueError(
            "Some portfolio weights are missing."
        )

    if (
        weights < 0
    ).any():
        raise ValueError(
            "Negative portfolio weights are not supported in this version."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):
        raise ValueError(
            "Portfolio weights must sum to 100%."
        )

    (
        prices,
        prices_local,
        asset_currencies,
        portfolio_quality,
    ) = download_portfolio_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio_currency=portfolio_currency,
    )

    daily_returns = (
        prices
        .pct_change(
            fill_method=None
        )
        .dropna(
            how="any"
        )
    )

    if daily_returns.empty:
        raise ValueError(
            "No daily returns could be calculated."
        )

    weights = weights.reindex(
        daily_returns.columns
    )

    portfolio_returns = (
        daily_returns
        .dot(
            weights
        )
    )

    portfolio_returns.name = (
        "Portfolio Return"
    )

    if rolling_window >= len(
        portfolio_returns
    ):
        raise ValueError(
            "The rolling window must be smaller "
            "than the available observations."
        )

    (
        benchmark_prices,
        benchmark_prices_local,
        benchmark_currency,
        benchmark_quality,
    ) = download_benchmark_prices(
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio_currency=portfolio_currency,
    )

    benchmark_returns_full = (
        benchmark_prices
        .pct_change(
            fill_method=None
        )
        .dropna()
    )

    risk_free_table, risk_free_metadata = (
        build_risk_free_return_table(
            portfolio_currency=portfolio_currency,
            market_dates=portfolio_returns.index,
            risk_free_mode=risk_free_mode,
            manual_annual_risk_free_rate=annual_risk_free_rate,
        )
    )

    daily_risk_free_returns = (
        risk_free_table[
            "Daily Risk-Free Return"
        ]
        .reindex(
            portfolio_returns.index
        )
    )

    if daily_risk_free_returns.isna().any():
        raise ValueError(
            "Risk-free returns could not be aligned with all "
            "portfolio-return dates."
        )

    excess_returns = (
        portfolio_returns
        - daily_risk_free_returns
    )

    if np.isclose(
        excess_returns.std(
            ddof=1
        ),
        0,
    ):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = float(
            excess_returns.mean()
            / excess_returns.std(
                ddof=1
            )
            * np.sqrt(
                252
            )
        )

    regression_result = (
        calculate_benchmark_regression(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns_full,
            daily_risk_free_returns=daily_risk_free_returns,
            confidence_level=regression_confidence_level,
        )
    )

    comparison_returns = (
        regression_result[
            "regression_frame"
        ][
            [
                "Portfolio Return",
                "Benchmark Return",
            ]
        ]
        .copy()
    )

    aligned_portfolio_returns = (
        comparison_returns[
            "Portfolio Return"
        ]
    )

    aligned_benchmark_returns = (
        comparison_returns[
            "Benchmark Return"
        ]
    )

    beta = float(
        regression_result[
            "beta"
        ]
    )

    alpha_annualized = float(
        regression_result[
            "alpha_annualized_percent"
        ]
    )

    asset_betas = (
        calculate_asset_betas_from_excess_returns(
            daily_returns=daily_returns,
            benchmark_returns=benchmark_returns_full,
            daily_risk_free_returns=daily_risk_free_returns,
        )
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    wealth_index = (
        1
        + portfolio_returns
    ).cumprod()

    cumulative_returns = (
        wealth_index
        - 1
    ) * 100

    cumulative_returns.name = (
        "Cumulative Return (%)"
    )

    portfolio_value = (
        initial_investment
        * wealth_index
    )

    portfolio_value.name = (
        "Portfolio Value"
    )

    final_value = float(
        portfolio_value.iloc[-1]
    )

    profit_loss = float(
        final_value
        - initial_investment
    )

    cumulative_return = float(
        cumulative_returns.iloc[-1]
    )

    annualized_volatility = float(
        portfolio_returns.std(
            ddof=1
        )
        * np.sqrt(
            252
        )
        * 100
    )

    annualized_asset_returns = (
        daily_returns.mean()
        * 252
    )

    return_contribution = (
        weights
        * annualized_asset_returns
    )

    annualized_portfolio_return = float(
        return_contribution.sum()
        * 100
    )

    annualized_covariance_matrix = (
        daily_returns.cov()
        * 252
    )

    portfolio_volatility_decimal = float(
        np.sqrt(
            weights.to_numpy()
            @ annualized_covariance_matrix.to_numpy()
            @ weights.to_numpy()
        )
    )

    if np.isclose(
        portfolio_volatility_decimal,
        0,
    ):

        marginal_risk_contribution = pd.Series(
            np.nan,
            index=weights.index,
        )

        component_risk_contribution = pd.Series(
            np.nan,
            index=weights.index,
        )

        risk_contribution_percentage = pd.Series(
            np.nan,
            index=weights.index,
        )

    else:

        marginal_risk_contribution = pd.Series(
            (
                annualized_covariance_matrix.to_numpy()
                @ weights.to_numpy()
            )
            / portfolio_volatility_decimal,
            index=weights.index,
        )

        component_risk_contribution = (
            weights
            * marginal_risk_contribution
        )

        risk_contribution_percentage = (
            component_risk_contribution
            / portfolio_volatility_decimal
        )

    contribution_table = pd.DataFrame({
        "Asset":
            weights.index,
        "Weight (%)":
            weights.values
            * 100,
        "Annualized Asset Return (%)":
            annualized_asset_returns.reindex(
                weights.index
            ).values
            * 100,
        "Return Contribution (p.p.)":
            return_contribution.reindex(
                weights.index
            ).values
            * 100,
        "Risk Contribution (p.p.)":
            component_risk_contribution.reindex(
                weights.index
            ).values
            * 100,
        "Risk Contribution (%)":
            risk_contribution_percentage.reindex(
                weights.index
            ).values
            * 100,
    })

    benchmark_wealth_index = (
        1
        + aligned_benchmark_returns
    ).cumprod()

    benchmark_cumulative_returns = (
        benchmark_wealth_index
        - 1
    ) * 100

    benchmark_cumulative_returns.name = (
        "Benchmark Cumulative Return (%)"
    )

    aligned_portfolio_wealth_index = (
        1
        + aligned_portfolio_returns
    ).cumprod()

    aligned_portfolio_cumulative_returns = (
        aligned_portfolio_wealth_index
        - 1
    ) * 100

    aligned_portfolio_cumulative_returns.name = (
        "Portfolio Cumulative Return (%)"
    )

    benchmark_comparison = pd.concat(
        [
            aligned_portfolio_cumulative_returns,
            benchmark_cumulative_returns,
        ],
        axis=1,
    )

    benchmark_cumulative_return = float(
        benchmark_cumulative_returns.iloc[-1]
    )

    active_return = float(
        aligned_portfolio_cumulative_returns.iloc[-1]
        - benchmark_cumulative_return
    )

    # ========================================================
    # DIVERSIFICATION
    # ========================================================

    asset_correlation_matrix = (
        daily_returns.corr()
    )

    asset_benchmark_comparison = pd.concat(
        [
            daily_returns,
            aligned_benchmark_returns.rename(
                benchmark_ticker
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    asset_benchmark_correlations = (
        asset_benchmark_comparison
        .corr()[
            benchmark_ticker
        ]
        .drop(
            labels=[
                benchmark_ticker
            ]
        )
    )

    portfolio_benchmark_correlation = float(
        aligned_portfolio_returns.corr(
            aligned_benchmark_returns
        )
    )

    if len(
        asset_correlation_matrix.columns
    ) > 1:

        correlation_without_diagonal_values = (
            asset_correlation_matrix
            .to_numpy(
                copy=True
            )
        )

        np.fill_diagonal(
            correlation_without_diagonal_values,
            np.nan,
        )

        correlation_without_diagonal = (
            pd.DataFrame(
                correlation_without_diagonal_values,
                index=asset_correlation_matrix.index,
                columns=asset_correlation_matrix.columns,
            )
        )

        average_asset_correlations = (
            correlation_without_diagonal.mean()
        )

        most_diversifying_asset = (
            average_asset_correlations.idxmin()
        )

        most_correlated_asset = (
            average_asset_correlations.idxmax()
        )

        average_portfolio_correlation = float(
            correlation_without_diagonal
            .stack()
            .mean()
        )

    else:

        average_asset_correlations = pd.Series(
            np.nan,
            index=asset_correlation_matrix.columns,
        )

        most_diversifying_asset = (
            asset_correlation_matrix.columns[
                0
            ]
        )

        most_correlated_asset = (
            asset_correlation_matrix.columns[
                0
            ]
        )

        average_portfolio_correlation = np.nan

    diversification_summary = pd.DataFrame({
        "Asset":
            daily_returns.columns,
        "Average Correlation with Other Assets":
            average_asset_correlations.reindex(
                daily_returns.columns
            ).values,
        f"Correlation with {benchmark_ticker}":
            asset_benchmark_correlations.reindex(
                daily_returns.columns
            ).values,
    })

    # ========================================================
    # STRESS TESTS
    # ========================================================

    if benchmark_stress_shock < -1.0:
        raise ValueError(
            "The benchmark stress shock cannot be lower than -100%."
        )

    market_asset_shocks = (
        asset_betas
        * benchmark_stress_shock
    ).clip(
        lower=-1.0
    )

    market_scenario_name = (
        f"Market Correction — "
        f"{benchmark_ticker} "
        f"{benchmark_stress_shock * 100:.1f}%"
    )

    (
        market_stress_summary,
        market_stress_detail,
    ) = calculate_stress_scenario(
        weights=weights,
        current_portfolio_value=final_value,
        asset_shocks=market_asset_shocks,
        scenario_name=market_scenario_name,
        scenario_type="Beta-Based Market Correction",
        asset_betas=asset_betas,
    )

    if custom_stress_shocks is None:
        custom_stress_shocks = pd.Series(
            -0.10,
            index=weights.index,
            dtype=float,
        )

    stress_scenario_name = (
        stress_scenario_name.strip()
        or "Custom Scenario"
    )

    (
        custom_stress_summary,
        custom_stress_detail,
    ) = calculate_stress_scenario(
        weights=weights,
        current_portfolio_value=final_value,
        asset_shocks=custom_stress_shocks,
        scenario_name=stress_scenario_name,
        scenario_type="Custom Asset Shocks",
        asset_betas=asset_betas,
    )

    stress_test_summary = pd.DataFrame(
        [
            market_stress_summary,
            custom_stress_summary,
        ]
    )

    stress_test_details = {
        "Market Correction":
            market_stress_detail,
        "Custom Scenario":
            custom_stress_detail,
    }

    # ========================================================
    # VALUE AT RISK AND EXPECTED SHORTFALL
    # ========================================================

    if (
        confidence_level <= 0
        or confidence_level >= 1
    ):
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    historical_var_threshold = float(
        np.quantile(
            portfolio_returns,
            1
            - confidence_level,
            method="linear",
        )
    )

    historical_var_return = float(
        -historical_var_threshold
    )

    historical_var_money = float(
        historical_var_return
        * final_value
    )

    historical_tail_returns = (
        portfolio_returns[
            portfolio_returns
            <= historical_var_threshold
        ]
    )

    historical_es_return = float(
        -historical_tail_returns.mean()
    )

    historical_es_money = float(
        historical_es_return
        * final_value
    )

    portfolio_mean_daily = float(
        portfolio_returns.mean()
    )

    portfolio_std_daily = float(
        portfolio_returns.std(
            ddof=1
        )
    )

    z_score = float(
        norm.ppf(
            1
            - confidence_level
        )
    )

    parametric_var_return = float(
        -(
            portfolio_mean_daily
            + z_score
            * portfolio_std_daily
        )
    )

    parametric_var_money = float(
        parametric_var_return
        * final_value
    )

    tail_probability = (
        1
        - confidence_level
    )

    parametric_es_return = float(
        -(
            portfolio_mean_daily
            - portfolio_std_daily
            * norm.pdf(
                z_score
            )
            / tail_probability
        )
    )

    parametric_es_money = float(
        parametric_es_return
        * final_value
    )

    running_maximum = (
        wealth_index.cummax()
    )

    drawdown = (
        wealth_index
        / running_maximum
        - 1
    ) * 100

    maximum_drawdown = float(
        drawdown.min()
    )

    rolling_volatility = (
        portfolio_returns
        .rolling(
            window=rolling_window
        )
        .std(
            ddof=1
        )
        * np.sqrt(
            252
        )
        * 100
    ).dropna()

    rolling_volatility.name = (
        "Annualized Rolling Volatility (%)"
    )

    allocation_table = pd.DataFrame({
        "Asset":
            weights.index,
        "Weight (%)":
            weights.values
            * 100,
        "Initial Invested Value":
            initial_investment
            * weights.values,
        "Current Estimated Value":
            final_value
            * weights.values,
    })

    initial_portfolio_construction = (
        calculate_initial_portfolio_construction(
            tickers=tickers,
            weights=weights,
            initial_investment=initial_investment,
            requested_start_date=start_date,
            portfolio_currency=portfolio_currency,
            asset_currencies=asset_currencies,
        )
    )

    data_quality_results = (
        build_data_quality_results(
            portfolio_quality=portfolio_quality,
            benchmark_quality=benchmark_quality,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns_full,
            regression_result=regression_result,
            risk_free_table=risk_free_table,
            risk_free_metadata=risk_free_metadata,
        )
    )

    return {
        "portfolio_currency":
            portfolio_currency,
        "currency_symbol":
            currency_symbol,
        "prices":
            prices,
        "prices_local":
            prices_local,
        "asset_currencies":
            asset_currencies,
        "daily_returns":
            daily_returns,
        "portfolio_returns":
            portfolio_returns,
        "cumulative_returns":
            cumulative_returns,
        "portfolio_value":
            portfolio_value,
        "rolling_volatility":
            rolling_volatility,
        "drawdown":
            drawdown,
        "allocation_table":
            allocation_table,
        "initial_portfolio_construction":
            initial_portfolio_construction,
        "initial_investment":
            float(
                initial_investment
            ),
        "final_value":
            final_value,
        "profit_loss":
            profit_loss,
        "cumulative_return":
            cumulative_return,
        "annualized_volatility":
            annualized_volatility,
        "sharpe_ratio":
            sharpe_ratio,
        "maximum_drawdown":
            maximum_drawdown,
        "confidence_level":
            float(
                confidence_level
                * 100
            ),
        "historical_var_return":
            float(
                historical_var_return
                * 100
            ),
        "historical_var_money":
            historical_var_money,
        "parametric_var_return":
            float(
                parametric_var_return
                * 100
            ),
        "parametric_var_money":
            parametric_var_money,
        "historical_es_return":
            float(
                historical_es_return
                * 100
            ),
        "historical_es_money":
            historical_es_money,
        "parametric_es_return":
            float(
                parametric_es_return
                * 100
            ),
        "parametric_es_money":
            parametric_es_money,
        "annualized_asset_returns":
            annualized_asset_returns,
        "return_contribution":
            return_contribution,
        "annualized_portfolio_return":
            annualized_portfolio_return,
        "annualized_covariance_matrix":
            annualized_covariance_matrix,
        "marginal_risk_contribution":
            marginal_risk_contribution,
        "component_risk_contribution":
            component_risk_contribution,
        "risk_contribution_percentage":
            risk_contribution_percentage,
        "contribution_table":
            contribution_table,
        "benchmark_ticker":
            benchmark_ticker,
        "benchmark_currency":
            benchmark_currency,
        "benchmark_prices":
            benchmark_prices,
        "benchmark_prices_local":
            benchmark_prices_local,
        "benchmark_returns":
            aligned_benchmark_returns,
        "benchmark_comparison":
            benchmark_comparison,
        "benchmark_cumulative_return":
            benchmark_cumulative_return,
        "active_return":
            active_return,
        "beta":
            beta,
        "alpha_annualized":
            alpha_annualized,
        "r_squared":
            float(
                regression_result[
                    "r_squared"
                ]
            ),
        "adjusted_r_squared":
            float(
                regression_result[
                    "adjusted_r_squared"
                ]
            ),
        "beta_standard_error_hac":
            float(
                regression_result[
                    "beta_standard_error_hac"
                ]
            ),
        "beta_p_value_hac":
            float(
                regression_result[
                    "beta_p_value_hac"
                ]
            ),
        "beta_confidence_interval":
            regression_result[
                "beta_confidence_interval"
            ],
        "alpha_standard_error_hac":
            float(
                regression_result[
                    "alpha_standard_error_hac"
                ]
            ),
        "alpha_p_value_hac":
            float(
                regression_result[
                    "alpha_p_value_hac"
                ]
            ),
        "alpha_confidence_interval_daily":
            regression_result[
                "alpha_confidence_interval_daily"
            ],
        "regression_observation_count":
            int(
                regression_result[
                    "observation_count"
                ]
            ),
        "regression_hac_lags":
            int(
                regression_result[
                    "hac_lags"
                ]
            ),
        "regression_plot_data":
            regression_result[
                "regression_plot_data"
            ],
        "regression_coefficients":
            regression_result[
                "regression_coefficients"
            ],
        "regression_diagnostics":
            regression_result[
                "regression_diagnostics"
            ],
        "regression_residuals":
            regression_result[
                "residuals"
            ],
        "residual_volatility_annualized":
            float(
                regression_result[
                    "residual_volatility_annualized_percent"
                ]
            ),
        "durbin_watson":
            float(
                regression_result[
                    "durbin_watson"
                ]
            ),
        "ljung_box_p_value":
            float(
                regression_result[
                    "ljung_box_p_value"
                ]
            ),
        "breusch_pagan_p_value":
            float(
                regression_result[
                    "breusch_pagan_p_value"
                ]
            ),
        "jarque_bera_p_value":
            float(
                regression_result[
                    "jarque_bera_p_value"
                ]
            ),
        "reset_p_value":
            float(
                regression_result[
                    "reset_p_value"
                ]
            )
            if pd.notna(
                regression_result[
                    "reset_p_value"
                ]
            )
            else np.nan,
        "influential_observation_count":
            int(
                regression_result[
                    "influential_observation_count"
                ]
            ),
        "risk_free_table":
            risk_free_table,
        **risk_free_metadata,
        "asset_correlation_matrix":
            asset_correlation_matrix,
        "asset_benchmark_correlations":
            asset_benchmark_correlations,
        "portfolio_benchmark_correlation":
            portfolio_benchmark_correlation,
        "average_asset_correlations":
            average_asset_correlations,
        "average_portfolio_correlation":
            average_portfolio_correlation,
        "most_diversifying_asset":
            most_diversifying_asset,
        "most_correlated_asset":
            most_correlated_asset,
        "diversification_summary":
            diversification_summary,
        "asset_betas":
            asset_betas,
        "benchmark_stress_shock":
            float(
                benchmark_stress_shock
                * 100
            ),
        "market_stress_summary":
            market_stress_summary,
        "market_stress_detail":
            market_stress_detail,
        "custom_stress_summary":
            custom_stress_summary,
        "custom_stress_detail":
            custom_stress_detail,
        "stress_test_summary":
            stress_test_summary,
        "stress_test_details":
            stress_test_details,
        **data_quality_results,
    }



