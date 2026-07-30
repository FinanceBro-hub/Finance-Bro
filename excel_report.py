from __future__ import annotations

from datetime import datetime
from io import BytesIO
import math
import re
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import xlsxwriter
except ImportError as error:  # pragma: no cover - executed in the user's environment
    raise ImportError(
        "The Excel export requires XlsxWriter. Install it with: "
        "pip install XlsxWriter"
    ) from error


# ============================================================
# FINANCE BRO — PROFESSIONAL EXCEL REPORT
# ============================================================

BRAND_NAVY = "#0A1F44"
BRAND_BLUE = "#1663F0"
BRAND_SKY = "#23A8F2"
BRAND_GREEN = "#2DB24A"
BRAND_RED = "#DC2626"
BRAND_GOLD = "#D97706"
LIGHT_BLUE = "#EEF4FF"
LIGHT_GREEN = "#ECF8EF"
LIGHT_RED = "#FDECEC"
LIGHT_GOLD = "#FFF7E6"
LIGHT_GREY = "#F5F8FC"
MID_GREY = "#D1D5DB"
DARK_GREY = "#374151"
WHITE = "#FFFFFF"
BLACK = "#000000"
INPUT_BLUE = "#1663F0"
LINK_GREEN = "#2DB24A"

CHART_COLORS = [
    BRAND_BLUE,
    BRAND_SKY,
    BRAND_GREEN,
    BRAND_GOLD,
    "#7C3AED",
    "#DB2777",
    "#0891B2",
    "#4F46E5",
]

CHART_WIDTH = 640
CHART_HEIGHT = 300
CHART_X_OFFSET = 12
CHART_Y_OFFSET = 10


# ============================================================
# PUBLIC FUNCTION
# ============================================================


def create_excel_report(
    results: dict[str, Any],
    rolling_window: int,
    report_title: str = "Finance Bro Portfolio Report",
) -> bytes:
    """
    Creates a polished multi-sheet Excel report and returns its bytes.

    Parameters
    ----------
    results:
        Dictionary returned by ``analyze_portfolio``.
    rolling_window:
        Rolling-volatility window selected in the Streamlit sidebar.
    report_title:
        Main title displayed in the workbook.
    """

    output = BytesIO()

    workbook = xlsxwriter.Workbook(
        output,
        {
            "in_memory": True,
            "nan_inf_to_errors": True,
        },
    )

    workbook.set_properties(
        {
            "title": report_title,
            "subject": "Portfolio analytics, risk and education",
            "author": "Finance Bro",
            "company": "Finance Bro",
            "comments": (
                "Educational portfolio report generated from Yahoo Finance "
                "market data and ECB reference exchange rates."
            ),
        }
    )

    context = _build_report_context(
        workbook=workbook,
        results=results,
        rolling_window=rolling_window,
    )

    _build_model_data_sheet(context)
    _build_executive_summary_sheet(context, report_title)
    _build_data_quality_sheet(context)
    _build_regression_sheet(context)
    _build_portfolio_construction_sheet(context)
    _build_performance_sheet(context)
    _build_risk_sheet(context)
    _build_allocation_sheet(context)
    _build_diversification_sheet(context)
    _build_stress_sheet(context)
    _build_educational_guide_sheet(context)

    # Keep the technical data sheet hidden while preserving formulas and auditability.
    context["model_sheet"].hide()

    workbook.close()
    output.seek(0)
    return output.getvalue()


# ============================================================
# CONTEXT AND FORMATS
# ============================================================


def _build_report_context(
    workbook: xlsxwriter.Workbook,
    results: dict[str, Any],
    rolling_window: int,
) -> dict[str, Any]:

    currency = str(
        results.get("portfolio_currency", "EUR")
    ).upper()

    currency_symbol = str(
        results.get(
            "currency_symbol",
            "€" if currency == "EUR" else "$",
        )
    )

    currency_number_format = (
        f'{currency_symbol}#,##0.00;'
        f'[Red]({currency_symbol}#,##0.00);-'
    )

    currency_number_format_0 = (
        f'{currency_symbol}#,##0;'
        f'[Red]({currency_symbol}#,##0);-'
    )

    percentage_points_format = (
        '0.00"%";[Red](0.00"%");-'
    )

    decimal_format = (
        '0.00;[Red](0.00);-'
    )

    formats = {
        "title": workbook.add_format(
            {
                "bold": True,
                "font_size": 24,
                "font_color": WHITE,
                "bg_color": BRAND_NAVY,
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "subtitle": workbook.add_format(
            {
                "font_size": 12,
                "font_color": WHITE,
                "bg_color": BRAND_NAVY,
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "section": workbook.add_format(
            {
                "bold": True,
                "font_size": 13,
                "font_color": WHITE,
                "bg_color": BRAND_NAVY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "subsection": workbook.add_format(
            {
                "bold": True,
                "font_size": 11,
                "font_color": WHITE,
                "bg_color": BRAND_BLUE,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": WHITE,
                "bg_color": BRAND_BLUE,
                "border": 1,
                "border_color": WHITE,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "header_dark": workbook.add_format(
            {
                "bold": True,
                "font_color": WHITE,
                "bg_color": BRAND_NAVY,
                "border": 1,
                "border_color": WHITE,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "label": workbook.add_format(
            {
                "bold": True,
                "font_color": DARK_GREY,
                "bg_color": LIGHT_GREY,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "text": workbook.add_format(
            {
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "text_wrap": workbook.add_format(
            {
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "left",
                "valign": "top",
                "text_wrap": True,
            }
        ),
        "input_text": workbook.add_format(
            {
                "font_color": INPUT_BLUE,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "linked_text": workbook.add_format(
            {
                "font_color": LINK_GREEN,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "money": workbook.add_format(
            {
                "num_format": currency_number_format,
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "money_0": workbook.add_format(
            {
                "num_format": currency_number_format_0,
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "money_input": workbook.add_format(
            {
                "num_format": currency_number_format,
                "font_color": INPUT_BLUE,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "money_link": workbook.add_format(
            {
                "num_format": currency_number_format,
                "font_color": LINK_GREEN,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "pct_points": workbook.add_format(
            {
                "num_format": percentage_points_format,
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "pct_points_input": workbook.add_format(
            {
                "num_format": percentage_points_format,
                "font_color": INPUT_BLUE,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "pct_points_link": workbook.add_format(
            {
                "num_format": percentage_points_format,
                "font_color": LINK_GREEN,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "pct_decimal": workbook.add_format(
            {
                "num_format": "0.00%;[Red](0.00%);-",
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "decimal": workbook.add_format(
            {
                "num_format": decimal_format,
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "decimal_input": workbook.add_format(
            {
                "num_format": decimal_format,
                "font_color": INPUT_BLUE,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "decimal_link": workbook.add_format(
            {
                "num_format": decimal_format,
                "font_color": LINK_GREEN,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "integer": workbook.add_format(
            {
                "num_format": "0;[Red](0);-",
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "date": workbook.add_format(
            {
                "num_format": "dd/mm/yyyy",
                "font_color": BLACK,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "date_input": workbook.add_format(
            {
                "num_format": "dd/mm/yyyy",
                "font_color": INPUT_BLUE,
                "border": 1,
                "border_color": MID_GREY,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "note": workbook.add_format(
            {
                "font_color": DARK_GREY,
                "font_size": 9,
                "italic": True,
                "text_wrap": True,
                "valign": "top",
            }
        ),
        "disclaimer": workbook.add_format(
            {
                "font_color": DARK_GREY,
                "font_size": 9,
                "bg_color": LIGHT_GOLD,
                "border": 1,
                "border_color": BRAND_GOLD,
                "text_wrap": True,
                "valign": "vcenter",
            }
        ),
        "positive": workbook.add_format(
            {
                "font_color": BRAND_GREEN,
                "bg_color": LIGHT_GREEN,
                "bold": True,
                "border": 1,
                "border_color": BRAND_GREEN,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "negative": workbook.add_format(
            {
                "font_color": BRAND_RED,
                "bg_color": LIGHT_RED,
                "bold": True,
                "border": 1,
                "border_color": BRAND_RED,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "neutral": workbook.add_format(
            {
                "font_color": BRAND_GOLD,
                "bg_color": LIGHT_GOLD,
                "bold": True,
                "border": 1,
                "border_color": BRAND_GOLD,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "kpi_label": workbook.add_format(
            {
                "bold": True,
                "font_color": WHITE,
                "bg_color": BRAND_BLUE,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "border_color": WHITE,
            }
        ),
        "kpi_money": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": LINK_GREEN,
                "bg_color": LIGHT_BLUE,
                "num_format": currency_number_format,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "border_color": BRAND_BLUE,
            }
        ),
        "kpi_pct": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": LINK_GREEN,
                "bg_color": LIGHT_BLUE,
                "num_format": percentage_points_format,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "border_color": BRAND_BLUE,
            }
        ),
        "kpi_decimal": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": LINK_GREEN,
                "bg_color": LIGHT_BLUE,
                "num_format": decimal_format,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "border_color": BRAND_BLUE,
            }
        ),
        "kpi_integer": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": LINK_GREEN,
                "bg_color": LIGHT_BLUE,
                "num_format": "0;[Red](0);-",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "border_color": BRAND_BLUE,
            }
        ),
        "nav": workbook.add_format(
            {
                "font_color": BRAND_BLUE,
                "underline": True,
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "border_color": MID_GREY,
                "bg_color": WHITE,
            }
        ),
        "formula_text": workbook.add_format(
            {
                "font_name": "Cambria Math",
                "font_color": BRAND_NAVY,
                "bg_color": LIGHT_BLUE,
                "border": 1,
                "border_color": BRAND_BLUE,
                "text_wrap": True,
                "align": "center",
                "valign": "vcenter",
            }
        ),
    }

    prices = _as_dataframe(results.get("prices"))

    if not prices.empty:
        start_date = pd.Timestamp(prices.index.min()).to_pydatetime()
        end_date = pd.Timestamp(prices.index.max()).to_pydatetime()
    else:
        start_date = datetime.today()
        end_date = datetime.today()

    model_sheet = workbook.add_worksheet("Model Data")

    return {
        "workbook": workbook,
        "model_sheet": model_sheet,
        "results": results,
        "formats": formats,
        "rolling_window": int(rolling_window),
        "currency": currency,
        "currency_symbol": currency_symbol,
        "currency_number_format": currency_number_format,
        "currency_number_format_0": currency_number_format_0,
        "percentage_points_format": percentage_points_format,
        "decimal_format": decimal_format,
        "start_date": start_date,
        "end_date": end_date,
        "report_date": datetime.now(),
    }


# ============================================================
# SHARED HELPERS
# ============================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(number) or math.isinf(number):
        return default

    return number


def _as_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        return value.to_frame()

    return pd.DataFrame(value)


def _as_series(value: Any, name: str) -> pd.Series:
    if value is None:
        return pd.Series(dtype=float, name=name)

    if isinstance(value, pd.Series):
        result = value.copy()
        result.name = name
        return result

    if isinstance(value, pd.DataFrame) and value.shape[1] >= 1:
        result = value.iloc[:, 0].copy()
        result.name = name
        return result

    return pd.Series(value, name=name)


def _excel_safe_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.to_pydatetime()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()

    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return value


def _sheet_name_formula(sheet_name: str) -> str:
    return sheet_name.replace("'", "''")


def _sanitize_table_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"T_{cleaned}"
    return cleaned[:200]


def _configure_sheet(
    worksheet: xlsxwriter.worksheet.Worksheet,
    freeze_row: int = 4,
) -> None:
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(freeze_row, 0)
    worksheet.set_zoom(90)
    worksheet.set_landscape()
    worksheet.set_paper(9)  # A4
    worksheet.fit_to_pages(1, 0)
    worksheet.set_margins(0.3, 0.3, 0.5, 0.5)
    worksheet.repeat_rows(0, 3)
    worksheet.set_footer(
        "&LFinance Bro&CPage &P of &N&RGenerated &D"
    )


def _write_sheet_header(
    context: dict[str, Any],
    worksheet: xlsxwriter.worksheet.Worksheet,
    title: str,
    subtitle: str,
    last_col: int = 11,
) -> None:
    formats = context["formats"]
    worksheet.set_row(0, 28)
    worksheet.set_row(1, 20)
    worksheet.merge_range(0, 0, 0, last_col, title, formats["title"])
    worksheet.merge_range(1, 0, 1, last_col, subtitle, formats["subtitle"])
    worksheet.merge_range(
        2,
        0,
        2,
        2,
        "← Executive Summary",
        formats["nav"],
    )
    worksheet.write_url(
        2,
        0,
        "internal:'Executive Summary'!A1",
        formats["nav"],
        "← Executive Summary",
    )


def _write_section(
    context: dict[str, Any],
    worksheet: xlsxwriter.worksheet.Worksheet,
    row: int,
    title: str,
    first_col: int = 0,
    last_col: int = 7,
) -> None:
    worksheet.merge_range(
        row,
        first_col,
        row,
        last_col,
        title,
        context["formats"]["section"],
    )
    worksheet.set_row(row, 21)


def _write_value(
    worksheet: xlsxwriter.worksheet.Worksheet,
    row: int,
    col: int,
    value: Any,
    cell_format: Any,
) -> None:
    safe_value = _excel_safe_value(value)

    if isinstance(safe_value, datetime):
        worksheet.write_datetime(row, col, safe_value, cell_format)
    elif safe_value is None:
        worksheet.write_blank(row, col, None, cell_format)
    else:
        worksheet.write(row, col, safe_value, cell_format)


def _detect_column_format(
    context: dict[str, Any],
    column_name: str,
    series: pd.Series,
    hardcoded: bool = False,
) -> Any:
    formats = context["formats"]
    name = str(column_name).lower()

    if pd.api.types.is_datetime64_any_dtype(series):
        return formats["date_input"] if hardcoded else formats["date"]

    if (
        "date" in name
        or "time_period" in name
    ):
        return formats["date_input"] if hardcoded else formats["date"]

    if any(
        token in name
        for token in [
            "value",
            "amount",
            "profit",
            "loss",
            "price",
            "invested",
            "monetary",
        ]
    ) and not any(
        token in name
        for token in [
            "percentage",
            "return",
            "weight",
            "shock",
            "impact",
        ]
    ):
        return formats["money_input"] if hardcoded else formats["money"]

    if any(
        token in name
        for token in [
            "(%)",
            "p.p.",
            "return",
            "volatility",
            "drawdown",
            "weight",
            "shock",
            "impact",
            "confidence",
        ]
    ):
        return (
            formats["pct_points_input"]
            if hardcoded
            else formats["pct_points"]
        )

    if any(
        token in name
        for token in [
            "beta",
            "alpha",
            "correlation",
            "sharpe",
            "rate",
            "shares",
        ]
    ):
        return formats["decimal_input"] if hardcoded else formats["decimal"]

    if pd.api.types.is_numeric_dtype(series):
        return formats["decimal_input"] if hardcoded else formats["decimal"]

    return formats["input_text"] if hardcoded else formats["text"]


def _column_width(column_name: str, series: pd.Series) -> float:
    header_length = len(str(column_name))

    sample_lengths = []
    for value in series.head(100).tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        sample_lengths.append(len(str(value)))

    maximum = max([header_length] + sample_lengths + [8])

    if pd.api.types.is_datetime64_any_dtype(series) or "date" in str(column_name).lower():
        return 13

    if pd.api.types.is_numeric_dtype(series):
        return min(max(maximum + 2, 12), 22)

    return min(max(maximum + 2, 12), 34)


def _write_dataframe(
    context: dict[str, Any],
    worksheet: xlsxwriter.worksheet.Worksheet,
    dataframe: pd.DataFrame,
    start_row: int,
    start_col: int,
    table_name: str,
    hardcoded: bool = True,
    add_table: bool = True,
    column_formats: Optional[dict[str, Any]] = None,
) -> tuple[int, int, dict[str, int]]:
    formats = context["formats"]
    df = dataframe.copy()

    if not isinstance(df.index, pd.RangeIndex):
        index_name = df.index.name or "Date"
        df = df.reset_index(names=index_name)

    df.columns = [str(column) for column in df.columns]

    if df.empty:
        worksheet.write(
            start_row,
            start_col,
            "No data available",
            formats["note"],
        )
        return start_row, start_col, {}

    column_mapping = {
        column: start_col + offset
        for offset, column in enumerate(df.columns)
    }

    for offset, column in enumerate(df.columns):
        worksheet.write(
            start_row,
            start_col + offset,
            column,
            formats["header"],
        )

    worksheet.set_row(
        start_row,
        32,
    )

    detected_formats = {}

    for offset, column in enumerate(df.columns):
        if column_formats and column in column_formats:
            detected_formats[column] = column_formats[column]
        else:
            detected_formats[column] = _detect_column_format(
                context=context,
                column_name=column,
                series=df[column],
                hardcoded=hardcoded,
            )

        worksheet.set_column(
            start_col + offset,
            start_col + offset,
            _column_width(column, df[column]),
        )

    for row_offset, row_values in enumerate(
        df.itertuples(index=False, name=None),
        start=1,
    ):
        worksheet.set_row(
            start_row + row_offset,
            22,
        )

        for column_offset, value in enumerate(row_values):
            column_name = df.columns[column_offset]
            _write_value(
                worksheet=worksheet,
                row=start_row + row_offset,
                col=start_col + column_offset,
                value=value,
                cell_format=detected_formats[column_name],
            )

    last_row = start_row + len(df)
    last_col = start_col + len(df.columns) - 1

    if add_table:
        worksheet.add_table(
            start_row,
            start_col,
            last_row,
            last_col,
            {
                "name": _sanitize_table_name(table_name),
                "style": "Table Style Medium 2",
                "columns": [
                    {
                        "header": column,
                        "format": detected_formats[column],
                    }
                    for column in df.columns
                ],
            },
        )

    return last_row, last_col, column_mapping


def _add_source_comment(
    worksheet: xlsxwriter.worksheet.Worksheet,
    row: int,
    col: int,
    text: str,
) -> None:
    worksheet.write_comment(
        row,
        col,
        text,
        {
            "author": "Finance Bro",
            "width": 340,
            "height": 110,
        },
    )


def _add_line_chart(
    context: dict[str, Any],
    worksheet: xlsxwriter.worksheet.Worksheet,
    title: str,
    categories: tuple[int, int, int],
    series_specs: Iterable[dict[str, Any]],
    position: str,
    end_position: str,
    y_axis_name: str,
    y_axis_format: Optional[str] = None,
) -> None:
    workbook = context["workbook"]
    chart = workbook.add_chart({"type": "line"})

    for index, spec in enumerate(series_specs):
        chart.add_series(
            {
                "name": spec["name"],
                "categories": [
                    worksheet.name,
                    categories[0],
                    categories[1],
                    categories[2],
                    categories[1],
                ],
                "values": [
                    worksheet.name,
                    spec["first_row"],
                    spec["column"],
                    spec["last_row"],
                    spec["column"],
                ],
                "line": {
                    "color": spec.get(
                        "color",
                        CHART_COLORS[index % len(CHART_COLORS)],
                    ),
                    "width": 2.25,
                },
            }
        )

    chart.set_title(
        {
            "name": title,
            "name_font": {"size": 14, "bold": True, "color": BRAND_NAVY},
        }
    )
    chart.set_legend({"position": "bottom"})
    chart.set_chartarea({"border": {"none": True}})
    chart.set_plotarea(
        {
            "border": {"color": MID_GREY},
            "fill": {"color": WHITE},
        }
    )
    chart.set_x_axis(
        {
            "name": "Date",
            "date_axis": True,
            "num_format": "mmm-yy",
            "label_position": "low",
            "major_gridlines": {"visible": False},
        }
    )
    y_axis_options = {
        "name": y_axis_name,
        "major_gridlines": {"visible": True, "line": {"color": LIGHT_GREY}},
    }
    if y_axis_format:
        y_axis_options["num_format"] = y_axis_format
    chart.set_y_axis(y_axis_options)
    chart.set_size(
        {
            "width": CHART_WIDTH,
            "height": CHART_HEIGHT,
        }
    )
    worksheet.insert_chart(
        position,
        chart,
        {
            "x_offset": CHART_X_OFFSET,
            "y_offset": CHART_Y_OFFSET,
        },
    )


def _add_column_chart(
    context: dict[str, Any],
    worksheet: xlsxwriter.worksheet.Worksheet,
    title: str,
    categories: tuple[int, int, int],
    series_specs: Iterable[dict[str, Any]],
    position: str,
    y_axis_name: str,
    y_axis_format: Optional[str] = None,
    stacked: bool = False,
) -> None:
    workbook = context["workbook"]
    chart = workbook.add_chart(
        {
            "type": "column",
            "subtype": "stacked" if stacked else "clustered",
        }
    )

    for index, spec in enumerate(series_specs):
        chart.add_series(
            {
                "name": spec["name"],
                "categories": [
                    worksheet.name,
                    categories[0],
                    categories[1],
                    categories[2],
                    categories[1],
                ],
                "values": [
                    worksheet.name,
                    spec["first_row"],
                    spec["column"],
                    spec["last_row"],
                    spec["column"],
                ],
                "fill": {
                    "color": spec.get(
                        "color",
                        CHART_COLORS[index % len(CHART_COLORS)],
                    )
                },
                "border": {"none": True},
                "data_labels": spec.get("data_labels", {"value": False}),
            }
        )

    chart.set_title(
        {
            "name": title,
            "name_font": {"size": 14, "bold": True, "color": BRAND_NAVY},
        }
    )
    chart.set_legend({"position": "bottom"})
    chart.set_chartarea({"border": {"none": True}})
    chart.set_plotarea(
        {
            "border": {"color": MID_GREY},
            "fill": {"color": WHITE},
        }
    )
    chart.set_x_axis(
        {
            "major_gridlines": {"visible": False},
            "label_position": "low",
        }
    )
    y_axis_options = {
        "name": y_axis_name,
        "major_gridlines": {"visible": True, "line": {"color": LIGHT_GREY}},
    }
    if y_axis_format:
        y_axis_options["num_format"] = y_axis_format
    chart.set_y_axis(y_axis_options)
    chart.set_size(
        {
            "width": CHART_WIDTH,
            "height": CHART_HEIGHT,
        }
    )
    worksheet.insert_chart(
        position,
        chart,
        {
            "x_offset": CHART_X_OFFSET,
            "y_offset": CHART_Y_OFFSET,
        },
    )



def _build_executive_conclusion(
    results: dict[str, Any],
) -> str:

    cumulative_return = _safe_float(
        results.get(
            "cumulative_return"
        )
    )

    volatility = _safe_float(
        results.get(
            "annualized_volatility"
        )
    )

    sharpe = _safe_float(
        results.get(
            "sharpe_ratio"
        )
    )

    drawdown = _safe_float(
        results.get(
            "maximum_drawdown"
        )
    )

    beta = _safe_float(
        results.get(
            "beta"
        )
    )

    alpha = _safe_float(
        results.get(
            "alpha_annualized"
        )
    )

    r_squared = (
        _safe_float(
            results.get(
                "r_squared"
            )
        )
        * 100
    )

    retention = _safe_float(
        results.get(
            "data_quality_headline",
            {}
        ).get(
            "data_retention_percent"
        )
    )

    performance_text = (
        "The portfolio generated a positive cumulative return"
        if cumulative_return >= 0
        else "The portfolio generated a negative cumulative return"
    )

    if volatility < 10:
        risk_text = (
            "with relatively low historical volatility"
        )
    elif volatility < 20:
        risk_text = (
            "with moderate historical volatility"
        )
    elif volatility < 35:
        risk_text = (
            "with high historical volatility"
        )
    else:
        risk_text = (
            "with very high historical volatility"
        )

    if sharpe < 0:
        sharpe_text = (
            "The Sharpe Ratio was negative, indicating that historical "
            "risk-adjusted performance was weaker than the selected "
            "risk-free reference."
        )
    elif sharpe < 1:
        sharpe_text = (
            "The Sharpe Ratio was positive but below 1, indicating limited "
            "historical excess return per unit of volatility."
        )
    else:
        sharpe_text = (
            "The Sharpe Ratio was at least 1, indicating comparatively "
            "strong historical excess return per unit of volatility."
        )

    regression_text = (
        f"The excess-return regression estimated a beta of {beta:.2f}, "
        f"annualized alpha of {alpha:.2f}% and R-squared of "
        f"{r_squared:.2f}%."
    )

    quality_text = (
        f"Strict common-date alignment retained {retention:.2f}% of the "
        "raw portfolio market-date rows. Missing stock prices were not "
        "interpolated, and flagged return anomalies were retained for review."
    )

    return (
        f"{performance_text} of {cumulative_return:.2f}% {risk_text} "
        f"({volatility:.2f}%). The maximum historical drawdown was "
        f"{drawdown:.2f}%. {sharpe_text} {regression_text} {quality_text} "
        "All conclusions are sample-dependent and should be interpreted "
        "together with the detailed risk, data-quality and diagnostic sheets."
    )



# ============================================================
# MODEL DATA SHEET
# ============================================================



def _build_model_data_sheet(context: dict[str, Any]) -> None:
    worksheet = context["model_sheet"]
    results = context["results"]
    formats = context["formats"]

    worksheet.hide_gridlines(2)
    worksheet.set_column("A:A", 40)
    worksheet.set_column("B:B", 24)
    worksheet.set_column("C:C", 80)

    worksheet.write("A1", "Metric", formats["header_dark"])
    worksheet.write("B1", "Value", formats["header_dark"])
    worksheet.write("C1", "Source / Calculation", formats["header_dark"])

    metrics = [
        ("Initial Investment", results.get("initial_investment"), "User input"),
        ("Final Portfolio Value", results.get("final_value"), "Calculated from converted portfolio returns"),
        ("Profit / Loss", None, "Final Portfolio Value - Initial Investment"),
        ("Cumulative Return (%)", results.get("cumulative_return"), "Portfolio wealth-index change"),
        ("Annualized Volatility (%)", results.get("annualized_volatility"), "Sample daily volatility × √252"),
        ("Sharpe Ratio", results.get("sharpe_ratio"), "Mean daily excess return / sample standard deviation × √252"),
        ("Maximum Drawdown (%)", results.get("maximum_drawdown"), "Largest historical peak-to-trough decline"),
        ("Benchmark Return (%)", results.get("benchmark_cumulative_return"), "Converted benchmark cumulative return"),
        ("Active Return (%)", results.get("active_return"), "Portfolio cumulative return - benchmark cumulative return"),
        ("Portfolio Beta", results.get("beta"), "Excess-return OLS slope"),
        ("Annualized Alpha (%)", results.get("alpha_annualized"), "Daily OLS intercept × 252"),
        ("Historical VaR", results.get("historical_var_money"), "Historical return quantile × final value"),
        ("Historical Expected Shortfall", results.get("historical_es_money"), "Average historical loss beyond VaR"),
        ("Parametric VaR", results.get("parametric_var_money"), "Normal-distribution estimate"),
        ("Parametric Expected Shortfall", results.get("parametric_es_money"), "Normal-distribution tail estimate"),
        ("Confidence Level (%)", results.get("confidence_level"), "User input"),
        ("Average Asset Correlation", results.get("average_portfolio_correlation"), "Average off-diagonal correlation"),
        ("Market Stress Impact (%)", results.get("market_stress_summary", {}).get("Portfolio Change (%)"), "Beta-based stress scenario"),
        ("Custom Stress Impact (%)", results.get("custom_stress_summary", {}).get("Portfolio Change (%)"), "User-defined stress scenario"),
        ("Portfolio Currency", context["currency"], "User input"),
        ("Benchmark", results.get("benchmark_ticker"), "User input"),
        ("Start Date", context["start_date"], "First common converted price observation"),
        ("End Date", context["end_date"], "Last common converted price observation"),
        ("Report Date", context["report_date"], "Workbook generation timestamp"),
        ("R-Squared (%)", _safe_float(results.get("r_squared")) * 100, "Excess-return OLS coefficient of determination"),
        ("Adjusted R-Squared (%)", _safe_float(results.get("adjusted_r_squared")) * 100, "OLS R-squared adjusted for model dimension"),
        ("Beta HAC p-value", results.get("beta_p_value_hac"), "Newey-West HAC inference"),
        ("Alpha HAC p-value", results.get("alpha_p_value_hac"), "Newey-West HAC inference"),
        ("Regression Observations", results.get("regression_observation_count"), "Common portfolio, benchmark and risk-free observations"),
        ("Risk-Free Source", results.get("risk_free_source"), "Official series selected by portfolio currency"),
        ("Average Annual Risk-Free Rate (%)", results.get("risk_free_average_annual_rate_percent"), "Average aligned annual reference yield"),
        ("Latest Annual Risk-Free Rate (%)", results.get("risk_free_latest_annual_rate_percent"), "Latest aligned official reference yield"),
        ("Common-Date Data Retention (%)", results.get("data_quality_headline", {}).get("data_retention_percent"), "Common price dates / raw market-date rows"),
        ("Potential Anomalies Flagged", results.get("data_quality_headline", {}).get("potential_anomaly_count"), "Robust review flags; observations retained"),
    ]

    money_rows = {
        2,
        3,
        13,
        14,
        15,
        16,
    }

    pct_rows = {
        5,
        6,
        8,
        9,
        10,
        12,
        17,
        19,
        20,
        26,
        27,
        32,
        33,
        34,
    }

    decimal_rows = {
        7,
        11,
        18,
        28,
        29,
    }

    integer_rows = {
        30,
        35,
    }

    date_rows = {
        23,
        24,
        25,
    }

    for row_number, (
        label,
        value,
        calculation_source,
    ) in enumerate(
        metrics,
        start=2,
    ):

        excel_row = (
            row_number
            - 1
        )

        worksheet.write(
            excel_row,
            0,
            label,
            formats["label"],
        )

        if row_number == 4:

            worksheet.write_formula(
                excel_row,
                1,
                "=B3-B2",
                formats["money"],
                _safe_float(
                    results.get(
                        "profit_loss"
                    )
                ),
            )

        elif row_number in money_rows:

            _write_value(
                worksheet,
                excel_row,
                1,
                value,
                formats["money_input"],
            )

        elif row_number in pct_rows:

            _write_value(
                worksheet,
                excel_row,
                1,
                value,
                formats["pct_points_input"],
            )

        elif row_number in decimal_rows:

            _write_value(
                worksheet,
                excel_row,
                1,
                value,
                formats["decimal_input"],
            )

        elif row_number in integer_rows:

            _write_value(
                worksheet,
                excel_row,
                1,
                value,
                formats["integer"],
            )

        elif row_number in date_rows:

            _write_value(
                worksheet,
                excel_row,
                1,
                value,
                formats["date_input"],
            )

        else:

            _write_value(
                worksheet,
                excel_row,
                1,
                value,
                formats["input_text"],
            )

        worksheet.write(
            excel_row,
            2,
            calculation_source,
            formats["text"],
        )

    _add_source_comment(
        worksheet,
        1,
        1,
        (
            "Portfolio inputs and analytics originate from the Finance Bro "
            "Streamlit analysis. Market prices are downloaded from Yahoo "
            "Finance."
        ),
    )

    _add_source_comment(
        worksheet,
        19,
        1,
        (
            "Foreign-exchange conversions use European Central Bank daily "
            "reference rates: https://data.ecb.europa.eu/"
        ),
    )

    _add_source_comment(
        worksheet,
        29,
        1,
        (
            "EUR automatic risk-free series: ECB 3-month compounded €STR, "
            "series EST.B.EU000A2QQF32.CR. USD automatic series: Federal "
            "Reserve / FRED DGS3MO."
        ),
    )



# ============================================================
# EXECUTIVE SUMMARY
# ============================================================



def _build_executive_summary_sheet(
    context: dict[str, Any],
    report_title: str,
) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Executive Summary")
    formats = context["formats"]
    results = context["results"]

    _configure_sheet(
        worksheet,
        freeze_row=4,
    )

    worksheet.set_column("A:A", 20)
    worksheet.set_column("B:H", 15)
    worksheet.set_column("I:L", 14)

    worksheet.set_row(0, 34)
    worksheet.merge_range(
        "A1:L2",
        report_title,
        formats["title"],
    )

    worksheet.merge_range(
        "A3:L3",
        "Understand Today. Invest Better Tomorrow.",
        formats["subtitle"],
    )

    worksheet.merge_range(
        "A5:L5",
        "Report Overview",
        formats["section"],
    )

    overview_items = [
        (
            "Report Date",
            context["report_date"],
            formats["date_input"],
        ),
        (
            "Portfolio Currency",
            context["currency"],
            formats["input_text"],
        ),
        (
            "Benchmark",
            results.get("benchmark_ticker"),
            formats["input_text"],
        ),
        (
            "Analysis Period",
            (
                f"{context['start_date']:%d/%m/%Y} – "
                f"{context['end_date']:%d/%m/%Y}"
            ),
            formats["input_text"],
        ),
        (
            "Assets",
            ", ".join(
                _as_dataframe(
                    results.get(
                        "prices"
                    )
                ).columns.astype(
                    str
                )
            ),
            formats["input_text"],
        ),
        (
            "Risk-Free Source",
            results.get("risk_free_source"),
            formats["input_text"],
        ),
    ]

    for index, (
        label,
        value,
        value_format,
    ) in enumerate(
        overview_items
    ):

        row = (
            5
            + index // 3
        )

        group = (
            index % 3
        )

        label_col = (
            group * 4
        )

        worksheet.write(
            row,
            label_col,
            label,
            formats["label"],
        )

        worksheet.merge_range(
            row,
            label_col + 1,
            row,
            label_col + 3,
            value,
            value_format,
        )

    kpis = [
        (
            "Initial Investment",
            "='Model Data'!B2",
            results.get("initial_investment"),
            formats["kpi_money"],
        ),
        (
            "Final Portfolio Value",
            "='Model Data'!B3",
            results.get("final_value"),
            formats["kpi_money"],
        ),
        (
            "Profit / Loss",
            "='Model Data'!B4",
            results.get("profit_loss"),
            formats["kpi_money"],
        ),
        (
            "Cumulative Return",
            "='Model Data'!B5",
            results.get("cumulative_return"),
            formats["kpi_pct"],
        ),
        (
            "Annualized Volatility",
            "='Model Data'!B6",
            results.get("annualized_volatility"),
            formats["kpi_pct"],
        ),
        (
            "Sharpe Ratio",
            "='Model Data'!B7",
            results.get("sharpe_ratio"),
            formats["kpi_decimal"],
        ),
        (
            "Maximum Drawdown",
            "='Model Data'!B8",
            results.get("maximum_drawdown"),
            formats["kpi_pct"],
        ),
        (
            "Historical VaR",
            "='Model Data'!B13",
            results.get("historical_var_money"),
            formats["kpi_money"],
        ),
    ]

    worksheet.merge_range(
        "A9:L9",
        "Key Metrics",
        formats["section"],
    )

    for index, (
        label,
        formula,
        cached_value,
        value_format,
    ) in enumerate(
        kpis
    ):

        row_group = (
            index // 4
        )

        col_group = (
            index % 4
        )

        first_col = (
            col_group * 3
        )

        label_row = (
            9
            + row_group * 3
        )

        value_row = (
            label_row
            + 1
        )

        worksheet.merge_range(
            label_row,
            first_col,
            label_row,
            first_col + 2,
            label,
            formats["kpi_label"],
        )

        worksheet.merge_range(
            value_row,
            first_col,
            value_row + 1,
            first_col + 2,
            "",
            value_format,
        )

        worksheet.write_formula(
            value_row,
            first_col,
            formula,
            value_format,
            _safe_float(
                cached_value
            ),
        )

    worksheet.merge_range(
        "A17:L17",
        "Regression Summary",
        formats["section"],
    )

    regression_kpis = [
        (
            "Beta",
            results.get("beta"),
            formats["kpi_decimal"],
        ),
        (
            "Annualized Alpha",
            results.get("alpha_annualized"),
            formats["kpi_pct"],
        ),
        (
            "R-Squared",
            _safe_float(
                results.get(
                    "r_squared"
                )
            )
            * 100,
            formats["kpi_pct"],
        ),
        (
            "Regression Observations",
            results.get(
                "regression_observation_count"
            ),
            formats["kpi_integer"],
        ),
    ]

    for index, (
        label,
        value,
        value_format,
    ) in enumerate(
        regression_kpis
    ):

        first_col = (
            index * 3
        )

        worksheet.merge_range(
            17,
            first_col,
            17,
            first_col + 2,
            label,
            formats["kpi_label"],
        )

        worksheet.merge_range(
            18,
            first_col,
            19,
            first_col + 2,
            "",
            value_format,
        )

        _write_value(
            worksheet,
            18,
            first_col,
            value,
            value_format,
        )

    worksheet.merge_range(
        "A22:L22",
        "Executive Interpretation",
        formats["section"],
    )

    worksheet.merge_range(
        "A23:L26",
        _build_executive_conclusion(
            results
        ),
        formats["text_wrap"],
    )

    worksheet.set_row(
        22,
        24,
    )

    for row in range(
        23,
        27,
    ):
        worksheet.set_row(
            row,
            31,
        )

    worksheet.merge_range(
        "A28:L28",
        "Workbook Navigation",
        formats["section"],
    )

    navigation = [
        "Data Quality",
        "Regression Analysis",
        "Portfolio Construction",
        "Performance",
        "Risk Analysis",
        "Allocation",
        "Diversification",
        "Stress Tests",
        "Educational Guide",
    ]

    for index, sheet_name in enumerate(
        navigation
    ):

        start_col = (
            index % 4
            * 3
        )

        row = (
            28
            + index // 4
        )

        worksheet.merge_range(
            row,
            start_col,
            row,
            start_col + 2,
            sheet_name,
            formats["nav"],
        )

        worksheet.write_url(
            row,
            start_col,
            (
                f"internal:'"
                f"{_sheet_name_formula(sheet_name)}"
                f"'!A1"
            ),
            formats["nav"],
            sheet_name,
        )

    worksheet.merge_range(
        "A33:L36",
        (
            "Educational use only. Results are based on historical market "
            "data, official reference exchange rates, official or manually "
            "selected risk-free rates, a one-factor benchmark regression "
            "and hypothetical stress assumptions. They do not constitute "
            "investment advice, a recommendation or a forecast."
        ),
        formats["disclaimer"],
    )




# ============================================================
# DATA QUALITY
# ============================================================


def _build_data_quality_sheet(
    context: dict[str, Any],
) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Data Quality")
    formats = context["formats"]
    results = context["results"]

    _configure_sheet(
        worksheet
    )

    _write_sheet_header(
        context,
        worksheet,
        "Data Quality",
        (
            "Strict common-date alignment, coverage statistics and "
            "non-destructive anomaly review"
        ),
        last_col=11,
    )

    headline = (
        results.get(
            "data_quality_headline",
            {}
        )
    )

    kpis = [
        (
            "Raw Market Dates",
            headline.get(
                "raw_market_dates"
            ),
        ),
        (
            "Common Price Dates",
            headline.get(
                "common_price_dates"
            ),
        ),
        (
            "Return Observations",
            headline.get(
                "portfolio_return_observations"
            ),
        ),
        (
            "Common-Date Retention",
            headline.get(
                "data_retention_percent"
            ),
        ),
        (
            "Price Cell Coverage",
            headline.get(
                "overall_price_cell_coverage_percent"
            ),
        ),
        (
            "Potential Anomalies",
            headline.get(
                "potential_anomaly_count"
            ),
        ),
    ]

    _write_section(
        context,
        worksheet,
        4,
        "Data Quality Summary",
        0,
        11,
    )

    for index, (
        label,
        value,
    ) in enumerate(
        kpis
    ):

        group = (
            index % 3
        )

        row_group = (
            index // 3
        )

        first_col = (
            group * 4
        )

        label_row = (
            5
            + row_group * 3
        )

        value_row = (
            label_row
            + 1
        )

        worksheet.merge_range(
            label_row,
            first_col,
            label_row,
            first_col + 3,
            label,
            formats["kpi_label"],
        )

        value_format = (
            formats["kpi_pct"]
            if "Retention" in label
            or "Coverage" in label
            else formats["kpi_integer"]
        )

        worksheet.merge_range(
            value_row,
            first_col,
            value_row + 1,
            first_col + 3,
            "",
            value_format,
        )

        _write_value(
            worksheet,
            value_row,
            first_col,
            value,
            value_format,
        )

    alignment = _as_dataframe(
        results.get(
            "data_quality_alignment_table"
        )
    )

    _write_section(
        context,
        worksheet,
        12,
        "Observation Alignment",
        0,
        7,
    )

    alignment_last, _, _ = _write_dataframe(
        context,
        worksheet,
        alignment,
        start_row=13,
        start_col=0,
        table_name="DataQualityAlignment",
        hardcoded=False,
        column_formats={
            "Stage":
                formats["text"],
            "Observations":
                formats["integer"],
            "Removed from Previous Stage":
                formats["integer"],
            "Retention from Previous Stage (%)":
                formats["pct_points"],
        },
    )

    asset_quality = _as_dataframe(
        results.get(
            "data_quality_asset_table"
        )
    )

    asset_start = (
        alignment_last
        + 3
    )

    _write_section(
        context,
        worksheet,
        asset_start,
        "Coverage by Asset",
        0,
        11,
    )

    asset_last, _, asset_mapping = _write_dataframe(
        context,
        worksheet,
        asset_quality,
        start_row=asset_start + 1,
        start_col=0,
        table_name="AssetDataQuality",
        hardcoded=False,
        column_formats={
            "Asset":
                formats["text"],
            "Trading Currency":
                formats["text"],
            "Portfolio Currency":
                formats["text"],
            "First Valid Date":
                formats["date"],
            "Last Valid Date":
                formats["date"],
            "Raw Date Rows":
                formats["integer"],
            "Valid Prices":
                formats["integer"],
            "Missing Prices":
                formats["integer"],
            "Coverage (%)":
                formats["pct_points"],
            "Dates Removed by Common Alignment":
                formats["integer"],
            "Potential Return Anomalies":
                formats["integer"],
            "Largest Absolute Daily Return (%)":
                formats["pct_points"],
        },
    )

    if (
        "Coverage (%)"
        in asset_mapping
        and not asset_quality.empty
    ):

        coverage_col = asset_mapping[
            "Coverage (%)"
        ]

        worksheet.conditional_format(
            asset_start + 2,
            coverage_col,
            asset_last,
            coverage_col,
            {
                "type":
                    "3_color_scale",
                "min_color":
                    LIGHT_RED,
                "mid_color":
                    LIGHT_GOLD,
                "max_color":
                    LIGHT_GREEN,
            },
        )

    anomalies = _as_dataframe(
        results.get(
            "data_quality_anomalies"
        )
    )

    anomaly_start = (
        asset_last
        + 3
    )

    _write_section(
        context,
        worksheet,
        anomaly_start,
        "Potential Return Anomalies — Retained for Review",
        0,
        11,
    )

    anomaly_last, _, _ = _write_dataframe(
        context,
        worksheet,
        anomalies,
        start_row=anomaly_start + 1,
        start_col=0,
        table_name="PotentialReturnAnomalies",
        hardcoded=False,
        column_formats={
            "Date":
                formats["date"],
            "Series":
                formats["text"],
            "Return Basis":
                formats["text_wrap"],
            "Return (%)":
                formats["pct_points"],
            "Modified Z-Score":
                formats["decimal"],
            "Flag Rule":
                formats["text_wrap"],
        },
    )

    methodology = _as_dataframe(
        results.get(
            "data_quality_methodology_table"
        )
    )

    methodology_start = (
        anomaly_last
        + 3
    )

    _write_section(
        context,
        worksheet,
        methodology_start,
        "Cleaning and Alignment Methodology",
        0,
        11,
    )

    methodology_last, _, _ = _write_dataframe(
        context,
        worksheet,
        methodology,
        start_row=methodology_start + 1,
        start_col=0,
        table_name="DataQualityMethodology",
        hardcoded=False,
        column_formats={
            "Process":
                formats["label"],
            "Method":
                formats["text_wrap"],
        },
    )

    worksheet.merge_range(
        methodology_last + 3,
        0,
        methodology_last + 6,
        11,
        (
            "No stock-price interpolation, forward filling, backward "
            "filling, winsorization or automatic anomaly deletion is "
            "performed. A flagged observation may be a genuine market event. "
            "ECB foreign-exchange and official risk-free series are aligned "
            "with the last published observation available on or before the "
            "market date; this is not linear interpolation."
        ),
        formats["disclaimer"],
    )

    _add_source_comment(
        worksheet,
        5,
        0,
        (
            "Market prices: Yahoo Finance. FX and EUR risk-free data: "
            "European Central Bank Data Portal. USD risk-free data: "
            "Federal Reserve / FRED DGS3MO."
        ),
    )


# ============================================================
# REGRESSION ANALYSIS
# ============================================================


def _build_regression_sheet(
    context: dict[str, Any],
) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Regression Analysis")
    formats = context["formats"]
    results = context["results"]

    _configure_sheet(
        worksheet
    )

    _write_sheet_header(
        context,
        worksheet,
        "Regression Analysis",
        (
            "Excess-return OLS model with Newey-West HAC inference "
            "and advanced diagnostics"
        ),
        last_col=15,
    )

    _write_section(
        context,
        worksheet,
        4,
        "Model Summary",
        0,
        15,
    )

    summary_items = [
        (
            "Beta",
            results.get(
                "beta"
            ),
            formats["kpi_decimal"],
        ),
        (
            "Annualized Alpha",
            results.get(
                "alpha_annualized"
            ),
            formats["kpi_pct"],
        ),
        (
            "R-Squared",
            _safe_float(
                results.get(
                    "r_squared"
                )
            )
            * 100,
            formats["kpi_pct"],
        ),
        (
            "Observations",
            results.get(
                "regression_observation_count"
            ),
            formats["kpi_integer"],
        ),
    ]

    for index, (
        label,
        value,
        value_format,
    ) in enumerate(
        summary_items
    ):

        first_col = (
            index * 4
        )

        worksheet.merge_range(
            5,
            first_col,
            5,
            first_col + 3,
            label,
            formats["kpi_label"],
        )

        worksheet.merge_range(
            6,
            first_col,
            7,
            first_col + 3,
            "",
            value_format,
        )

        _write_value(
            worksheet,
            6,
            first_col,
            value,
            value_format,
        )

    worksheet.merge_range(
        9,
        0,
        10,
        15,
        (
            "Model: R_p,t − R_f,t = α + β(R_m,t − R_f,t) + ε_t. "
            f"Risk-free source: {results.get('risk_free_source')}. "
            f"Average aligned annual rate: "
            f"{_safe_float(results.get('risk_free_average_annual_rate_percent')):.4f}%. "
            f"Newey-West HAC lags: "
            f"{results.get('regression_hac_lags')}."
        ),
        formats["text_wrap"],
    )

    coefficients = _as_dataframe(
        results.get(
            "regression_coefficients"
        )
    )

    _write_section(
        context,
        worksheet,
        12,
        "Robust Coefficient Estimates",
        0,
        7,
    )

    coefficient_last, _, _ = _write_dataframe(
        context,
        worksheet,
        coefficients,
        start_row=13,
        start_col=0,
        table_name="RegressionCoefficients",
        hardcoded=False,
        column_formats={
            "Coefficient":
                formats["text"],
            "Estimate":
                formats["decimal"],
            "Robust Standard Error":
                formats["decimal"],
            "Robust p-value":
                formats["decimal"],
            **{
                column:
                    formats["decimal"]
                for column in coefficients.columns
                if "CI Lower" in str(column)
                or "CI Upper" in str(column)
            },
        },
    )

    diagnostics = _as_dataframe(
        results.get(
            "regression_diagnostics"
        )
    )

    diagnostic_start = (
        coefficient_last
        + 3
    )

    _write_section(
        context,
        worksheet,
        diagnostic_start,
        "Advanced Diagnostics",
        0,
        7,
    )

    diagnostic_last, _, _ = _write_dataframe(
        context,
        worksheet,
        diagnostics,
        start_row=diagnostic_start + 1,
        start_col=0,
        table_name="RegressionDiagnostics",
        hardcoded=False,
        column_formats={
            "Diagnostic":
                formats["text"],
            "Value":
                formats["decimal"],
        },
    )

    regression_data = _as_dataframe(
        results.get(
            "regression_plot_data"
        )
    )

    regression_data_start = 12

    _write_section(
        context,
        worksheet,
        regression_data_start,
        "Regression Observations",
        9,
        15,
    )

    regression_last, _, mapping = _write_dataframe(
        context,
        worksheet,
        regression_data,
        start_row=regression_data_start + 1,
        start_col=9,
        table_name="RegressionObservations",
        hardcoded=False,
        add_table=True,
        column_formats={
            "Date":
                formats["date"],
            "Portfolio Excess Return":
                formats["pct_points"],
            "Benchmark Excess Return":
                formats["pct_points"],
            "Fitted Portfolio Excess Return":
                formats["pct_points"],
            "Residual":
                formats["pct_points"],
        },
    )

    required_columns = {
        "Date",
        "Portfolio Excess Return",
        "Benchmark Excess Return",
        "Fitted Portfolio Excess Return",
        "Residual",
    }

    if (
        required_columns.issubset(
            mapping
        )
        and not regression_data.empty
    ):

        sorted_line_data = (
            regression_data
            .sort_values(
                "Benchmark Excess Return"
            )
            .reset_index(
                drop=True
            )
        )

        line_start_col = 17

        line_last, _, line_mapping = _write_dataframe(
            context,
            worksheet,
            sorted_line_data[
                [
                    "Benchmark Excess Return",
                    "Fitted Portfolio Excess Return",
                ]
            ],
            start_row=regression_data_start + 1,
            start_col=line_start_col,
            table_name="RegressionLineData",
            hardcoded=False,
            add_table=False,
            column_formats={
                "Benchmark Excess Return":
                    formats["pct_points"],
                "Fitted Portfolio Excess Return":
                    formats["pct_points"],
            },
        )

        scatter_chart = workbook.add_chart(
            {
                "type":
                    "scatter",
                "subtype":
                    "straight_with_markers",
            }
        )

        scatter_chart.add_series(
            {
                "name":
                    "Daily Observations",
                "categories": [
                    worksheet.name,
                    regression_data_start + 2,
                    mapping[
                        "Benchmark Excess Return"
                    ],
                    regression_last,
                    mapping[
                        "Benchmark Excess Return"
                    ],
                ],
                "values": [
                    worksheet.name,
                    regression_data_start + 2,
                    mapping[
                        "Portfolio Excess Return"
                    ],
                    regression_last,
                    mapping[
                        "Portfolio Excess Return"
                    ],
                ],
                "marker": {
                    "type":
                        "circle",
                    "size":
                        4,
                    "border": {
                        "color":
                            BRAND_BLUE,
                    },
                    "fill": {
                        "color":
                            LIGHT_BLUE,
                    },
                },
                "line": {
                    "none":
                        True,
                },
            }
        )

        scatter_chart.add_series(
            {
                "name":
                    "OLS Regression Line",
                "categories": [
                    worksheet.name,
                    regression_data_start + 2,
                    line_mapping[
                        "Benchmark Excess Return"
                    ],
                    line_last,
                    line_mapping[
                        "Benchmark Excess Return"
                    ],
                ],
                "values": [
                    worksheet.name,
                    regression_data_start + 2,
                    line_mapping[
                        "Fitted Portfolio Excess Return"
                    ],
                    line_last,
                    line_mapping[
                        "Fitted Portfolio Excess Return"
                    ],
                ],
                "marker": {
                    "type":
                        "none",
                },
                "line": {
                    "color":
                        BRAND_RED,
                    "width":
                        2.25,
                },
            }
        )

        scatter_chart.set_title(
            {
                "name":
                    "Portfolio vs. Benchmark Excess Returns",
                "name_font": {
                    "size":
                        14,
                    "bold":
                        True,
                    "color":
                        BRAND_NAVY,
                },
            }
        )

        scatter_chart.set_x_axis(
            {
                "name":
                    "Benchmark Excess Return (%)",
                "major_gridlines": {
                    "visible":
                        True,
                    "line": {
                        "color":
                            LIGHT_GREY,
                    },
                },
            }
        )

        scatter_chart.set_y_axis(
            {
                "name":
                    "Portfolio Excess Return (%)",
                "major_gridlines": {
                    "visible":
                        True,
                    "line": {
                        "color":
                            LIGHT_GREY,
                    },
                },
            }
        )

        scatter_chart.set_legend(
            {
                "position":
                    "bottom",
            }
        )

        scatter_chart.set_chartarea(
            {
                "border": {
                    "none":
                        True,
                },
            }
        )

        scatter_chart.set_plotarea(
            {
                "border": {
                    "color":
                        MID_GREY,
                },
                "fill": {
                    "color":
                        WHITE,
                },
            }
        )

        scatter_chart.set_size(
            {
                "width": CHART_WIDTH,
                "height": 320,
            }
        )

        worksheet.insert_chart(
            "A47",
            scatter_chart,
            {
                "x_offset": CHART_X_OFFSET,
                "y_offset": CHART_Y_OFFSET,
            },
        )

        residual_chart = workbook.add_chart(
            {
                "type":
                    "line",
            }
        )

        residual_chart.add_series(
            {
                "name":
                    "Residual",
                "categories": [
                    worksheet.name,
                    regression_data_start + 2,
                    mapping[
                        "Date"
                    ],
                    regression_last,
                    mapping[
                        "Date"
                    ],
                ],
                "values": [
                    worksheet.name,
                    regression_data_start + 2,
                    mapping[
                        "Residual"
                    ],
                    regression_last,
                    mapping[
                        "Residual"
                    ],
                ],
                "line": {
                    "color":
                        BRAND_GOLD,
                    "width":
                        1.5,
                },
            }
        )

        residual_chart.set_title(
            {
                "name":
                    "Regression Residuals Through Time",
            }
        )

        residual_chart.set_x_axis(
            {
                "name":
                    "Date",
                "date_axis":
                    True,
                "num_format":
                    "mmm-yy",
            }
        )

        residual_chart.set_y_axis(
            {
                "name":
                    "Residual (%)",
                "major_gridlines": {
                    "visible":
                        True,
                    "line": {
                        "color":
                            LIGHT_GREY,
                    },
                },
            }
        )

        residual_chart.set_legend(
            {
                "none":
                    True,
            }
        )

        residual_chart.set_size(
            {
                "width": CHART_WIDTH,
                "height": 320,
            }
        )

        worksheet.insert_chart(
            "J47",
            residual_chart,
            {
                "x_offset": CHART_X_OFFSET,
                "y_offset": CHART_Y_OFFSET,
            },
        )

    note_row = max(
        diagnostic_last,
        regression_last,
    ) + 3

    worksheet.merge_range(
        note_row,
        0,
        note_row + 4,
        15,
        (
            "OLS estimates the conditional linear relationship in the sample. "
            "Newey-West HAC standard errors make confidence intervals and "
            "p-values more robust to heteroscedasticity and limited serial "
            "correlation, but they do not correct an economically "
            "misspecified model. R-squared is descriptive, not causal, and "
            "all results depend on the benchmark, currency, risk-free proxy "
            "and historical period."
        ),
        formats["disclaimer"],
    )

    _add_source_comment(
        worksheet,
        9,
        0,
        (
            "EUR automatic risk-free proxy: ECB 3-month compounded €STR "
            "(EST.B.EU000A2QQF32.CR). USD automatic proxy: Federal Reserve / "
            "FRED 3-month Treasury constant maturity (DGS3MO)."
        ),
    )

# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================


def _build_portfolio_construction_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Portfolio Construction")
    results = context["results"]
    formats = context["formats"]
    currency = context["currency"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Portfolio Construction",
        (
            "Historical entry prices, ECB cross rates and theoretical "
            "fractional-share purchases"
        ),
        last_col=11,
    )

    construction = _as_dataframe(
        results.get("initial_portfolio_construction")
    )

    rename_columns = {
        "Entry Price (Portfolio Currency)": f"Entry Price ({currency})",
        "Amount Invested (Portfolio Currency)": f"Amount Invested ({currency})",
        "ECB Cross Rate (Portfolio per Local)": (
            f"ECB Cross Rate ({currency} per Local Currency)"
        ),
    }
    construction = construction.rename(columns=rename_columns)

    _write_section(context, worksheet, 4, "Initial Portfolio Construction", 0, 11)

    last_row, last_col, mapping = _write_dataframe(
        context,
        worksheet,
        construction,
        start_row=5,
        start_col=0,
        table_name="InitialPortfolioConstruction",
        hardcoded=True,
    )

    if mapping:
        amount_column_name = f"Amount Invested ({currency})"
        shares_column_name = "Fractional Shares Purchased"

        total_row = last_row + 2
        worksheet.write(total_row, 0, "Total / Portfolio", formats["label"])

        if amount_column_name in mapping:
            amount_col = mapping[amount_column_name]
            worksheet.write_formula(
                total_row,
                amount_col,
                f"=SUM({xlsxwriter.utility.xl_range(6, amount_col, last_row, amount_col)})",
                formats["money"],
                _safe_float(results.get("initial_investment")),
            )

        if shares_column_name in mapping:
            worksheet.write(total_row, mapping[shares_column_name], "—", formats["text"])

        worksheet.set_row(total_row, 20)

        if "Entry Price (Local)" in mapping:
            _add_source_comment(
                worksheet,
                5,
                mapping["Entry Price (Local)"],
                (
                    "Source: Yahoo Finance unadjusted historical closing "
                    "price on the entry date used. https://finance.yahoo.com/"
                ),
            )

        fx_column = f"ECB Cross Rate ({currency} per Local Currency)"
        if fx_column in mapping:
            _add_source_comment(
                worksheet,
                5,
                mapping[fx_column],
                (
                    "Source: European Central Bank daily reference exchange "
                    "rates. https://data.ecb.europa.eu/"
                ),
            )

    explanation_row = last_row + 5
    _write_section(context, worksheet, explanation_row, "Methodology", 0, 11)
    worksheet.merge_range(
        explanation_row + 1,
        0,
        explanation_row + 4,
        11,
        (
            "The app allocates the initial portfolio value according to the "
            "selected target weights. Each asset's first available "
            "unadjusted closing price on or after the requested start date "
            "is converted into the portfolio currency with an ECB daily "
            "reference cross rate. Fractional shares are then calculated as "
            "allocated amount divided by converted entry price. This is a "
            "theoretical reconstruction and may differ from broker execution "
            "prices, spreads and commissions."
        ),
        formats["text_wrap"],
    )


# ============================================================
# PERFORMANCE
# ============================================================


def _build_performance_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Performance")
    results = context["results"]
    formats = context["formats"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Performance Analysis",
        (
            "Portfolio value, cumulative performance, normalized asset "
            "prices and daily returns"
        ),
        last_col=13,
    )

    portfolio_value = _as_series(
        results.get("portfolio_value"),
        "Portfolio Value",
    )
    cumulative_returns = _as_series(
        results.get("cumulative_returns"),
        "Portfolio Cumulative Return (%)",
    )
    benchmark_comparison = _as_dataframe(
        results.get("benchmark_comparison")
    )

    performance_table = pd.concat(
        [
            portfolio_value,
            cumulative_returns,
            benchmark_comparison,
        ],
        axis=1,
    )
    performance_table = performance_table.loc[
        :, ~performance_table.columns.duplicated()
    ]
    performance_table.index.name = "Date"

    _write_section(context, worksheet, 4, "Portfolio and Benchmark Performance", 0, 4)
    performance_last_row, _, performance_mapping = _write_dataframe(
        context,
        worksheet,
        performance_table,
        start_row=5,
        start_col=0,
        table_name="PerformanceTable",
        hardcoded=True,
        column_formats={
            "Date": formats["date_input"],
            "Portfolio Value": formats["money_input"],
            "Portfolio Cumulative Return (%)": formats["pct_points_input"],
            "Benchmark Cumulative Return (%)": formats["pct_points_input"],
        },
    )

    if performance_mapping and performance_last_row > 6:
        date_col = performance_mapping["Date"]
        series_specs = []

        for column_name, color in [
            ("Portfolio Cumulative Return (%)", BRAND_BLUE),
            ("Benchmark Cumulative Return (%)", BRAND_GOLD),
        ]:
            if column_name in performance_mapping:
                series_specs.append(
                    {
                        "name": column_name,
                        "first_row": 6,
                        "last_row": performance_last_row,
                        "column": performance_mapping[column_name],
                        "color": color,
                    }
                )

        if series_specs:
            _add_line_chart(
                context,
                worksheet,
                "Portfolio vs. Benchmark Cumulative Return",
                (6, date_col, performance_last_row),
                series_specs,
                "G6",
                "N22",
                "Cumulative Return (%)",
                context["percentage_points_format"],
            )

        if "Portfolio Value" in performance_mapping:
            _add_line_chart(
                context,
                worksheet,
                "Portfolio Value Over Time",
                (6, date_col, performance_last_row),
                [
                    {
                        "name": "Portfolio Value",
                        "first_row": 6,
                        "last_row": performance_last_row,
                        "column": performance_mapping["Portfolio Value"],
                        "color": BRAND_GREEN,
                    }
                ],
                "G25",
                "N41",
                f"Portfolio Value ({context['currency']})",
                context["currency_number_format"],
            )

    prices = _as_dataframe(results.get("prices"))
    normalized_prices = pd.DataFrame()

    if not prices.empty:
        normalized_prices = prices.div(prices.iloc[0]).mul(100)
        normalized_prices.index.name = "Date"
        normalized_prices.columns = [
            f"{column} (Base 100)"
            for column in normalized_prices.columns
        ]

    normalized_start = performance_last_row + 3
    _write_section(
        context,
        worksheet,
        normalized_start,
        "Normalized Asset Prices",
        0,
        max(4, min(13, len(normalized_prices.columns))),
    )
    normalized_last_row, _, normalized_mapping = _write_dataframe(
        context,
        worksheet,
        normalized_prices,
        start_row=normalized_start + 1,
        start_col=0,
        table_name="NormalizedPrices",
        hardcoded=True,
        column_formats={
            "Date": formats["date_input"],
        },
    )

    if normalized_mapping and normalized_last_row > normalized_start + 2:
        normalized_date_col = normalized_mapping["Date"]
        series_specs = []
        for index, column_name in enumerate(normalized_prices.columns):
            series_specs.append(
                {
                    "name": column_name.replace(" (Base 100)", ""),
                    "first_row": normalized_start + 2,
                    "last_row": normalized_last_row,
                    "column": normalized_mapping[column_name],
                    "color": CHART_COLORS[index % len(CHART_COLORS)],
                }
            )

        _add_line_chart(
            context,
            worksheet,
            "Normalized Asset Price Performance",
            (normalized_start + 2, normalized_date_col, normalized_last_row),
            series_specs,
            "G44",
            "N60",
            "Normalized Price (Base 100)",
            "0.00",
        )

    daily_returns = _as_dataframe(results.get("daily_returns"))
    portfolio_returns = _as_series(
        results.get("portfolio_returns"),
        "Portfolio Return",
    )
    benchmark_returns = _as_series(
        results.get("benchmark_returns"),
        "Benchmark Return",
    )

    returns_table = pd.concat(
        [daily_returns, portfolio_returns, benchmark_returns],
        axis=1,
    ).dropna(how="all")
    returns_table.index.name = "Date"

    returns_start = normalized_last_row + 3
    _write_section(
        context,
        worksheet,
        returns_start,
        "Daily Returns",
        0,
        max(4, min(13, len(returns_table.columns))),
    )
    _write_dataframe(
        context,
        worksheet,
        returns_table,
        start_row=returns_start + 1,
        start_col=0,
        table_name="DailyReturns",
        hardcoded=True,
        column_formats={
            "Date": formats["date_input"],
            **{
                column: formats["pct_decimal"]
                for column in returns_table.columns
            },
        },
    )

    _add_source_comment(
        worksheet,
        5,
        0,
        (
            "Converted adjusted price and return data originate from Yahoo "
            "Finance, with ECB historical exchange-rate conversion for "
            "assets whose trading currency differs from the portfolio "
            "currency."
        ),
    )


# ============================================================
# RISK ANALYSIS
# ============================================================


def _build_risk_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Risk Analysis")
    results = context["results"]
    formats = context["formats"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Risk Analysis",
        "Volatility, drawdown, VaR, Expected Shortfall and return distribution",
        last_col=14,
    )

    _write_section(context, worksheet, 4, "Risk Summary", 0, 5)

    risk_metrics = [
        ("Annualized Volatility (%)", results.get("annualized_volatility"), formats["pct_points_input"]),
        ("Maximum Drawdown (%)", results.get("maximum_drawdown"), formats["pct_points_input"]),
        ("Sharpe Ratio", results.get("sharpe_ratio"), formats["decimal_input"]),
        ("Confidence Level (%)", results.get("confidence_level"), formats["pct_points_input"]),
        ("Historical VaR", results.get("historical_var_money"), formats["money_input"]),
        ("Historical Expected Shortfall", results.get("historical_es_money"), formats["money_input"]),
        ("Parametric VaR", results.get("parametric_var_money"), formats["money_input"]),
        ("Parametric Expected Shortfall", results.get("parametric_es_money"), formats["money_input"]),
    ]

    for index, (label, value, value_format) in enumerate(risk_metrics):
        row = 5 + index // 2
        col = (index % 2) * 3
        worksheet.write(row, col, label, formats["label"])
        worksheet.merge_range(row, col + 1, row, col + 2, "", value_format)
        _write_value(worksheet, row, col + 1, value, value_format)

    risk_comparison = pd.DataFrame(
        {
            "Measure": [
                "Historical VaR",
                "Parametric VaR",
                "Historical ES",
                "Parametric ES",
            ],
            "Estimated Loss": [
                results.get("historical_var_money"),
                results.get("parametric_var_money"),
                results.get("historical_es_money"),
                results.get("parametric_es_money"),
            ],
        }
    )

    risk_table_last, _, risk_mapping = _write_dataframe(
        context,
        worksheet,
        risk_comparison,
        start_row=5,
        start_col=7,
        table_name="RiskMeasureComparison",
        hardcoded=True,
        column_formats={
            "Measure": formats["input_text"],
            "Estimated Loss": formats["money_input"],
        },
    )

    if risk_mapping:
        _add_column_chart(
            context,
            worksheet,
            "VaR and Expected Shortfall Comparison",
            (6, risk_mapping["Measure"], risk_table_last),
            [
                {
                    "name": "Estimated Loss",
                    "first_row": 6,
                    "last_row": risk_table_last,
                    "column": risk_mapping["Estimated Loss"],
                    "color": BRAND_RED,
                    "data_labels": {"value": True, "num_format": context["currency_number_format"]},
                }
            ],
            "J6",
            f"Estimated Loss ({context['currency']})",
            context["currency_number_format"],
        )

    portfolio_returns = _as_series(
        results.get("portfolio_returns"),
        "Portfolio Return",
    )
    portfolio_value = _as_series(
        results.get("portfolio_value"),
        "Portfolio Value",
    )
    rolling_volatility = _as_series(
        results.get("rolling_volatility"),
        f"{context['rolling_window']}-Day Rolling Volatility (%)",
    )

    if not portfolio_value.empty:
        wealth_index = portfolio_value / portfolio_value.iloc[0]
        drawdown = (
            wealth_index / wealth_index.cummax() - 1
        ) * 100
        drawdown.name = "Drawdown (%)"
    else:
        drawdown = pd.Series(dtype=float, name="Drawdown (%)")

    risk_time_series = pd.concat(
        [rolling_volatility, drawdown],
        axis=1,
    )
    risk_time_series.index.name = "Date"

    time_start = max(risk_table_last + 4, 13)
    _write_section(context, worksheet, time_start, "Risk Through Time", 0, 4)
    time_last, _, time_mapping = _write_dataframe(
        context,
        worksheet,
        risk_time_series,
        start_row=time_start + 1,
        start_col=0,
        table_name="RiskTimeSeries",
        hardcoded=True,
        column_formats={
            "Date": formats["date_input"],
            f"{context['rolling_window']}-Day Rolling Volatility (%)": formats["pct_points_input"],
            "Drawdown (%)": formats["pct_points_input"],
        },
    )

    if time_mapping and time_last > time_start + 2:
        date_col = time_mapping["Date"]
        rolling_name = f"{context['rolling_window']}-Day Rolling Volatility (%)"

        if rolling_name in time_mapping:
            _add_line_chart(
                context,
                worksheet,
                f"{context['rolling_window']}-Day Rolling Volatility",
                (time_start + 2, date_col, time_last),
                [
                    {
                        "name": rolling_name,
                        "first_row": time_start + 2,
                        "last_row": time_last,
                        "column": time_mapping[rolling_name],
                        "color": BRAND_BLUE,
                    }
                ],
                "J25",
                "Q41",
                "Annualized Volatility (%)",
                context["percentage_points_format"],
            )

        if "Drawdown (%)" in time_mapping:
            _add_line_chart(
                context,
                worksheet,
                "Portfolio Drawdown",
                (time_start + 2, date_col, time_last),
                [
                    {
                        "name": "Drawdown (%)",
                        "first_row": time_start + 2,
                        "last_row": time_last,
                        "column": time_mapping["Drawdown (%)"],
                        "color": BRAND_RED,
                    }
                ],
                "J44",
                "Q60",
                "Drawdown (%)",
                context["percentage_points_format"],
            )

    # Histogram data.
    clean_returns = portfolio_returns.dropna().to_numpy(dtype=float)

    if clean_returns.size >= 2:
        number_of_bins = min(30, max(10, int(np.sqrt(clean_returns.size))))
        frequencies, bin_edges = np.histogram(clean_returns, bins=number_of_bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        histogram = pd.DataFrame(
            {
                "Daily Return Bin": bin_centres,
                "Frequency": frequencies,
            }
        )
    else:
        histogram = pd.DataFrame(
            {"Daily Return Bin": [], "Frequency": []}
        )

    histogram_start = time_last + 3
    _write_section(context, worksheet, histogram_start, "Return Distribution", 0, 5)
    hist_last, _, hist_mapping = _write_dataframe(
        context,
        worksheet,
        histogram,
        start_row=histogram_start + 1,
        start_col=0,
        table_name="ReturnHistogram",
        hardcoded=True,
        column_formats={
            "Daily Return Bin": formats["pct_decimal"],
            "Frequency": formats["integer"],
        },
    )

    if hist_mapping and hist_last > histogram_start + 2:
        _add_column_chart(
            context,
            worksheet,
            "Daily Portfolio Return Distribution",
            (histogram_start + 2, hist_mapping["Daily Return Bin"], hist_last),
            [
                {
                    "name": "Frequency",
                    "first_row": histogram_start + 2,
                    "last_row": hist_last,
                    "column": hist_mapping["Frequency"],
                    "color": BRAND_SKY,
                }
            ],
            "J63",
            "Frequency",
            "0",
        )

    worksheet.merge_range(
        histogram_start + 1,
        7,
        histogram_start + 4,
        14,
        (
            "VaR estimates a loss threshold at the selected confidence "
            "level. Expected Shortfall estimates the average loss beyond "
            "that threshold. Historical methods use observed portfolio "
            "returns; parametric methods use a normal-distribution model. "
            "Neither measure represents the maximum possible loss."
        ),
        formats["text_wrap"],
    )


# ============================================================
# ALLOCATION
# ============================================================


def _build_allocation_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Allocation")
    results = context["results"]
    formats = context["formats"]
    currency = context["currency"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Allocation and Contributions",
        "Portfolio weights, position values and contributions to return and risk",
        last_col=13,
    )

    allocation = _as_dataframe(results.get("allocation_table"))
    allocation = allocation.rename(
        columns={
            "Initial Invested Value": f"Initial Invested Value ({currency})",
            "Current Estimated Value": f"Current Estimated Value ({currency})",
        }
    )

    _write_section(context, worksheet, 4, "Portfolio Allocation", 0, 4)
    allocation_last, _, allocation_mapping = _write_dataframe(
        context,
        worksheet,
        allocation,
        start_row=5,
        start_col=0,
        table_name="PortfolioAllocation",
        hardcoded=True,
    )

    if allocation_mapping and allocation_last > 6:
        chart = workbook.add_chart({"type": "doughnut"})
        asset_col = allocation_mapping["Asset"]
        weight_col = allocation_mapping["Weight (%)"]
        chart.add_series(
            {
                "name": "Portfolio Allocation",
                "categories": [worksheet.name, 6, asset_col, allocation_last, asset_col],
                "values": [worksheet.name, 6, weight_col, allocation_last, weight_col],
                "points": [
                    {"fill": {"color": CHART_COLORS[index % len(CHART_COLORS)]}}
                    for index in range(len(allocation))
                ],
                "data_labels": {
                    "category": True,
                    "percentage": True,
                    "leader_lines": True,
                },
            }
        )
        chart.set_title(
            {
                "name": "Portfolio Allocation",
                "name_font": {"size": 14, "bold": True, "color": BRAND_NAVY},
            }
        )
        chart.set_hole_size(55)
        chart.set_legend({"position": "bottom"})
        chart.set_chartarea({"border": {"none": True}})
        chart.set_size(
            {
                "width": CHART_WIDTH,
                "height": CHART_HEIGHT,
            }
        )
        worksheet.insert_chart(
            "I6",
            chart,
            {
                "x_offset": CHART_X_OFFSET,
                "y_offset": CHART_Y_OFFSET,
            },
        )

    contribution = _as_dataframe(results.get("contribution_table"))
    contribution_start = allocation_last + 4
    _write_section(context, worksheet, contribution_start, "Return and Risk Contributions", 0, 7)
    contribution_last, _, contribution_mapping = _write_dataframe(
        context,
        worksheet,
        contribution,
        start_row=contribution_start + 1,
        start_col=0,
        table_name="PortfolioContributions",
        hardcoded=True,
    )

    if contribution_mapping and contribution_last > contribution_start + 2:
        asset_col = contribution_mapping["Asset"]

        _add_column_chart(
            context,
            worksheet,
            "Contribution to Annualized Return",
            (contribution_start + 2, asset_col, contribution_last),
            [
                {
                    "name": "Return Contribution (p.p.)",
                    "first_row": contribution_start + 2,
                    "last_row": contribution_last,
                    "column": contribution_mapping["Return Contribution (p.p.)"],
                    "color": BRAND_GREEN,
                    "data_labels": {"value": True, "num_format": '0.00"%"'},
                }
            ],
            "I26",
            "Return Contribution (p.p.)",
            context["percentage_points_format"],
        )

        _add_column_chart(
            context,
            worksheet,
            "Share of Portfolio Risk",
            (contribution_start + 2, asset_col, contribution_last),
            [
                {
                    "name": "Risk Contribution (%)",
                    "first_row": contribution_start + 2,
                    "last_row": contribution_last,
                    "column": contribution_mapping["Risk Contribution (%)"],
                    "color": BRAND_RED,
                    "data_labels": {"value": True, "num_format": '0.00"%"'},
                }
            ],
            "I46",
            "Risk Contribution (%)",
            context["percentage_points_format"],
        )

    total_row = contribution_last + 2
    worksheet.write(total_row, 0, "Portfolio Total", formats["label"])

    if "Weight (%)" in contribution_mapping:
        weight_col = contribution_mapping["Weight (%)"]
        worksheet.write_formula(
            total_row,
            weight_col,
            f"=SUM({xlsxwriter.utility.xl_range(contribution_start + 2, weight_col, contribution_last, weight_col)})",
            formats["pct_points"],
            100.0,
        )

    if "Return Contribution (p.p.)" in contribution_mapping:
        return_col = contribution_mapping["Return Contribution (p.p.)"]
        worksheet.write_formula(
            total_row,
            return_col,
            f"=SUM({xlsxwriter.utility.xl_range(contribution_start + 2, return_col, contribution_last, return_col)})",
            formats["pct_points"],
            _safe_float(results.get("annualized_portfolio_return")),
        )

    if "Risk Contribution (%)" in contribution_mapping:
        risk_col = contribution_mapping["Risk Contribution (%)"]
        worksheet.write_formula(
            total_row,
            risk_col,
            f"=SUM({xlsxwriter.utility.xl_range(contribution_start + 2, risk_col, contribution_last, risk_col)})",
            formats["pct_points"],
            100.0,
        )


# ============================================================
# DIVERSIFICATION
# ============================================================


def _build_diversification_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Diversification")
    results = context["results"]
    formats = context["formats"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Diversification Analysis",
        "Asset correlations, benchmark relationships and diversification diagnostics",
        last_col=13,
    )

    correlation_matrix = _as_dataframe(
        results.get("asset_correlation_matrix")
    )
    correlation_matrix.index.name = "Asset"

    _write_section(context, worksheet, 4, "Asset Correlation Matrix", 0, max(4, len(correlation_matrix.columns)))
    matrix_last, matrix_last_col, matrix_mapping = _write_dataframe(
        context,
        worksheet,
        correlation_matrix,
        start_row=5,
        start_col=0,
        table_name="AssetCorrelationMatrix",
        hardcoded=True,
        add_table=False,
        column_formats={
            "Asset": formats["input_text"],
            **{
                column: formats["decimal_input"]
                for column in correlation_matrix.columns
            },
        },
    )

    if matrix_mapping and matrix_last > 6:
        first_numeric_col = 1
        worksheet.conditional_format(
            6,
            first_numeric_col,
            matrix_last,
            matrix_last_col,
            {
                "type": "3_color_scale",
                "min_type": "num",
                "min_value": -1,
                "min_color": "#D73027",
                "mid_type": "num",
                "mid_value": 0,
                "mid_color": "#FFF7BC",
                "max_type": "num",
                "max_value": 1,
                "max_color": "#1A9850",
            },
        )

    diversification_summary = _as_dataframe(
        results.get("diversification_summary")
    )

    summary_start = matrix_last + 4
    _write_section(context, worksheet, summary_start, "Diversification Summary", 0, 5)
    summary_last, _, summary_mapping = _write_dataframe(
        context,
        worksheet,
        diversification_summary,
        start_row=summary_start + 1,
        start_col=0,
        table_name="DiversificationSummary",
        hardcoded=True,
    )

    diagnostics_col = 7
    worksheet.merge_range(
        summary_start,
        diagnostics_col,
        summary_start,
        diagnostics_col + 4,
        "Portfolio Diagnostics",
        formats["section"],
    )

    diagnostics = [
        ("Average Portfolio Correlation", results.get("average_portfolio_correlation"), formats["decimal_input"]),
        ("Portfolio / Benchmark Correlation", results.get("portfolio_benchmark_correlation"), formats["decimal_input"]),
        ("Most Diversifying Asset", results.get("most_diversifying_asset"), formats["input_text"]),
        ("Most Correlated Asset", results.get("most_correlated_asset"), formats["input_text"]),
    ]

    for index, (label, value, value_format) in enumerate(diagnostics, start=1):
        row = summary_start + index
        worksheet.write(row, diagnostics_col, label, formats["label"])
        worksheet.merge_range(
            row,
            diagnostics_col + 1,
            row,
            diagnostics_col + 4,
            "",
            value_format,
        )
        _write_value(
            worksheet,
            row,
            diagnostics_col + 1,
            value,
            value_format,
        )

    if summary_mapping and summary_last > summary_start + 2:
        asset_col = summary_mapping["Asset"]
        series_specs = []

        for index, column in enumerate(diversification_summary.columns):
            if column == "Asset":
                continue
            series_specs.append(
                {
                    "name": column,
                    "first_row": summary_start + 2,
                    "last_row": summary_last,
                    "column": summary_mapping[column],
                    "color": CHART_COLORS[index % len(CHART_COLORS)],
                    "data_labels": {"value": True, "num_format": "0.00"},
                }
            )

        if series_specs:
            _add_column_chart(
                context,
                worksheet,
                "Asset Correlation Diagnostics",
                (summary_start + 2, asset_col, summary_last),
                series_specs,
                f"I{summary_start + 7}",
                "Correlation",
                "0.00",
            )

    worksheet.merge_range(
        summary_last + 3,
        0,
        summary_last + 6,
        13,
        (
            "Correlation ranges from -1 to +1 and measures historical linear "
            "co-movement. Lower correlations can improve diversification, "
            "but correlations are unstable and frequently rise during market "
            "stress. Holding many assets does not guarantee diversification "
            "when they share the same sectors, regions or risk factors."
        ),
        formats["text_wrap"],
    )


# ============================================================
# STRESS TESTS
# ============================================================


def _build_stress_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Stress Tests")
    results = context["results"]
    formats = context["formats"]
    currency = context["currency"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Portfolio Stress Tests",
        "Beta-based market correction and custom asset-by-asset scenario",
        last_col=14,
    )

    stress_summary = _as_dataframe(results.get("stress_test_summary"))
    stress_summary = stress_summary.rename(
        columns={
            "Portfolio Change": f"Portfolio Change ({currency})",
            "Stressed Portfolio Value": f"Stressed Portfolio Value ({currency})",
        }
    )

    _write_section(context, worksheet, 4, "Scenario Comparison", 0, 8)
    summary_last, _, summary_mapping = _write_dataframe(
        context,
        worksheet,
        stress_summary,
        start_row=5,
        start_col=0,
        table_name="StressScenarioSummary",
        hardcoded=True,
    )

    if summary_mapping and summary_last > 6:
        _add_column_chart(
            context,
            worksheet,
            "Estimated Portfolio Impact by Scenario",
            (6, summary_mapping["Scenario"], summary_last),
            [
                {
                    "name": "Portfolio Change (%)",
                    "first_row": 6,
                    "last_row": summary_last,
                    "column": summary_mapping["Portfolio Change (%)"],
                    "color": BRAND_RED,
                    "data_labels": {"value": True, "num_format": '0.00"%"'},
                }
            ],
            "J6",
            "Portfolio Change (%)",
            context["percentage_points_format"],
        )

    market_detail = _as_dataframe(results.get("market_stress_detail"))
    custom_detail = _as_dataframe(results.get("custom_stress_detail"))

    detail_rename = {
        "Current Position Value": f"Current Position Value ({currency})",
        "Value Change": f"Value Change ({currency})",
        "Stressed Position Value": f"Stressed Position Value ({currency})",
    }

    market_detail = market_detail.rename(columns=detail_rename)
    custom_detail = custom_detail.rename(columns=detail_rename)

    market_start = summary_last + 4
    _write_section(context, worksheet, market_start, "Beta-Based Market Correction", 0, 9)
    market_last, _, market_mapping = _write_dataframe(
        context,
        worksheet,
        market_detail,
        start_row=market_start + 1,
        start_col=0,
        table_name="MarketStressDetail",
        hardcoded=True,
    )

    if market_mapping and market_last > market_start + 2:
        _add_column_chart(
            context,
            worksheet,
            "Market Correction — Value Change by Asset",
            (market_start + 2, market_mapping["Asset"], market_last),
            [
                {
                    "name": f"Value Change ({currency})",
                    "first_row": market_start + 2,
                    "last_row": market_last,
                    "column": market_mapping[f"Value Change ({currency})"],
                    "color": BRAND_RED,
                    "data_labels": {"value": True, "num_format": context["currency_number_format"]},
                }
            ],
            "J26",
            f"Value Change ({currency})",
            context["currency_number_format"],
        )

    custom_start = market_last + 4
    _write_section(context, worksheet, custom_start, "Custom Asset-by-Asset Scenario", 0, 9)
    custom_last, _, custom_mapping = _write_dataframe(
        context,
        worksheet,
        custom_detail,
        start_row=custom_start + 1,
        start_col=0,
        table_name="CustomStressDetail",
        hardcoded=True,
    )

    if custom_mapping and custom_last > custom_start + 2:
        _add_column_chart(
            context,
            worksheet,
            "Custom Scenario — Value Change by Asset",
            (custom_start + 2, custom_mapping["Asset"], custom_last),
            [
                {
                    "name": f"Value Change ({currency})",
                    "first_row": custom_start + 2,
                    "last_row": custom_last,
                    "column": custom_mapping[f"Value Change ({currency})"],
                    "color": BRAND_GOLD,
                    "data_labels": {"value": True, "num_format": context["currency_number_format"]},
                }
            ],
            "J46",
            f"Value Change ({currency})",
            context["currency_number_format"],
        )

    explanation_row = custom_last + 4
    worksheet.merge_range(
        explanation_row,
        0,
        explanation_row + 3,
        14,
        (
            "Stress testing estimates the impact of deliberately severe "
            "hypothetical shocks. The beta-based scenario applies each "
            "asset's historical benchmark sensitivity to the selected market "
            "shock. The custom scenario applies the user's asset-specific "
            "shocks. These scenarios are not probability forecasts and do "
            "not model liquidity, changing correlations, taxes, transaction "
            "costs or dynamic rebalancing."
        ),
        formats["disclaimer"],
    )


# ============================================================
# EDUCATIONAL GUIDE
# ============================================================


def _build_educational_guide_sheet(context: dict[str, Any]) -> None:
    workbook = context["workbook"]
    worksheet = workbook.add_worksheet("Educational Guide")
    formats = context["formats"]

    _configure_sheet(worksheet)
    _write_sheet_header(
        context,
        worksheet,
        "Educational Guide",
        "Simple intuition, professional formulas, interpretation and limitations",
        last_col=7,
    )

    guide = pd.DataFrame(
        [
            {
                "Metric": "Cumulative Return",
                "Simple Meaning": "Total gain or loss over the selected period.",
                "Simple Formula": "(Final Value ÷ Initial Value − 1) × 100",
                "Professional Formula": "R_cum = (V_T / V_0 − 1) × 100",
                "How to Interpret": "Positive values indicate growth; negative values indicate a decline.",
                "Common Mistake": "Treating cumulative return as an annual return.",
                "Limitation": "Depends heavily on the selected start and end dates.",
            },
            {
                "Metric": "Annualized Volatility",
                "Simple Meaning": "How widely portfolio returns fluctuated.",
                "Simple Formula": "Daily return volatility × √252",
                "Professional Formula": "σ_annual = s_daily × √252",
                "How to Interpret": "Higher values indicate larger historical return movements.",
                "Common Mistake": "Assuming volatility is the same as loss.",
                "Limitation": "Treats upside and downside movements equally and is backward-looking.",
            },
            {
                "Metric": "Sharpe Ratio",
                "Simple Meaning": "Excess return earned per unit of volatility.",
                "Simple Formula": "Return above risk-free rate ÷ Portfolio risk",
                "Professional Formula": "SR_p = [E(R_p) − R_f] / σ_p",
                "How to Interpret": "Higher positive values indicate stronger historical risk-adjusted performance.",
                "Common Mistake": "Believing a high Sharpe Ratio eliminates crash risk.",
                "Limitation": "Can be distorted by non-normal returns and short samples.",
            },
            {
                "Metric": "Maximum Drawdown",
                "Simple Meaning": "Deepest fall from a previous portfolio peak.",
                "Simple Formula": "Current value ÷ Previous peak − 1",
                "Professional Formula": "MDD = min_t[V_t / max_(s≤t)(V_s) − 1]",
                "How to Interpret": "A more negative value indicates a more severe historical decline.",
                "Common Mistake": "Confusing drawdown with the total start-to-end return.",
                "Limitation": "Does not show how long recovery took.",
            },
            {
                "Metric": "Value at Risk",
                "Simple Meaning": "Estimated loss threshold at a chosen confidence level.",
                "Simple Formula": "Historical quantile or normal-distribution threshold",
                "Professional Formula": "VaR_c = −Q_(1−c)(R_p)",
                "How to Interpret": "Losses may exceed VaR in the remaining tail probability.",
                "Common Mistake": "Treating VaR as the maximum possible loss.",
                "Limitation": "Does not describe the severity of losses beyond the threshold.",
            },
            {
                "Metric": "Expected Shortfall",
                "Simple Meaning": "Average loss in the tail beyond VaR.",
                "Simple Formula": "Average of outcomes worse than VaR",
                "Professional Formula": "ES_c = −E[R_p | R_p ≤ Q_(1−c)(R_p)]",
                "How to Interpret": "Higher ES means more severe estimated tail losses.",
                "Common Mistake": "Adding ES to VaR as if they were separate losses.",
                "Limitation": "Depends on sample history or distributional assumptions.",
            },
            {
                "Metric": "Beta",
                "Simple Meaning": "Historical sensitivity to benchmark movements.",
                "Simple Formula": "Portfolio–benchmark co-movement ÷ Benchmark risk",
                "Professional Formula": "β_p = Cov(R_p,R_m) / Var(R_m)",
                "How to Interpret": "Beta above 1 indicates greater historical benchmark sensitivity.",
                "Common Mistake": "Treating beta as a forecast or measure of total risk.",
                "Limitation": "Benchmark sensitivity can change across regimes.",
            },
            {
                "Metric": "Alpha",
                "Simple Meaning": "Return not explained by benchmark exposure.",
                "Simple Formula": "Actual excess return − Beta-explained return",
                "Professional Formula": "α_p = E(R_p−R_f) − β_pE(R_m−R_f)",
                "How to Interpret": "Positive alpha indicates historical outperformance relative to the model.",
                "Common Mistake": "Assuming positive alpha proves skill.",
                "Limitation": "Highly dependent on benchmark and model selection.",
            },
            {
                "Metric": "Correlation",
                "Simple Meaning": "Strength and direction of linear co-movement.",
                "Simple Formula": "Shared movement ÷ Combined volatility",
                "Professional Formula": "ρ_ij = Cov(R_i,R_j) / (σ_iσ_j)",
                "How to Interpret": "Values near +1 move together; near −1 move oppositely.",
                "Common Mistake": "Confusing correlation with beta.",
                "Limitation": "Correlations often increase during crises.",
            },
            {
                "Metric": "Portfolio Variance",
                "Simple Meaning": "Total risk after weights and asset interactions.",
                "Simple Formula": "Weights + Volatilities + Correlations",
                "Professional Formula": "σ_p² = wᵀΣw",
                "How to Interpret": "Diversification reduces risk when assets do not move perfectly together.",
                "Common Mistake": "Using a weighted average of standalone volatilities.",
                "Limitation": "Covariance estimates are historically unstable.",
            },
            {
                "Metric": "Risk Contribution",
                "Simple Meaning": "How much each position contributes to portfolio volatility.",
                "Simple Formula": "Weight × Marginal impact on portfolio risk",
                "Professional Formula": "CRC_i = w_i(Σw)_i / σ_p",
                "How to Interpret": "A small weight can still dominate risk if highly volatile and correlated.",
                "Common Mistake": "Assuming portfolio weight equals risk contribution.",
                "Limitation": "Depends on historical covariance estimates.",
            },
            {
                "Metric": "FX Conversion",
                "Simple Meaning": "Converts all assets into one portfolio currency.",
                "Simple Formula": "Converted Price = Local Price × Cross Rate",
                "Professional Formula": "P_i^B = P_i^L × X_(B/L)",
                "How to Interpret": "Foreign-asset returns include both asset and currency movements.",
                "Common Mistake": "Changing only the currency symbol without converting prices through time.",
                "Limitation": "ECB reference rates can differ from broker execution rates.",
            },
            {
                "Metric": "Stress Testing",
                "Simple Meaning": "Hypothetical portfolio impact under severe shocks.",
                "Simple Formula": "Position impact = Position value × Asset shock",
                "Professional Formula": "ΔV_p = Σ_i(V_i × s_i)",
                "How to Interpret": "Shows vulnerability under chosen assumptions, not probability.",
                "Common Mistake": "Treating the scenario as a forecast.",
                "Limitation": "Does not model liquidity, contagion or changing correlations.",
            },
            {
                "Metric": "Data Quality",
                "Simple Meaning": "Documents which observations enter the analysis.",
                "Simple Formula": "Common dates = Dates valid for every selected asset",
                "Professional Formula": "T_common = ⋂_(i=1)^N T_i",
                "How to Interpret": "Higher retention means less data was lost during strict alignment.",
                "Common Mistake": "Filling missing stock prices and treating invented values as observed prices.",
                "Limitation": "Strict alignment can shorten samples when markets have different calendars.",
            },
            {
                "Metric": "Modified Z-Score",
                "Simple Meaning": "Robust flag for unusual return observations.",
                "Simple Formula": "Distance from median ÷ Median absolute deviation",
                "Professional Formula": "M_t = 0.67448975(r_t − median(r)) / MAD(r)",
                "How to Interpret": "Large absolute values identify observations that deserve review.",
                "Common Mistake": "Automatically deleting every flagged observation.",
                "Limitation": "A genuine market event can appear statistically unusual.",
            },
            {
                "Metric": "Benchmark Regression",
                "Simple Meaning": "Separates benchmark-linked return from residual return.",
                "Simple Formula": "Portfolio excess return = Alpha + Beta × Benchmark excess return + Residual",
                "Professional Formula": "R_p,t − R_f,t = α + β(R_m,t − R_f,t) + ε_t",
                "How to Interpret": "Beta measures sensitivity, alpha is the intercept and R² measures explained sample variation.",
                "Common Mistake": "Interpreting regression association as causality.",
                "Limitation": "A one-factor model can omit important systematic exposures.",
            },
            {
                "Metric": "Newey-West HAC",
                "Simple Meaning": "More robust uncertainty estimates for regression coefficients.",
                "Simple Formula": "OLS coefficients + HAC covariance matrix",
                "Professional Formula": "Var_HAC(θ̂) = (X'X)⁻¹ S_HAC (X'X)⁻¹",
                "How to Interpret": "Used for robust standard errors, confidence intervals and p-values.",
                "Common Mistake": "Believing robust standard errors repair a misspecified model.",
                "Limitation": "Results remain dependent on lag selection and sample size.",
            },
        ]
    )

    _write_section(context, worksheet, 4, "Metric Reference", 0, 7)
    guide_last, _, mapping = _write_dataframe(
        context,
        worksheet,
        guide,
        start_row=5,
        start_col=0,
        table_name="EducationalGuide",
        hardcoded=False,
        column_formats={
            "Metric": formats["label"],
            "Simple Meaning": formats["text_wrap"],
            "Simple Formula": formats["formula_text"],
            "Professional Formula": formats["formula_text"],
            "How to Interpret": formats["text_wrap"],
            "Common Mistake": formats["text_wrap"],
            "Limitation": formats["text_wrap"],
        },
    )

    worksheet.set_column("A:A", 25)
    worksheet.set_column("B:B", 35)
    worksheet.set_column("C:D", 40)
    worksheet.set_column("E:G", 42)

    for row in range(6, guide_last + 1):
        worksheet.set_row(row, 58)

    worksheet.merge_range(
        guide_last + 3,
        0,
        guide_last + 5,
        7,
        (
            "The guide provides educational interpretations rather than "
            "universal investment rules. Metric thresholds vary across asset "
            "classes, market regimes, horizons and investor objectives."
        ),
        formats["disclaimer"],
    )
