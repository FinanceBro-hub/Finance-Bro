from pathlib import Path
import base64

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from portfolio_analysis import analyze_portfolio
from excel_report import create_excel_report
from learning_centre import render_learning_centre
from stock_research import render_stock_research
from live_investment import render_live_investment


# ============================================================
# FINANCE BRO — BRAND SYSTEM
# ============================================================

BRAND_NAVY = "#0A1F44"
BRAND_BLUE = "#1663F0"
BRAND_GREEN = "#2DB24A"
BRAND_WHITE = "#FFFFFF"
BRAND_BACKGROUND = "#F5F8FC"
BRAND_BORDER = "#DDE7F5"
BRAND_MUTED = "#64748B"

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

APP_ICON_PATH = ASSETS_DIR / "finance_bro_app_icon.png"
PRIMARY_LOGO_PATH = ASSETS_DIR / "finance_bro_primary_logo.png"
HORIZONTAL_LOGO_PATH = ASSETS_DIR / "finance_bro_horizontal_logo.png"


def load_base64_image(image_path: Path) -> str:
    """Loads a local image as a base64 string for branded HTML blocks."""

    if not image_path.exists():
        return ""

    return base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")


app_icon_image = (
    Image.open(APP_ICON_PATH)
    if APP_ICON_PATH.exists()
    else "📈"
)

app_icon_base64 = load_base64_image(
    APP_ICON_PATH
)

primary_logo_base64 = load_base64_image(
    PRIMARY_LOGO_PATH
)


def render_html(html_content: str) -> None:
    """
    Renderiza HTML sem deixar a indentação do Python ser interpretada
    pelo Markdown como um bloco de código.
    """

    cleaned_html = "\n".join(
        line.strip()
        for line in html_content.splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Finance Bro | Portfolio Intelligence",
    page_icon=app_icon_image,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL BRAND STYLES
# ============================================================

render_html(
    """
    <style>
        :root {
            --fb-navy: #0A1F44;
            --fb-blue: #1663F0;
            --fb-green: #2DB24A;
            --fb-white: #FFFFFF;
            --fb-background: #F5F8FC;
            --fb-surface: #FFFFFF;
            --fb-border: #DDE7F5;
            --fb-muted: #64748B;
            --fb-soft-blue: #EEF4FF;
            --fb-soft-green: #ECF8EF;
            --fb-red: #D92D20;
            --fb-shadow: 0 14px 34px rgba(10, 31, 68, 0.09);
            --fb-shadow-soft: 0 8px 22px rgba(10, 31, 68, 0.07);
        }

        html,
        body,
        [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 90% 2%,
                    rgba(22, 99, 240, 0.08),
                    transparent 26rem
                ),
                linear-gradient(
                    180deg,
                    #F7F9FD 0%,
                    #FFFFFF 38%,
                    #F7F9FD 100%
                );
            color: var(--fb-navy);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 249, 253, 0.82);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(221, 231, 245, 0.72);
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1540px;
            padding-top: 1.55rem;
            padding-bottom: 3.5rem;
        }

        h1,
        h2,
        h3,
        h4 {
            color: var(--fb-navy) !important;
            letter-spacing: -0.025em;
        }

        h1 {
            font-size: 3rem !important;
            line-height: 1.12 !important;
        }

        h2 {
            font-size: 2rem !important;
            line-height: 1.2 !important;
        }

        h3 {
            font-size: 1.48rem !important;
            line-height: 1.28 !important;
        }

        h4 {
            font-size: 1.18rem !important;
        }

        .stMarkdown p,
        .stMarkdown li {
            font-size: 17px;
            line-height: 1.68;
            color: #34445E;
        }

        hr {
            border: none !important;
            height: 1px !important;
            margin: 1.55rem 0 !important;
            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(22, 99, 240, 0.35),
                    rgba(45, 178, 74, 0.35),
                    transparent
                ) !important;
        }

        /* Hero */
        .fb-hero {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 190px;
            gap: 2rem;
            align-items: center;
            padding: 2.35rem 2.6rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(22, 99, 240, 0.18);
            border-radius: 28px;
            background:
                radial-gradient(
                    circle at 87% 16%,
                    rgba(22, 99, 240, 0.17),
                    transparent 17rem
                ),
                radial-gradient(
                    circle at 70% 100%,
                    rgba(45, 178, 74, 0.12),
                    transparent 15rem
                ),
                rgba(255, 255, 255, 0.96);
            box-shadow: var(--fb-shadow);
        }

        .fb-hero::after {
            content: "";
            position: absolute;
            right: -78px;
            bottom: -100px;
            width: 250px;
            height: 250px;
            border: 1px solid rgba(22, 99, 240, 0.12);
            border-radius: 50%;
            box-shadow:
                0 0 0 26px rgba(22, 99, 240, 0.035),
                0 0 0 54px rgba(45, 178, 74, 0.025);
        }

        .fb-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.42rem 0.72rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(22, 99, 240, 0.18);
            border-radius: 999px;
            background: var(--fb-soft-blue);
            color: var(--fb-blue);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .fb-eyebrow-dot {
            width: 0.52rem;
            height: 0.52rem;
            border-radius: 50%;
            background: var(--fb-green);
            box-shadow: 0 0 0 5px rgba(45, 178, 74, 0.11);
        }

        .fb-brand-name {
            margin: 0;
            color: var(--fb-navy);
            font-size: clamp(2.8rem, 5vw, 4.7rem);
            font-weight: 850;
            line-height: 0.98;
            letter-spacing: -0.055em;
        }

        .fb-brand-name span {
            color: var(--fb-blue);
        }

        .fb-hero-title {
            margin: 1rem 0 0.48rem;
            color: var(--fb-navy);
            font-size: clamp(1.45rem, 2.2vw, 2.05rem);
            font-weight: 760;
            line-height: 1.22;
            letter-spacing: -0.035em;
        }

        .fb-hero-copy {
            max-width: 830px;
            margin: 0;
            color: #52627A;
            font-size: 1.03rem;
            line-height: 1.65;
        }

        .fb-hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.62rem;
            margin-top: 1.25rem;
        }

        .fb-tag {
            padding: 0.5rem 0.78rem;
            border: 1px solid var(--fb-border);
            border-radius: 999px;
            background: #FFFFFF;
            color: var(--fb-navy);
            font-size: 0.82rem;
            font-weight: 720;
            box-shadow: 0 5px 14px rgba(10, 31, 68, 0.05);
        }

        .fb-tag strong {
            color: var(--fb-green);
        }

        .fb-hero-logo-wrap {
            position: relative;
            z-index: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .fb-hero-logo {
            width: 164px;
            height: 164px;
            object-fit: contain;
            border-radius: 34px;
            filter: drop-shadow(0 18px 24px rgba(10, 31, 68, 0.22));
        }

        .fb-values-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.9rem 0 1.8rem;
        }

        .fb-value-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-height: 64px;
            padding: 0.8rem 1rem;
            border: 1px solid var(--fb-border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: var(--fb-shadow-soft);
        }

        .fb-value-icon {
            display: grid;
            place-items: center;
            flex: 0 0 36px;
            width: 36px;
            height: 36px;
            border-radius: 11px;
            background:
                linear-gradient(
                    135deg,
                    var(--fb-navy),
                    var(--fb-blue)
                );
            color: #FFFFFF;
            font-size: 1rem;
            font-weight: 850;
        }

        .fb-value-title {
            color: var(--fb-navy);
            font-size: 0.94rem;
            font-weight: 800;
        }

        .fb-value-copy {
            margin-top: 0.08rem;
            color: var(--fb-muted);
            font-size: 0.78rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] > div:first-child {
            background:
                radial-gradient(
                    circle at 50% -10%,
                    rgba(22, 99, 240, 0.65),
                    transparent 21rem
                ),
                linear-gradient(
                    180deg,
                    #071A37 0%,
                    #0A1F44 58%,
                    #102A55 100%
                );
            border-right: 1px solid rgba(255, 255, 255, 0.09);
        }

        [data-testid="stSidebar"] {
            color: #FFFFFF;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.3rem;
        }

        .fb-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.82rem;
            padding: 0.82rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 17px;
            background: rgba(255, 255, 255, 0.07);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
        }

        .fb-sidebar-logo {
            width: 54px;
            height: 54px;
            object-fit: contain;
            border-radius: 14px;
        }

        .fb-sidebar-name {
            color: #FFFFFF;
            font-size: 1.1rem;
            font-weight: 850;
            letter-spacing: -0.025em;
        }

        .fb-sidebar-subtitle {
            margin-top: 0.08rem;
            color: rgba(255, 255, 255, 0.67);
            font-size: 0.74rem;
            font-weight: 650;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .fb-sidebar-section {
            margin: 1rem 0 0.75rem;
            color: #FFFFFF;
            font-size: 1.2rem;
            font-weight: 800;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] summary p {
            color: rgba(255, 255, 255, 0.90) !important;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: var(--fb-navy) !important;
            background: #FFFFFF !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            color: var(--fb-navy) !important;
            background: #FFFFFF !important;
            border-color: transparent !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="input"] * {
            color: var(--fb-navy) !important;
        }

        [data-testid="stSidebar"] details[data-testid="stExpander"] {
            border-color: rgba(255, 255, 255, 0.15) !important;
            background: rgba(255, 255, 255, 0.06) !important;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.94);
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: var(--fb-navy) !important;
        }

        /* Form fields */
        label,
        [data-testid="stWidgetLabel"] p {
            font-size: 15px !important;
            font-weight: 720 !important;
        }

        input,
        textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            border-radius: 11px !important;
        }

        input:focus,
        textarea:focus {
            border-color: var(--fb-blue) !important;
            box-shadow: 0 0 0 3px rgba(22, 99, 240, 0.12) !important;
        }

        /* Buttons */
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stDownloadButton"] button,
        .stButton > button {
            min-height: 46px;
            border: none !important;
            border-radius: 12px !important;
            font-size: 0.94rem !important;
            font-weight: 800 !important;
            transition:
                transform 150ms ease,
                box-shadow 150ms ease,
                filter 150ms ease;
        }

        [data-testid="stFormSubmitButton"] button {
            color: #FFFFFF !important;
            background:
                linear-gradient(
                    100deg,
                    var(--fb-blue) 0%,
                    #2476F4 56%,
                    var(--fb-green) 120%
                ) !important;
            box-shadow: 0 12px 24px rgba(22, 99, 240, 0.28) !important;
        }

        [data-testid="stDownloadButton"] button {
            color: #FFFFFF !important;
            background: var(--fb-navy) !important;
            box-shadow: 0 10px 20px rgba(10, 31, 68, 0.20) !important;
        }

        [data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover,
        .stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.035);
            box-shadow: 0 14px 26px rgba(10, 31, 68, 0.22) !important;
        }

        /* Tabs */
        [data-baseweb="tab-list"] {
            gap: 0.42rem;
            padding: 0.42rem;
            border: 1px solid var(--fb-border);
            border-radius: 16px;
            background: rgba(238, 244, 255, 0.72);
            box-shadow: 0 7px 18px rgba(10, 31, 68, 0.045);
            overflow-x: auto;
        }

        button[data-baseweb="tab"] {
            min-height: 43px;
            padding: 0 0.92rem !important;
            border-radius: 11px;
            color: #5D6C82 !important;
            font-size: 0.9rem !important;
            font-weight: 760 !important;
            white-space: nowrap;
        }

        button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
            color: inherit !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--fb-navy) !important;
            background: #FFFFFF !important;
            box-shadow: 0 7px 16px rgba(10, 31, 68, 0.09);
        }

        button[data-baseweb="tab"][aria-selected="true"]::after {
            content: "";
            position: absolute;
            right: 20%;
            bottom: 2px;
            left: 20%;
            height: 3px;
            border-radius: 999px;
            background:
                linear-gradient(
                    90deg,
                    var(--fb-blue),
                    var(--fb-green)
                );
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            min-height: 132px;
            padding: 1.05rem 1.1rem;
            border: 1px solid var(--fb-border);
            border-radius: 17px;
            background:
                linear-gradient(
                    145deg,
                    #FFFFFF 0%,
                    #FBFDFF 100%
                );
            box-shadow: var(--fb-shadow-soft);
            transition:
                transform 150ms ease,
                border-color 150ms ease,
                box-shadow 150ms ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(22, 99, 240, 0.38);
            box-shadow: var(--fb-shadow);
        }

        [data-testid="stMetricLabel"] p {
            color: #637189 !important;
            font-size: 0.82rem !important;
            font-weight: 780 !important;
            letter-spacing: 0.025em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--fb-navy) !important;
            font-size: clamp(1.48rem, 2.1vw, 2rem) !important;
            font-weight: 850 !important;
            letter-spacing: -0.035em;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.82rem !important;
            font-weight: 740 !important;
        }

        /* Charts, tables and expanders */
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--fb-border);
            border-radius: 18px;
            background: #FFFFFF;
            box-shadow: var(--fb-shadow-soft);
        }

        [data-testid="stPlotlyChart"] {
            padding: 0.5rem;
        }

        details[data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid var(--fb-border) !important;
            border-radius: 15px !important;
            background: rgba(255, 255, 255, 0.92) !important;
            box-shadow: 0 6px 17px rgba(10, 31, 68, 0.045);
        }

        details[data-testid="stExpander"] summary {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem;
        }

        details[data-testid="stExpander"] summary p {
            color: var(--fb-navy) !important;
            font-size: 0.95rem !important;
            font-weight: 780 !important;
        }

        [data-testid="stAlert"] {
            border-radius: 14px;
            border-width: 1px;
            box-shadow: 0 5px 14px rgba(10, 31, 68, 0.045);
        }

        [data-testid="stCaptionContainer"] p {
            color: var(--fb-muted) !important;
            font-size: 0.85rem !important;
            line-height: 1.5 !important;
        }

        /* Landing content */
        .fb-welcome-panel {
            margin: 1.4rem 0 1rem;
            padding: 1.45rem 1.55rem;
            border: 1px solid rgba(22, 99, 240, 0.17);
            border-radius: 20px;
            background:
                linear-gradient(
                    110deg,
                    rgba(238, 244, 255, 0.88),
                    rgba(236, 248, 239, 0.76)
                );
        }

        .fb-welcome-title {
            color: var(--fb-navy);
            font-size: 1.2rem;
            font-weight: 820;
        }

        .fb-welcome-copy {
            margin-top: 0.3rem;
            color: #52627A;
            line-height: 1.6;
        }

        .fb-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .fb-feature-card {
            min-height: 175px;
            padding: 1.2rem;
            border: 1px solid var(--fb-border);
            border-radius: 18px;
            background: #FFFFFF;
            box-shadow: var(--fb-shadow-soft);
        }

        .fb-feature-number {
            display: inline-grid;
            place-items: center;
            width: 36px;
            height: 36px;
            border-radius: 11px;
            color: #FFFFFF;
            background:
                linear-gradient(
                    135deg,
                    var(--fb-navy),
                    var(--fb-blue)
                );
            font-weight: 850;
        }

        .fb-feature-card h4 {
            margin: 0.85rem 0 0.38rem;
            color: var(--fb-navy);
            font-size: 1.03rem;
        }

        .fb-feature-card p {
            margin: 0;
            color: var(--fb-muted);
            font-size: 0.88rem;
            line-height: 1.55;
        }

        /* Footer */
        .fb-footer {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1rem;
            align-items: center;
            margin-top: 3rem;
            padding: 1.1rem 1.3rem;
            border-radius: 17px;
            background:
                linear-gradient(
                    100deg,
                    #071A37,
                    #0A1F44 68%,
                    #12418F
                );
            color: #FFFFFF;
            box-shadow: var(--fb-shadow);
        }

        .fb-footer-brand {
            font-size: 0.95rem;
            font-weight: 820;
        }

        .fb-footer-copy {
            margin-top: 0.15rem;
            color: rgba(255, 255, 255, 0.67);
            font-size: 0.78rem;
        }

        .fb-footer-pill {
            padding: 0.48rem 0.76rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.08);
            font-size: 0.76rem;
            font-weight: 720;
            white-space: nowrap;
        }

        .fb-footer-pill .fb-footer-green {
            color: #2DB24A;
            font-weight: 850;
        }


        /* =====================================================
           STANDARD DESKTOP SCALE
           Uses the original compact Finance Bro sizing above.
           ===================================================== */

        @media (max-width: 1100px) {
            [data-testid="stSidebar"] {
                min-width: 320px !important;
                max-width: 320px !important;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                width: 320px !important;
            }
        }

        @media (max-width: 900px) {
            .fb-hero {
                grid-template-columns: 1fr;
                padding: 1.65rem;
            }

            .fb-hero-logo-wrap {
                justify-content: flex-start;
            }

            .fb-hero-logo {
                width: 112px;
                height: 112px;
            }

            .fb-values-strip,
            .fb-feature-grid {
                grid-template-columns: 1fr;
            }

            .fb-footer {
                grid-template-columns: 1fr;
            }
        }

        /* =====================================================
           SIDEBAR CONTRAST FIX
           Keeps every label readable on the dark navy background.
           ===================================================== */

        [data-testid="stSidebar"] {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] legend,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] label p,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
            color: rgba(255, 255, 255, 0.96) !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: rgba(255, 255, 255, 0.76) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] details[data-testid="stExpander"] {
            overflow: hidden !important;
            border: 1px solid rgba(255, 255, 255, 0.24) !important;
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.075) !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] details[data-testid="stExpander"] summary {
            min-height: 58px !important;
            padding: 0.65rem 0.9rem !important;
            color: #FFFFFF !important;
            background: rgba(255, 255, 255, 0.10) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg,
        [data-testid="stSidebar"] details[data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] details[data-testid="stExpander"] summary span,
        [data-testid="stSidebar"] details[data-testid="stExpander"] summary svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [role="radiogroup"] label span,
        [data-testid="stSidebar"] [role="radiogroup"] label p {
            color: #FFFFFF !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"],
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
        [data-testid="stSidebar"] button[aria-label*="help" i],
        [data-testid="stSidebar"] button[aria-label*="help" i] svg {
            color: rgba(255, 255, 255, 0.88) !important;
            fill: rgba(255, 255, 255, 0.88) !important;
            stroke: rgba(255, 255, 255, 0.88) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="input"] *,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button *,
        [data-testid="stSidebar"] [data-testid="stDateInput"] button,
        [data-testid="stSidebar"] [data-testid="stDateInput"] button * {
            color: #0A1F44 !important;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] > div,
        [data-testid="stSidebar"] [data-testid="stDateInput"] > div {
            background: #FFFFFF !important;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            border: 1px solid rgba(22, 99, 240, 0.18) !important;
            background: #EAF2FF !important;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] p,
        [data-testid="stSidebar"] [data-testid="stAlert"] span,
        [data-testid="stSidebar"] [data-testid="stAlert"] svg {
            color: #0A1F44 !important;
            fill: #1663F0 !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h4 {
            color: #FFFFFF !important;
        }


        /* =====================================================
           REFINED TYPOGRAPHY
           Large and readable without excessive visual weight.
           ===================================================== */

        html,
        body,
        [class*="css"],
        .stApp {
            font-family:
                "Segoe UI Variable Text",
                "Segoe UI Variable",
                "Segoe UI",
                Roboto,
                Helvetica,
                Arial,
                sans-serif !important;
            font-synthesis: none;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        html {
            font-size: 17px;
        }

        h1,
        h2,
        h3,
        h4 {
            font-family:
                "Segoe UI Variable Display",
                "Segoe UI Variable",
                "Segoe UI",
                Roboto,
                Helvetica,
                Arial,
                sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.018em !important;
        }

        h1 {
            font-size: clamp(2.65rem, 4vw, 3.45rem) !important;
            line-height: 1.1 !important;
        }

        h2 {
            font-size: clamp(2rem, 3vw, 2.45rem) !important;
            line-height: 1.16 !important;
        }

        h3 {
            font-size: clamp(1.5rem, 2.2vw, 1.82rem) !important;
            line-height: 1.24 !important;
        }

        h4 {
            font-size: 1.28rem !important;
            line-height: 1.3 !important;
        }

        .stMarkdown p,
        .stMarkdown li,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            font-size: 18px !important;
            line-height: 1.68 !important;
            font-weight: 400 !important;
            letter-spacing: 0 !important;
        }

        label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-size: 16.5px !important;
            line-height: 1.42 !important;
            font-weight: 650 !important;
            letter-spacing: 0 !important;
        }

        input,
        textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input {
            font-family:
                "Segoe UI Variable Text",
                "Segoe UI",
                Roboto,
                Arial,
                sans-serif !important;
            font-size: 17px !important;
            font-weight: 450 !important;
        }

        [data-testid="stMetric"] {
            min-height: 148px;
            padding: 1.2rem 1.25rem;
        }

        [data-testid="stMetricLabel"] p {
            color: #52627A !important;
            font-size: 0.95rem !important;
            line-height: 1.35 !important;
            font-weight: 650 !important;
            letter-spacing: 0.01em !important;
            text-transform: none !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--fb-navy) !important;
            font-family:
                "Segoe UI Variable Display",
                "Segoe UI Variable",
                "Segoe UI",
                Roboto,
                Arial,
                sans-serif !important;
            font-size: clamp(1.9rem, 2.5vw, 2.45rem) !important;
            line-height: 1.15 !important;
            font-weight: 700 !important;
            letter-spacing: -0.018em !important;
            font-variant-numeric: tabular-nums lining-nums;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.94rem !important;
            font-weight: 600 !important;
        }

        button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stDownloadButton"] button,
        .stButton > button {
            font-family:
                "Segoe UI Variable Text",
                "Segoe UI",
                Roboto,
                Arial,
                sans-serif !important;
            font-size: 1.02rem !important;
            font-weight: 650 !important;
            letter-spacing: 0 !important;
        }

        button[data-baseweb="tab"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            letter-spacing: 0 !important;
        }

        details[data-testid="stExpander"] summary p {
            font-size: 1.08rem !important;
            line-height: 1.42 !important;
            font-weight: 650 !important;
            letter-spacing: 0 !important;
        }

        [data-testid="stAlert"] p {
            font-size: 17px !important;
            line-height: 1.58 !important;
            font-weight: 430 !important;
        }

        [data-testid="stCaptionContainer"] p {
            font-size: 0.94rem !important;
            line-height: 1.55 !important;
            font-weight: 400 !important;
        }

        .fb-brand-name {
            font-weight: 760 !important;
            letter-spacing: -0.032em !important;
        }

        .fb-hero-title {
            font-weight: 650 !important;
            letter-spacing: -0.018em !important;
        }

        .fb-hero-copy,
        .fb-welcome-copy,
        .fb-feature-card p {
            font-weight: 400 !important;
        }

        .fb-sidebar-name,
        .fb-footer-brand {
            font-weight: 700 !important;
        }

    </style>
    """
)


# ============================================================
# PLOTLY BRAND TEMPLATE
# ============================================================

pio.templates["finance_bro"] = go.layout.Template(
    layout=go.Layout(
        colorway=[
            BRAND_BLUE,
            BRAND_GREEN,
            BRAND_NAVY,
            "#23A8F2",
            "#7C5CE5",
            "#E3A008",
            "#D92D20",
            "#0F8B8D",
        ],
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            size=18,
            color="#34445E",
        ),
        title=dict(
            font=dict(
                family="Inter, Segoe UI, sans-serif",
                size=26,
                color=BRAND_NAVY,
            ),
            x=0.02,
            xanchor="left",
        ),
        legend=dict(
            font=dict(
                size=15,
                color="#52627A",
            ),
            title=dict(
                font=dict(
                    size=16,
                    color=BRAND_NAVY,
                )
            ),
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="#DDE7F5",
            borderwidth=1,
        ),
        xaxis=dict(
            showline=False,
            gridcolor="#E8EEF7",
            zerolinecolor="#CBD7E8",
            tickfont=dict(
                size=14,
                color="#64748B",
            ),
            title=dict(
                font=dict(
                    size=14,
                    color=BRAND_NAVY,
                )
            ),
        ),
        yaxis=dict(
            showline=False,
            gridcolor="#E8EEF7",
            zerolinecolor="#CBD7E8",
            tickfont=dict(
                size=12,
                color="#64748B",
            ),
            title=dict(
                font=dict(
                    size=14,
                    color=BRAND_NAVY,
                )
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(
            bgcolor=BRAND_NAVY,
            bordercolor=BRAND_NAVY,
            font=dict(
                color="#FFFFFF",
                size=15,
            ),
        ),
        margin=dict(
            l=45,
            r=25,
            t=75,
            b=45,
        ),
    )
)

pio.templates.default = "plotly_white+finance_bro"


# ============================================================
# REUSABLE BRAND COMPONENTS
# ============================================================

def render_brand_footer() -> None:
    render_html(
        """
        <div class="fb-footer">
            <div>
                <div class="fb-footer-brand">
                    Finance Bro — Clear insights. Smarter decisions.
                </div>
                <div class="fb-footer-copy">
                    Historical analytics and education, not investment advice.
                </div>
            </div>
            <div class="fb-footer-pill">
                Smart Decisions.
                <span class="fb-footer-green">Stronger Futures.</span>
            </div>
        </div>
        """
    )


# ============================================================
# HERO HEADER
# ============================================================

render_html(
    f"""
    <section class="fb-hero">
        <div>
            <div class="fb-eyebrow">
                <span class="fb-eyebrow-dot"></span>
                Portfolio Intelligence &amp; Education
            </div>

            <div class="fb-brand-name">
                Finance <span>Bro</span>
            </div>

            <div class="fb-hero-title">
                Understand Today. Invest&nbsp;Better&nbsp;Tomorrow.
            </div>

            <p class="fb-hero-copy">
                Turn market data into clear portfolio insights, rigorous risk
                analysis and practical financial education — all in one place.
            </p>

            <div class="fb-hero-tags">
                <span class="fb-tag">
                    <strong>✓</strong> Multi-Currency Analytics
                </span>
                <span class="fb-tag">
                    <strong>✓</strong> Data Quality &amp; Regression
                </span>
                <span class="fb-tag">
                    <strong>✓</strong> Professional Excel Reports
                </span>
            </div>
        </div>

        <div class="fb-hero-logo-wrap">
            <img
                class="fb-hero-logo"
                src="data:image/png;base64,{app_icon_base64}"
                alt="Finance Bro app icon"
            />
        </div>
    </section>

    <div class="fb-values-strip">
        <div class="fb-value-item">
            <div class="fb-value-icon">✓</div>
            <div>
                <div class="fb-value-title">Clear Insights</div>
                <div class="fb-value-copy">Transparent data and methodology.</div>
            </div>
        </div>

        <div class="fb-value-item">
            <div class="fb-value-icon">↗</div>
            <div>
                <div class="fb-value-title">Smarter Decisions</div>
                <div class="fb-value-copy">Performance, risk and benchmark context.</div>
            </div>
        </div>

        <div class="fb-value-item">
            <div class="fb-value-icon">◎</div>
            <div>
                <div class="fb-value-title">Better Outcomes</div>
                <div class="fb-value-copy">Learn the meaning behind every metric.</div>
            </div>
        </div>
    </div>
    """
)


# ============================================================
# WORKSPACE SELECTION
# ============================================================

if "pending_analysis_mode" in st.session_state:
    st.session_state["analysis_mode_selector"] = st.session_state.pop(
        "pending_analysis_mode"
    )

with st.sidebar:

    render_html(
        f"""
        <div class="fb-sidebar-brand">
            <img
                class="fb-sidebar-logo"
                src="data:image/png;base64,{app_icon_base64}"
                alt="Finance Bro logo"
            />
            <div>
                <div class="fb-sidebar-name">Finance Bro</div>
                <div class="fb-sidebar-subtitle">Research &amp; Portfolio Intelligence</div>
            </div>
        </div>
        """
    )

    analysis_mode = st.radio(
        "Workspace",
        options=[
            "Stock Research",
            "Portfolio Analysis",
        ],
        index=0,
        help=(
            "Research companies or analyze hypothetical portfolios using "
            "the latest available prices."
        ),
        key="analysis_mode_selector",
    )

    render_html(
        """
        <div
            aria-disabled="true"
            title="Live Investment is being prepared for a future release."
            style="
                margin-top: 0.45rem;
                padding: 0.70rem 0.80rem;
                border: 1px solid rgba(10, 31, 68, 0.14);
                border-radius: 0.70rem;
                background: rgba(10, 31, 68, 0.045);
                color: rgba(10, 31, 68, 0.55);
                cursor: not-allowed;
                user-select: none;
                pointer-events: none;
            "
        >
            <div style="display:flex; align-items:center; justify-content:space-between; gap:0.75rem;">
                <span style="font-weight:700;">Live Investment</span>
                <span
                    style="
                        padding: 0.15rem 0.45rem;
                        border-radius: 999px;
                        background: rgba(22, 99, 240, 0.10);
                        color: rgba(22, 99, 240, 0.68);
                        font-size: 0.72rem;
                        font-weight: 800;
                        letter-spacing: 0.02em;
                        white-space: nowrap;
                    "
                >
                    Coming soon
                </span>
            </div>
        </div>
        """
    )

if analysis_mode == "Stock Research":
    render_stock_research()
    render_brand_footer()
    st.stop()

# ============================================================
# PORTFOLIO SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="fb-sidebar-section">
            Build Your Portfolio
        </div>
        """
    )

    portfolio_currency = st.selectbox(
        "Portfolio Currency",
        options=[
            "EUR",
            "USD",
        ],
        index=0,
        format_func=lambda currency: (
            "EUR — Euro (€)"
            if currency == "EUR"
            else "USD — US Dollar ($)"
        ),
        help=(
            "All portfolio values, monetary risk metrics and "
            "historical prices are converted into this currency."
        ),
        key="portfolio_currency_selector",
    )

    selected_currency_symbol = {
        "EUR": "€",
        "USD": "$",
    }[portfolio_currency]

    with st.container():

        initial_investment = st.number_input(
            f"Initial Investment ({portfolio_currency})",
            min_value=1.0,
            value=10000.0,
            step=500.0
        )

        ticker_input = st.text_input(
            "Tickers",
            value="",
            placeholder="e.g. AAPL, MSFT, GOOGL",
            help="Enter the tickers separated by commas.",
            key="portfolio_ticker_input_v2",
        )

        preview_tickers = [
            ticker.strip().upper()
            for ticker in ticker_input.split(",")
            if ticker.strip()
        ]

        equal_weights = st.checkbox(
            "Use equal weights",
            value=True
        )

        custom_weight_inputs = {}

        if not equal_weights:

            st.markdown("#### Custom Weights")

            default_weight = (
                round(100 / len(preview_tickers), 2)
                if preview_tickers
                else 0.0
            )

            for ticker in preview_tickers:

                custom_weight_inputs[ticker] = st.number_input(
                    f"{ticker} Weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight,
                    step=0.5,
                    key=f"weight_{ticker}"
                )

        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2023-01-01"),
            max_value=pd.Timestamp.today().date(),
            help=(
                "Choose the beginning of the analysis period. "
                "When the selected day is not a trading day, the analysis "
                "uses the first valid market observation available after it."
            )
        )

        use_latest_date = st.checkbox(
            "Use latest available date",
            value=True,
            help=(
                "When enabled, Finance Bro downloads data up to the latest "
                "market observation available whenever the analysis is run. "
                "When disabled, an End Date field appears below."
            )
        )

        if use_latest_date:

            manual_end_date = None

            st.caption(
                "ⓘ End Date: latest available market observation. "
                "The result updates whenever newer data becomes available."
            )

        else:

            manual_end_date = st.date_input(
                "End Date",
                value=pd.Timestamp.today().date(),
                min_value=start_date,
                max_value=pd.Timestamp.today().date(),
                help=(
                    "Choose the final calendar date to include in the "
                    "analysis. If that date is a weekend or market holiday, "
                    "the app uses the last valid market observation on or "
                    "before it."
                ),
                key="manual_end_date_input",
            )

            st.caption(
                "ⓘ The selected End Date is inclusive. When no market data "
                "exists on that day, the previous available trading day is "
                "used."
            )

        rolling_window = st.number_input(
            "Rolling Window (Trading Days)",
            min_value=2,
            max_value=252,
            value=20,
            step=1,
            help=(
                "On each date, the rolling calculation uses that date's "
                "return plus the previous trading-day returns until the "
                "selected window is completed. For example, a 20-day window "
                "uses the current return and the previous 19 daily returns. "
                "The app does not fetch extra market prices before the chosen "
                "start date, so the first rolling result only appears after "
                "enough observations have accumulated."
            )
        )

        st.caption(
            "ⓘ Example: 20 means that each rolling value uses the most "
            "recent 20 daily returns — the current day plus the previous "
            "19 trading days. It does not use 20 days before the selected "
            "start date. The first 19 return observations therefore have no "
            "rolling result; because one return needs two prices, 20 returns "
            "require 21 price observations."
        )

        confidence_level = st.selectbox(
            "Confidence Level",
            options=[90, 95, 99],
            index=1
        )

        benchmark_names = {
            "^GSPC": "S&P 500",
            "^STOXX50E": "EURO STOXX 50",
            "^IXIC": "Nasdaq Composite",
            "^DJI": "Dow Jones",
        }

        benchmark_ticker = st.selectbox(
            "Benchmark",
            options=list(
                benchmark_names.keys()
            ),
            index=0,
            format_func=lambda ticker: (
                f"{benchmark_names[ticker]} ({ticker})"
            ),
            help=(
                "Choose the market index used for comparison, regression, "
                "beta estimation and the beta-based market stress scenario."
            ),
            key="benchmark_selector",
        )

        selected_benchmark_name = benchmark_names[
            benchmark_ticker
        ]

        with st.expander(
            "Risk-Free Rate Settings",
            expanded=False
        ):

            risk_free_mode = st.radio(
                "Risk-Free Rate Method",
                options=[
                    "Automatic",
                    "Manual",
                ],
                index=0,
                horizontal=True,
                help=(
                    "Automatic uses an official currency-specific "
                    "3-month reference series. Manual applies one "
                    "constant annual rate to the entire sample."
                ),
                key="risk_free_mode_selector",
            )

            if risk_free_mode == "Automatic":

                if portfolio_currency == "EUR":

                    selected_risk_free_source = (
                        "ECB 3-month compounded €STR average rate"
                    )

                    selected_risk_free_description = (
                        "Because the portfolio currency is EUR, Finance Bro "
                        "automatically uses the ECB 3-month compounded €STR "
                        "average rate."
                    )

                else:

                    selected_risk_free_source = (
                        "3-month U.S. Treasury constant-maturity "
                        "yield (DGS3MO)"
                    )

                    selected_risk_free_description = (
                        "Because the portfolio currency is USD, Finance Bro "
                        "automatically uses the 3-month U.S. Treasury "
                        "constant-maturity yield (DGS3MO)."
                    )

                st.info(
                    f"**Automatic risk-free source:**\n\n"
                    f"{selected_risk_free_description}\n\n"
                    f"**Selected source:** {selected_risk_free_source}"
                )

                manual_annual_risk_free_rate = None

            else:

                manual_default_rate = (
                    2.00
                    if portfolio_currency == "EUR"
                    else 3.85
                )

                st.info(
                    "Manual mode ignores the automatic EUR or USD reference "
                    "series and uses the annual percentage entered below for "
                    "the entire analysis period."
                )

                manual_annual_risk_free_rate = st.number_input(
                    "Manual Annual Risk-Free Rate (%)",
                    min_value=-5.0,
                    max_value=25.0,
                    value=float(
                        manual_default_rate
                    ),
                    step=0.05,
                    help=(
                        "Enter an annual risk-free rate in percentage terms. "
                        "Finance Bro converts it into an equivalent daily "
                        "return before calculating the Sharpe Ratio, alpha "
                        "and benchmark regression."
                    ),
                    key="manual_risk_free_rate_input",
                )

        with st.expander(
            "Stress Test Settings",
            expanded=False
        ):

            st.markdown("#### Beta-Based Market Correction")

            benchmark_stress_shock = st.number_input(
                (
                    f"{selected_benchmark_name} "
                    f"({benchmark_ticker}) Shock (%)"
                ),
                min_value=-100.0,
                max_value=100.0,
                value=-10.0,
                step=1.0,
                help=(
                    "This shock applies to the benchmark selected just "
                    "above. Each asset shock is then estimated as the asset's "
                    "historical beta multiplied by this benchmark shock."
                )
            )

            st.markdown("#### Custom Asset Shocks")

            stress_scenario_name = st.text_input(
                "Custom Scenario Name",
                value="My Custom Stress Scenario",
                help=(
                    "Examples: Technology Sell-Off, Energy Shock "
                    "or Company-Specific Crisis."
                )
            )

            custom_stress_inputs = {}

            for ticker in preview_tickers:

                custom_stress_inputs[ticker] = st.number_input(
                    f"{ticker} Custom Shock (%)",
                    min_value=-100.0,
                    max_value=100.0,
                    value=-10.0,
                    step=1.0,
                    key=f"stress_shock_{ticker}",
                    help=(
                        "Use a negative value for a price fall and "
                        "a positive value for a price increase."
                    )
                )

        analyze_button = st.button(
            "Analyze Portfolio",
            use_container_width=True,
            type="primary"
        )


# ============================================================
# LANDING STATE
# ============================================================

if not analyze_button:

    render_html(
        """
        <div class="fb-welcome-panel">
            <div class="fb-welcome-title">
                Build your first Finance Bro analysis
            </div>
            <div class="fb-welcome-copy">
                Choose the portfolio currency, enter the tickers, define the
                weights and press <strong>Analyze Portfolio</strong>. Finance
                Bro will align the data, convert currencies, estimate risk and
                benchmark relationships, and prepare a professional Excel
                report.
            </div>
        </div>

        <div class="fb-feature-grid">
            <div class="fb-feature-card">
                <div class="fb-feature-number">1</div>
                <h4>Measure Performance</h4>
                <p>
                    Follow portfolio value, cumulative return, benchmark
                    performance, alpha and beta in one consistent currency.
                </p>
            </div>

            <div class="fb-feature-card">
                <div class="fb-feature-number">2</div>
                <h4>Understand Risk</h4>
                <p>
                    Explore volatility, drawdown, VaR, Expected Shortfall,
                    diversification and scenario-based stress testing.
                </p>
            </div>

            <div class="fb-feature-card">
                <div class="fb-feature-number">3</div>
                <h4>Learn the Logic</h4>
                <p>
                    See formulas, interpretations, limitations and advanced
                    diagnostics instead of receiving unexplained numbers.
                </p>
            </div>
        </div>
        """
    )

    render_brand_footer()


# ============================================================
# PROCESSAR INPUTS
# ============================================================

if analyze_button:

    tickers = [
        ticker.strip().upper()
        for ticker in ticker_input.split(",")
        if ticker.strip()
    ]

    if len(tickers) == 0:

        st.error(
            "Enter at least one valid ticker."
        )

        st.stop()

    if equal_weights:

        weights = pd.Series(
            1 / len(tickers),
            index=tickers
        )

    else:

        weights = pd.Series(
            {
                ticker: custom_weight_inputs[ticker] / 100
                for ticker in tickers
            }
        )

        if not np.isclose(weights.sum(), 1.0):

            st.error(
                "The weights must sum to 100%. "
                f"Current total: {weights.sum() * 100:.2f}%"
            )

            st.stop()

    # ========================================================
    # EXECUTAR A ANÁLISE REAL
    # ========================================================

    if (
        not use_latest_date
        and manual_end_date <= start_date
    ):

        st.error(
            "End Date must be later than Start Date so that returns "
            "can be calculated."
        )

        st.stop()

    selected_end_date = (
        None
        if use_latest_date
        else manual_end_date
    )

    try:

        with st.spinner(
            "Downloading market data and analyzing the portfolio..."
        ):

            results = analyze_portfolio(
                tickers=tickers,
                weights=weights,
                initial_investment=initial_investment,
                start_date=start_date,
                end_date=selected_end_date,
                portfolio_currency=portfolio_currency,
                rolling_window=int(rolling_window),
                annual_risk_free_rate=(
                    manual_annual_risk_free_rate
                ),
                risk_free_mode=risk_free_mode,
                confidence_level=confidence_level / 100,
                regression_confidence_level=0.95,
                benchmark_ticker=benchmark_ticker,
                benchmark_stress_shock=(
                    benchmark_stress_shock / 100
                ),
                custom_stress_shocks=pd.Series(
                    {
                        ticker:
                            custom_stress_inputs[ticker] / 100
                        for ticker in tickers
                    }
                ),
                stress_scenario_name=stress_scenario_name,
            )

    except Exception as error:

        st.error(
            f"Portfolio analysis failed: {error}"
        )

        st.stop()

    st.success(
        "Portfolio analysis completed successfully."
    )

    portfolio_currency = results[
        "portfolio_currency"
    ]

    currency_symbol = results[
        "currency_symbol"
    ]

    def format_money(
        value,
        decimals=2,
        show_plus=False,
    ):

        if pd.isna(value):
            return "N/A"

        value = float(
            value
        )

        if value < 0:
            return (
                f"-{currency_symbol}"
                f"{abs(value):,.{decimals}f}"
            )

        if value > 0 and show_plus:
            return (
                f"+{currency_symbol}"
                f"{value:,.{decimals}f}"
            )

        return (
            f"{currency_symbol}"
            f"{value:,.{decimals}f}"
        )

    portfolio_value_column = (
        "Portfolio Value"
    )

    initial_invested_value_column = (
        "Initial Invested Value"
    )

    current_estimated_value_column = (
        "Current Estimated Value"
    )


    # ========================================================
    # EXPORTAÇÃO PROFISSIONAL PARA EXCEL
    # ========================================================

    try:

        with st.spinner(
            "Preparing the professional Excel report..."
        ):

            excel_report_bytes = create_excel_report(
                results=results,
                rolling_window=int(rolling_window),
            )

    except Exception as excel_export_error:

        st.warning(
            "The portfolio analysis was completed, but the Excel report "
            f"could not be prepared: {excel_export_error}"
        )

    else:

        export_column_1, export_column_2 = st.columns(
            [1.2, 2.8]
        )

        report_file_name = (
            "Finance_Bro_Portfolio_Report_"
            f"{portfolio_currency}_"
            f"{pd.Timestamp.now():%Y%m%d}.xlsx"
        )

        with export_column_1:

            st.download_button(
                label="Download Professional Excel Report",
                data=excel_report_bytes,
                file_name=report_file_name,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                help=(
                    "Downloads a complete multi-sheet report containing "
                    "data quality, regression, performance, risk, "
                    "allocation, diversification, stress tests and "
                    "an educational guide."
                ),
            )

        with export_column_2:

            st.info(
                "The report automatically uses the selected portfolio "
                f"currency ({portfolio_currency}), historical FX conversion, "
                "official risk-free data, regression diagnostics, "
                "professional number formats, charts and educational notes."
            )


    # ========================================================
    # SEPARADORES PRINCIPAIS
    # ========================================================

    (
        overview_tab,
        data_quality_tab,
        performance_tab,
        regression_tab,
        risk_tab,
        allocation_tab,
        diversification_tab,
        stress_tab,
        learn_tab,
        about_tab,
    ) = st.tabs(
        [
            "Overview",
            "Data Quality",
            "Performance",
            "Regression",
            "Risk",
            "Allocation",
            "Diversification",
            "Stress Tests",
            "Learn",
            "About",
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with overview_tab:

        st.subheader("Portfolio Overview")

        column_1, column_2, column_3, column_4 = st.columns(4)

        with column_1:
            st.metric(
                label="Initial Investment",
                value=format_money(results["initial_investment"])
            )

        with column_2:
            st.metric(
                label="Final Portfolio Value",
                value=format_money(results["final_value"]),
                delta=f"{results['cumulative_return']:.2f}%"
            )

        with column_3:
            st.metric(
                label="Profit / Loss",
                value=format_money(results["profit_loss"], show_plus=True),
                delta=format_money(
                    results["profit_loss"],
                    show_plus=True,
                )
            )

        with column_4:
            st.metric(
                label="Maximum Drawdown",
                value=f"{results['maximum_drawdown']:.2f}%"
            )

        st.subheader("Key Performance and Risk Metrics")

        metric_column_1, metric_column_2, metric_column_3 = st.columns(3)

        with metric_column_1:
            st.metric(
                label="Cumulative Return",
                value=f"{results['cumulative_return']:.2f}%"
            )

        with metric_column_2:
            st.metric(
                label="Annualized Volatility",
                value=f"{results['annualized_volatility']:.2f}%"
            )

        with metric_column_3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{results['sharpe_ratio']:.2f}"
            )

        st.info(
            "Use the tabs above to inspect data quality, regression, "
            "performance, risk, allocation, stress testing and "
            "educational explanations."
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    with data_quality_tab:

        st.subheader("Data Quality and Preparation")

        st.write(
            "This panel documents the observations downloaded, the dates "
            "retained after strict alignment and potential return anomalies. "
            "No stock-price interpolation is performed."
        )

        quality_headline = results[
            "data_quality_headline"
        ]

        quality_column_1, quality_column_2, (
            quality_column_3
        ), quality_column_4 = st.columns(4)

        with quality_column_1:
            st.metric(
                label="Raw Market Dates",
                value=f"{quality_headline['raw_market_dates']:,}"
            )

        with quality_column_2:
            st.metric(
                label="Common Price Dates",
                value=f"{quality_headline['common_price_dates']:,}",
                delta=(
                    f"-{quality_headline['dates_removed']:,} dates"
                    if quality_headline[
                        "dates_removed"
                    ] > 0
                    else "No dates removed"
                ),
                delta_color="normal"
            )

        with quality_column_3:
            st.metric(
                label="Return Observations",
                value=(
                    f"{quality_headline['portfolio_return_observations']:,}"
                )
            )

        with quality_column_4:
            st.metric(
                label="Common-Date Retention",
                value=(
                    f"{quality_headline['data_retention_percent']:.2f}%"
                )
            )

        st.info(
            f"The requested sample produced "
            f"{quality_headline['raw_market_dates']:,} raw market-date rows. "
            f"After requiring a valid price for every portfolio asset, "
            f"{quality_headline['common_price_dates']:,} dates were retained. "
            f"The final portfolio analysis uses "
            f"{quality_headline['portfolio_return_observations']:,} daily "
            "return observations."
        )

        actual_period_column_1, actual_period_column_2, (
            actual_period_column_3
        ) = st.columns(3)

        with actual_period_column_1:
            st.metric(
                label="Actual Common Start Date",
                value=pd.Timestamp(
                    quality_headline[
                        "common_first_date"
                    ]
                ).strftime(
                    "%d/%m/%Y"
                )
            )

        with actual_period_column_2:
            st.metric(
                label="Actual Common End Date",
                value=pd.Timestamp(
                    quality_headline[
                        "common_last_date"
                    ]
                ).strftime(
                    "%d/%m/%Y"
                )
            )

        with actual_period_column_3:
            st.metric(
                label="Potential Anomalies Flagged",
                value=str(
                    quality_headline[
                        "potential_anomaly_count"
                    ]
                )
            )

        st.divider()

        st.subheader("Observation Alignment")

        alignment_table_display = (
            results[
                "data_quality_alignment_table"
            ]
            .copy()
            .round(
                {
                    "Retention from Previous Stage (%)": 2,
                }
            )
        )

        st.dataframe(
            alignment_table_display,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The first return observation is naturally lost because a "
            "return requires two consecutive prices."
        )

        st.divider()

        st.subheader("Data Coverage by Asset")

        asset_quality_display = (
            results[
                "data_quality_asset_table"
            ]
            .copy()
            .round(
                {
                    "Coverage (%)": 2,
                    "Largest Absolute Daily Return (%)": 2,
                }
            )
        )

        st.dataframe(
            asset_quality_display,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Potential Return Anomalies")

        anomalies = (
            results[
                "data_quality_anomalies"
            ]
            .copy()
        )

        if anomalies.empty:

            st.success(
                "No observations were flagged by the robust anomaly rules."
            )

        else:

            st.warning(
                f"{len(anomalies)} observations were flagged for review. "
                "They remain in the analysis and are not deleted "
                "automatically."
            )

            anomalies_display = (
                anomalies
                .sort_values(
                    by="Date",
                    ascending=False
                )
                .round(
                    {
                        "Return (%)": 2,
                        "Modified Z-Score": 2,
                    }
                )
            )

            st.dataframe(
                anomalies_display,
                use_container_width=True,
                hide_index=True
            )

        with st.expander(
            "Cleaning methodology and mathematical anomaly rule",
            expanded=False
        ):

            st.dataframe(
                results[
                    "data_quality_methodology_table"
                ],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**Robust modified z-score**")

            st.latex(
                r"""
                M_t =
                0.67448975
                \frac{
                    r_t - \operatorname{median}(r)
                }{
                    \operatorname{median}
                    \left(
                        \left|
                            r_t - \operatorname{median}(r)
                        \right|
                    \right)
                }
                """
            )

            st.write(
                "An observation is flagged when the absolute modified "
                "z-score exceeds 3.5, when the absolute simple return exceeds "
                "50%, or when the return is less than or equal to -100%. "
                "These rules identify observations for review; they do not "
                "prove that the observation is erroneous."
            )

            st.warning(
                "Finance Bro does not interpolate missing stock prices and "
                "does not winsorize or delete extreme returns automatically."
            )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    with performance_tab:

        st.subheader("Performance vs. Benchmark")

        benchmark_column_1, benchmark_column_2, benchmark_column_3 = (
            st.columns(3)
        )

        with benchmark_column_1:
            st.metric(
                label="Portfolio Return",
                value=f"{results['cumulative_return']:.2f}%"
            )

        with benchmark_column_2:
            st.metric(
                label=(
                    f"Benchmark Return "
                    f"({results['benchmark_ticker']})"
                ),
                value=f"{results['benchmark_cumulative_return']:.2f}%"
            )

        with benchmark_column_3:
            st.metric(
                label="Active Return",
                value=f"{results['active_return']:.2f}%"
            )

        (
            benchmark_risk_column_1,
            benchmark_risk_column_2,
            benchmark_risk_column_3,
        ) = st.columns(3)

        with benchmark_risk_column_1:
            st.metric(
                label="Portfolio Beta",
                value=f"{results['beta']:.3f}"
            )

            st.caption(
                "Estimated from the excess-return regression."
            )

        with benchmark_risk_column_2:
            st.metric(
                label="Annualized Alpha",
                value=f"{results['alpha_annualized']:.2f}%"
            )

            st.caption(
                "Arithmetic annualization of the daily regression intercept."
            )

        with benchmark_risk_column_3:
            st.metric(
                label="R-Squared",
                value=f"{results['r_squared'] * 100:.2f}%"
            )

            st.caption(
                "Share of portfolio excess-return variation explained "
                "by the benchmark excess return."
            )

        st.info(
            f"Risk-free source: {results['risk_free_source']}. "
            f"Average annual rate in the sample: "
            f"{results['risk_free_average_annual_rate_percent']:.3f}%. "
            "Official observations are aligned using the latest value "
            "available on or before each market date."
        )

        st.divider()

        benchmark_comparison_data = (
            results["benchmark_comparison"]
            .reset_index()
            .melt(
                id_vars="Date",
                var_name="Series",
                value_name="Cumulative Return (%)"
            )
        )

        benchmark_name_map = {
            "Portfolio Cumulative Return (%)": "Portfolio",
            "Benchmark Cumulative Return (%)": (
                f"Benchmark ({results['benchmark_ticker']})"
            ),
        }

        benchmark_comparison_data["Series"] = (
            benchmark_comparison_data["Series"]
            .replace(benchmark_name_map)
        )

        figure_benchmark_comparison = px.line(
            benchmark_comparison_data,
            x="Date",
            y="Cumulative Return (%)",
            color="Series",
            title="Portfolio vs. Benchmark Cumulative Return"
        )

        figure_benchmark_comparison.update_traces(
            line=dict(width=3),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}"
                "<br>Cumulative Return: %{y:.2f}%"
                "<extra></extra>"
            )
        )

        figure_benchmark_comparison.update_layout(
            legend_title="Series"
        )

        st.plotly_chart(
            figure_benchmark_comparison,
            use_container_width=True
        )

        st.caption(
            "Both series are rebased to the same starting point, "
            "making their historical performance directly comparable."
        )

        st.divider()

        st.subheader("Portfolio Value Over Time")

        portfolio_value_data = (
            results["portfolio_value"]
            .reset_index()
        )

        figure_portfolio_value = px.line(
            portfolio_value_data,
            x="Date",
            y=portfolio_value_column,
            title=(
                f"Portfolio Value Over Time "
                f"({portfolio_currency})"
            )
        )

        figure_portfolio_value.update_traces(
            line=dict(width=3),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}"
                f"<br>Portfolio Value: "
                f"{currency_symbol}%{{y:,.2f}}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(
            figure_portfolio_value,
            use_container_width=True
        )

        st.caption(
            "This chart shows how the selected initial investment "
            "would have evolved over the historical period."
        )

    # ========================================================
    # REGRESSION
    # ========================================================

    with regression_tab:

        st.subheader("Benchmark Regression Analysis")

        st.latex(
            r"R_{p,t}-R_{f,t}"
            r"=\alpha+\beta(R_{m,t}-R_{f,t})+\varepsilon_t"
        )

        regression_metric_1, regression_metric_2, (
            regression_metric_3
        ) = st.columns(3)

        with regression_metric_1:
            st.metric(
                label="Beta",
                value=f"{results['beta']:.3f}"
            )

        with regression_metric_2:
            st.metric(
                label="Annualized Alpha",
                value=f"{results['alpha_annualized']:.2f}%"
            )

        with regression_metric_3:
            st.metric(
                label="R-Squared",
                value=f"{results['r_squared'] * 100:.2f}%"
            )

        beta_lower, beta_upper = results[
            "beta_confidence_interval"
        ]

        alpha_significance_text = (
            "statistically different from zero at the 5% level"
            if results[
                "alpha_p_value_hac"
            ] < 0.05
            else "not statistically different from zero at the 5% level"
        )

        st.info(
            f"The estimated beta is {results['beta']:.3f}, with a robust "
            f"95% confidence interval from {beta_lower:.3f} to "
            f"{beta_upper:.3f}. The model R² is "
            f"{results['r_squared'] * 100:.2f}%, meaning that this proportion "
            "of the historical variation in portfolio excess returns is "
            "explained by the benchmark excess return in the linear model. "
            f"The estimated alpha is {alpha_significance_text}."
        )

        regression_plot_data = (
            results[
                "regression_plot_data"
            ]
            .copy()
        )

        regression_line_data = (
            regression_plot_data
            .sort_values(
                "Benchmark Excess Return"
            )
        )

        regression_figure = px.scatter(
            regression_plot_data,
            x="Benchmark Excess Return",
            y="Portfolio Excess Return",
            hover_data={
                "Date": "|%Y-%m-%d",
                "Benchmark Excess Return": ":.3f",
                "Portfolio Excess Return": ":.3f",
                "Residual": ":.3f",
            },
            title=(
                "Portfolio Excess Return vs. "
                "Benchmark Excess Return"
            )
        )

        regression_figure.add_trace(
            go.Scatter(
                x=regression_line_data[
                    "Benchmark Excess Return"
                ],
                y=regression_line_data[
                    "Fitted Portfolio Excess Return"
                ],
                mode="lines",
                name="OLS Regression Line",
                hovertemplate=(
                    "Benchmark Excess Return: %{x:.3f}%"
                    "<br>Fitted Portfolio Excess Return: %{y:.3f}%"
                    "<extra></extra>"
                )
            )
        )

        regression_figure.add_hline(
            y=0,
            line_dash="dot",
            opacity=0.5
        )

        regression_figure.add_vline(
            x=0,
            line_dash="dot",
            opacity=0.5
        )

        regression_figure.update_traces(
            marker=dict(
                size=7,
                opacity=0.65
            ),
            selector=dict(
                mode="markers"
            )
        )

        regression_figure.update_layout(
            xaxis_title="Benchmark Excess Return (%)",
            yaxis_title="Portfolio Excess Return (%)",
            legend_title="Model"
        )

        st.plotly_chart(
            regression_figure,
            use_container_width=True
        )

        st.caption(
            "Each point represents one common daily observation. The line "
            "is the OLS conditional-mean estimate; statistical inference "
            "uses Newey-West HAC standard errors."
        )

        with st.expander(
            "Advanced Regression Diagnostics",
            expanded=False
        ):

            diagnostic_column_1, diagnostic_column_2, (
                diagnostic_column_3
            ), diagnostic_column_4 = st.columns(4)

            with diagnostic_column_1:
                st.metric(
                    label="Observations",
                    value=str(
                        results[
                            "regression_observation_count"
                        ]
                    )
                )

            with diagnostic_column_2:
                st.metric(
                    label="Adjusted R-Squared",
                    value=(
                        f"{results['adjusted_r_squared'] * 100:.2f}%"
                    )
                )

            with diagnostic_column_3:
                st.metric(
                    label="Beta p-value (HAC)",
                    value=f"{results['beta_p_value_hac']:.4f}"
                )

            with diagnostic_column_4:
                st.metric(
                    label="Alpha p-value (HAC)",
                    value=f"{results['alpha_p_value_hac']:.4f}"
                )

            st.markdown("#### Robust Coefficient Estimates")

            coefficient_display = (
                results[
                    "regression_coefficients"
                ]
                .copy()
                .round(6)
            )

            st.dataframe(
                coefficient_display,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("#### Diagnostic Tests")

            diagnostics_display = (
                results[
                    "regression_diagnostics"
                ]
                .copy()
                .round(6)
            )

            st.dataframe(
                diagnostics_display,
                use_container_width=True,
                hide_index=True
            )

            diagnostic_messages = []

            if results[
                "ljung_box_p_value"
            ] < 0.05:
                diagnostic_messages.append(
                    "The Ljung-Box test indicates statistically significant "
                    "residual autocorrelation at the reported lag."
                )
            else:
                diagnostic_messages.append(
                    "The Ljung-Box test does not reject residual "
                    "independence at the 5% level."
                )

            if results[
                "breusch_pagan_p_value"
            ] < 0.05:
                diagnostic_messages.append(
                    "The Breusch-Pagan test indicates heteroscedasticity."
                )
            else:
                diagnostic_messages.append(
                    "The Breusch-Pagan test does not detect "
                    "heteroscedasticity at the 5% level."
                )

            if results[
                "jarque_bera_p_value"
            ] < 0.05:
                diagnostic_messages.append(
                    "The Jarque-Bera test rejects normal residuals."
                )
            else:
                diagnostic_messages.append(
                    "The Jarque-Bera test does not reject residual "
                    "normality at the 5% level."
                )

            if pd.notna(
                results[
                    "reset_p_value"
                ]
            ):

                if results[
                    "reset_p_value"
                ] < 0.05:
                    diagnostic_messages.append(
                        "The Ramsey RESET test suggests possible "
                        "functional-form misspecification."
                    )
                else:
                    diagnostic_messages.append(
                        "The Ramsey RESET test does not indicate "
                        "functional-form misspecification at the 5% level."
                    )

            for message in diagnostic_messages:
                st.write(
                    f"• {message}"
                )

            st.warning(
                "Diagnostic tests are sample-dependent. A non-significant "
                "result does not prove that an assumption is true, and a "
                "significant result does not automatically invalidate every "
                "use of the model. Newey-West HAC standard errors reduce the "
                "impact of heteroscedasticity and autocorrelation on "
                "coefficient inference, but they do not repair a misspecified "
                "economic model."
            )

            regression_residual_data = (
                results[
                    "regression_plot_data"
                ][
                    [
                        "Date",
                        "Residual",
                    ]
                ]
                .copy()
            )

            residual_figure = px.line(
                regression_residual_data,
                x="Date",
                y="Residual",
                title="Regression Residuals Through Time"
            )

            residual_figure.add_hline(
                y=0,
                line_dash="dot"
            )

            residual_figure.update_traces(
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}"
                    "<br>Residual: %{y:.3f}%"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                residual_figure,
                use_container_width=True
            )

    # ========================================================
    # RISK
    # ========================================================

    with risk_tab:

        st.subheader("Risk Overview")

        risk_summary_1, risk_summary_2 = st.columns(2)

        with risk_summary_1:
            st.metric(
                label="Annualized Volatility",
                value=f"{results['annualized_volatility']:.2f}%"
            )

        with risk_summary_2:
            st.metric(
                label="Maximum Drawdown",
                value=f"{results['maximum_drawdown']:.2f}%"
            )

        st.divider()

        st.subheader(
            f"Value at Risk — "
            f"{results['confidence_level']:.0f}% Confidence"
        )

        var_column_1, var_column_2 = st.columns(2)

        with var_column_1:

            st.markdown("#### Historical VaR")

            st.metric(
                label="Estimated 1-Day Loss",
                value=format_money(results["historical_var_money"])
            )

            st.caption(
                f"Equivalent to {results['historical_var_return']:.2f}% "
                "of the current portfolio value."
            )

            st.caption(
                "Based on the worst historical portfolio returns "
                "within the selected period."
            )

        with var_column_2:

            st.markdown("#### Parametric VaR")

            st.metric(
                label="Estimated 1-Day Loss",
                value=format_money(results["parametric_var_money"])
            )

            st.caption(
                f"Equivalent to {results['parametric_var_return']:.2f}% "
                "of the current portfolio value."
            )

            st.caption(
                "Estimated using the portfolio mean, volatility "
                "and a normal distribution."
            )

        st.divider()

        st.subheader(
            f"Expected Shortfall — "
            f"{results['confidence_level']:.0f}% Confidence"
        )

        es_column_1, es_column_2 = st.columns(2)

        with es_column_1:

            st.markdown("#### Historical Expected Shortfall")

            st.metric(
                label="Average 1-Day Loss Beyond VaR",
                value=format_money(results["historical_es_money"])
            )

            st.caption(
                f"Equivalent to {results['historical_es_return']:.2f}% "
                "of the current portfolio value."
            )

            st.caption(
                "Average loss observed in the worst historical "
                "returns beyond the Historical VaR threshold."
            )

        with es_column_2:

            st.markdown("#### Parametric Expected Shortfall")

            st.metric(
                label="Average 1-Day Loss Beyond VaR",
                value=format_money(results["parametric_es_money"])
            )

            st.caption(
                f"Equivalent to {results['parametric_es_return']:.2f}% "
                "of the current portfolio value."
            )

            st.caption(
                "Estimated average loss beyond Parametric VaR "
                "assuming normally distributed returns."
            )

        st.divider()

        st.subheader("VaR and Expected Shortfall Comparison")

        risk_comparison_data = pd.DataFrame({
            "Metric": [
                "Historical VaR",
                "Parametric VaR",
                "Historical ES",
                "Parametric ES",
            ],
            "Loss (%)": [
                results["historical_var_return"],
                results["parametric_var_return"],
                results["historical_es_return"],
                results["parametric_es_return"],
            ],
            "Loss": [
                results["historical_var_money"],
                results["parametric_var_money"],
                results["historical_es_money"],
                results["parametric_es_money"],
            ],
        })

        figure_risk_comparison = px.bar(
            risk_comparison_data,
            x="Metric",
            y="Loss",
            text="Loss",
            title="Estimated 1-Day Loss by Risk Measure",
            hover_data={
                "Loss (%)": ":.2f",
                "Loss": ":,.2f",
            },
        )

        figure_risk_comparison.update_traces(
            texttemplate=(
                f"{currency_symbol}%{{text:,.2f}}"
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b>"
                f"<br>Estimated Loss: "
                f"{currency_symbol}%{{y:,.2f}}"
                "<br>Portfolio Share: %{customdata[0]:.2f}%"
                "<extra></extra>"
            ),
        )

        figure_risk_comparison.update_layout(
            yaxis_title=(
                f"Estimated Loss ({portfolio_currency})"
            ),
            xaxis_title="Risk Measure",
        )

        st.plotly_chart(
            figure_risk_comparison,
            use_container_width=True,
        )

        st.caption(
            "Expected Shortfall should normally be greater than VaR "
            "because it measures the average loss beyond the VaR threshold."
        )

        st.divider()

        st.subheader("Daily Return Distribution")

        return_distribution_data = pd.DataFrame({
            "Daily Return (%)": (
                results["portfolio_returns"]
                * 100
            )
        })

        histogram_values = (
            return_distribution_data["Daily Return (%)"]
            .dropna()
        )

        histogram_counts, _ = np.histogram(
            histogram_values,
            bins=50
        )

        y_max = max(
            float(histogram_counts.max()) * 1.15,
            1.0
        )

        figure_return_distribution = go.Figure()

        figure_return_distribution.add_trace(
            go.Histogram(
                x=histogram_values,
                nbinsx=50,
                name="Daily Returns",
                marker=dict(
                    color="#2F80ED"
                ),
                opacity=0.82,
                hovertemplate=(
                    "Daily return: %{x:.2f}%"
                    "<br>Number of days: %{y}"
                    "<extra></extra>"
                )
            )
        )

        figure_return_distribution.add_vrect(
            x0=float(histogram_values.min()),
            x1=-results["historical_var_return"],
            fillcolor="rgba(220, 53, 69, 0.12)",
            line_width=0,
            layer="below"
        )

        figure_return_distribution.add_trace(
            go.Scatter(
                x=[
                    -results["historical_var_return"],
                    -results["historical_var_return"]
                ],
                y=[0, y_max],
                mode="lines",
                name="Historical VaR",
                line=dict(
                    color="#D62728",
                    width=3,
                    dash="dash"
                ),
                hovertemplate=(
                    "Historical VaR"
                    "<br>Threshold: %{x:.2f}%"
                    "<extra></extra>"
                )
            )
        )

        figure_return_distribution.add_trace(
            go.Scatter(
                x=[
                    -results["parametric_var_return"],
                    -results["parametric_var_return"]
                ],
                y=[0, y_max],
                mode="lines",
                name="Parametric VaR",
                line=dict(
                    color="#FF7F0E",
                    width=3,
                    dash="dash"
                ),
                hovertemplate=(
                    "Parametric VaR"
                    "<br>Threshold: %{x:.2f}%"
                    "<extra></extra>"
                )
            )
        )

        figure_return_distribution.add_trace(
            go.Scatter(
                x=[
                    -results["historical_es_return"],
                    -results["historical_es_return"]
                ],
                y=[0, y_max],
                mode="lines",
                name="Historical ES",
                line=dict(
                    color="#9467BD",
                    width=3,
                    dash="dot"
                ),
                hovertemplate=(
                    "Historical ES"
                    "<br>Average tail loss: %{x:.2f}%"
                    "<extra></extra>"
                )
            )
        )

        figure_return_distribution.add_trace(
            go.Scatter(
                x=[
                    -results["parametric_es_return"],
                    -results["parametric_es_return"]
                ],
                y=[0, y_max],
                mode="lines",
                name="Parametric ES",
                line=dict(
                    color="#17A2B8",
                    width=3,
                    dash="dot"
                ),
                hovertemplate=(
                    "Parametric ES"
                    "<br>Average tail loss: %{x:.2f}%"
                    "<extra></extra>"
                )
            )
        )

        figure_return_distribution.update_layout(
            title="Distribution of Daily Portfolio Returns",
            xaxis_title="Daily Portfolio Return (%)",
            yaxis_title="Number of Days",
            bargap=0.05,
            legend=dict(
                title="Risk Measures",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
            margin=dict(
                t=120
            )
        )

        st.plotly_chart(
            figure_return_distribution,
            use_container_width=True
        )

        st.caption(
            "The histogram shows daily portfolio returns. "
            "Dashed lines represent Value at Risk, dotted lines represent "
            "Expected Shortfall, and the shaded area highlights the "
            "historical loss tail."
        )

        st.divider()

        st.subheader(
            f"{rolling_window}-Day Rolling Volatility"
        )

        rolling_volatility_data = (
            results["rolling_volatility"]
            .reset_index()
        )

        figure_rolling_volatility = px.line(
            rolling_volatility_data,
            x="Date",
            y="Annualized Rolling Volatility (%)",
            title=(
                f"{rolling_window}-Day "
                f"Annualized Rolling Volatility"
            )
        )

        st.plotly_chart(
            figure_rolling_volatility,
            use_container_width=True
        )

    # ========================================================
    # ALLOCATION
    # ========================================================

    with allocation_tab:

        st.subheader("Portfolio Allocation")

        allocation_column_1, allocation_column_2 = st.columns([1.1, 1])

        with allocation_column_1:

            allocation_table_display = (
                results["allocation_table"]
                .copy()
                .rename(
                    columns={
                        "Initial Invested Value":
                            f"Initial Invested Value "
                            f"({portfolio_currency})",
                        "Current Estimated Value":
                            f"Current Estimated Value "
                            f"({portfolio_currency})",
                    }
                )
                .round(
                    {
                        f"Initial Invested Value "
                        f"({portfolio_currency})": 2,
                        f"Current Estimated Value "
                        f"({portfolio_currency})": 2,
                        "Weight (%)": 2,
                    }
                )
            )

            st.dataframe(
                allocation_table_display,
                use_container_width=True,
                hide_index=True
            )

        with allocation_column_2:

            figure_allocation = px.pie(
                results["allocation_table"],
                names="Asset",
                values=initial_invested_value_column,
                hole=0.55,
                title="Portfolio Allocation"
            )

            figure_allocation.update_traces(
                textposition="inside",
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b>"
                    "<br>Weight: %{percent}"
                    f"<br>Initial Value: "
                    f"{currency_symbol}%{{value:,.2f}}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                figure_allocation,
                use_container_width=True
            )

        st.divider()

        st.subheader("Initial Portfolio Construction")

        st.caption(
            "This table reconstructs the theoretical initial purchase using "
            "fractional shares, the first available unadjusted closing price "
            "on or after the selected start date, and ECB daily reference "
            "exchange rates whenever an asset currency differs from the "
            "selected portfolio currency."
        )

        initial_construction_display = (
            results[
                "initial_portfolio_construction"
            ]
            .copy()
            .rename(
                columns={
                    "Entry Price (Portfolio Currency)":
                        f"Entry Price ({portfolio_currency})",
                    "Amount Invested (Portfolio Currency)":
                        f"Amount Invested ({portfolio_currency})",
                    "ECB Cross Rate (Portfolio per Local)":
                        f"ECB Cross Rate "
                        f"({portfolio_currency} per Local Currency)",
                }
            )
            .round(
                {
                    "Entry Price (Local)": 4,
                    f"ECB Cross Rate "
                    f"({portfolio_currency} per Local Currency)": 6,
                    f"Entry Price ({portfolio_currency})": 4,
                    "Target Weight (%)": 2,
                    f"Amount Invested ({portfolio_currency})": 2,
                    "Fractional Shares Purchased": 6,
                }
            )
        )

        construction_metric_1, construction_metric_2, (
            construction_metric_3
        ) = st.columns(3)

        with construction_metric_1:
            st.metric(
                label="Purchase Method",
                value="Fractional Shares"
            )

        with construction_metric_2:
            st.metric(
                label="Portfolio Currency",
                value=(
                    f"{portfolio_currency} "
                    f"({currency_symbol})"
                )
            )

        with construction_metric_3:
            converted_assets = int(
                (
                    initial_construction_display[
                        "Trading Currency"
                    ]
                    != portfolio_currency
                ).sum()
            )

            st.metric(
                label="Assets Converted",
                value=str(
                    converted_assets
                )
            )

        st.dataframe(
            initial_construction_display,
            use_container_width=True,
            hide_index=True
        )

        with st.expander(
            "How entry prices and exchange rates are selected"
        ):

            st.markdown(
                r"""
                **Entry price**

                The app uses the first available unadjusted closing price on
                or after the requested start date. When the selected date is
                a weekend or market holiday, the next available trading day
                is used.

                **Currency conversion**

                ECB rates are first expressed as currency units per one euro.
                The app then calculates a cross rate from the asset's trading
                currency into the selected portfolio currency.

                **Professional formula**
                """
            )

            st.latex(
                r"X_{B/L,t}"
                r"=\frac{X_{B/EUR,t}}{X_{L/EUR,t}}"
            )

            st.latex(
                r"P_{i,t}^{B}"
                r"=P_{i,t}^{L}X_{B/L,t}"
            )

            st.latex(
                r"N_{i,0}"
                r"=\frac{V_0w_i}{P_{i,0}^{B}}"
            )

            st.markdown(
                r"""
                where $B$ is the selected portfolio currency and $L$ is the
                asset's local trading currency.

                Reference rates are used for educational reconstruction.
                They may differ from the rate, spread and fees applied by a
                bank or broker.
                """
            )

        st.info(
            "The performance calculations are also carried out in the "
            f"selected portfolio currency ({portfolio_currency}). This means "
            "that returns on foreign assets include the historical effect of "
            "exchange-rate movements."
        )

        st.warning(
            "The initial-purchase table represents a theoretical fractional-"
            "share transaction. The portfolio-return model continues to use "
            "constant portfolio weights, so it should be interpreted as a "
            "periodically rebalanced portfolio rather than a strict "
            "buy-and-hold share ledger."
        )

        st.divider()

        st.subheader("Asset Contribution Analysis")

        st.caption(
            "This section shows how much each asset contributes to the "
            "portfolio's annualized return and total volatility."
        )

        contribution_table_display = (
            results["contribution_table"]
            .copy()
            .round(
                {
                    "Weight (%)": 2,
                    "Annualized Asset Return (%)": 2,
                    "Return Contribution (p.p.)": 2,
                    "Risk Contribution (p.p.)": 2,
                    "Risk Contribution (%)": 2,
                }
            )
        )

        st.dataframe(
            contribution_table_display,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        contribution_chart_column_1, contribution_chart_column_2 = (
            st.columns(2)
        )

        with contribution_chart_column_1:

            figure_return_contribution = px.bar(
                contribution_table_display,
                x="Asset",
                y="Return Contribution (p.p.)",
                text="Return Contribution (p.p.)",
                title="Contribution to Annualized Return"
            )

            figure_return_contribution.update_traces(
                texttemplate="%{text:.2f} p.p.",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b>"
                    "<br>Return Contribution: %{y:.2f} p.p."
                    "<extra></extra>"
                )
            )

            figure_return_contribution.update_layout(
                xaxis_title="Asset",
                yaxis_title="Return Contribution (percentage points)"
            )

            st.plotly_chart(
                figure_return_contribution,
                use_container_width=True
            )

        with contribution_chart_column_2:

            figure_risk_contribution = px.bar(
                contribution_table_display,
                x="Asset",
                y="Risk Contribution (%)",
                text="Risk Contribution (%)",
                title="Contribution to Portfolio Risk"
            )

            figure_risk_contribution.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b>"
                    "<br>Risk Contribution: %{y:.2f}%"
                    "<extra></extra>"
                )
            )

            figure_risk_contribution.update_layout(
                xaxis_title="Asset",
                yaxis_title="Share of Total Portfolio Risk (%)"
            )

            st.plotly_chart(
                figure_risk_contribution,
                use_container_width=True
            )

        st.info(
            "A large portfolio weight does not always mean a large risk "
            "contribution. Correlation and volatility also determine how "
            "much risk each asset adds to the portfolio."
        )

    # ========================================================
    # DIVERSIFICATION
    # ========================================================

    with diversification_tab:

        st.subheader("Portfolio Diversification")

        st.caption(
            "Diversification depends on how assets move relative to one "
            "another. Lower correlations can help reduce portfolio risk."
        )

        diversification_metric_1, diversification_metric_2, (
            diversification_metric_3
        ) = st.columns(3)

        with diversification_metric_1:
            st.metric(
                label="Portfolio vs. Benchmark Correlation",
                value=(
                    f"{results['portfolio_benchmark_correlation']:.2f}"
                )
            )

        with diversification_metric_2:
            average_correlation_column = (
                "Average Correlation with Other Assets"
            )

            absolute_correlation_table = (
                results["diversification_summary"][
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
            else:
                absolute_correlation_table[
                    "Absolute Average Correlation"
                ] = (
                    absolute_correlation_table[
                        average_correlation_column
                    ].abs()
                )

                lowest_absolute_correlation_asset = str(
                    absolute_correlation_table.loc[
                        absolute_correlation_table[
                            "Absolute Average Correlation"
                        ].idxmin(),
                        "Asset",
                    ]
                )

            st.metric(
                label=(
                    "Lowest Absolute Average "
                    "Correlation Asset"
                ),
                value=lowest_absolute_correlation_asset,
            )

        with diversification_metric_3:
            average_correlation = results[
                "average_portfolio_correlation"
            ]

            average_correlation_text = (
                "N/A"
                if pd.isna(average_correlation)
                else f"{average_correlation:.2f}"
            )

            st.metric(
                label="Average Correlation Between Assets",
                value=average_correlation_text
            )

        st.divider()

        st.subheader("Asset Correlation Matrix")

        asset_correlation_matrix = (
            results["asset_correlation_matrix"]
            .round(2)
        )

        figure_correlation_heatmap = px.imshow(
            asset_correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )

        figure_correlation_heatmap.update_layout(
            xaxis_title="Asset",
            yaxis_title="Asset",
            coloraxis_colorbar_title="Correlation"
        )

        st.plotly_chart(
            figure_correlation_heatmap,
            use_container_width=True
        )

        st.caption(
            "Values close to 1 indicate that assets tended to move in the "
            "same direction. Values close to 0 indicate weak linear "
            "co-movement. Negative values indicate opposite movements."
        )

        st.divider()

        st.subheader(
            f"Correlation with Benchmark "
            f"({results['benchmark_ticker']})"
        )

        diversification_summary_display = (
            results["diversification_summary"]
            .copy()
            .round(2)
        )

        st.dataframe(
            diversification_summary_display,
            use_container_width=True,
            hide_index=True
        )

        benchmark_correlation_column = (
            f"Correlation with {results['benchmark_ticker']}"
        )

        figure_benchmark_correlation = px.bar(
            diversification_summary_display,
            x="Asset",
            y=benchmark_correlation_column,
            text=benchmark_correlation_column,
            title=(
                f"Asset Correlation with "
                f"{results['benchmark_ticker']}"
            )
        )

        figure_benchmark_correlation.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b>"
                "<br>Correlation: %{y:.2f}"
                "<extra></extra>"
            )
        )

        figure_benchmark_correlation.update_layout(
            xaxis_title="Asset",
            yaxis_title="Correlation",
            yaxis=dict(
                range=[-1, 1]
            )
        )

        st.plotly_chart(
            figure_benchmark_correlation,
            use_container_width=True
        )

        st.info(
            "Correlation measures how closely two return series moved "
            "together. It does not imply causation and can change over time."
        )

    # ========================================================
    # STRESS TESTS
    # ========================================================

    with stress_tab:

        st.subheader("Professional Portfolio Stress Testing")

        st.caption(
            "Compare a beta-based market correction with a fully "
            "customizable asset-by-asset stress scenario."
        )

        stress_summary_display = (
            results["stress_test_summary"]
            .copy()
            .rename(
                columns={
                    "Portfolio Change":
                        f"Portfolio Change ({portfolio_currency})",
                    "Stressed Portfolio Value":
                        f"Stressed Portfolio Value "
                        f"({portfolio_currency})",
                }
            )
            .round(
                {
                    "Portfolio Change (%)": 2,
                    f"Portfolio Change ({portfolio_currency})": 2,
                    f"Stressed Portfolio Value "
                    f"({portfolio_currency})": 2,
                }
            )
        )

        (
            stress_comparison_tab,
            market_correction_tab,
            custom_scenario_tab,
        ) = st.tabs(
            [
                "Scenario Comparison",
                "Market Correction",
                "Custom Scenario",
            ]
        )

        # ----------------------------------------------------
        # SCENARIO COMPARISON
        # ----------------------------------------------------

        with stress_comparison_tab:

            st.subheader("Scenario Comparison")

            st.dataframe(
                stress_summary_display,
                use_container_width=True,
                hide_index=True
            )

            scenario_value_comparison = pd.DataFrame({
                "Scenario": [
                    "Current Portfolio",
                    results["market_stress_summary"]["Scenario"],
                    results["custom_stress_summary"]["Scenario"],
                ],
                f"Portfolio Value ({portfolio_currency})": [
                    results["final_value"],
                    results["market_stress_summary"][
                        "Stressed Portfolio Value"
                    ],
                    results["custom_stress_summary"][
                        "Stressed Portfolio Value"
                    ],
                ],
            })

            figure_scenario_values = px.bar(
                scenario_value_comparison,
                x="Scenario",
                y=f"Portfolio Value ({portfolio_currency})",
                text=f"Portfolio Value ({portfolio_currency})",
                title="Current vs. Stressed Portfolio Value"
            )

            figure_scenario_values.update_traces(
                texttemplate=f"{currency_symbol}%{{text:,.2f}}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b>"
                    f"<br>Portfolio Value: {currency_symbol}%{{y:,.2f}}"
                    "<extra></extra>"
                )
            )

            figure_scenario_values.update_layout(
                xaxis_title="Scenario",
                yaxis_title="Portfolio Value"
            )

            st.plotly_chart(
                figure_scenario_values,
                use_container_width=True
            )

            scenario_impact_data = (
                stress_summary_display[
                    [
                        "Scenario",
                        "Portfolio Change (%)",
                        f"Portfolio Change ({portfolio_currency})",
                    ]
                ]
            )

            figure_scenario_impact = px.bar(
                scenario_impact_data,
                x="Scenario",
                y="Portfolio Change (%)",
                text="Portfolio Change (%)",
                title="Estimated Portfolio Impact"
            )

            figure_scenario_impact.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b>"
                    "<br>Portfolio Change: %{y:.2f}%"
                    "<extra></extra>"
                )
            )

            figure_scenario_impact.update_layout(
                xaxis_title="Scenario",
                yaxis_title="Portfolio Change (%)"
            )

            st.plotly_chart(
                figure_scenario_impact,
                use_container_width=True
            )

        # ----------------------------------------------------
        # REUSABLE SCENARIO DISPLAY
        # ----------------------------------------------------

        def display_stress_scenario(
            scenario_summary,
            scenario_detail,
            show_beta_chart=False,
        ):

            scenario_detail_display = (
                scenario_detail
                .copy()
                .rename(
                    columns={
                        "Current Position Value":
                            f"Current Position Value "
                            f"({portfolio_currency})",
                        "Value Change":
                            f"Value Change ({portfolio_currency})",
                        "Stressed Position Value":
                            f"Stressed Position Value "
                            f"({portfolio_currency})",
                    }
                )
                .round(
                    {
                        "Weight (%)": 2,
                        "Beta": 2,
                        "Shock (%)": 2,
                        "Portfolio Impact (p.p.)": 2,
                        f"Current Position Value "
                        f"({portfolio_currency})": 2,
                        f"Value Change ({portfolio_currency})": 2,
                        "Loss Contribution (%)": 2,
                        f"Stressed Position Value "
                        f"({portfolio_currency})": 2,
                    }
                )
            )

            current_value = results["final_value"]

            stressed_value = scenario_summary[
                "Stressed Portfolio Value"
            ]

            portfolio_change_money = scenario_summary[
                "Portfolio Change"
            ]

            portfolio_change_percent = scenario_summary[
                "Portfolio Change (%)"
            ]

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)

            with metric_1:
                st.metric(
                    label="Current Portfolio Value",
                    value=format_money(current_value)
                )

            with metric_2:
                st.metric(
                    label="Stressed Portfolio Value",
                    value=format_money(stressed_value)
                )

            with metric_3:
                st.metric(
                    label="Portfolio Change",
                    value=f"{portfolio_change_percent:.2f}%",
                    delta=f"{portfolio_change_percent:.2f}%",
                    delta_color="normal"
                )

            with metric_4:

                profit_loss_text = format_money(
                    portfolio_change_money,
                    show_plus=True,
                )

                st.metric(
                    label="Estimated Profit / Loss",
                    value=profit_loss_text,
                    delta=profit_loss_text,
                    delta_color="normal"
                )

            st.info(
                "Largest loss contributor: "
                f"**{scenario_summary['Largest Loss Contributor']}**  "
                "— Most resilient asset: "
                f"**{scenario_summary['Most Resilient Asset']}**"
            )

            st.divider()

            st.subheader("Results by Asset")

            st.dataframe(
                scenario_detail_display,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Portfolio Impact (p.p.) shows how many percentage points "
                "each asset adds to the total portfolio change. Loss "
                "Contribution (%) divides the scenario's gross losses "
                "between the loss-making positions."
            )

            st.divider()

            chart_column_1, chart_column_2 = st.columns(2)

            with chart_column_1:

                if show_beta_chart:

                    figure_first_chart = px.bar(
                        scenario_detail_display,
                        x="Asset",
                        y="Beta",
                        text="Beta",
                        title=(
                            f"Asset Beta vs. "
                            f"{results['benchmark_ticker']}"
                        )
                    )

                    figure_first_chart.update_traces(
                        texttemplate="%{text:.2f}",
                        textposition="outside",
                        hovertemplate=(
                            "<b>%{x}</b>"
                            "<br>Beta: %{y:.2f}"
                            "<extra></extra>"
                        )
                    )

                    figure_first_chart.update_layout(
                        xaxis_title="Asset",
                        yaxis_title="Beta"
                    )

                else:

                    figure_first_chart = px.bar(
                        scenario_detail_display,
                        x="Asset",
                        y="Shock (%)",
                        text="Shock (%)",
                        title="Selected Custom Shock by Asset"
                    )

                    figure_first_chart.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside",
                        hovertemplate=(
                            "<b>%{x}</b>"
                            "<br>Selected Shock: %{y:.2f}%"
                            "<extra></extra>"
                        )
                    )

                    figure_first_chart.update_layout(
                        xaxis_title="Asset",
                        yaxis_title="Shock (%)"
                    )

                st.plotly_chart(
                    figure_first_chart,
                    use_container_width=True
                )

            with chart_column_2:

                figure_asset_impact = px.bar(
                    scenario_detail_display,
                    x="Asset",
                    y=f"Value Change ({portfolio_currency})",
                    text=f"Value Change ({portfolio_currency})",
                    title="Estimated Value Change by Asset"
                )

                figure_asset_impact.update_traces(
                    texttemplate=f"{currency_symbol}%{{text:,.2f}}",
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b>"
                        f"<br>Estimated Change: {currency_symbol}%{{y:,.2f}}"
                        "<extra></extra>"
                    )
                )

                figure_asset_impact.update_layout(
                    xaxis_title="Asset",
                    yaxis_title="Value Change"
                )

                st.plotly_chart(
                    figure_asset_impact,
                    use_container_width=True
                )

            st.divider()

            current_vs_stressed_positions = (
                scenario_detail_display[
                    [
                        "Asset",
                        f"Current Position Value ({portfolio_currency})",
                        f"Stressed Position Value ({portfolio_currency})",
                    ]
                ]
                .melt(
                    id_vars="Asset",
                    var_name="Portfolio State",
                    value_name=f"Position Value ({portfolio_currency})"
                )
            )

            figure_position_comparison = px.bar(
                current_vs_stressed_positions,
                x="Asset",
                y=f"Position Value ({portfolio_currency})",
                color="Portfolio State",
                barmode="group",
                title="Current vs. Stressed Position Values"
            )

            figure_position_comparison.update_traces(
                hovertemplate=(
                    "<b>%{x}</b>"
                    f"<br>Position Value: {currency_symbol}%{{y:,.2f}}"
                    "<extra></extra>"
                )
            )

            figure_position_comparison.update_layout(
                xaxis_title="Asset",
                yaxis_title="Position Value",
                legend_title="Portfolio State"
            )

            st.plotly_chart(
                figure_position_comparison,
                use_container_width=True
            )

            st.divider()

            waterfall_assets = (
                scenario_detail["Asset"].tolist()
            )

            waterfall_changes = (
                scenario_detail["Value Change"].tolist()
            )

            figure_stress_waterfall = go.Figure(
                go.Waterfall(
                    name=scenario_summary["Scenario"],
                    orientation="v",
                    measure=(
                        ["absolute"]
                        + ["relative"] * len(waterfall_assets)
                        + ["total"]
                    ),
                    x=(
                        ["Current Portfolio"]
                        + waterfall_assets
                        + ["Stressed Portfolio"]
                    ),
                    y=(
                        [current_value]
                        + waterfall_changes
                        + [0]
                    ),
                    text=(
                        [format_money(current_value)]
                        + [
                            format_money(value_change, show_plus=True)
                            for value_change in waterfall_changes
                        ]
                        + [format_money(stressed_value)]
                    ),
                    textposition="outside",
                    connector=dict(
                        line=dict(
                            width=1
                        )
                    ),
                    hovertemplate=(
                        "<b>%{x}</b>"
                        f"<br>Value / Change: {currency_symbol}%{{y:,.2f}}"
                        "<extra></extra>"
                    )
                )
            )

            figure_stress_waterfall.update_layout(
                title="Portfolio Value Waterfall",
                xaxis_title="Portfolio Component",
                yaxis_title=(f"Portfolio Value / Change ({portfolio_currency})"),
                showlegend=False
            )

            st.plotly_chart(
                figure_stress_waterfall,
                use_container_width=True
            )

        # ----------------------------------------------------
        # MARKET CORRECTION
        # ----------------------------------------------------

        with market_correction_tab:

            st.subheader("Beta-Based Market Correction")

            st.write(
                f"The selected benchmark shock is "
                f"**{results['benchmark_stress_shock']:.2f}%** for "
                f"**{results['benchmark_ticker']}**. Each asset's estimated "
                "shock is calculated as beta multiplied by the benchmark "
                "shock."
            )

            display_stress_scenario(
                scenario_summary=results[
                    "market_stress_summary"
                ],
                scenario_detail=results[
                    "market_stress_detail"
                ],
                show_beta_chart=True,
            )

            st.warning(
                "Beta is estimated from historical linear relationships. "
                "During real market crises, betas, correlations and "
                "liquidity conditions can change substantially."
            )

        # ----------------------------------------------------
        # CUSTOM SCENARIO
        # ----------------------------------------------------

        with custom_scenario_tab:

            st.subheader("Custom Asset-by-Asset Scenario")

            st.write(
                "The shocks shown below were selected individually in "
                "the sidebar. Negative values represent price falls and "
                "positive values represent price increases."
            )

            display_stress_scenario(
                scenario_summary=results[
                    "custom_stress_summary"
                ],
                scenario_detail=results[
                    "custom_stress_detail"
                ],
                show_beta_chart=False,
            )

            st.warning(
                "This scenario assumes immediate one-period price shocks. "
                "It does not model recovery, rebalancing, taxes, liquidity "
                "constraints or second-round market effects."
            )

    # ========================================================
    # LEARN
    # ========================================================

    with learn_tab:

        render_learning_centre(
            results=results,
            portfolio_currency=portfolio_currency,
            currency_symbol=currency_symbol,
            rolling_window=int(rolling_window),
        )

    # ========================================================
    # ABOUT
    # ========================================================

    with about_tab:

        st.subheader("The Finance Bro Story")

        st.markdown(
            """
            **Finance Bro began as a personal summer project with one
            simple objective: learn Python by building something directly
            connected to finance.**

            Instead of studying programming only through isolated
            exercises, the project started with a practical challenge:
            creating a portfolio analysis tool capable of transforming
            market data into useful, visual and understandable financial
            information.

            The first version focused on the foundations — prices, daily
            returns, portfolio weights and cumulative performance. From
            there, Finance Bro evolved step by step. Volatility, drawdown,
            Value at Risk, Expected Shortfall, benchmark comparison, beta
            and alpha were gradually added as the project became more
            ambitious.

            What started as a learning exercise is now becoming a complete
            portfolio analytics and financial education platform.
            """
        )

        st.divider()

        st.subheader("Why Finance Bro Exists")

        st.markdown(
            """
            Financial analysis can often feel unnecessarily complex.
            Important ideas are frequently hidden behind technical language,
            formulas and professional tools that are difficult for beginners
            to understand.

            Finance Bro was created to make those concepts clearer.

            The platform does not only display a number. It aims to explain
            what that number means, how it should be interpreted and where
            its limitations are.
            """
        )

        st.divider()

        st.subheader("The Three Main Objectives")

        objective_1, objective_2, objective_3 = st.columns(3)

        with objective_1:
            st.markdown("### Analyze Performance")
            st.write(
                "Help users understand how their portfolios performed "
                "over time and how that performance compares with a "
                "selected market benchmark."
            )

        with objective_2:
            st.markdown("### Understand Risk")
            st.write(
                "Transform volatility, drawdown, VaR, Expected Shortfall "
                "and other risk measures into clear visual and monetary "
                "insights."
            )

        with objective_3:
            st.markdown("### Learn Finance")
            st.write(
                "Combine financial analysis with simple explanations so "
                "users can understand the logic behind every metric."
            )

        st.divider()

        st.subheader("The Purpose")

        st.markdown(
            """
            Finance Bro is not designed to predict the future or tell users
            which assets they should buy.

            Its purpose is to help users make more informed decisions by
            understanding the historical performance, behaviour and risk of
            their portfolios.
            """
        )

        st.divider()

        st.subheader("The Vision")

        st.markdown(
            """
            The long-term vision is to build an accessible and complete
            portfolio analytics platform that combines professional
            financial analysis with education and clear visual explanations.

            Future development can include individual stock analysis,
            deeper financial-statement tools, reusable scenario libraries,
            portfolio optimization and additional features that make financial
            analysis even more useful and approachable.
            """
        )

        st.info(
            "Built with curiosity. Improved through learning. "
            "Designed for better financial decisions."
        )

    render_brand_footer()




