import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Overview Dashboard", layout="wide")

# =========================
# OPTIONS
# =========================
BENCHMARK = "SPY"
PRICE_HISTORY_PERIOD = "2y"

# If you truly want to HIDE the sparkline column entirely, set this to False
SHOW_SPARKLINE = True

def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# =========================
# CSS
# =========================
CSS = """
<style>
.block-container {max-width: 1750px; padding-top: 1.0rem; padding-bottom: 2rem;}
.section-title {font-weight: 900; font-size: 1.15rem; margin: 0.65rem 0 0.4rem 0;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.12); margin: 14px 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 10px;
}
[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}

/* Try to allow wrapping so long group names are readable */
[data-testid="stDataFrame"] div[role="gridcell"] > div {
  white-space: normal !important;
  line-height: 1.15 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# YOUR TICKER LIST (1 per line)
# ============================================================
TICKERS_RAW = r"""
SPY
QQQ
DIA
IWM
RSP
QQQE
EDOW
MDY
IWN
IWO
XLC
XLY
XLP
XLE
XLF
XLV
XLI
XLB
XLRE
XLK
XLU
SOXX
SMH
XSD
IGV
XSW
IGM
VGT
XT
CIBR
BOTZ
AIQ
XTL
VOX
FCOM
FDN
SOCL
XRT
IBUY
CARZ
IDRV
ITB
XHB
PEJ
VDC
FSTA
KXI
PBJ
VPU
FUTY
IDU
IYE
VDE
XOP
IEO
OIH
IXC
IBB
XBI
PBE
IDNA
IHI
XHE
XHS
XPH
FHLC
PINK
KBE
KRE
IAT
KIE
IAI
KCE
IYG
VFH
ITA
PPA
XAR
IYT
XTN
VIS
FIDU
XME
GDX
SIL
SLX
PICK
VAW
VNQ
IYR
REET
SRVR
HOMZ
SCHH
NETL
GLD
SLV
UNG
USO
DBA
CORN
DBB
PALL
URA
UGA
CPER
CATL
HOGS
SOYB
WEAT
DBC
IEMG
EUE
C6E
FEZ
E40
DAX
ISF
FXI
EEM
EWJ
EWU
EWZ
EWG
EWT
EWH
EWI
EWW
PIN
IDX
EWY
EWA
EWM
EWS
EWC
EWP
EZA
EWL
UUP
FXE
FXY
FXB
FXA
FXF
FXC
IBIT
ETHA
TLT
BND
SHY
IEF
SGOV
IEI
TLH
AGG
MUB
GOVT
IGSB
USHY
IGIB
""".strip()

def parse_ticker_list(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        t = ln.strip().upper()
        if t:
            out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq

ALL_TICKERS = parse_ticker_list(TICKERS_RAW)
ALL_TICKERS_SET = set(ALL_TICKERS)

# Excel structure
MAJOR = ALL_TICKERS[:10]
SECTORS = ALL_TICKERS[10:21]

# ===========================
# SUB-SECTOR / INDUSTRY GROUP MAP (EXCEL-LIKE)
# ===========================
SUBSECTOR_LEFT = {
    "Semiconductors": ["SOXX","SMH","XSD"],
    "Software / Cloud / Broad Tech": ["IGV","XSW","IGM","VGT","XT"],
    "Cyber Security": ["CIBR"],
    "AI / Robotics / Automation": ["BOTZ","AIQ"],
    "Telecom & Communication": ["XTL","VOX","FCOM"],
    "Internet / Media / Social": ["FDN","SOCL"],
    "Retail": ["XRT","IBUY"],
    "Autos / EV": ["IDRV","CARZ"],
    "Homebuilders / Construction": ["ITB","XHB"],
    "Leisure & Entertainment": ["PEJ"],
    "Consumer Staples": ["VDC","FSTA","KXI","PBJ"],
    "Utilities": ["VPU","FUTY","IDU"],
    "Energy": ["IYE","VDE"],
    "Exploration & Production": ["XOP","IEO"],
    "Oil Services": ["OIH"],
    "Global Energy": ["IXC"],
}

SUBSECTOR_RIGHT = {
    "Biotechnology / Genomics": ["IBB","XBI","PBE","IDNA"],
    "Medical Equipment": ["IHI","XHE"],
    "Health Care Providers / Services": ["XHS"],
    "Pharmaceuticals": ["XPH"],
    "Broad / Alternative Health": ["FHLC","PINK"],
    "Banks": ["KBE","KRE","IAT"],
    "Insurance": ["KIE"],
    "Capital Markets / Brokerage": ["IAI","KCE"],
    "Diversified Financial Services": ["IYG"],
    "Broad Financials": ["VFH"],
    "Aerospace & Defense": ["ITA","PPA","XAR"],
    "Transportation": ["IYT","XTN"],
    "Broad Industrials": ["VIS","FIDU"],
    "Materials": ["XME","GDX","SIL","SLX","PICK","VAW"],
    "Real Estate": ["VNQ","IYR","REET"],
    "Specialty REITs": ["SRVR","HOMZ","SCHH","NETL"],
}

SUBSECTOR_ALL = {}
SUBSECTOR_ALL.update(SUBSECTOR_LEFT)
SUBSECTOR_ALL.update(SUBSECTOR_RIGHT)

# -----------------------------
# Data pulls
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_prices(tickers, period=PRICE_HISTORY_PERIOD):
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from price source.")

    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        close_df = pd.DataFrame(closes)
    else:
        close_df = pd.DataFrame({tickers[0]: df["Close"]})

    return close_df.dropna(how="all").ffill()

@st.cache_data(show_spinner=False, ttl=24*60*60)
def fetch_names(tickers: list[str]) -> dict[str, str]:
    names = {t: t for t in tickers}
    for t in tickers:
        try:
            inf = yf.Ticker(t).info
            n = inf.get("shortName") or inf.get("longName")
            if n:
                names[t] = str(n)
        except Exception:
            pass
    names["SPY"] = "S&P 500"
    names["QQQ"] = "Nasdaq-100"
    names["DIA"] = "Dow"
    names["IWM"] = "Russell 2000"
    names["RSP"] = "S&P 500 EW"
    return names

# -----------------------------
# Sparkline
# -----------------------------
SPARK_CHARS = "▁▂▃▄▅▆▇█"
SPARK_MAP = {c: i for i, c in enumerate(SPARK_CHARS)}

def sparkline_from_series(s: pd.Series, n=26) -> str:
    s = s.dropna().tail(n)
    if s.empty:
        return ""
    if s.nunique() == 1:
        return SPARK_CHARS[len(SPARK_CHARS)//2] * len(s)

    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return ""
    scaled = (s - lo) / (hi - lo)
    idx = (scaled * (len(SPARK_CHARS)-1)).round().astype(int).clip(0, len(SPARK_CHARS)-1)
    return "".join(SPARK_CHARS[i] for i in idx)

def spark_strength(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return np.nan
    vals = [SPARK_MAP.get(ch, np.nan) for ch in s]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals) / (len(SPARK_CHARS) - 1))

# -----------------------------
# Metrics / Table
# -----------------------------
def _ret(close: pd.Series, periods: int):
    return close.pct_change(periods=periods)

def _ratio_rs(close_t: pd.Series, close_b: pd.Series, periods: int):
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    return (t / b) - 1

def build_table(p: pd.DataFrame, tickers: list[str], name_map: dict[str, str]) -> pd.DataFrame:
    horizons_ret = {"% 1D": 1, "% 1W": 5, "% 1M": 21, "% 3M": 63, "% 6M": 126, "% 1Y": 252}
    horizons_rs  = {"RS 1W": 5, "RS 1M": 21, "RS 3M": 63, "RS 6M": 126, "RS 1Y": 252}

    b = p[BENCHMARK]

    rows = []
    for t in tickers:
        if t not in p.columns:
            continue

        close = p[t]
        last_price = float(close.dropna().iloc[-1]) if close.dropna().shape[0] else np.nan

        rs_ratio_series = (close / close.shift(21)) / (b / b.shift(21))
        spark = sparkline_from_series(rs_ratio_series, n=26)

        rec = {
            "Ticker": t,
            "Name": name_map.get(t, t),
            "Price": last_price,
            "Relative Strength 1M": spark,
        }

        for col, n in horizons_rs.items():
            rr = _ratio_rs(close, b, n)
            rec[col] = float(rr.dropna().iloc[-1]) if rr.dropna().shape[0] else np.nan

        for col, n in horizons_ret.items():
            r = _ret(close, n)
            rec[col] = float(r.dropna().iloc[-1]) if r.dropna().shape[0] else np.nan

        rows.append(rec)

    df = pd.DataFrame(rows)

    # Convert RS columns into 1-99 rank scale
    for col in horizons_rs.keys():
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = (s.rank(pct=True) * 99).round().clip(1, 99)

    return df

# -----------------------------
# Color helpers
# -----------------------------
def _heat_rs(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    x = (v - 1) / 98.0
    if x < 0.5:
        r = 255
        g = int(80 + (x/0.5) * (180-80))
    else:
        r = int(255 - ((x-0.5)/0.5) * (255-40))
        g = 200
    b = 60
    return f"background-color: rgb({r},{g},{b}); color:#0B0B0B; font-weight:900;"

def _pct_text(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    if v > 0:
        return "color: #7CFC9A; font-weight: 800;"
    if v < 0:
        return "color: #FF6B6B; font-weight: 800;"
    return "opacity:0.9; font-weight:700;"

def _spark_color(s):
    """
    IMPORTANT CHANGE:
    Color the SPARKLINE ITSELF (text color), NOT the background.
    """
    strength = spark_strength(s)
    if np.isnan(strength):
        return ""
    x = float(strength)
    if x < 0.5:
        r = 255
        g = int(80 + (x/0.5) * (180-80))
    else:
        r = int(255 - ((x-0.5)/0.5) * (255-40))
        g = 200
    b = 60
    return f"color: rgb({r},{g},{b}); font-weight:900;"

def style_df(df: pd.DataFrame):
    fmt = {
        "Price": "${:,.2f}",
        "% 1D": "{:.2%}", "% 1W": "{:.2%}", "% 1M": "{:.2%}",
        "% 3M": "{:.2%}", "% 6M": "{:.2%}", "% 1Y": "{:.2%}",
        "RS 1W": "{:.0f}", "RS 1M": "{:.0f}", "RS 3M": "{:.0f}", "RS 6M": "{:.0f}", "RS 1Y": "{:.0f}",
    }

    rs_cols = ["RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y"]
    pct_cols = ["% 1D", "% 1W", "% 1M", "% 3M", "% 6M", "% 1Y"]

    sty = df.style.format(fmt, na_rep="").hide(axis="index")

    for c in rs_cols:
        if c in df.columns:
            sty = sty.applymap(_heat_rs, subset=[c])

    for c in pct_cols:
        if c in df.columns:
            sty = sty.applymap(_pct_text, subset=[c])

    # Sparkline styling (only if visible)
    if SHOW_SPARKLINE and "Relative Strength 1M" in df.columns:
        sty = sty.applymap(_spark_color, subset=["Relative Strength 1M"])
        sty = sty.set_properties(
            subset=["Relative Strength 1M"],
            **{"font-family": "monospace", "font-weight": "900"}
        )

    return sty

# -----------------------------
# Manual inputs
# -----------------------------
DEFAULT_RIGHT = {
    "Market Exposure": {"IBD Exposure": "40-60%", "Selected": "X"},
    "Market Type": {"Type": "Bull Quiet"},
    "Trend Condition (QQQ)": {"Above 5DMA": "Yes", "Above 10DMA": "Yes", "Above 20DMA": "Yes", "Above 50DMA": "Yes", "Above 200DMA": "No"},
    "52-Week High/Low": {"Daily": 231, "Weekly": 811, "Monthly": -828},
    "Market Indicators": {"VIX": 16.34, "PCC": 0.67, "Up/Down Vol Ratio": 2.36, "A/D Ratio": 2.20},
    "Macro": {"Fed Funds": 4.09, "M2 Money": 22.2, "10yr": 4.02},
    "Breadth & Participation": {"% Price Above 10DMA": 56, "% Price Above 20DMA": 49, "% Price Above 50DMA": 58, "% Price Above 200DMA": 68},
    "Composite Model": {"Monetary Policy": "Neutral", "Liquidity Flow": "Good", "Rates & Credit": "Good", "Tape Strength": "Good", "Sentiment": "Neutral", "Total Score": 8.5},
    "Hot Sectors / Industry Groups": {"Notes": "Type here..."},
    "Market Correlations": {"Correlated": "Dow, Nasdaq", "Uncorrelated": "Dollar, Bonds"},
}

def init_right_state():
    if "right_panel" not in st.session_state:
        st.session_state.right_panel = DEFAULT_RIGHT

def right_panel_ui():
    init_right_state()
    rp = st.session_state.right_panel

    st.markdown(
        '<div class="card"><div style="font-weight:900; font-size:1.0rem;">Manual Inputs</div>'
        '<div class="small-muted">You update only these. Everything else pulls & calculates automatically.</div></div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "Download settings.json",
        data=json.dumps(rp, indent=2),
        file_name="dashboard_settings.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("Import settings JSON (optional)", type=["json"])
    if up is not None:
        try:
            st.session_state.right_panel = json.loads(up.read().decode("utf-8"))
            rp = st.session_state.right_panel
            st.success("Imported settings.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    for block in rp.keys():
        st.markdown(
            f'<div class="card"><div style="font-weight:900; margin-bottom:8px;">{block}</div>',
            unsafe_allow_html=True
        )
        data = rp.get(block, {})
        kv = pd.DataFrame({"Metric": list(data.keys()), "Value": list(data.values())})
        edited = st.data_editor(
            kv,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key=f"ed_{block}"
        )
        rp[block] = dict(zip(edited["Metric"], edited["Value"]))
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.right_panel = rp

# -----------------------------
# Grouped sub-sector helpers (clean headers)
# -----------------------------
def grouped_block(groups: dict[str, list[str]], df_by_ticker: dict[str, dict]) -> pd.DataFrame:
    out_rows = []
    for group_name, ticks in groups.items():
        # Header row: only Name populated; everything else NaN/blank (so NO "None")
        out_rows.append({
            "Ticker": "",
            "Name": group_name,
            "Price": np.nan,
            "Relative Strength 1M": "",
            "RS 1W": np.nan, "RS 1M": np.nan, "RS 3M": np.nan, "RS 6M": np.nan, "RS 1Y": np.nan,
            "% 1D": np.nan, "% 1W": np.nan, "% 1M": np.nan, "% 3M": np.nan, "% 6M": np.nan, "% 1Y": np.nan,
        })
        for t in ticks:
            if t in df_by_ticker:
                out_rows.append(df_by_ticker[t])
    return pd.DataFrame(out_rows)

def style_grouped(df: pd.DataFrame):
    sty = style_df(df)

    def _header_row_styles(row):
        is_header = (str(row.get("Ticker", "")).strip() == "") and (str(row.get("Name", "")).strip() != "")
        if not is_header:
            return ["" for _ in row.index]

        styles = []
        for col in row.index:
            if col == "Name":
                styles.append(
                    "font-weight:950;"
                    "background-color: rgba(0,0,0,0.65);"
                    "color: #FFFFFF;"
                )
            else:
                # black out the rest of the row
                styles.append(
                    "background-color: rgba(0,0,0,0.65);"
                    "color: rgba(0,0,0,0);"   # hide any stray text if it ever appears
                )
        return styles

    return sty.apply(_header_row_styles, axis=1)

# =========================
# UI
# =========================
st.title("Market Overview Dashboard")
st.caption(f"As of: {_asof_ts()} • Auto data: Yahoo Finance • RS Benchmark: {BENCHMARK}")

with st.sidebar:
    st.subheader("Controls")
    if st.button("Refresh Data"):
        fetch_prices.clear()
        fetch_names.clear()
        st.rerun()

pull_list = list(dict.fromkeys(ALL_TICKERS + [BENCHMARK]))

try:
    price_df = fetch_prices(pull_list, period=PRICE_HISTORY_PERIOD)
except Exception as e:
    st.error(f"Data pull failed: {e}")
    st.stop()

name_map = fetch_names(pull_list)

df_major = build_table(price_df, MAJOR, name_map)
df_sectors = build_table(price_df, SECTORS, name_map)

all_sub_ticks = []
for v in list(SUBSECTOR_ALL.values()):
    all_sub_ticks.extend(v)
all_sub_ticks = [t for t in all_sub_ticks if t in ALL_TICKERS_SET]

df_sub_master = build_table(price_df, all_sub_ticks, name_map)
df_by_ticker = {r["Ticker"]: r.to_dict() for _, r in df_sub_master.iterrows()}
df_sub_all = grouped_block(SUBSECTOR_ALL, df_by_ticker)

# Column order
base_cols = [
    "Ticker", "Name", "Price",
    "RS 1W", "RS 1M", "RS 3M", "RS 6M", "RS 1Y",
    "% 1D", "% 1W", "% 1M", "% 3M", "% 6M", "% 1Y"
]
show_cols = (["Ticker", "Name", "Price", "Relative Strength 1M"] + base_cols[3:]) if SHOW_SPARKLINE else base_cols

# ---- Major Indexes ----
st.markdown('<div class="section-title">Major U.S. Indexes</div>', unsafe_allow_html=True)
st.dataframe(
    style_df(df_major[show_cols]),
    use_container_width=True,
    height=330,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Name": st.column_config.TextColumn(width="large"),
        "Relative Strength 1M": st.column_config.TextColumn(width="large"),
    },
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---- Sectors ----
st.markdown('<div class="section-title">U.S. Sectors</div>', unsafe_allow_html=True)
st.dataframe(
    style_df(df_sectors[show_cols]),
    use_container_width=True,
    height=360,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Name": st.column_config.TextColumn(width="large"),
        "Relative Strength 1M": st.column_config.TextColumn(width="large"),
    },
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---- Sub-sectors ----
st.markdown('<div class="section-title">U.S. Sub-Sectors / Industry Groups</div>', unsafe_allow_html=True)
st.dataframe(
    style_grouped(df_sub_all[show_cols]),
    use_container_width=True,
    height=1100,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Name": st.column_config.TextColumn(width="large"),
        "Relative Strength 1M": st.column_config.TextColumn(width="large"),
    },
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---- Manual Inputs UNDER everything ----
right_panel_ui()

