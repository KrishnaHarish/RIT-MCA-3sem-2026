import time
from datetime import datetime

import requests
import streamlit as st


DEFAULT_ESP_URL = "http://192.168.1.20/data"  # change if needed

DEMO_DATA = {
    "temp": 26.5,
    "hum": 61.0,
    "note": "Demo values (switch to Live to read ESP8266)",
}


def _rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _read_esp(url: str) -> tuple[float, float]:
    r = requests.get(url, timeout=2)
    r.raise_for_status()
    j = r.json()
    temp = float(j["temp"])
    hum = float(j["hum"])
    return temp, hum


def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))


def _circle_card(label: str, value: float, unit: str, *, vmin: float, vmax: float, color: str) -> str:
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        pct = ((_clamp(value, vmin, vmax) - vmin) / denom) * 100.0
        pct = _clamp(pct, 0.0, 100.0)
        value_text = f"{value:.1f}{unit}" if unit else f"{value:.1f}"
        return f"""
        <div class="card">
            <div class="card-top">
                <div class="label">{label}</div>
                <div class="value">{value_text}</div>
            </div>
            <div class="ring" style="--p:{pct:.2f}; --c:{color};">
                <div class="ring-inner">
                    <div class="ring-big">{pct:.0f}%</div>
                    <div class="ring-sub">of range</div>
                </div>
            </div>
        </div>
        """


st.set_page_config(page_title="ESP8266 DHT22", page_icon="DHT22", layout="centered")

theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

if theme == "Light":
        theme_vars = """
        :root {
            --bg1: #f8fafc;
            --bg2: #eef2ff;
            --card: rgba(255,255,255,0.72);
            --card2: rgba(255,255,255,0.92);
            --stroke: rgba(15,23,42,0.10);
            --text: rgba(15,23,42,0.92);
            --muted: rgba(15,23,42,0.64);
            --ringTrack: rgba(15,23,42,0.10);
            --ringInner: rgba(255,255,255,0.86);
            --sidebarBg: rgba(255,255,255,0.60);
            --btnBg: rgba(15,23,42,0.04);
            --btnBgHover: rgba(15,23,42,0.07);
            --btnBorder: rgba(15,23,42,0.14);
        }
        """
        app_bg = """
            background: radial-gradient(1100px 600px at 0% 0%, rgba(99,102,241,0.20) 0%, rgba(99,102,241,0.0) 60%),
                                    radial-gradient(900px 500px at 100% 0%, rgba(14,165,233,0.18) 0%, rgba(14,165,233,0.0) 55%),
                                    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
        """
else:
        theme_vars = """
        :root {
            --bg1: #0b1220;
            --bg2: #0f1d36;
            --card: rgba(255,255,255,0.06);
            --card2: rgba(255,255,255,0.08);
            --stroke: rgba(255,255,255,0.12);
            --text: rgba(255,255,255,0.92);
            --muted: rgba(255,255,255,0.66);
            --ringTrack: rgba(255,255,255,0.10);
            --ringInner: rgba(12,18,32,0.65);
            --sidebarBg: rgba(0,0,0,0.18);
            --btnBg: rgba(255,255,255,0.06);
            --btnBgHover: rgba(255,255,255,0.10);
            --btnBorder: rgba(255,255,255,0.14);
        }
        """
        app_bg = """
            background: radial-gradient(1200px 600px at 10% 10%, #1a2a52 0%, rgba(26,42,82,0.0) 60%),
                                    radial-gradient(900px 500px at 90% 0%, #2b1a4a 0%, rgba(43,26,74,0.0) 55%),
                                    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
        """

st.markdown(
        f"""
<style>
    {theme_vars}
    .stApp {{
        {app_bg}
        color: var(--text);
    }}
    h1, h2, h3, p, label {{ color: var(--text) !important; }}
    .small-muted {{ color: var(--muted); font-size: 0.95rem; }}

    .grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 10px;
    }}
    @media (max-width: 640px) {{
        .grid {{ grid-template-columns: 1fr; }}
    }}

    .card {{
        background: linear-gradient(180deg, var(--card2), var(--card));
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        backdrop-filter: blur(8px);
    }}
    .card-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }}
    .label {{ font-size: 0.95rem; letter-spacing: 0.02em; color: var(--muted); }}
    .value {{ font-size: 1.35rem; font-weight: 700; }}

    .ring {{
        --p: 0;
        --c: #2dd4bf;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        margin: 14px auto 0 auto;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--ringTrack) 0);
        position: relative;
        display: grid;
        place-items: center;
    }}
    .ring::before {{
        content: "";
        position: absolute;
        inset: 10px;
        border-radius: 50%;
        background: var(--ringInner);
        border: 1px solid var(--stroke);
    }}
    .ring-inner {{
        position: relative;
        text-align: center;
        line-height: 1.05;
    }}
    .ring-big {{ font-size: 1.55rem; font-weight: 800; }}
    .ring-sub {{ font-size: 0.85rem; color: var(--muted); margin-top: 2px; }}

    section[data-testid="stSidebar"] {{
        background: var(--sidebarBg);
        border-right: 1px solid var(--stroke);
    }}

    .stButton > button {{
        border-radius: 999px;
        border: 1px solid var(--btnBorder);
        background: var(--btnBg);
        color: var(--text);
        padding: 0.55rem 0.9rem;
    }}
    .stButton > button:hover {{
        border-color: var(--btnBorder);
        background: var(--btnBgHover);
    }}
</style>
""",
        unsafe_allow_html=True,
)


st.title("ESP8266 DHT22 - First Demo (Streamlit)")
st.markdown("<div class='small-muted'>Modern circular dashboard • Demo data shows first</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Source")
    mode = st.radio("Data mode", ["Demo", "Live"], index=0)
    esp_url = st.text_input("ESP URL", value=DEFAULT_ESP_URL, disabled=(mode == "Demo"))
    auto_refresh = st.checkbox("Auto refresh", value=(mode == "Live"))
    refresh_seconds = st.slider("Refresh interval (seconds)", 1, 30, 2, disabled=not auto_refresh)

gauges = st.empty()
status = st.empty()

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if mode == "Demo":
    temp = float(DEMO_DATA["temp"])
    hum = float(DEMO_DATA["hum"])
    gauges.markdown(
        f"""
        <div class="grid">
          {_circle_card("Temperature", temp, "°C", vmin=0, vmax=50, color="#fb7185")}
          {_circle_card("Humidity", hum, "%", vmin=0, vmax=100, color="#2dd4bf")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    status.info(f"Demo data shown first • {now}")
    with st.expander("Demo payload"):
        st.json(DEMO_DATA)
else:
    try:
        temp, hum = _read_esp(esp_url)
        gauges.markdown(
            f"""
            <div class="grid">
              {_circle_card("Temperature", temp, "°C", vmin=0, vmax=50, color="#fb7185")}
              {_circle_card("Humidity", hum, "%", vmin=0, vmax=100, color="#2dd4bf")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        status.success(f"Live • {now}")
    except Exception as e:
        status.error(f"Not reading from ESP: {e}")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Refresh now"):
            _rerun()
    with cols[1]:
        st.caption("Tip: enable Auto refresh for continuous updates")

    if auto_refresh:
        time.sleep(refresh_seconds)
        _rerun()