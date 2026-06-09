import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta as ta_lib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
 
# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
 
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 
    .stApp { background-color: #0d1117; color: #e6edf3; }
 
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }
 
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
 
    .signal-buy {
        background: linear-gradient(135deg, #0d2818, #0f3d1f);
        border: 1px solid #238636;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(135deg, #2d0f0f, #3d1515);
        border: 1px solid #da3633;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .signal-hold {
        background: linear-gradient(135deg, #1c1a0d, #2d2a0f);
        border: 1px solid #d29922;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
 
    .signal-label {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin: 8px 0;
    }
 
    .reason-card {
        background: #161b22;
        border-left: 3px solid #388bfd;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
 
    .reason-card.positive { border-left-color: #3fb950; }
    .reason-card.negative { border-left-color: #f85149; }
    .reason-card.neutral  { border-left-color: #d29922; }
 
    .stat-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        color: #e6edf3;
    }
 
    .accuracy-badge {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        color: #79c0ff;
        display: inline-block;
        margin: 4px;
    }
 
    h1, h2, h3 { color: #e6edf3 !important; }
    .stSelectbox label, .stSlider label { color: #8b949e !important; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)
 
 
# ─── Helper Functions ────────────────────────────────────────────────────────
 
@st.cache_data(ttl=3600)
def load_stock_data(symbol, period="1y"):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df
 
def compute_indicators(df):
    df = df.copy()
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()
 
    df["EMA_20"]  = ta.ema(c, length=20)
    df["EMA_50"]  = ta.ema(c, length=50)
    df["EMA_200"] = ta.ema(c, length=200)
    df["SMA_20"]  = ta.sma(c, length=20)
 
    rsi = ta.rsi(c, length=14)
    df["RSI"] = rsi
 
    macd_df = ta.macd(c)
    if macd_df is not None:
        df["MACD"]        = macd_df.iloc[:, 0]
        df["MACD_Signal"] = macd_df.iloc[:, 2]
        df["MACD_Hist"]   = macd_df.iloc[:, 1]
 
    bb = ta.bbands(c, length=20)
    if bb is not None:
        df["BB_Upper"] = bb.iloc[:, 2]
        df["BB_Mid"]   = bb.iloc[:, 1]
        df["BB_Lower"] = bb.iloc[:, 0]
 
    df["ATR"]  = ta.atr(h, l, c, length=14)
    df["OBV"]  = ta.obv(c, v)
    df["VWAP"] = (c * v).cumsum() / v.cumsum()
 
    df["Return_1d"] = c.pct_change(1)
    df["Return_5d"] = c.pct_change(5)
    df["Volatility"]= c.pct_change().rolling(20).std()
    df["Vol_Ratio"] = v / v.rolling(20).mean()
 
    df["Price_vs_EMA20"]  = (c - df["EMA_20"])  / df["EMA_20"]
    df["Price_vs_EMA50"]  = (c - df["EMA_50"])  / df["EMA_50"]
    df["Price_vs_EMA200"] = (c - df["EMA_200"]) / df["EMA_200"]
 
    df["future_return"] = c.shift(-5) / c - 1
    df["label"] = 0
    df.loc[df["future_return"] >  0.02, "label"] =  1
    df.loc[df["future_return"] < -0.02, "label"] = -1
 
    return df.dropna()
 
def train_model(df):
    EXCLUDE = ["label", "future_return", "Open", "High", "Low", "Close", "Volume"]
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
 
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"]
 
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
 
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_enc[:split], y_enc[split:]
 
    model = LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train)
 
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                      target_names=le.classes_.astype(str),
                                      output_dict=True)
 
    importances = pd.Series(model.feature_importances_, index=feat_cols)\
                    .sort_values(ascending=False)
 
    return model, le, feat_cols, accuracy, report, importances, X_test, y_test, y_pred
 
def generate_signal(model, le, feat_cols, df):
    latest = df[feat_cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)
    pred_enc  = model.predict(latest)[0]
    prob      = model.predict_proba(latest)[0]
    signal    = le.inverse_transform([pred_enc])[0]
    confidence= float(prob.max()) * 100
    return signal, confidence, prob
 
def generate_reasons(df, signal):
    row = df.iloc[-1]
    reasons = []
 
    # RSI
    rsi = row.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            reasons.append(("positive", f"RSI is oversold at {rsi:.1f} — potential reversal upward"))
        elif rsi > 70:
            reasons.append(("negative", f"RSI is overbought at {rsi:.1f} — potential pullback ahead"))
        else:
            reasons.append(("neutral", f"RSI at {rsi:.1f} — neutral momentum zone"))
 
    # MACD
    macd = row.get("MACD", np.nan)
    macd_sig = row.get("MACD_Signal", np.nan)
    if not np.isnan(macd) and not np.isnan(macd_sig):
        if macd > macd_sig:
            reasons.append(("positive", "MACD crossed above signal line — bullish momentum building"))
        else:
            reasons.append(("negative", "MACD below signal line — bearish pressure present"))
 
    # EMA trend
    close = row.get("Close", np.nan)
    ema20 = row.get("EMA_20", np.nan)
    ema50 = row.get("EMA_50", np.nan)
    if not np.isnan(close) and not np.isnan(ema20) and not np.isnan(ema50):
        if close > ema20 > ema50:
            reasons.append(("positive", "Price above EMA20 & EMA50 — strong uptrend confirmed"))
        elif close < ema20 < ema50:
            reasons.append(("negative", "Price below EMA20 & EMA50 — downtrend in progress"))
        else:
            reasons.append(("neutral", "Mixed EMA signals — no clear trend direction"))
 
    # Volume
    vol_ratio = row.get("Vol_Ratio", np.nan)
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            reasons.append(("positive" if signal == 1 else "negative",
                            f"Volume is {vol_ratio:.1f}x the 20-day average — strong conviction move"))
        else:
            reasons.append(("neutral", f"Volume at {vol_ratio:.1f}x average — normal activity"))
 
    # Bollinger Bands
    bb_upper = row.get("BB_Upper", np.nan)
    bb_lower = row.get("BB_Lower", np.nan)
    if not np.isnan(close) and not np.isnan(bb_upper) and not np.isnan(bb_lower):
        if close > bb_upper:
            reasons.append(("negative", "Price above Bollinger upper band — potentially overextended"))
        elif close < bb_lower:
            reasons.append(("positive", "Price below Bollinger lower band — potential bounce zone"))
 
    # 5-day return
    ret5 = row.get("Return_5d", np.nan)
    if not np.isnan(ret5):
        direction = "gained" if ret5 > 0 else "lost"
        tone = "positive" if ret5 > 0 else "negative"
        reasons.append((tone, f"Stock has {direction} {abs(ret5)*100:.1f}% over the last 5 days"))
 
    return reasons
 
 
def candlestick_chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
 
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#3fb950",
        decreasing_line_color="#f85149"
    ), row=1, col=1)
 
    for ema, color in [("EMA_20","#79c0ff"),("EMA_50","#d2a8ff"),("EMA_200","#ffa657")]:
        if ema in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ema],
                name=ema, line=dict(color=color, width=1.2)), row=1, col=1)
 
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            name="BB Upper", line=dict(color="#8b949e", width=0.8, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            name="BB Lower", line=dict(color="#8b949e", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(139,148,158,0.05)"), row=1, col=1)
 
    # Volume
    colors = ["#3fb950" if c >= o else "#f85149"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)
 
    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
            name="RSI", line=dict(color="#d2a8ff", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#f85149", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#3fb950", opacity=0.5, row=3, col=1)
 
    fig.update_layout(
        template="plotly_dark", height=600,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter", color="#8b949e")
    )
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    fig.update_xaxes(gridcolor="#21262d")
    return fig
 
 
def accuracy_chart(report):
    classes = [k for k in report.keys() if k not in ["accuracy","macro avg","weighted avg"]]
    metrics = ["precision","recall","f1-score"]
    fig = go.Figure()
    colors = ["#79c0ff","#3fb950","#d2a8ff"]
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=m.capitalize(), x=classes,
            y=[report[c][m] for c in classes],
            marker_color=colors[i]
        ))
    fig.update_layout(
        barmode="group", template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h"),
        font=dict(family="Inter", color="#8b949e"),
        yaxis=dict(range=[0,1], gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d")
    )
    return fig
 
 
def feature_importance_chart(importances):
    top = importances.head(12)
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker=dict(color=top.values, colorscale="Blues")
    ))
    fig.update_layout(
        template="plotly_dark", height=320,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(l=0,r=0,t=10,b=0),
        font=dict(family="Inter", color="#8b949e"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(autorange="reversed", gridcolor="#21262d")
    )
    return fig
 
 
# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockSense AI")
    st.markdown("---")
 
    market = st.selectbox("Market", ["🇮🇳 NSE India", "🇺🇸 US Market"])
 
    if "NSE" in market:
        popular = ["RELIANCE","TCS","INFY","HDFCBANK","WIPRO",
                   "ICICIBANK","SBIN","BHARTIARTL","AXISBANK","KOTAKBANK"]
        suffix = ".NS"
    else:
        popular = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","NFLX","AMD","INTC"]
        suffix = ""
 
    ticker_input = st.text_input("Stock Symbol", value=popular[0])
    symbol = ticker_input.upper() + (suffix if suffix not in ticker_input.upper() else "")
 
    period = st.select_slider("History Period",
        options=["3mo","6mo","1y","2y","5y"], value="1y")
 
    st.markdown("---")
    st.markdown("**Quick Picks**")
    cols = st.columns(2)
    for i, p in enumerate(popular[:6]):
        if cols[i % 2].button(p, use_container_width=True):
            symbol = p + suffix
 
    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Stock", use_container_width=True, type="primary")
 
    st.markdown("""
    <div style='color:#8b949e; font-size:0.75rem; margin-top:20px; line-height:1.6'>
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Not financial advice. Always do your own research before investing.
    </div>
    """, unsafe_allow_html=True)
 
 
# ─── Main Dashboard ───────────────────────────────────────────────────────────
st.markdown("# StockSense AI &nbsp; <span style='font-size:1rem;color:#8b949e;font-weight:400'>Powered by LightGBM + Technical Analysis</span>", unsafe_allow_html=True)
 
if analyze_btn or True:
    with st.spinner(f"Fetching data for **{symbol}**..."):
        raw_df = load_stock_data(symbol, period)
 
    if raw_df is None or len(raw_df) < 60:
        st.error(f"Could not load data for **{symbol}**. Check the symbol and try again.")
        st.stop()
 
    with st.spinner("Computing indicators & training model..."):
        df = compute_indicators(raw_df)
        model, le, feat_cols, accuracy, report, importances, X_test, y_test, y_pred = train_model(df)
        signal, confidence, probs = generate_signal(model, le, feat_cols, df)
        reasons = generate_reasons(df, signal)
 
    # ── Live Price Bar ──────────────────────────────────────────────────────
    ticker_info = yf.Ticker(symbol)
    info = ticker_info.info or {}
 
    latest    = df.iloc[-1]
    prev      = df.iloc[-2]
    price     = float(latest["Close"])
    chg       = price - float(prev["Close"])
    chg_pct   = chg / float(prev["Close"]) * 100
    currency  = "₹" if ".NS" in symbol else "$"
    arrow     = "▲" if chg >= 0 else "▼"
    chg_color = "#3fb950" if chg >= 0 else "#f85149"
 
    name = info.get("longName", symbol)
 
    st.markdown(f"""
    <div class="metric-card" style="display:flex; align-items:center; gap:32px; flex-wrap:wrap">
        <div>
            <div class="stat-label">Company</div>
            <div style="font-size:1.2rem; font-weight:600; color:#e6edf3">{name}</div>
            <div style="color:#8b949e; font-size:0.85rem">{symbol}</div>
        </div>
        <div>
            <div class="stat-label">Price</div>
            <div class="stat-value">{currency}{price:,.2f}</div>
        </div>
        <div>
            <div class="stat-label">Change</div>
            <div class="stat-value" style="color:{chg_color}">{arrow} {currency}{abs(chg):.2f} ({chg_pct:+.2f}%)</div>
        </div>
        <div>
            <div class="stat-label">52W High</div>
            <div class="stat-value">{currency}{info.get('fiftyTwoWeekHigh', df['High'].max()):,.2f}</div>
        </div>
        <div>
            <div class="stat-label">52W Low</div>
            <div class="stat-value">{currency}{info.get('fiftyTwoWeekLow', df['Low'].min()):,.2f}</div>
        </div>
        <div>
            <div class="stat-label">Model Accuracy</div>
            <div class="stat-value" style="color:#79c0ff">{accuracy:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Main Row: Signal + Chart ─────────────────────────────────────────────
    col_signal, col_chart = st.columns([1, 2.8])
 
    with col_signal:
        sig_class = {1: "buy", -1: "sell", 0: "hold"}[signal]
        sig_label = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
        sig_emoji = {1: "🟢", -1: "🔴", 0: "🟡"}[signal]
 
        st.markdown(f"""
        <div class="signal-{sig_class}">
            <div class="stat-label" style="color:#8b949e">AI SIGNAL</div>
            <div class="signal-label" style="color:{'#3fb950' if signal==1 else '#f85149' if signal==-1 else '#d29922'}">
                {sig_emoji} {sig_label}
            </div>
            <div style="font-size:1.5rem; font-weight:700; color:#e6edf3; font-family:'JetBrains Mono',monospace">
                {confidence:.1f}%
            </div>
            <div class="stat-label" style="margin-top:4px">confidence</div>
        </div>
        """, unsafe_allow_html=True)
 
        # Probability breakdown
        st.markdown("<div style='margin-top:16px'>", unsafe_allow_html=True)
        class_map = {-1: "SELL 🔴", 0: "HOLD 🟡", 1: "BUY 🟢"}
        for enc_idx, cls in enumerate(le.classes_):
            label = class_map.get(int(cls), str(cls))
            prob_val = float(probs[enc_idx]) * 100
            bar_color = "#3fb950" if int(cls)==1 else "#f85149" if int(cls)==-1 else "#d29922"
            st.markdown(f"""
            <div style="margin:8px 0">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px">
                    <span style="font-size:0.8rem; color:#8b949e">{label}</span>
                    <span style="font-size:0.8rem; font-family:'JetBrains Mono',monospace; color:#e6edf3">{prob_val:.1f}%</span>
                </div>
                <div style="background:#21262d; border-radius:4px; height:6px">
                    <div style="background:{bar_color}; width:{prob_val}%; height:100%; border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
        # Key indicators snapshot
        st.markdown("<br>**Key Indicators**", unsafe_allow_html=True)
        ind_data = {
            "RSI (14)":    f"{latest.get('RSI', 0):.1f}",
            "MACD":        f"{latest.get('MACD', 0):.3f}",
            "EMA 20":      f"{currency}{latest.get('EMA_20', 0):,.2f}",
            "EMA 50":      f"{currency}{latest.get('EMA_50', 0):,.2f}",
            "Vol Ratio":   f"{latest.get('Vol_Ratio', 0):.2f}x",
            "Volatility":  f"{latest.get('Volatility', 0)*100:.2f}%",
        }
        for k, v in ind_data.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #21262d">
                <span style="color:#8b949e; font-size:0.82rem">{k}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#e6edf3">{v}</span>
            </div>
            """, unsafe_allow_html=True)
 
    with col_chart:
        st.plotly_chart(candlestick_chart(df.tail(120), symbol),
                        use_container_width=True, config={"displayModeBar": False})
 
    # ── Reasoning Section ─────────────────────────────────────────────────────
    st.markdown("### 🧠 Why this signal?")
    reason_cols = st.columns(2)
    for i, (tone, text) in enumerate(reasons):
        with reason_cols[i % 2]:
            icon = "✅" if tone == "positive" else "❌" if tone == "negative" else "⚠️"
            st.markdown(f"""
            <div class="reason-card {tone}">
                {icon} {text}
            </div>
            """, unsafe_allow_html=True)
 
    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
 
    col_acc, col_feat = st.columns(2)
 
    with col_acc:
        st.markdown("**Per-Class Accuracy (Precision / Recall / F1)**")
        st.plotly_chart(accuracy_chart(report), use_container_width=True,
                        config={"displayModeBar": False})
 
        badges = ""
        for cls in [k for k in report if k not in ["accuracy","macro avg","weighted avg"]]:
            label_map = {"-1":"SELL","0":"HOLD","1":"BUY"}
            label = label_map.get(cls, cls)
            f1 = report[cls]["f1-score"]
            badges += f'<span class="accuracy-badge">{label} F1: {f1:.2f}</span>'
        st.markdown(f"<div style='margin-top:8px'>{badges}</div>", unsafe_allow_html=True)
 
    with col_feat:
        st.markdown("**Top Feature Importances**")
        st.plotly_chart(feature_importance_chart(importances), use_container_width=True,
                        config={"displayModeBar": False})
 
    # ── Recent Signal History ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Recent Signal History (Last 30 Days)")
 
    recent = df.tail(30).copy()
    EXCLUDE = ["label","future_return","Open","High","Low","Close","Volume"]
    fc = [c for c in df.columns if c not in EXCLUDE]
    X_recent = recent[fc].replace([np.inf,-np.inf], np.nan).fillna(0)
    preds_enc = model.predict(X_recent)
    preds     = le.inverse_transform(preds_enc)
 
    history_rows = []
    for date, row, pred in zip(recent.index, recent.itertuples(), preds):
        sig_txt = {1:"🟢 BUY", -1:"🔴 SELL", 0:"🟡 HOLD"}[int(pred)]
        history_rows.append({
            "Date":    str(date.date()),
            "Close":   f"{currency}{row.Close:,.2f}",
            "RSI":     f"{row.RSI:.1f}" if hasattr(row,'RSI') else "-",
            "Signal":  sig_txt,
            "5D Ret":  f"{row.Return_5d*100:+.1f}%" if hasattr(row,'Return_5d') else "-"
        })
 
    hist_df = pd.DataFrame(history_rows[::-1])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
 
    st.markdown("""
    <div style='color:#8b949e; font-size:0.75rem; margin-top:24px; text-align:center'>
    ⚠️ StockSense AI is for educational & research purposes only. Past model performance does not guarantee future results. Always consult a SEBI-registered financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
 