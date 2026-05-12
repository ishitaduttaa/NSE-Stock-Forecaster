 # add app.py file
# FRONTEND
# used for streamlit-

import os, re, time, json, pickle
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

groq_key     = os.environ.get('GROQ_API_KEY', '')
newsdata_key = os.environ.get('NEWSDATA_API_KEY', '')
gnews_key    = os.environ.get('GNEWS_API_KEY', '')

st.set_page_config(
    page_title="Stock Analysis & RAG Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body { background-color: #0e1117; }
.metric-card {
    background: #1c2333; border-radius: 12px; padding: 16px 20px;
    margin: 6px 0; border-left: 4px solid #00d4ff;
}
.signal-buy   { color: #26a69a; font-size: 1.5rem; font-weight: 700; }
.signal-sell  { color: #ef5350; font-size: 1.5rem; font-weight: 700; }
.signal-hold  { color: #ffd700; font-size: 1.5rem; font-weight: 700; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #00d4ff;
    margin-top: 1.2rem; margin-bottom: .4rem;
}
.chat-user { background:#1c2333; border-radius:10px; padding:10px 14px; margin:6px 0; }
.chat-bot  { background:#0d2137; border-radius:10px; padding:10px 14px; margin:6px 0;
             border-left:3px solid #00d4ff; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.screener-card {
    background: #1c2333; border-radius: 12px; padding: 16px 20px;
    margin: 8px 0; border-left: 4px solid #26a69a;
}
</style>
""", unsafe_allow_html=True)

for key, default in {
    "headlines": [],
    "pdf_summary": "",
    "screener_data": {},          # stores scraped Screener.in data
    "screener_fetched": False,    # flag
    "documents_extra": [],
    "ohlcv": None,
    "df_feat": None,
    "documents": [],
    "faiss_index": None,
    "embedder": None,
    "models_trained": False,
    "results": {},
    "chat_history": [],
    "groq_client": None,
    "company_name": "",
    "ticker": "",
    "start_date": "2020-01-01",
    "end_date": datetime.today().strftime("%Y-%m-%d"),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("### 🏢 Stock Details")
    company_name = st.text_input("Company Name", placeholder="e.g. Tata Consumer")
    ticker       = st.text_input("NSE Ticker",   placeholder="e.g. TATACONSUM.NS")

    st.markdown("### 📅 Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=pd.to_datetime("2020-01-01"))
    with col_e:
        end_date   = st.date_input("End",   value=pd.to_datetime("today"))

    news_source = st.selectbox("News Source", ["Both (recommended)", "NewsData.io only", "GNews only"])
    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Models: ARIMA · SARIMA · Prophet · XGBoost · LightGBM · RF · LSTM")


def get_screener_pdf_url(ticker_symbol):
    """Scrape Screener.in to find the latest BSE Annual Report PDF link."""
    url = f"https://www.screener.in/company/{ticker_symbol}/"
    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }, timeout=15)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.text.strip().lower()
            if "financial year" in text and href.lower().endswith(".pdf"):
                return href
    except Exception as e:
        st.warning(f"Screener PDF scrape failed: {e}")
    return None


def fetch_screener_data(ticker):
    """Scrape Screener.in for fundamentals + auto-find annual report PDF."""
    from bs4 import BeautifulSoup
    slug = ticker.replace(".NS", "").replace(".BO", "").strip().upper()
    url  = f"https://www.screener.in/company/{slug}/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
    result = {
        "url": url, "about": "", "ratios": {},
        "pros": [], "cons": [], "quarterly_sales": [],
        "annual_report_url": "",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 404:
            url = f"https://www.screener.in/company/{slug}/"
            result["url"] = url
            r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except Exception as e:
        st.warning(f"Screener.in fetch failed: {e}")
        return result

    soup = BeautifulSoup(r.text, "html.parser")

    # About
    about_tag = soup.select_one("div.company-profile p") or soup.select_one("#about p")
    if about_tag:
        result["about"] = about_tag.get_text(strip=True)

    # Key ratios
    for li in soup.select("ul#top-ratios li"):
        name_tag  = li.select_one("span.name")
        value_tag = li.select_one("span.number") or li.select_one("span.value")
        if name_tag and value_tag:
            result["ratios"][name_tag.get_text(strip=True)] = value_tag.get_text(strip=True)

    # Pros & Cons
    for li in soup.select("div.pros ul li"):
        result["pros"].append(li.get_text(strip=True))
    for li in soup.select("div.cons ul li"):
        result["cons"].append(li.get_text(strip=True))

    # Quarterly sales
    for table in soup.select("section#quarters table"):
        rows = table.select("tr")
        for row in rows:
            cols = row.select("td")
            if cols and "Sales" in cols[0].get_text(strip=True):
                headers_row = table.select_one("thead tr") or rows[0]
                periods = [th.get_text(strip=True) for th in headers_row.select("th")][1:]
                values  = [td.get_text(strip=True) for td in cols][1:]
                result["quarterly_sales"] = list(zip(periods, values))[:8]
                break

    # PDF link — uses your exact notebook logic
    result["annual_report_url"] = get_screener_pdf_url(slug) or ""
    return result


def download_and_extract_pdf(pdf_url, groq_key):
    """Download a PDF from URL, extract text, summarise with Groq."""
    import pdfplumber, tempfile, os
    from groq import Groq

    r = requests.get(pdf_url, headers={
        "User-Agent": "Mozilla/5.0"
    }, timeout=60, stream=True)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    for chunk in r.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.flush(); tmp.close()

    pages = []
    with pdfplumber.open(tmp.name) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= 20: break
            txt = page.extract_text()
            if txt: pages.append(txt)
    os.unlink(tmp.name)

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        return "No extractable text found in the PDF."

    client = Groq(api_key=groq_key)
    chunks = [full_text[i:i+15000] for i in range(0, min(len(full_text), 60000), 15000)]
    chunk_summaries = []
    for chunk in chunks:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content":
                       f"Summarise this annual report section in bullet points "
                       f"(revenue, profit, risks, outlook):\n\n{chunk}"}],
            temperature=0.2, max_tokens=512,
        )
        chunk_summaries.append(resp.choices[0].message.content.strip())

    final = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content":
                   "Combine into one executive summary with sections: "
                   "Financial Performance, Business Highlights, Risks, Outlook:\n\n" +
                   "\n---\n".join(chunk_summaries)}],
        temperature=0.2, max_tokens=800,
    )
    return final.choices[0].message.content.strip()



def lazy_import():
    global yf, Prophet, auto_arima, xgb, lgb, RandomForestClassifier
    global accuracy_score, precision_score, recall_score, TimeSeriesSplit
    global RobustScaler, SentenceTransformer, faiss, TextBlob, Groq
    global pdfplumber, BeautifulSoup, pipeline, LABEL_MAP

    import yfinance as yf
    from prophet import Prophet
    from pmdarima import auto_arima
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    from sentence_transformers import SentenceTransformer
    import faiss
    from textblob import TextBlob
    from groq import Groq
    import pdfplumber
    from bs4 import BeautifulSoup
    try:
        from transformers import pipeline
        LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}
    except Exception:
        pipeline = None
        LABEL_MAP = {}


def build_query(name, tick):
    clean = tick.replace(".NS", "").replace(".BO", "").strip()
    return f"{name} stock" if name and name != clean else f"{clean} stock India"


def clean_headlines(raw):
    seen, out = set(), []
    for h in raw:
        h = h.strip()
        if h and h not in seen:
            seen.add(h); out.append(h)
    return out


def fetch_newsdata(query, api_key, max_results=10):
    try:
        r = requests.get("https://newsdata.io/api/1/news",
                         params={"apikey": api_key, "q": query,
                                 "language": "en", "category": "business",
                                 "size": max_results}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "success":
            return [a.get("title") or a.get("description", "") for a in data.get("results", []) if a.get("title")]
    except Exception as e:
        st.warning(f"NewsData.io error: {e}")
    return []


def fetch_gnews(query, api_key, max_results=10):
    try:
        r = requests.get("https://gnews.io/api/v4/search",
                         params={"token": api_key, "q": query, "lang": "en",
                                 "country": "in", "max": max_results,
                                 "sortby": "publishedAt"}, timeout=15)
        r.raise_for_status()
        return [a.get("title") or a.get("description", "") for a in r.json().get("articles", []) if a.get("title")]
    except Exception as e:
        st.warning(f"GNews error: {e}")
    return []


def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def engineer_features(ohlcv, macro_df, start_date, end_date):
    import numpy as np
    df = ohlcv.copy()
    df["Returns"]     = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["RSI"]         = compute_rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = bb_mid + 2*bb_std
    df["BB_Lower"] = bb_mid - 2*bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
    df["BB_PctB"]  = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"]     = pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"]
    df["OBV"]     = (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()
    for w in [7, 14, 21, 50]:
        df[f"Roll_Mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"Roll_Std_{w}"]  = df["Close"].rolling(w).std()
        df[f"RVol_{w}"]      = df["Returns"].rolling(w).std() * np.sqrt(252)
    df["Momentum_5"]  = df["Close"] / df["Close"].shift(5) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["HL_Spread"]   = (df["High"] - df["Low"])   / df["Close"]
    df["CO_Spread"]   = (df["Close"] - df["Open"]) / df["Open"]
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"]    = 100*(df["Close"] - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"]    = df["Stoch_K"].rolling(3).mean()
    df["Williams_R"] = -100*(high14 - df["Close"]) / (high14 - low14 + 1e-9)
    df["Vol_MA_20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_MA_20"] + 1)
    for lag in [1, 2, 3, 5, 10]:
        df[f"Ret_Lag_{lag}"]   = df["Returns"].shift(lag)
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"]     = df.index.month
    df["Quarter"]   = df.index.quarter
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    df["Regime"] = np.where(ema20 > ema50, 1, -1)
    if not macro_df.empty:
        for col in macro_df.columns:
            df[f"{col}_Ret"] = macro_df[col].pct_change()
    for col in ["Returns", "Log_Returns"]:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)
    df["Target"]     = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Next_Close"] = df["Close"].shift(-1)
    df.ffill(inplace=True); df.bfill(inplace=True); df.dropna(inplace=True)
    return df


def run_prophet(df, ohlcv):
    regressors = ["RSI", "MACD", "MACD_Hist", "BB_Width",
                  "ATR_Pct", "Momentum_5", "Momentum_20",
                  "Vol_Ratio", "HL_Spread", "CO_Spread"]
    df_p = pd.DataFrame({
        "ds": pd.to_datetime(ohlcv.index[:len(df)]),
        "y":  (df["Log_Returns"] - df["Log_Returns"].rolling(5).mean()).values
    }).reset_index(drop=True)
    for r in regressors:
        if r in df.columns:
            df_p[r] = df[r].values
    split = int(len(df_p) * 0.70)
    train = df_p.iloc[:split]
    m = Prophet(changepoint_prior_scale=0.1, seasonality_mode="multiplicative",
                yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for r in regressors:
        if r in train.columns:
            m.add_regressor(r)
    m.fit(train)
    forecast = m.predict(df_p)
    test_f = forecast.iloc[split:]
    test_a = df_p.iloc[split:]
    p_dir  = (test_f["yhat"].values > 0).astype(int)
    a_dir  = (test_a["y"].values > 0).astype(int)
    p_acc  = float(np.mean(p_dir == a_dir))
    df["Prophet_Return"]     = forecast["yhat"].values[:len(df)]
    df["Prophet_Trend"]      = forecast["trend"].values[:len(df)]
    df["Prophet_Upper"]      = forecast["yhat_upper"].values[:len(df)]
    df["Prophet_Lower"]      = forecast["yhat_lower"].values[:len(df)]
    df["Prophet_Confidence"] = (1 / (forecast["yhat_upper"].values[:len(df)]
                                     - forecast["yhat_lower"].values[:len(df)] + 1e-6))
    df["Prophet_Direction"]  = (df["Prophet_Return"].shift(-1) > 0).astype(int)
    current_price = float(df["Close"].iloc[-1])
    next_ret      = float(df["Prophet_Return"].iloc[-1])
    prophet_price = float(current_price * np.exp(next_ret))
    return df, p_acc, prophet_price


def run_arima_sarima(df):
    close_series = df["Close"].copy()
    log_close    = np.log(close_series)
    train_end    = int(len(log_close) * 0.80)
    train_log    = log_close.iloc[:train_end]
    test_log     = log_close.iloc[train_end:]
    arima = auto_arima(train_log, seasonal=False, stepwise=True,
                       suppress_warnings=True, error_action="ignore",
                       max_p=5, max_q=5, max_d=2, information_criterion="aic")
    ap_log  = arima.predict(n_periods=len(test_log))
    ap      = np.exp(ap_log)
    aa      = np.exp(test_log.values)
    a_dir   = float(np.mean(np.sign(np.diff(ap)) == np.sign(np.diff(aa))))
    an_log  = float(arima.predict(n_periods=1).iloc[0])
    a_price = float(np.exp(an_log))
    try:
        sarima = auto_arima(train_log, seasonal=True, m=5, stepwise=True,
                            suppress_warnings=True, error_action="ignore",
                            max_p=3, max_q=3, max_d=2,
                            max_P=2, max_Q=2, max_D=1, information_criterion="aic")
        sp_log  = sarima.predict(n_periods=len(test_log))
        sp      = np.exp(sp_log)
        s_dir   = float(np.mean(np.sign(np.diff(sp)) == np.sign(np.diff(aa))))
        sn_log  = sarima.predict(n_periods=1)[0]
        s_price = float(np.exp(sn_log))
    except Exception:
        s_dir   = a_dir
        s_price = a_price
    return a_dir, a_price, s_dir, s_price


def boost_features(df):
    feat = df.copy()
    for w in [5, 10, 20, 50, 200]:
        ma = feat["Close"].rolling(w).mean()
        feat[f"Price_vs_MA{w}"] = (feat["Close"] - ma) / ma
    feat["RSI_OB"]         = (feat["RSI"] > 65).astype(int)
    feat["RSI_OS"]         = (feat["RSI"] < 35).astype(int)
    feat["RSI_Momentum"]   = feat["RSI"] - feat["RSI"].shift(3)
    feat["MACD_Cross"]     = np.sign(feat["MACD"] - feat["MACD_Signal"])
    feat["MACD_Cross_Chg"] = feat["MACD_Cross"].diff()
    feat["BB_Squeeze"]     = (feat["BB_Width"] < feat["BB_Width"].rolling(20).mean()).astype(int)
    feat["BB_Position"]    = feat["BB_PctB"].clip(0, 1)
    feat["Vol_Surge"]      = (feat["Volume"] > feat["Volume"].rolling(20).mean()*1.5).astype(int)
    feat["Vol_Trend"]      = feat["Volume"].rolling(5).mean() / feat["Volume"].rolling(20).mean()
    feat["Up_Days"]        = feat["Returns"].gt(0).astype(int)
    feat["Ret_Accel"]      = feat["Returns"] - feat["Returns"].shift(1)
    feat["Ret_3d_Sum"]     = feat["Returns"].rolling(3).sum()
    feat["Ret_5d_Sum"]     = feat["Returns"].rolling(5).sum()
    feat["Ret_10d_Sum"]    = feat["Returns"].rolling(10).sum()
    feat["Gap"]            = (feat["Open"] - feat["Close"].shift(1)) / feat["Close"].shift(1)
    feat["ATR_Move"]       = feat["Returns"].abs() / (feat["ATR_Pct"] + 1e-9)
    feat["Stoch_Cross"]    = np.sign(feat["Stoch_K"] - feat["Stoch_D"])
    high52 = feat["High"].rolling(252).max()
    low52  = feat["Low"].rolling(252).min()
    feat["Dist_52w_High"] = (feat["Close"] - high52) / high52
    feat["Dist_52w_Low"]  = (feat["Close"] - low52)  / low52
    feat["ADX_Proxy"]     = feat["ATR"].rolling(14).mean() / feat["Close"]
    feat.ffill(inplace=True); feat.bfill(inplace=True); feat.dropna(inplace=True)
    COLS = [
        "Price_vs_MA5","Price_vs_MA10","Price_vs_MA20","Price_vs_MA50","Price_vs_MA200",
        "RSI","RSI_OB","RSI_OS","RSI_Momentum",
        "MACD","MACD_Signal","MACD_Hist","MACD_Cross","MACD_Cross_Chg",
        "BB_PctB","BB_Width","BB_Squeeze","BB_Position",
        "Vol_Ratio","Vol_Surge","Vol_Trend",
        "Returns","Ret_3d_Sum","Ret_5d_Sum","Ret_10d_Sum","Ret_Accel",
        "Ret_Lag_1","Ret_Lag_2","Ret_Lag_3","Ret_Lag_5",
        "Momentum_5","Momentum_20","Gap",
        "ATR_Pct","ATR_Move","HL_Spread","CO_Spread",
        "Stoch_K","Stoch_D","Stoch_Cross","Williams_R",
        "Dist_52w_High","Dist_52w_Low","Regime",
        "DayOfWeek","Month","Quarter",
    ]
    COLS = [c for c in COLS if c in feat.columns]
    return feat, COLS


def run_ensemble(feat, COLS):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    import xgboost as xgb
    import lightgbm as lgb

    fc = feat[COLS + ["Target"]].dropna().copy()
    X  = fc[COLS].values.astype(np.float32)
    y  = fc["Target"].values.astype(int)
    n  = len(X)
    n_tr = int(n*0.70); n_vl = int(n*0.10)
    X_tr, X_vl, X_te = X[:n_tr], X[n_tr:n_tr+n_vl], X[n_tr+n_vl:]
    y_tr, y_vl, y_te = y[:n_tr], y[n_tr:n_tr+n_vl], y[n_tr+n_vl:]
    sc = RobustScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xvl_s = sc.transform(X_vl)
    Xte_s = sc.transform(X_te)
    xgb1 = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                               subsample=0.75, colsample_bytree=0.75,
                               eval_metric="logloss", use_label_encoder=False,
                               random_state=42, verbosity=0)
    xgb1.fit(Xtr_s, y_tr, eval_set=[(Xvl_s, y_vl)], verbose=False)
    xgb_acc = accuracy_score(y_te, xgb1.predict(Xte_s))
    lgb1 = lgb.LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                                subsample=0.75, colsample_bytree=0.75,
                                random_state=42, verbose=-1)
    lgb1.fit(Xtr_s, y_tr, eval_set=[(Xvl_s, y_vl)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    lgb_acc = accuracy_score(y_te, lgb1.predict(Xte_s))
    rf1 = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    rf1.fit(Xtr_s, y_tr)
    rf_acc = accuracy_score(y_te, rf1.predict(Xte_s))
    w_total  = xgb_acc + lgb_acc + rf_acc
    pb_xgb   = xgb1.predict_proba(Xte_s)[:, 1]
    pb_lgb   = lgb1.predict_proba(Xte_s)[:, 1]
    pb_rf    = rf1.predict_proba(Xte_s)[:, 1]
    prob_ens = (pb_xgb*xgb_acc + pb_lgb*lgb_acc + pb_rf*rf_acc) / w_total
    pred_ens = (prob_ens >= 0.50).astype(int)
    ens_acc  = accuracy_score(y_te, pred_ens)
    ens_prec = precision_score(y_te, pred_ens, zero_division=0)
    ens_rec  = recall_score(y_te, pred_ens, zero_division=0)
    tscv    = TimeSeriesSplit(n_splits=5)
    wf_accs = []
    for tr_idx, te_idx in tscv.split(X):
        Xtr_f = sc.fit_transform(X[tr_idx])
        Xte_f = sc.transform(X[te_idx])
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                  eval_metric="logloss", use_label_encoder=False,
                                  random_state=42, verbosity=0)
        clf.fit(Xtr_f, y[tr_idx])
        wf_accs.append(accuracy_score(y[te_idx], clf.predict(Xte_f)))
    wf_mean = float(np.mean(wf_accs))
    live_f = feat[COLS].iloc[-1:].values.astype(np.float32)
    live_s = sc.transform(live_f)
    live_prob = float((xgb1.predict_proba(live_s)[0,1]*xgb_acc +
                       lgb1.predict_proba(live_s)[0,1]*lgb_acc +
                       rf1.predict_proba(live_s)[0,1]*rf_acc) / w_total)
    return {"xgb_acc": xgb_acc, "lgb_acc": lgb_acc, "rf_acc": rf_acc,
            "ens_acc": ens_acc, "ens_prec": ens_prec, "ens_rec": ens_rec,
            "wf_mean": wf_mean, "live_prob_ens": live_prob,
            "sc": sc, "feat_cols": COLS}


def compute_sentiment(headlines):
    from textblob import TextBlob
    tb_scores = [TextBlob(h).sentiment.polarity for h in headlines]
    tb_score  = float(np.mean(tb_scores)) if headlines else 0.0
    fb_score = 0.0
    try:
        from transformers import pipeline as hf_pipeline
        LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}
        finbert = hf_pipeline("text-classification", model="ProsusAI/finbert",
                               top_k=None, truncation=True, max_length=512)
        scores = []
        for h in headlines[:10]:
            res = finbert(h)[0]
            w = sum(LABEL_MAP.get(r["label"].lower(), 0)*r["score"] for r in res)
            scores.append(w)
        fb_score = float(np.mean(scores))
    except Exception:
        fb_score = tb_score
    return tb_score, fb_score


def ask_rag(question, embedder, index, documents, groq_client, company_name):
    q_vec = embedder.encode([question], normalize_embeddings=True).astype(np.float32)
    _, ids = index.search(q_vec, k=5)   # bumped k from 3→5 since we have more docs now
    context = "\n\n".join([documents[i] for i in ids[0] if i < len(documents)])
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content":
             f"You are a stock analyst for {company_name}. "
             "Answer only using the context provided. Be concise and specific."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2, max_tokens=400
    )
    return resp.choices[0].message.content.strip()


def price_signal(pred, cur, thresh=1.0):
    pct = (pred - cur) / cur * 100
    return "BUY" if pct >= thresh else "SELL" if pct <= -thresh else "HOLD"


def chunk_text(text, size=500, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]

if run_btn:
    if not company_name or not ticker:
        st.sidebar.error("Please enter Company Name and Ticker.")
    else:
        st.session_state.company_name = company_name
        st.session_state.ticker       = ticker
        st.session_state.start_date   = str(start_date)
        st.session_state.end_date     = str(end_date)

        with st.spinner("Importing libraries…"):
            lazy_import()

        progress = st.progress(0, text="Starting…")

        # News
        progress.progress(5, "Fetching news headlines…")
        q = build_query(company_name, ticker)
        nd, gn = [], []
        if news_source != "GNews only":
            nd = fetch_newsdata(q, newsdata_key)
        if news_source != "NewsData.io only":
            gn = fetch_gnews(q, gnews_key)
        st.session_state.headlines = clean_headlines(nd + gn)

        # Screener.in Auto-Fetch
        progress.progress(12, "Fetching Screener.in data…")
        screener_data = fetch_screener_data(ticker)
        st.session_state.screener_data    = screener_data
        st.session_state.screener_fetched = True
        st.session_state.documents_extra  = (
            [screener_data["about"]] if screener_data.get("about") else []
        ) + screener_data.get("pros", []) + screener_data.get("cons", [])

        # Stock Data
        progress.progress(15, "Downloading OHLCV data…")
        import yfinance as yf
        raw = yf.download(ticker, start=str(start_date), end=str(end_date),
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None)
        ohlcv.dropna(inplace=True)
        st.session_state.ohlcv = ohlcv

        if ohlcv.empty:
            st.error("No data returned. Check ticker symbol.")
            st.stop()

        # Macro
        macro_syms = {"^NSEI": "Nifty50", "USDINR=X": "USDINR", "GC=F": "Gold"}
        macro_df   = pd.DataFrame()
        for sym, name in macro_syms.items():
            try:
                m = yf.download(sym, start=str(start_date), end=str(end_date),
                                auto_adjust=True, progress=False)
                if isinstance(m.columns, pd.MultiIndex):
                    m.columns = m.columns.get_level_values(0)
                s = m["Close"].rename(name)
                s.index = pd.to_datetime(s.index).tz_localize(None)
                macro_df[name] = s
            except Exception:
                pass
        macro_df = macro_df.reindex(ohlcv.index, method="ffill").ffill().bfill()

        # Feature Engineering
        progress.progress(25, "Engineering features…")
        df = engineer_features(ohlcv, macro_df, str(start_date), str(end_date))
        current_price = float(df["Close"].iloc[-1])

        # Prophet
        progress.progress(35, "Running Prophet…")
        from prophet import Prophet as _Prophet
        df, prophet_acc, prophet_price = run_prophet(df, ohlcv)

        # ARIMA / SARIMA
        progress.progress(50, "Fitting ARIMA/SARIMA…")
        from pmdarima import auto_arima as _auto_arima
        arima_acc, arima_price, sarima_acc, sarima_price = run_arima_sarima(df)

        # Boosted Ensemble
        progress.progress(65, "Training XGB/LGB/RF ensemble…")
        feat_df, BOOST_COLS = boost_features(df)
        ens_res = run_ensemble(feat_df, BOOST_COLS)
        live_prob_ens = ens_res["live_prob_ens"]
        wf_mean       = ens_res["wf_mean"]
        ens_acc       = ens_res["ens_acc"]

        # Sentiment
        progress.progress(78, "Running sentiment analysis…")
        headlines = st.session_state.headlines
        tb_score, fb_score = compute_sentiment(headlines)

        bull_words = ["growth","positive","strong","bullish","increase","profit","revenue"]
        bear_words = ["risk","decline","bearish","negative","loss","weak","concern"]
        dummy_ans  = " ".join(headlines).lower()
        bc = sum(1 for w in bull_words if w in dummy_ans)
        br = sum(1 for w in bear_words if w in dummy_ans)
        rag_score = float(np.clip((bc - br) / (bc + br + 1e-9), -1, 1))

        # RAG Index now includes Screener data
        progress.progress(88, "Building RAG knowledge base…")

        # Start with PDF summary (auto-fetched or previously uploaded)
        documents = chunk_text(st.session_state.pdf_summary) if st.session_state.pdf_summary else []

        # Add Screener.in raw text (ratios, pros, cons, financials)
        screener_data = st.session_state.get("screener_data", {})
        if screener_data.get("raw_text"):
            documents.extend(chunk_text(screener_data["raw_text"], size=600, overlap=100))

        # Add news headlines
        documents.extend(headlines)
        documents.extend(st.session_state.get("documents_extra", []))


        if not documents:
            documents = [f"{company_name} stock analysis context."]

        from sentence_transformers import SentenceTransformer
        import faiss as _faiss
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = embedder.encode(documents, normalize_embeddings=True).astype(np.float32)
        idx  = _faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        st.session_state.faiss_index = idx
        st.session_state.embedder    = embedder
        st.session_state.documents   = documents

        from groq import Groq
        st.session_state.groq_client = Groq(api_key=groq_key)

        # Final Signal
        progress.progress(95, "Computing final signal…")
        a_sig   = price_signal(arima_price,   current_price)
        sa_sig  = price_signal(sarima_price,  current_price)
        pr_sig  = price_signal(prophet_price, current_price)
        ens_sig = "BUY" if live_prob_ens >= 0.60 else "SELL" if live_prob_ens <= 0.40 else "HOLD"

        all_sigs = [a_sig, sa_sig, pr_sig, ens_sig]
        buy_c    = all_sigs.count("BUY")
        sell_c   = all_sigs.count("SELL")
        majority = "BUY" if buy_c > sell_c else "SELL" if sell_c > buy_c else "HOLD"

        if fb_score > 0.3 and majority == "HOLD":  majority = "BUY"
        elif fb_score < -0.3 and majority == "HOLD": majority = "SELL"
        if rag_score > 0.6:   majority = "BUY"
        elif rag_score < -0.6: majority = "SELL"
        regime_val = int(df["Regime"].iloc[-1])
        if regime_val == -1 and majority == "BUY": majority = "HOLD"

        agreement = max(buy_c, sell_c) / len(all_sigs)
        latest = df.iloc[-1]
        tech_sigs = []
        if latest["RSI"] < 30:                      tech_sigs.append("BUY")
        elif latest["RSI"] > 70:                    tech_sigs.append("SELL")
        else:                                       tech_sigs.append("HOLD")
        if latest["MACD"] > latest["MACD_Signal"]: tech_sigs.append("BUY")
        else:                                       tech_sigs.append("SELL")
        if latest["Stoch_K"] > latest["Stoch_D"]:  tech_sigs.append("BUY")
        else:                                       tech_sigs.append("SELL")
        tech_agreement = tech_sigs.count(majority) / len(tech_sigs)

        meta_conf = abs(live_prob_ens - 0.5) * 2
        sent_avg  = (fb_score + tb_score) / 2
        sent_conf = max(0, sent_avg) if majority == "BUY" else max(0, -sent_avg) if majority == "SELL" else 1.0 - abs(sent_avg)
        confidence = min(0.25*meta_conf + 0.20*agreement + 0.15*tech_agreement +
                         0.10*sent_conf + 0.10*min(wf_mean, 1.0), 0.95)

        prices_flat = df["Close"].values
        running_max = np.maximum.accumulate(prices_flat)
        max_dd      = float(((prices_flat - running_max) / running_max * 100).min())
        ann_vol     = float(df["Returns"].std() * np.sqrt(252) * 100)
        sharpe      = float((df["Returns"].mean()*252 - 0.065) / (df["Returns"].std()*np.sqrt(252) + 1e-9))
        risk_level  = "LOW" if abs(max_dd) < 10 else "MEDIUM" if abs(max_dd) < 25 else "HIGH"
        ensemble_price = float(np.mean([arima_price, sarima_price, prophet_price]))

        st.session_state.df_feat = df
        st.session_state.models_trained = True
        st.session_state.results = {
            "current_price":   current_price,
            "ensemble_price":  ensemble_price,
            "arima_price":     arima_price,
            "sarima_price":    sarima_price,
            "prophet_price":   prophet_price,
            "arima_acc":       arima_acc,
            "sarima_acc":      sarima_acc,
            "prophet_acc":     prophet_acc,
            "ens_acc":         ens_acc,
            "wf_mean":         wf_mean,
            "live_prob_ens":   live_prob_ens,
            "tb_score":        tb_score,
            "fb_score":        fb_score,
            "rag_score":       rag_score,
            "majority":        majority,
            "confidence":      confidence,
            "max_dd":          max_dd,
            "ann_vol":         ann_vol,
            "sharpe":          sharpe,
            "risk_level":      risk_level,
            "regime":          regime_val,
            "a_sig": a_sig, "sa_sig": sa_sig, "pr_sig": pr_sig, "ens_sig": ens_sig,
        }
        progress.progress(100, "Done!")
        time.sleep(0.4)
        progress.empty()
        st.success("✅ Analysis complete!")



tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", "📈 Charts", "🤖 Models", "📰 Sentiment", "🔍 Screener Data", "💬 RAG Chatbot"
])

# DASHBOARD
with tab1:
    if not st.session_state.models_trained:
        st.info("👈 Enter company details in the sidebar and click **Run Full Analysis** to begin.")
        st.markdown("""
        ### What this app does
        - **News Fetching**: pulls latest headlines from NewsData.io and GNews
        - **Screener.in Auto-Fetch**:  automatically pulls ratios, pros/cons, financials & annual report PDF
        - **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, Stochastics
        - **Models**: ARIMA, SARIMA, Prophet, XGBoost, LightGBM, Random Forest
        - **Sentiment**:  TextBlob + FinBERT on news headlines
        - **Final Signal**: ensemble BUY / SELL / HOLD with confidence score
        - **RAG Chatbot**: ask any question grounded in Screener data + annual report + news
        """)
    else:
        r  = st.session_state.results
        cn = st.session_state.company_name
        tk = st.session_state.ticker

        st.title(f"📊 {cn} ({tk})")
        st.caption(f"Analysis as of {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[r["majority"]]
        st.markdown(f"""
        <div class="metric-card">
          <span style="color:#aaa">FINAL SIGNAL</span><br>
          <span class="{sig_class}">{r['majority']}</span>
          &nbsp;&nbsp;
          <span style="color:#ccc;font-size:1rem">Confidence: <b>{r['confidence']:.1%}</b></span>
          &nbsp;&nbsp;
          <span style="color:#ccc;font-size:.9rem">P(UP): {r['live_prob_ens']:.2%}</span>
          &nbsp;&nbsp;
          <span style="color:#ccc;font-size:.9rem">Regime: {'🐂 BULL' if r['regime']==1 else '🐻 BEAR'}</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",   f"₹{r['current_price']:,.2f}")
        c2.metric("Ensemble Target", f"₹{r['ensemble_price']:,.2f}",
                  f"{(r['ensemble_price']-r['current_price'])/r['current_price']*100:+.2f}%")
        c3.metric("Sharpe Ratio",    f"{r['sharpe']:.3f}")
        c4.metric("Risk Level",      r["risk_level"])

        st.markdown("#### Individual Model Signals")
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, label, sig, price in [
            (sc1, "ARIMA",   r["a_sig"],   r["arima_price"]),
            (sc2, "SARIMA",  r["sa_sig"],  r["sarima_price"]),
            (sc3, "Prophet", r["pr_sig"],  r["prophet_price"]),
            (sc4, "Ensemble",r["ens_sig"], r["ensemble_price"]),
        ]:
            color = "#26a69a" if sig=="BUY" else "#ef5350" if sig=="SELL" else "#ffd700"
            col.markdown(f"""
            <div class="metric-card">
              <span style="color:#aaa;font-size:.8rem">{label}</span><br>
              <span style="color:{color};font-size:1.2rem;font-weight:700">{sig}</span><br>
              <span style="color:#ccc;font-size:.85rem">₹{price:,.2f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Risk Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Max Drawdown",    f"{r['max_dd']:.2f}%")
        r2.metric("Ann. Volatility", f"{r['ann_vol']:.2f}%")
        r3.metric("Walk-Fwd Acc",    f"{r['wf_mean']:.2%}")
        r4.metric("Ensemble Acc",    f"{r['ens_acc']:.2%}")


# CHARTS
with tab2:
    if st.session_state.df_feat is not None:
        df     = st.session_state.df_feat
        cn     = st.session_state.company_name
        tk     = st.session_state.ticker
        window = st.slider("Days to display", 60, min(len(df), 756), 252, 30)
        plot   = df.tail(window).copy()

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.45, 0.2, 0.2, 0.15],
                            subplot_titles=["Price + Bollinger Bands", "RSI (14)", "MACD", "Volume"],
                            vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["Close"], name="Close",
                                  line=dict(color="#00d4ff", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["BB_Upper"], name="BB Upper",
                                  line=dict(color="steelblue", dash="dash", width=0.9)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["BB_Lower"], name="BB Lower",
                                  line=dict(color="steelblue", dash="dash", width=0.9),
                                  fill="tonexty", fillcolor="rgba(70,130,180,0.07)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["Roll_Mean_21"], name="21d MA",
                                  line=dict(color="#ffd700", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["RSI"], name="RSI",
                                  line=dict(color="#ff6b6b", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_color="red",   line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_color="green", line_dash="dash", row=2, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["MACD"],        name="MACD",
                                  line=dict(color="#00d4ff")), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot.index, y=plot["MACD_Signal"], name="Signal",
                                  line=dict(color="#ff6b6b")), row=3, col=1)
        colors_hist = ["#26a69a" if v >= 0 else "#ef5350" for v in plot["MACD_Hist"]]
        fig.add_trace(go.Bar(x=plot.index, y=plot["MACD_Hist"], name="Hist",
                              marker_color=colors_hist, opacity=0.7), row=3, col=1)
        vol_colors = ["#26a69a" if r >= 0 else "#ef5350" for r in plot["Returns"]]
        fig.add_trace(go.Bar(x=plot.index, y=plot["Volume"], name="Volume",
                              marker_color=vol_colors, opacity=0.7), row=4, col=1)
        fig.update_layout(template="plotly_dark", height=760,
                          title=f"{cn} ({tk}) — Technical Dashboard",
                          showlegend=True, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.01))
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run analysis first to see charts.")


# MODELS
with tab3:
    if st.session_state.models_trained:
        r  = st.session_state.results
        cn = st.session_state.company_name
        st.subheader("Model Performance Summary")
        models = ["ARIMA", "SARIMA", "Prophet", "XGB/LGB/RF Ensemble"]
        accs   = [r["arima_acc"]*100, r["sarima_acc"]*100, r["prophet_acc"]*100, r["ens_acc"]*100]
        fig_cmp = go.Figure(go.Bar(
            x=models, y=accs,
            marker_color=["#26a69a" if a >= 55 else "#ef5350" for a in accs],
            text=[f"{a:.1f}%" for a in accs], textposition="outside"
        ))
        fig_cmp.add_hline(y=55, line_dash="dash", line_color="#ffd700",
                          annotation_text="Target 55%", annotation_font_color="#ffd700")
        fig_cmp.update_layout(template="plotly_dark", height=380,
                               title="Directional Accuracy Comparison",
                               yaxis=dict(title="Accuracy %", range=[0, 100]))
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.markdown("#### Price Forecasts")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ARIMA",   f"₹{r['arima_price']:,.2f}",
                     f"{(r['arima_price']-r['current_price'])/r['current_price']*100:+.2f}%")
        col_b.metric("SARIMA",  f"₹{r['sarima_price']:,.2f}",
                     f"{(r['sarima_price']-r['current_price'])/r['current_price']*100:+.2f}%")
        col_c.metric("Prophet", f"₹{r['prophet_price']:,.2f}",
                     f"{(r['prophet_price']-r['current_price'])/r['current_price']*100:+.2f}%")
        st.markdown("#### Walk-Forward Validation (5-fold)")
        st.metric("Mean Walk-Forward Accuracy", f"{r['wf_mean']:.2%}")
        st.caption("Uses time-series cross-validation to prevent data leakage.")
        st.markdown("#### Ensemble Classifier")
        e1, e2, e3 = st.columns(3)
        e1.metric("Ensemble Acc", f"{r['ens_acc']:.2%}")
        e2.metric("P(UP)",        f"{r['live_prob_ens']:.2%}")
        e3.metric("Signal",       r["ens_sig"])
    else:
        st.info("Run analysis first.")


# SENTIMENT
with tab4:
    if st.session_state.models_trained:
        r = st.session_state.results
        st.subheader("Sentiment Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("TextBlob Score", f"{r['tb_score']:+.4f}",
                  "Positive" if r["tb_score"] > 0 else "Negative")
        c2.metric("FinBERT Score",  f"{r['fb_score']:+.4f}",
                  "Positive" if r["fb_score"] > 0 else "Negative")
        c3.metric("RAG Score",      f"{r['rag_score']:+.4f}",
                  "Bullish" if r["rag_score"] > 0 else "Bearish")
        headlines = st.session_state.headlines
        st.markdown(f"#### News Headlines ({len(headlines)} fetched)")
        if headlines:
            for i, h in enumerate(headlines, 1):
                st.markdown(f"**{i}.** {h}")
        else:
            st.warning("No headlines fetched. Check API keys and company name.")
    else:
        st.info("Run analysis first.")


# SCREENER DATA (replaces old Annual Report tab)
with tab5:
    st.subheader("🔍 Screener.in — Auto-Fetched Company Data")

    if not st.session_state.screener_fetched:
        st.info("Run analysis to auto-fetch Screener.in data for this company.")

        st.markdown("#### Or upload Annual Report PDF manually")
        uploaded = st.file_uploader("Upload Annual Report PDF", type=["pdf"])
        if uploaded:
            with st.spinner("Extracting & summarising PDF…"):
                try:
                    import pdfplumber, tempfile
                    from groq import Groq
                    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                    tmp.write(uploaded.read()); tmp.flush()
                    pages = []
                    with pdfplumber.open(tmp.name) as pdf:
                        for i, page in enumerate(pdf.pages):
                            if i >= 20: break
                            txt = page.extract_text()
                            if txt: pages.append(txt)
                    full_text = "\n\n".join(pages)
                    client = Groq(api_key=groq_key)
                    chunks = [full_text[i:i+15000] for i in range(0, min(len(full_text), 60000), 15000)]
                    chunk_summaries = []
                    for chunk in chunks:
                        resp = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content":
                                       f"Summarise this annual report section in bullet points "
                                       f"(revenue, profit, risks, outlook):\n\n{chunk}"}],
                            temperature=0.2, max_tokens=512
                        )
                        chunk_summaries.append(resp.choices[0].message.content.strip())
                    final_resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content":
                                   "Combine these into one executive summary with "
                                   "Financial Performance, Business Highlights, Risks, Outlook:\n\n" +
                                   "\n---\n".join(chunk_summaries)}],
                        temperature=0.2, max_tokens=800
                    )
                    summary = final_resp.choices[0].message.content.strip()
                    st.session_state.pdf_summary = summary
                    st.success("PDF summarised and added to knowledge base! Re-run analysis to include it.")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"PDF processing failed: {e}")
    else:
        sd = st.session_state.screener_data
        cn = st.session_state.company_name

        st.markdown(f"**Source:** [{sd['url']}]({sd['url']})")

        # About
        if sd.get("about"):
            st.markdown(f"#### About {cn}")
            st.write(sd["about"])

        # Key Ratios
        if sd.get("ratios"):
            st.markdown("#### 📊 Key Ratios")
            ratio_items = list(sd["ratios"].items())
            cols = st.columns(4)
            for i, (k, v) in enumerate(ratio_items):
                cols[i % 4].metric(k, v)

        # Pros & Cons
        col_p, col_c = st.columns(2)
        with col_p:
            if sd.get("pros"):
                st.markdown("#### Strengths")
                for p in sd["pros"]:
                    st.markdown(f"- {p}")
            else:
                st.markdown("#### Strengths")
                st.caption("None found on page.")
        with col_c:
            if sd.get("cons"):
                st.markdown("#### Weaknesses")
                for c in sd["cons"]:
                    st.markdown(f"- {c}")
            else:
                st.markdown("#### Weaknesses")
                st.caption("None found on page.")

        # Quarterly financials
        if sd.get("quarterly_sales"):
            st.markdown("####  Quarterly Sales (₹ Cr)")
            periods = [p for p, _ in sd["quarterly_sales"]]
            values  = [v.replace(",", "") for _, v in sd["quarterly_sales"]]
            try:
                fig_q = go.Figure(go.Bar(
                    x=periods, y=[float(v) for v in values],
                    marker_color="#00d4ff", text=values, textposition="outside"
                ))
                fig_q.update_layout(template="plotly_dark", height=300,
                                    title="Quarterly Revenue", showlegend=False)
                st.plotly_chart(fig_q, use_container_width=True)
            except Exception:
                st.table(sd["quarterly_sales"])

        # Annual report PDF
        if sd.get("annual_report_url"):
            st.markdown("#### Annual Report")
            st.markdown(f" [Download Annual Report PDF]({sd['annual_report_url']})")
            if st.session_state.pdf_summary:
                st.markdown("#### AI Summary of Annual Report")
                st.markdown(st.session_state.pdf_summary)
            else:
                if st.button(" Summarise Annual Report Now"):
                    with st.spinner("Downloading & summarising…"):
                        try:
                            summary = download_and_extract_pdf(sd["annual_report_url"], groq_key)
                            st.session_state.pdf_summary = summary
                            st.success("Done! Re-run analysis to include in RAG.")
                            st.markdown(summary)
                        except Exception as e:
                            st.error(f"Failed: {e}")
        else:
            st.info("No annual report PDF link found on Screener.in. You can upload one manually below.")
            uploaded = st.file_uploader("Upload Annual Report PDF (fallback)", type=["pdf"])
            if uploaded:
                with st.spinner("Extracting & summarising PDF…"):
                    try:
                        import pdfplumber, tempfile
                        from groq import Groq
                        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                        tmp.write(uploaded.read()); tmp.flush()
                        pages = []
                        with pdfplumber.open(tmp.name) as pdf:
                            for i, page in enumerate(pdf.pages):
                                if i >= 20: break
                                txt = page.extract_text()
                                if txt: pages.append(txt)
                        full_text = "\n\n".join(pages)
                        client = Groq(api_key=groq_key)
                        chunks_list = [full_text[i:i+15000] for i in range(0, min(len(full_text), 60000), 15000)]
                        chunk_summaries = []
                        for chunk in chunks_list:
                            resp = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[{"role": "user", "content":
                                           f"Summarise this annual report section in bullet points:\n\n{chunk}"}],
                                temperature=0.2, max_tokens=512
                            )
                            chunk_summaries.append(resp.choices[0].message.content.strip())
                        final_resp = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content":
                                       "Combine into executive summary (Financial Performance, Risks, Outlook):\n\n" +
                                       "\n---\n".join(chunk_summaries)}],
                            temperature=0.2, max_tokens=800
                        )
                        st.session_state.pdf_summary = final_resp.choices[0].message.content.strip()
                        st.success("PDF summarised! Re-run analysis to include in RAG.")
                        st.markdown(st.session_state.pdf_summary)
                    except Exception as e:
                        st.error(f"PDF processing failed: {e}")


# RAG CHATBOT
with tab6:
    st.subheader(" RAG Chatbot")
    cn = st.session_state.company_name or "the company"

    if not st.session_state.models_trained:
        st.info("Run analysis first to initialise the knowledge base.")
    else:
        doc_count = len(st.session_state.documents)
        has_screener = bool(st.session_state.get("screener_data", {}).get("raw_text"))
        has_pdf      = bool(st.session_state.pdf_summary)

        st.caption(
            f"Knowledge base: **{doc_count} chunks** — "
            f"{'Screener data' if has_screener else 'No Screener data'} · "
            f"{' Annual report' if has_pdf else 'No annual report'} · "
            f"{' News headlines' if st.session_state.headlines else 'No news'}"
        )

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_q    = st.text_input("Your question", placeholder="What are the key risks for this company?")
            submitted = st.form_submit_button("Send ➤")

        if submitted and user_q.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Thinking…"):
                try:
                    ans = ask_rag(user_q, st.session_state.embedder, st.session_state.faiss_index,
                                  st.session_state.documents, st.session_state.groq_client, cn)
                except Exception as e:
                    ans = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("#### Quick Questions")
        qqs = [
            "What are the key risks?",
            "What are the growth drivers?",
            "What do the key ratios say about valuation?",
            "Summarise the latest news sentiment.",
            "Is the company financially healthy?",
            "What is the revenue trend?",
        ]
        cols = st.columns(2)
        for i, qq in enumerate(qqs):
            if cols[i % 2].button(qq, key=f"qq_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": qq})
                with st.spinner("Thinking…"):
                    try:
                        ans = ask_rag(qq, st.session_state.embedder, st.session_state.faiss_index,
                                      st.session_state.documents, st.session_state.groq_client, cn)
                    except Exception as e:
                        ans = f"Error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()
