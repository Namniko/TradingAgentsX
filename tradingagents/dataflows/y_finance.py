from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf
import os
from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry, load_ohlcv, filter_financials_by_date

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def _spr_classify_trend(e8, e13, e21, e48, e200, curling_down, curling_up):
    if e21 > e48 and e21 > e200:
        if e8 > e13 > e21:
            return "Max Bull"
        elif e8 > e21:
            return "Strong Bull"
        elif curling_down:
            return "Weak Bull"
        else:
            return "Moderate Bull"
    elif e21 < e48 and e21 < e200:
        if e8 < e13 < e21:
            return "Max Bear"
        elif e8 < e21:
            return "Strong Bear"
        elif curling_up:
            return "Weak Bear"
        else:
            return "Moderate Bear"
    return "Warning"


def _spr_classify_momentum(close, e8, e13, e21, trend):
    if "Bull" in trend:
        if close > e8:
            return "Strong"
        elif close > e13:
            return "Fading"
        elif close > e21:
            return "Weak"
        return "Warning"
    elif "Bear" in trend:
        if close < e8:
            return "Strong"
        elif close < e21:
            return "Fading"
        return "Counter-trend"
    return "Neutral"


def _spr_classify_200(close, e21, e48, e200, atr, prev_e21, prev_e200):
    if prev_e21 is not None and prev_e200 is not None:
        if prev_e21 <= prev_e200 and e21 > e200:
            return "Active - Golden Cross"
        if prev_e21 >= prev_e200 and e21 < e200:
            return "Active - Death Cross"
    if atr > 0 and abs(close - e200) <= atr:
        return "Active - Price Testing"
    if e200 > 0 and abs(e48 - e200) / e200 <= 0.015:
        return "Watch - 48 Approaching"
    return "Inactive"


def _spr_reversal_stages(df):
    n = len(df)
    stages = ["N/A"] * n

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        e8 = row["ema_8"]; e13 = row["ema_13"]
        e21 = row["ema_21"]; e48 = row["ema_48"]
        close = row["Close"]; trend = row["trend_state"]

        # Find prior clean trend context (look back up to 20 bars)
        prior_ctx = None
        for j in range(i - 1, max(i - 20, -1), -1):
            t = df.iloc[j]["trend_state"]
            if "Bull" in t:
                prior_ctx = "bull"; break
            elif "Bear" in t:
                prior_ctx = "bear"; break
        if prior_ctx is None:
            continue

        recent = [stages[j] for j in range(max(0, i - 5), i)]
        was_reversing = any(
            s in ("Stage 1 Warning", "Stage 2 Gradual", "Stage 2 Violent") for s in recent
        )

        # Stage 3: clean opposite trend emerged after a reversal progression
        if was_reversing:
            if prior_ctx == "bull" and "Bear" in trend:
                stages[i] = "Stage 3 Confirmed"; continue
            if prior_ctx == "bear" and "Bull" in trend:
                stages[i] = "Stage 3 Confirmed"; continue

        if prior_ctx == "bull":
            had_vomy = "Stage 2 Gradual" in recent
            # Stage 2 Violent: breaks below EMA48 with no prior Vomy consolidation
            if close < e48 and not had_vomy:
                stages[i] = "Stage 2 Violent"; continue
            # Stage 2 Gradual (Vomy): EMA8 crossed below EMA13, price in 13-21 zone 2-7 days
            if e8 < e13:
                days = 0
                for j in range(i, max(i - 8, -1), -1):
                    zr = df.iloc[j]
                    if zr["ema_21"] <= zr["Close"] <= zr["ema_13"] and zr["Close"] > zr["ema_48"]:
                        days += 1
                    else:
                        break
                if 2 <= days <= 7:
                    stages[i] = "Stage 2 Gradual"; continue
            # Stage 1 Warning: curling, crossed below EMA21, or fast cloud compressing
            crossed_below = close < e21 and prev["Close"] >= prev["ema_21"]
            cloud_compress = abs(e8 - e21) < abs(prev["ema_8"] - prev["ema_21"])
            if row["curling_down"] or crossed_below or cloud_compress:
                stages[i] = "Stage 1 Warning"

        elif prior_ctx == "bear":
            had_vomy = "Stage 2 Gradual" in recent
            if close > e48 and not had_vomy:
                stages[i] = "Stage 2 Violent"; continue
            if e8 > e13:
                days = 0
                for j in range(i, max(i - 8, -1), -1):
                    zr = df.iloc[j]
                    if zr["ema_13"] <= zr["Close"] <= zr["ema_21"] and zr["Close"] < zr["ema_48"]:
                        days += 1
                    else:
                        break
                if 2 <= days <= 7:
                    stages[i] = "Stage 2 Gradual"; continue
            crossed_above = close > e21 and prev["Close"] <= prev["ema_21"]
            cloud_compress = abs(e8 - e21) < abs(prev["ema_8"] - prev["ema_21"])
            if row["curling_up"] or crossed_above or cloud_compress:
                stages[i] = "Stage 1 Warning"

    return stages


def _spr_setups(df):
    _levels = ["Strongest", "Strong", "Moderate", "Weak", "Marginal"]
    _penalty = {"Max Bull": 0, "Strong Bull": 0, "Moderate Bull": 1, "Weak Bull": 2,
                "Max Bear": 0, "Strong Bear": 0, "Moderate Bear": 1, "Weak Bear": 2}
    setups = ["None"] * len(df)
    strengths = ["N/A"] * len(df)

    for i in range(1, len(df)):
        row = df.iloc[i]
        rev = row["reversal_stage"]
        if rev != "N/A":
            if rev == "Stage 2 Gradual":
                setups[i] = "Vomy Forming"
            elif rev == "Stage 2 Violent":
                setups[i] = "Violent Reversal"
            continue

        trend = row["trend_state"]; bias = row["bias"]
        e8 = row["ema_8"]; e13 = row["ema_13"]; e21 = row["ema_21"]
        close = row["Close"]; low = row["Low"]; high = row["High"]

        if bias == "Bullish" and "Bull" in trend:
            if low <= e8 and close >= e8:
                base = 0
            elif low <= e13 and close >= e13:
                base = 1
            elif low <= e21 and close >= e21:
                base = 2
            else:
                continue
            setups[i] = "Pullback Buy"
            strengths[i] = _levels[min(base + _penalty.get(trend, 0), 4)]

        elif bias == "Bearish" and "Bear" in trend:
            if high >= e8 and close <= e8:
                base = 0
            elif high >= e13 and close <= e13:
                base = 1
            elif high >= e21 and close <= e21:
                base = 2
            else:
                continue
            setups[i] = "Pullback Short"
            strengths[i] = _levels[min(base + _penalty.get(trend, 0), 4)]

    return setups, strengths


def get_saty_pivot_ribbon(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Compute Saty Pivot Ribbon Pro state: EMA stack trend, bias, momentum,
    reversal progression, and entry setups for the given symbol and date."""
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        return f"No OHLCV data available for {symbol} up to {curr_date}."

    df = data.copy().sort_values("Date").reset_index(drop=True)

    # EMAs — uses full history for accurate EMA200 seeding
    for span, col in [(8, "ema_8"), (13, "ema_13"), (21, "ema_21"), (48, "ema_48"), (200, "ema_200")]:
        df[col] = df["Close"].ewm(span=span, adjust=False).mean()

    # ATR14
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"] - df["Low"]), (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    # Curling signals: slope flip AND gap narrowing (both required)
    e8_d = df["ema_8"] - df["ema_8"].shift(1)
    gap = (df["ema_8"] - df["ema_13"]).abs()
    df["curling_down"] = ((e8_d.shift(1) > 0) & (e8_d <= 0)) & (gap < gap.shift(1)) & (df["ema_8"] >= df["ema_21"])
    df["curling_up"]   = ((e8_d.shift(1) < 0) & (e8_d >= 0)) & (gap < gap.shift(1)) & (df["ema_8"] <= df["ema_21"])

    # Per-row state computation
    date_strs, trends, biases, moms, s200s, stacks = [], [], [], [], [], []
    for i, row in df.iterrows():
        e8, e13 = row["ema_8"], row["ema_13"]
        e21, e48, e200 = row["ema_21"], row["ema_48"], row["ema_200"]
        close, atr = row["Close"], row["atr_14"]
        prev = df.iloc[i - 1] if i > 0 else None

        date_strs.append(row["Date"].strftime("%Y-%m-%d"))

        labeled = sorted([(e8,"8"),(e13,"13"),(e21,"21"),(e48,"48"),(e200,"200")], key=lambda x: x[0], reverse=True)
        stacks.append(" > ".join(x[1] for x in labeled))

        trend = _spr_classify_trend(e8, e13, e21, e48, e200, row["curling_down"], row["curling_up"])
        trends.append(trend)

        if prev is not None:
            above_now = close > e21
            above_prev = prev["Close"] > prev["ema_21"]
            bias = "Warning" if above_now != above_prev else ("Bullish" if above_now else "Bearish")
        else:
            bias = "Bullish" if close > e21 else "Bearish"
        biases.append(bias)

        moms.append(_spr_classify_momentum(close, e8, e13, e21, trend))
        s200s.append(_spr_classify_200(
            close, e21, e48, e200, atr,
            prev["ema_21"] if prev is not None else None,
            prev["ema_200"] if prev is not None else None,
        ))

    df["date_str"] = date_strs
    df["trend_state"] = trends
    df["bias"] = biases
    df["momentum"] = moms
    df["ema_200_status"] = s200s
    df["ema_stack"] = stacks
    df["reversal_stage"] = _spr_reversal_stages(df)
    df["setup"], df["setup_strength"] = _spr_setups(df)

    # Locate latest trading row on or before curr_date
    window = df[df["date_str"] <= curr_date]
    if window.empty:
        return f"No trading data found for {symbol} on or before {curr_date}."
    lr = window.iloc[-1]

    # History window
    before_str = (
        datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(days=look_back_days)
    ).strftime("%Y-%m-%d")
    hist = df[(df["date_str"] >= before_str) & (df["date_str"] <= curr_date)]

    def _summary(r):
        rev = r["reversal_stage"]; trend = r["trend_state"]; bias = r["bias"]
        mom = r["momentum"]; setup = r["setup"]; strength = r["setup_strength"]
        s200 = r["ema_200_status"]; sym = symbol.upper()
        line1 = (f"{sym} is in {rev} as of {r['date_str']}." if rev != "N/A"
                 else f"{sym} is in a {trend} trend with {bias.lower()} bias.")
        line2 = (f"Momentum is {mom.lower()}." if mom not in ("Neutral", "Warning")
                 else "Momentum is neutral or in warning — caution advised.")
        extras = []
        if setup not in ("None", "N/A"):
            extras.append(f"A {setup} ({strength.lower()} conviction) is present.")
        if s200 != "Inactive":
            extras.append(f"200 EMA: {s200}.")
        return f"{line1} {line2}" + (f" {' '.join(extras)}" if extras else "")

    # Current state block
    current = (
        f"SYMBOL: {symbol.upper()}\n"
        f"DATE: {lr['date_str']}\n\n"
        f"TREND STATE: {lr['trend_state']}\n\n"
        f"EMA STACK: {lr['ema_stack']}\n"
        f"EMA VALUES: 8={lr['ema_8']:.2f} | 13={lr['ema_13']:.2f} | "
        f"21={lr['ema_21']:.2f} | 48={lr['ema_48']:.2f} | 200={lr['ema_200']:.2f}\n"
        f"CLOSE: {lr['Close']:.2f}\n\n"
        f"BIAS: {lr['bias']}\n\n"
        f"MOMENTUM: {lr['momentum']}\n\n"
        f"200 EMA STATUS: {lr['ema_200_status']}\n\n"
        f"SETUP: {lr['setup']}\n\n"
        f"SETUP STRENGTH: {lr['setup_strength']}\n\n"
        f"REVERSAL STAGE: {lr['reversal_stage']}\n\n"
        f"SUMMARY: {_summary(lr)}"
    )

    # History table
    w = [10, 15, 10, 13, 21, 7]
    hdr = (f"{'Date':<{w[0]}} {'Trend State':<{w[1]}} {'Bias':<{w[2]}} "
           f"{'Momentum':<{w[3]}} {'Reversal Stage':<{w[4]}} {'Close':>{w[5]}}")
    sep = "-" * (sum(w) + 5)
    rows = []
    for _, r in hist.sort_values("date_str", ascending=False).iterrows():
        rows.append(
            f"{r['date_str']:<{w[0]}} {r['trend_state']:<{w[1]}} {r['bias']:<{w[2]}} "
            f"{r['momentum']:<{w[3]}} {r['reversal_stage']:<{w[4]}} {r['Close']:>{w[5]}.2f}"
        )

    return (
        f"## Saty Pivot Ribbon Pro — {symbol.upper()}\n\n"
        f"### Current State\n\n{current}\n\n---\n\n"
        f"### {look_back_days}-Day History\n\n"
        f"{hdr}\n{sep}\n" + "\n".join(rows) + "\n\n---\n\n"
        f"### Signal Reference\n"
        f"TREND STATE: Max/Strong/Moderate/Weak Bull|Bear, Warning (EMA 8/13/21/48 stack vs 200)\n"
        f"BIAS: Bullish=close>EMA21, Bearish=close<EMA21, Warning=just crossed\n"
        f"MOMENTUM: Strong=above EMA8, Fading=EMA8-13 zone, Weak=EMA13-21 zone, Counter-trend/Warning\n"
        f"SETUP: Pullback Buy/Short (no active reversal), Vomy Forming (Stage 2 Gradual), Violent Reversal\n"
        f"REVERSAL: Stage 1 Warning → Stage 2 Gradual|Violent → Stage 3 Confirmed\n"
        f"200 EMA: Inactive | Watch-48 Approaching | Active-Price Testing | Golden/Death Cross\n"
    )


def _ttm_linreg(series: pd.Series, length: int) -> pd.Series:
    """Linear regression value at last bar — equivalent to PineScript linreg(src, length, 0)."""
    def _lr(y):
        n = len(y)
        xm = (n - 1) / 2.0
        ym = y.mean()
        denom = sum((j - xm) ** 2 for j in range(n))
        if denom == 0:
            return ym
        slope = sum((j - xm) * (y[j] - ym) for j in range(n)) / denom
        return slope * (n - 1) + (ym - slope * xm)
    return series.rolling(length).apply(_lr, raw=True)


def get_ttm_squeeze_pro(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Compute TTM Squeeze Pro: volatility compression dots + momentum histogram."""
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        return f"No OHLCV data available for {symbol} up to {curr_date}."

    df = data.copy().sort_values("Date").reset_index(drop=True)
    length = 20

    # Bollinger Bands
    sma20 = df["Close"].rolling(length).mean()
    std20 = df["Close"].rolling(length).std(ddof=1)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20

    # True Range and Keltner Channels (SMA of TR — matches PineScript ta.sma(ta.tr, length))
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    dev_kc = tr.rolling(length).mean()
    kc_u1 = sma20 + dev_kc * 1.0;  kc_l1 = sma20 - dev_kc * 1.0
    kc_u15 = sma20 + dev_kc * 1.5; kc_l15 = sma20 - dev_kc * 1.5
    kc_u2 = sma20 + dev_kc * 2.0;  kc_l2 = sma20 - dev_kc * 2.0

    # ATR14 for LATE signal distance
    atr14 = tr.ewm(span=14, adjust=False).mean()

    # Momentum: linreg(close - avg(avg(HH,LL), SMA20), 20, 0)
    hh = df["High"].rolling(length).max()
    ll = df["Low"].rolling(length).min()
    delta = df["Close"] - ((hh + ll) / 2.0 + sma20) / 2.0
    mom = _ttm_linreg(delta, length)

    # --- Helper: classify dot ---
    def _dot(i):
        if pd.isna(bb_lower.iloc[i]) or pd.isna(kc_l1.iloc[i]):
            return "NoSqz"
        bbl, bbu = bb_lower.iloc[i], bb_upper.iloc[i]
        if bbl >= kc_l1.iloc[i] or bbu <= kc_u1.iloc[i]:   return "HighSqz"
        if bbl >= kc_l15.iloc[i] or bbu <= kc_u15.iloc[i]: return "MidSqz"
        if bbl >= kc_l2.iloc[i] or bbu <= kc_u2.iloc[i]:   return "LowSqz"
        return "NoSqz"

    # --- Helper: momentum zone ---
    def _zone(m, mp):
        if pd.isna(m) or mp is None or pd.isna(mp): return "Neutral"
        if m > 0 and mp <= 0: return "Crossing Up"
        if m <= 0 and mp > 0: return "Crossing Down"
        if m > 0 and m > mp:  return "Strong Bull"
        if m > 0:             return "Fading Bull"
        if m < mp:            return "Strong Bear"
        if m < 0:             return "Fading Bear"
        return "Neutral"

    # --- Helper: signal strength ---
    def _strength(comp_n, last_comp_dot, zone):
        accel = zone in ("Strong Bull", "Strong Bear")
        orange = last_comp_dot == "HighSqz"
        if comp_n >= 20 and orange and accel: return "Maximum"
        if comp_n >= 10 and orange and accel: return "Maximum"
        if comp_n >= 5  and orange and accel: return "Very Strong"
        if comp_n >= 10 and accel:            return "Strong"
        if comp_n >= 5  and accel:            return "Moderate"
        return "Weak"

    # --- Helper: duration category ---
    def _dur(n):
        if n == 0:   return ""
        if n <= 4:   return "Premature"
        if n <= 9:   return "Standard"
        if n <= 19:  return "Extended"
        return "Extreme"

    # --- Forward pass ---
    n = len(df)
    dots = [_dot(i) for i in range(n)]
    compr_counts = [0] * n
    transitions  = ["None"] * n
    mom_zones    = [None] * n
    signals      = ["NO-SIGNAL"] * n
    strengths    = ["N/A"] * n
    atr_notes    = [""] * n

    active_compr = 0
    active_start = None

    for i in range(n):
        dot      = dots[i]
        prev_dot = dots[i - 1] if i > 0 else "NoSqz"
        close_i  = df["Close"].iloc[i]
        atr_i    = atr14.iloc[i]
        m_i      = mom.iloc[i]
        m_prev   = mom.iloc[i - 1] if i > 0 else None

        # Compression tracking
        fired_start = None
        fired_count = 0
        if dot != "NoSqz":
            if prev_dot == "NoSqz":
                active_compr = 1
                active_start = close_i
                transitions[i] = "Compression Building"
            else:
                active_compr += 1
                transitions[i] = "Peak Compression" if dot == "HighSqz" else "Compression Building"
        else:
            if prev_dot != "NoSqz":
                fired_start  = active_start
                fired_count  = active_compr
                transitions[i] = "Just Fired"
                active_compr = 0
                active_start = None

        compr_counts[i] = active_compr

        # Momentum zone
        z = _zone(m_i, m_prev)
        mom_zones[i] = z
        pz  = mom_zones[i - 1] if i > 0 else None

        atr_note = ""

        # Signal — priority order
        # 1. EXIT
        if z == "Fading Bull":
            if pz == "Fading Bull":
                sig = "EXIT-LONG"
            elif pz == "Strong Bull":
                sig = "EXIT-WARNING-LONG"
            else:
                sig = "NO-SIGNAL"
        elif z == "Fading Bear":
            if pz == "Fading Bear":
                sig = "EXIT-SHORT"
            elif pz == "Strong Bear":
                sig = "EXIT-WARNING-SHORT"
            else:
                sig = "NO-SIGNAL"
        # 2. CONFIRMED / LATE (first green dot)
        elif dot == "NoSqz" and prev_dot != "NoSqz":
            if fired_count < 2:
                sig = "NO-SIGNAL"
            elif z in ("Strong Bull", "Fading Bull", "Crossing Up"):
                dist = abs(close_i - fired_start) / atr_i if (atr_i > 0 and fired_start) else 0.0
                if dist <= 1.0:
                    sig = "CONFIRMED-BULL"
                else:
                    sig = "LATE-BULL"
                    atr_note = (f"Price has moved {dist:.1f} ATR from squeeze start of "
                                f"{fired_start:.2f}. Current price {close_i:.2f}. "
                                f"Reward reduced, risk elevated.")
            elif z in ("Strong Bear", "Fading Bear", "Crossing Down"):
                dist = abs(close_i - fired_start) / atr_i if (atr_i > 0 and fired_start) else 0.0
                if dist <= 1.0:
                    sig = "CONFIRMED-BEAR"
                else:
                    sig = "LATE-BEAR"
                    atr_note = (f"Price has moved {dist:.1f} ATR from squeeze start of "
                                f"{fired_start:.2f}. Current price {close_i:.2f}. "
                                f"Reward reduced, risk elevated.")
            else:
                sig = "NO-SIGNAL"
        # 3. ANTICIPATORY / SETUP-WATCH (during compression)
        elif dot != "NoSqz":
            qualifies = active_compr >= 5 or dot == "HighSqz"
            if qualifies and z == "Strong Bull":
                sig = "ANTICIPATORY-BULL"
            elif qualifies and z == "Strong Bear":
                sig = "ANTICIPATORY-BEAR"
            elif active_compr >= 1:
                sig = "SETUP-WATCH"
            else:
                sig = "NO-SIGNAL"
        else:
            sig = "NO-SIGNAL"

        signals[i]   = sig
        atr_notes[i] = atr_note

        # Strength (entry signals only)
        if sig in ("ANTICIPATORY-BULL", "ANTICIPATORY-BEAR",
                   "CONFIRMED-BULL", "CONFIRMED-BEAR", "LATE-BULL", "LATE-BEAR"):
            comp_n = fired_count if (dot == "NoSqz" and prev_dot != "NoSqz") else active_compr
            ref_dot = prev_dot if (dot == "NoSqz" and prev_dot != "NoSqz") else dot
            strengths[i] = _strength(comp_n, ref_dot, z)

    # Assign to df
    df["date_str"]         = df["Date"].dt.strftime("%Y-%m-%d")
    df["dot_color"]        = dots
    df["compression_count"] = compr_counts
    df["squeeze_transition"] = transitions
    df["mom_zone"]         = mom_zones
    df["signal"]           = signals
    df["signal_strength"]  = strengths
    df["atr_note"]         = atr_notes

    # Locate latest row on or before curr_date
    window = df[df["date_str"] <= curr_date]
    if window.empty:
        return f"No trading data found for {symbol} on or before {curr_date}."
    lr = window.iloc[-1]

    # History window
    before_str = (
        datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(days=look_back_days)
    ).strftime("%Y-%m-%d")
    hist = df[(df["date_str"] >= before_str) & (df["date_str"] <= curr_date)]

    _dot_lbl = {"NoSqz": "No Squeeze (Fired)", "LowSqz": "Low Compression",
                "MidSqz": "Mid Compression",  "HighSqz": "High Compression"}
    _mom_dir = {"Strong Bull": "Above zero rising",   "Fading Bull": "Above zero falling",
                "Strong Bear": "Below zero falling",  "Fading Bear": "Below zero rising",
                "Crossing Up": "Crossing zero upward","Crossing Down": "Crossing zero downward",
                "Neutral": "Near zero / neutral"}

    def _summary(r):
        dot = r["dot_color"]; comp = int(r["compression_count"])
        z = r["mom_zone"]; sig = r["signal"]; sym = symbol.upper()
        if dot == "NoSqz":
            l1 = f"{sym} squeeze has fired — no active compression."
        else:
            l1 = f"{sym} is in {_dot_lbl[dot]} ({comp} bars, {_dur(comp)})."
        l2 = f"Momentum is {z.lower()}."
        l3 = f"Signal: {sig}." + (f" {r['atr_note']}" if r["atr_note"] else "")
        return f"{l1} {l2} {l3}"

    comp_lr = int(lr["compression_count"])
    dur_str = f"{comp_lr} bars — {_dur(comp_lr)}" if comp_lr > 0 else "0 bars (no active compression)"

    current = (
        f"SYMBOL: {symbol.upper()}\n"
        f"DATE: {lr['date_str']}\n\n"
        f"SQUEEZE STATE: {_dot_lbl.get(lr['dot_color'], lr['dot_color'])}\n"
        f"SQUEEZE DURATION: {dur_str}\n"
        f"SQUEEZE TRANSITION: {lr['squeeze_transition']}\n\n"
        f"MOMENTUM: {lr['mom_zone']}\n"
        f"MOMENTUM DIRECTION: {_mom_dir.get(lr['mom_zone'], lr['mom_zone'])}\n\n"
        f"SIGNAL: {lr['signal']}\n\n"
        f"SIGNAL STRENGTH: {lr['signal_strength']}\n\n"
        f"ATR NOTE: {lr['atr_note'] if lr['atr_note'] else 'N/A'}\n\n"
        f"SUMMARY: {_summary(lr)}"
    )

    w = [10, 22, 15, 14, 20, 7]
    hdr = (f"{'Date':<{w[0]}} {'Squeeze State':<{w[1]}} {'Duration':<{w[2]}} "
           f"{'Momentum':<{w[3]}} {'Signal':<{w[4]}} {'Close':>{w[5]}}")
    sep = "-" * (sum(w) + 5)
    rows = []
    for _, r in hist.sort_values("date_str", ascending=False).iterrows():
        c = int(r["compression_count"])
        dur_r = f"{c}b/{_dur(c)}" if c > 0 else "0"
        rows.append(
            f"{r['date_str']:<{w[0]}} {_dot_lbl.get(r['dot_color'], r['dot_color']):<{w[1]}} "
            f"{dur_r:<{w[2]}} {r['mom_zone']:<{w[3]}} {r['signal']:<{w[4]}} {r['Close']:>{w[5]}.2f}"
        )

    return (
        f"## TTM Squeeze Pro — {symbol.upper()}\n\n"
        f"### Current State\n\n{current}\n\n---\n\n"
        f"### {look_back_days}-Day History\n\n"
        f"{hdr}\n{sep}\n" + "\n".join(rows) + "\n\n---\n\n"
        f"### Signal Reference\n"
        f"DOTS: NoSqz=Green(fired) | LowSqz=Black | MidSqz=Red | HighSqz=Orange\n"
        f"DURATION: Premature=1-4 bars | Standard=5-9 | Extended=10-19 | Extreme=20+\n"
        f"MOMENTUM: Strong Bull/Bear=accelerating | Fading=decelerating | Crossing=zero cross\n"
        f"ANTICIPATORY: 5+ bars OR orange dot + directional non-crossing momentum\n"
        f"CONFIRMED: first green dot + momentum aligned + within 1 ATR of squeeze start price\n"
        f"LATE: first green dot + price beyond 1 ATR from squeeze start — report with ATR note\n"
        f"EXIT-LONG: 2nd consecutive Fading Bull | EXIT-SHORT: 2nd consecutive Fading Bear\n"
    )


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
        "ttm_squeeze_pro": (
            "TTM Squeeze Pro: Volatility compression (squeeze dots) + momentum histogram. "
            "Squeeze dots: NoSqz=Green(fired) | LowSqz=Black | MidSqz=Red | HighSqz=Orange. "
            "Momentum: linear regression oscillator — positive/rising=bullish, negative/falling=bearish. "
            "Signals: ANTICIPATORY (5+ bars or orange dot + directional momentum), "
            "CONFIRMED/LATE (first green dot, within/beyond 1 ATR from squeeze start), "
            "EXIT-LONG/SHORT (2nd consecutive decelerating momentum bar). "
            "Use alongside saty_pivot_ribbon for trend context. Use look_back_days=30 or more."
        ),
        "saty_pivot_ribbon": (
            "Saty Pivot Ribbon Pro: A multi-EMA ribbon system using EMAs 8/13/21/48/200. "
            "Produces structured state: Trend State (EMA stack ordering), Bias (price vs EMA21), "
            "Momentum (price position within stack), Entry Setups (Pullback Buy/Short when no reversal "
            "is active), Reversal Progression (Stage 1 Warning → Stage 2 Gradual/Violent → Stage 3 "
            "Confirmed), and 200 EMA status. Returns current state block plus history table. "
            "Use look_back_days=30 or more for full reversal progression context."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    # Custom computation paths (bypass stockstats)
    if indicator == "saty_pivot_ribbon":
        return get_saty_pivot_ribbon(symbol, curr_date, look_back_days)
    if indicator == "ttm_squeeze_pro":
        return get_ttm_squeeze_pro(symbol, curr_date, look_back_days)

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator
    
    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)

        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = []
        for label, value in fields:
            if value is not None:
                lines.append(f"{label}: {value}")

        header = f"# Company Fundamentals for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())

        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_income_stmt)
        else:
            data = yf_retry(lambda: ticker_obj.income_stmt)

        data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.insider_transactions)
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"