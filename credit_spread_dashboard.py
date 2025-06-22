import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, time
import math
import zoneinfo  # Python 3.9+ built-in, no install needed
import pytz
import pandas_market_calendars as mcal  # install via pip if needed

# Get current time in Eastern
eastern = pytz.timezone('US/Eastern')
now = datetime.now(eastern)

# Market hours
market_open = time(9, 30)
market_close = time(16, 0)

# Calendar setup
nyse = mcal.get_calendar('NYSE')

# Get today's trading schedule (set a 1-day range)
today_str = now.strftime('%Y-%m-%d')
schedule = nyse.schedule(start_date=today_str, end_date=today_str)

# Logic flags
is_weekday = now.weekday() < 5
is_market_day = not schedule.empty
is_market_hours = market_open <= now.time() <= market_close

# Set TEST_MODE
TEST_MODE = not (is_weekday and is_market_day and is_market_hours)

st.set_page_config(page_title="Credit Spread Dashboard with Presets & Orders", layout="wide")

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    st.error("Please install alpaca_trade_api: pip install alpaca-trade-api")
    st.stop()

# ---------------------------
# Streamlit sidebar for API keys
st.sidebar.header("Enter Your API Keys")

TRADIER_TOKEN = st.sidebar.text_input("Tradier Token", type="password")
ALPACA_API_KEY = st.sidebar.text_input("Alpaca API Key", type="password")
ALPACA_SECRET_KEY = st.sidebar.text_input("Alpaca Secret Key", type="password")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

if not (TRADIER_TOKEN and ALPACA_API_KEY and ALPACA_SECRET_KEY):
    st.warning("Please enter all API keys in the sidebar to use the app.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}

try:
    alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
    alpaca.get_account()  # Test connection
except Exception as e:
    st.error(f"Failed to connect to Alpaca: {e}")
    st.stop()

# ---------------------------
# Trade log state initialization
if "trade_log" not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=[
        "trade_id", "symbol", "entry_date", "exit_date", "qty",
        "entry_price", "exit_price", "realized_pl", "unrealized_pl", "status"
    ])
# ---------------------------
# Helper functions

@st.cache_data(ttl=60)
def get_expirations(symbol):
    if TEST_MODE:
        st.write("TEST_MODE: Returning mock expirations")
        mock_dates = ['2024-06-21']
        st.write(f"Mock expirations: {mock_dates}")
        return mock_dates
    url = "https://sandbox.tradier.com/v1/markets/options/expirations"
    params = {"symbol": symbol, "includeAllRoots": "true"}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    if 'expirations' in data and 'date' in data['expirations']:
        return data['expirations']['date']
    return []

@st.cache_data(ttl=30)
def get_chain(symbol, expiry):
    if TEST_MODE:
        st.write("TEST_MODE: Loading mock chain from CSV")
        df = pd.read_csv("mock_chain.csv")
        st.write(f"Mock chain head:\n{df.head()}")
        return df
    url = "https://sandbox.tradier.com/v1/markets/options/chains"
    params = {"symbol": symbol, "expiration": expiry, "greeks": "true"}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    if 'options' in data and 'option' in data['options']:
        df = pd.json_normalize(data['options']['option'])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_quote(symbol):
    if TEST_MODE:
        st.write("TEST_MODE: Returning mock quote")
        mock_quote = 435.00
        st.write(f"Mock quote: {mock_quote}")
        return mock_quote
    url = f"https://sandbox.tradier.com/v1/markets/quotes"
    params = {"symbols": symbol}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    try:
        q = data['quotes']['quote']
        if isinstance(q, list):
            return q[0]['last']
        return q['last']
    except:
        return None

@st.cache_data(ttl=300)
def get_earnings(symbol):
    if TEST_MODE:
        st.write("TEST_MODE: Returning mock earnings dates")
        mock_earnings = [datetime(2024, 6, 25)]
        st.write(f"Mock earnings: {mock_earnings}")
        return mock_earnings
    today = datetime.utcnow().strftime('%Y-%m-%d')
    url = f"https://sandbox.tradier.com/v1/markets/calendar/earnings"
    params = {"symbol": symbol, "start": today, "end": (datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d')}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    if 'earnings' in data and 'calendar' in data['earnings']:
        dates = [item['date'] for item in data['earnings']['calendar']]
        return [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    return []
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta).clip(lower=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, window=20, stds=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + stds * std
    lower = ma - stds * std
    return upper, lower

def expected_move(price, iv, dte):
    return price * iv * math.sqrt(dte / 365)

def find_spreads(
    df, 
    underlying_price,
    min_iv_rank=0,
    min_oi=0,
    min_volume=0,
    min_credit=0,
    delta_range=(0.2, 0.4),
    max_delta=None,
    exclude_price_move=False,
    exclude_earnings_days=None,
    earnings_dates=None,
    enable_rsi_filter=False,
    rsi=None,
    rsi_thresholds=(30, 70),
    enable_bollinger=False,
    boll_upper=None,
    boll_lower=None,
    debug=False,
    spread_width=None
):
    df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce').dt.tz_localize(None)
    df['bid_iv'] = df['greeks.bid_iv'].astype(float)
    df['delta'] = df['greeks.delta'].astype(float)
    df['volume'] = df['volume'].astype(int)
    df['open_interest'] = df['open_interest'].astype(int)

    df = df[df['bid_iv'] >= min_iv_rank]
    df = df[df['open_interest'] >= min_oi]
    df = df[df['volume'] >= min_volume]

    results = []
    for typ in ['put', 'call']:
        sub = df[df['option_type'] == typ].sort_values('strike')
        for i in range(len(sub) - 1):
            short = sub.iloc[i]
            long = sub.iloc[i + 1]
            width = abs(long['strike'] - short['strike'])

            if spread_width is not None and width != spread_width:
                continue

            credit = short['bid'] - long['ask']
            if credit < min_credit:
                continue

            max_loss = width - credit
            roi = credit / max_loss if max_loss else 0

            if not (delta_range[0] <= abs(short['delta']) <= delta_range[1]):
                continue

            if max_delta is not None and abs(short['delta']) > max_delta:
                continue

            if short['volume'] < min_volume or short['open_interest'] < min_oi:
                continue

            if exclude_price_move:
                try:
                    dte = (short['expiration_date'] - pd.Timestamp.utcnow().replace(tzinfo=None)).days
                except Exception as e:
                    if debug:
                        print("Error calculating DTE:", e)
                    continue

                if dte <= 0:
                    continue

                iv = short['bid_iv']
                move = expected_move(underlying_price, iv, dte)
                dist = abs(short['strike'] - underlying_price)
                if dist < move:
                    continue

            if exclude_earnings_days is not None and earnings_dates is not None:
                exclude = False
                for edate in earnings_dates:
                    days_to_earnings = (edate - pd.Timestamp.utcnow().replace(tzinfo=None)).days
                    if 0 <= days_to_earnings <= exclude_earnings_days:
                        if abs((short['expiration_date'] - edate).days) <= exclude_earnings_days:
                            exclude = True
                            break
                if exclude:
                    continue

            if enable_rsi_filter and rsi is not None:
                if typ == 'put' and rsi[-1] > rsi_thresholds[0]:
                    continue
                if typ == 'call' and rsi[-1] < rsi_thresholds[1]:
                    continue

            if enable_bollinger and boll_upper is not None and boll_lower is not None:
                if typ == 'put' and underlying_price < boll_lower[-1]:
                    continue
                if typ == 'call' and underlying_price > boll_upper[-1]:
                    continue

            results.append(dict(
                type="Bull Put" if typ == 'put' else "Bear Call",
                sell_strike=short['strike'],
                buy_strike=long['strike'],
                expiry=short['expiration_date'].strftime('%Y-%m-%d'),
                credit=round(credit, 2),
                max_loss=round(max_loss, 2),
                roi=round(roi * 100, 1),
                iv_rank=round(short['bid_iv'], 2),
                delta=round(short['delta'], 2),
                volume=short['volume'],
                oi=short['open_interest']
            ))

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        return df_results.sort_values('roi', ascending=False)
    else:
        if debug:
            print("No spreads found with current filters.")
        return df_results

def get_positions():
    try:
        positions = alpaca.list_positions()
        data = []
        for p in positions:
            if p.asset_class == 'option':
                data.append({
                    'symbol': p.symbol,
                    'qty': int(p.qty),
                    'market_value': float(p.market_value),
                    'cost_basis': float(p.cost_basis),
                    'unrealized_pl': float(p.unrealized_pl)
                })
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not fetch Alpaca positions: {e}")
        return pd.DataFrame()

def place_credit_spread_order(symbol, sell_strike, buy_strike, expiry, spread_type, qty):
    try:
        option_type = 'put' if spread_type == "Bull Put" else 'call'
        def build_occ_symbol(sym, exp_date, opt_type, strike_price):
            dt = datetime.strptime(exp_date, "%Y-%m-%d")
            yymmdd = dt.strftime("%y%m%d")
            strike_int = int(round(strike_price * 1000))
            strike_str = f"{strike_int:08d}"
            return f"{sym}{yymmdd}{opt_type[0].upper()}{strike_str}"
        sell_symbol = build_occ_symbol(symbol, expiry, option_type, sell_strike)
        buy_symbol = build_occ_symbol(symbol, expiry, option_type, buy_strike)
        legs = [
            {"side": "sell", "option_symbol": sell_symbol, "quantity": qty},
            {"side": "buy", "option_symbol": buy_symbol, "quantity": qty}
        ]
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='vertical',  # Check Alpaca API docs for correct multi-leg order syntax
            time_in_force='day',
            order_class='vertical',
            legs=legs
        )
        return order
    except Exception as e:
        return str(e)

def fetch_live_positions_with_unrealized_pl():
    try:
        positions = alpaca.list_positions()
        data = []
        for p in positions:
            unrealized = float(p.unrealized_pl)
            data.append({
                "trade_id": p.id if hasattr(p, "id") else p.symbol + str(p.qty) + p.asset_id,
                "symbol": p.symbol,
                "qty": int(p.qty),
                "entry_date": p.asset_class,  # Placeholder for lack of entry date from Alpaca
                "entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": unrealized,
                "status": "OPEN"
            })
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not fetch Alpaca positions: {e}")
        return pd.DataFrame()

def log_exit_trade():
    st.subheader("Log Exit Trade")

    # Move these checks outside the form so the form always has a submit button
    if st.session_state.trade_log.empty:
        st.info("No open trades in log to close.")
        return

    open_trades = st.session_state.trade_log[st.session_state.trade_log["status"] == "OPEN"]

    if open_trades.empty:
        st.info("No open trades to exit.")
        return

    with st.form("exit_trade_form"):
        trade_to_exit = st.selectbox(
            "Select trade to exit",
            open_trades["trade_id"] + " - " + open_trades["symbol"],
            format_func=lambda x: x
        )

        exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01)
        exit_date = st.date_input("Exit Date", datetime.utcnow().date())

        submitted = st.form_submit_button("Log Exit")

        if submitted:
            trade_id = trade_to_exit.split(" - ")[0]
            idx = st.session_state.trade_log.index[st.session_state.trade_log["trade_id"] == trade_id].tolist()
            if not idx:
                st.error("Trade not found.")
                return
            idx = idx[0]
            entry_price = st.session_state.trade_log.at[idx, "entry_price"]
            qty = st.session_state.trade_log.at[idx, "qty"]
            realized_pl = (exit_price - entry_price) * qty

            st.session_state.trade_log.at[idx, "exit_price"] = exit_price
            st.session_state.trade_log.at[idx, "exit_date"] = pd.to_datetime(exit_date)
            st.session_state.trade_log.at[idx, "realized_pl"] = realized_pl
            st.session_state.trade_log.at[idx, "status"] = "CLOSED"
            st.success(f"Trade {trade_id} closed with P/L: ${realized_pl:.2f}")

def show_trade_log_and_export():
    st.subheader("Trade Log")
    if st.session_state.trade_log.empty:
        st.info("No trades logged yet.")
    else:
        live_positions = fetch_live_positions_with_unrealized_pl()
        trade_log = st.session_state.trade_log.copy()
        for idx, row in trade_log.iterrows():
            if row["status"] == "OPEN":
                match = live_positions[live_positions["symbol"] == row["symbol"]]
                if not match.empty:
                    trade_log.at[idx, "unrealized_pl"] = match.iloc[0]["unrealized_pl"]

        st.session_state.trade_log = trade_log

        st.dataframe(st.session_state.trade_log)

        csv = st.session_state.trade_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Trade Log CSV",
            data=csv,
            file_name="trade_log.csv",
            mime="text/csv"
        )

def show_performance_analytics():
    st.subheader("Performance Analytics")
    if st.session_state.trade_log.empty or st.session_state.trade_log[st.session_state.trade_log["status"] == "CLOSED"].empty:
        st.info("No closed trades to analyze.")
        return

    closed = st.session_state.trade_log[st.session_state.trade_log["status"] == "CLOSED"]

    win_rate = (closed["realized_pl"] > 0).mean()
    avg_roi = (closed["realized_pl"] / (closed["entry_price"] * closed["qty"])).mean()

    closed["entry_date"] = pd.to_datetime(closed["entry_date"], errors='coerce')
    closed["exit_date"] = pd.to_datetime(closed["exit_date"], errors='coerce')
    closed = closed.dropna(subset=["entry_date", "exit_date"])
    closed["days_in_trade"] = (closed["exit_date"] - closed["entry_date"]).dt.days
    avg_days_in_trade = closed["days_in_trade"].mean()

    st.markdown(f"**Win Rate:** {win_rate:.1%}")
    st.markdown(f"**Average ROI per trade:** {avg_roi:.2%}")
    st.markdown(f"**Average Time in Trade:** {avg_days_in_trade:.1f} days")

# ---------------------------
# Streamlit UI

if 'filters' not in st.session_state:
    st.session_state.filters = {
        'min_iv_rank': 0.3,
        'min_oi': 50,
        'min_volume': 10,
        'min_credit': 0.3,
        'delta_low': 0.2,
        'delta_high': 0.4,
        'max_delta': 0.3,
        'enable_pop_filter': True,
        'exclude_price_move': True,
        'exclude_earnings_days': 7,
        'enable_earnings_filter': True,
        'enable_rsi_filter': False,
        'rsi_low': 30,
        'rsi_high': 70,
        'enable_bollinger': False,
    }
if 'custom_presets' not in st.session_state:
    st.session_state.custom_presets = {}

def reset_to_default():
    st.session_state.filters = {
        'min_iv_rank': 0.3,
        'min_oi': 50,
        'min_volume': 10,
        'min_credit': 0.3,
        'delta_low': 0.2,
        'delta_high': 0.4,
        'max_delta': 0.3,
        'enable_pop_filter': True,
        'exclude_price_move': True,
        'exclude_earnings_days': 7,
        'enable_earnings_filter': True,
        'enable_rsi_filter': False,
        'rsi_low': 30,
        'rsi_high': 70,
        'enable_bollinger': False,
    }

PRESETS = {
    "Neutral Income": {
        'enable_pop_filter': True,
        'max_delta': 0.3,
        'exclude_price_move': True,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 7,
        'enable_rsi_filter': False,
        'enable_bollinger': False,
        'min_iv_rank': 0.3,
        'min_oi': 50,
        'min_volume': 10,
        'min_credit': 0.3,
        'delta_low': 0.2,
        'delta_high': 0.4,
    },
    "High IV Reversion": {
        'enable_pop_filter': True,
        'max_delta': 0.25,
        'exclude_price_move': True,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 5,
        'enable_rsi_filter': True,
        'rsi_low': 20,
        'rsi_high': 60,
        'enable_bollinger': True,
        'min_iv_rank': 0.5,
        'min_oi': 100,
        'min_volume': 20,
        'min_credit': 0.5,
        'delta_low': 0.15,
        'delta_high': 0.35,
    },
    "Aggressive Weeklies": {
        'enable_pop_filter': False,
        'exclude_price_move': False,
        'enable_earnings_filter': False,
        'enable_rsi_filter': False,
        'enable_bollinger': False,
        'min_iv_rank': 0.0,
        'min_oi': 10,
        'min_volume': 5,
        'min_credit': 0.1,
        'delta_low': 0.1,
        'delta_high': 0.5,
    },
    # Your new presets added here:
    "Defensive Income": {
        'enable_pop_filter': True,
        'max_delta': 0.25,
        'exclude_price_move': True,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 10,
        'enable_rsi_filter': True,
        'rsi_low': 35,
        'rsi_high': 65,
        'enable_bollinger': True,
        'min_iv_rank': 0.4,
        'min_oi': 80,
        'min_volume': 15,
        'min_credit': 0.25,
        'delta_low': 0.15,
        'delta_high': 0.35,
    },
    "Aggressive Income": {
        'enable_pop_filter': False,
        'exclude_price_move': False,
        'enable_earnings_filter': False,
        'enable_rsi_filter': False,
        'enable_bollinger': False,
        'min_iv_rank': 0.2,
        'min_oi': 30,
        'min_volume': 10,
        'min_credit': 0.2,
        'delta_low': 0.25,
        'delta_high': 0.5,
    },
    "Earnings Play": {
        'enable_pop_filter': False,
        'exclude_price_move': False,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 0,
        'enable_rsi_filter': False,
        'enable_bollinger': False,
        'min_iv_rank': 0.6,
        'min_oi': 50,
        'min_volume': 20,
        'min_credit': 0.4,
        'delta_low': 0.1,
        'delta_high': 0.3,
    },
    "Volatility Crush Hunter": {
        'enable_pop_filter': True,
        'max_delta': 0.2,
        'exclude_price_move': True,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 5,
        'enable_rsi_filter': True,
        'rsi_low': 40,
        'rsi_high': 60,
        'enable_bollinger': True,
        'min_iv_rank': 0.7,
        'min_oi': 120,
        'min_volume': 25,
        'min_credit': 0.6,
        'delta_low': 0.1,
        'delta_high': 0.3,
    },
    "Weekly Scalper": {
        'enable_pop_filter': False,
        'exclude_price_move': False,
        'enable_earnings_filter': False,
        'enable_rsi_filter': False,
        'enable_bollinger': False,
        'min_iv_rank': 0.0,
        'min_oi': 20,
        'min_volume': 10,
        'min_credit': 0.15,
        'delta_low': 0.1,
        'delta_high': 0.4,
    },
    "Hybrid Defensive-Aggressive": {
        'enable_pop_filter': True,
        'max_delta': 0.3,
        'exclude_price_move': True,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 7,
        'enable_rsi_filter': True,
        'rsi_low': 30,
        'rsi_high': 65,
        'enable_bollinger': True,
        'min_iv_rank': 0.35,
        'min_oi': 70,
        'min_volume': 15,
        'min_credit': 0.3,
        'delta_low': 0.2,
        'delta_high': 0.4,
    },
    "Hybrid Income-Scalper": {
        'enable_pop_filter': False,
        'exclude_price_move': False,
        'enable_earnings_filter': True,
        'exclude_earnings_days': 3,
        'enable_rsi_filter': False,
        'enable_bollinger': False,
        'min_iv_rank': 0.25,
        'min_oi': 30,
        'min_volume': 10,
        'min_credit': 0.2,
        'delta_low': 0.15,
        'delta_high': 0.35,
    }
}

def apply_preset(name):
    preset = PRESETS.get(name) or st.session_state.custom_presets.get(name)
    if preset:
        for k, v in preset.items():
            st.session_state.filters[k] = v

tabs = st.tabs(["Scan", "Live Trades", "Strategy Builder", "Trade Log"])

with tabs[0]:
    st.header("Credit Spread Scanner")
    symbol = st.text_input("Ticker symbol (e.g. SPY)", "SPY").upper()

    expirations = get_expirations(symbol)
    if not expirations:
        st.warning("No expirations found or invalid symbol.")
    else:
        expiry = st.selectbox("Choose Expiration Date", expirations)
        price = get_quote(symbol)
        if price is None:
            st.warning("Could not fetch price for symbol.")
        else:
            df_chain = get_chain(symbol, expiry)
            if df_chain.empty:
                st.warning("No option chain data found.")
            else:
                dummy_series = pd.Series(np.random.normal(price, price * 0.01, 100))
                rsi_series = rsi(dummy_series)
                boll_upper, boll_lower = bollinger_bands(dummy_series)

                f = st.session_state.filters

                # ✅ NEW SPREAD WIDTH INPUT
                spread_width = st.selectbox("Choose spread width:", options=[1, 5, 10], index=1)

                filtered_spreads = find_spreads(
                    df_chain,
                    underlying_price=price,
                    min_iv_rank=f['min_iv_rank'],
                    min_oi=f['min_oi'],
                    min_volume=f['min_volume'],
                    min_credit=f['min_credit'],
                    delta_range=(f['delta_low'], f['delta_high']),
                    max_delta=f['max_delta'] if f['enable_pop_filter'] else None,
                    exclude_price_move=f['exclude_price_move'],
                    exclude_earnings_days=f['exclude_earnings_days'] if f['enable_earnings_filter'] else None,
                    earnings_dates=get_earnings(symbol) if f['enable_earnings_filter'] else None,
                    enable_rsi_filter=f['enable_rsi_filter'],
                    rsi=rsi_series,
                    rsi_thresholds=(f['rsi_low'], f['rsi_high']),
                    enable_bollinger=f['enable_bollinger'],
                    boll_upper=boll_upper,
                    boll_lower=boll_lower,
                    spread_width=None  # ✅ INCLUDED HERE
                )

                if filtered_spreads.empty:
                    st.info("No spreads found with current filters.")
                else:
                    st.dataframe(filtered_spreads)

with tabs[1]:
    st.header("Live Alpaca Option Positions & Order Placement")

    positions_df = get_positions()
    if positions_df.empty:
        st.info("No open option positions found in Alpaca paper account.")
    else:
        st.dataframe(positions_df)

    st.markdown("---")
    st.subheader("Place a Credit Spread Order")

    with st.form("order_form"):
        symbol_order = st.text_input("Underlying Symbol (e.g. SPY)", value=symbol if 'symbol' in locals() else "SPY").upper()
        spread_type = st.selectbox("Spread Type", ["Bull Put", "Bear Call"])
        sell_strike = st.number_input("Sell Strike Price", min_value=0.0, step=0.5)
        buy_strike = st.number_input("Buy Strike Price", min_value=0.0, step=0.5)
        expiry_order = st.text_input("Expiration Date (YYYY-MM-DD)", value=expiry if 'expiry' in locals() else "")
        qty = st.number_input("Quantity (number of spreads)", min_value=1, step=1, value=1)

        submitted = st.form_submit_button("Place Order")
        if submitted:
            try:
                datetime.strptime(expiry_order, "%Y-%m-%d")
            except:
                st.error("Expiration date format must be YYYY-MM-DD")
                st.stop()
            if buy_strike <= sell_strike:
                st.error("Buy strike must be greater than sell strike")
                st.stop()

            order_response = place_credit_spread_order(
                symbol_order, sell_strike, buy_strike, expiry_order, spread_type, qty
            )
            if isinstance(order_response, str):
                st.error(f"Order failed: {order_response}")
            else:
                st.success(f"Order placed! ID: {order_response.id}")

with tabs[2]:
    st.header("Strategy Builder & Presets")

    preset_name = st.selectbox("Select a Preset", options=["-- Select --"] + list(PRESETS.keys()) + list(st.session_state.custom_presets.keys()))

    if preset_name in PRESETS or preset_name in st.session_state.custom_presets:
        if st.button("Apply Preset"):
            apply_preset(preset_name)
            st.success(f"Applied preset: {preset_name}")

    st.markdown("### Current Filters")
    f = st.session_state.filters

    col1, col2 = st.columns(2)

    with col1:
        f['min_iv_rank'] = st.slider("Min IV Rank", 0.0, 1.0, f['min_iv_rank'], 0.05, help="Minimum implied volatility rank")
        f['min_oi'] = st.number_input("Min Open Interest", 0, 10000, f['min_oi'], step=10, help="Minimum open interest for option legs")
        f['min_volume'] = st.number_input("Min Volume", 0, 10000, f['min_volume'], step=10, help="Minimum volume for option legs")
        f['min_credit'] = st.number_input("Min Credit ($)", 0.0, 10.0, f['min_credit'], 0.05, help="Minimum credit received for spread")
        f['delta_low'] = st.slider("Min Delta", 0.0, 1.0, f['delta_low'], 0.05, help="Minimum delta of short leg")
        f['delta_high'] = st.slider("Max Delta", 0.0, 1.0, f['delta_high'], 0.05, help="Maximum delta of short leg")
        f['spread_width'] = st.selectbox("Spread Width", options=[1, 2, 5, 10], index=2, help="Width between strikes for the credit spread")

    with col2:
        f['enable_pop_filter'] = st.checkbox("Enable Probability of Profit (Max Delta) Filter", f['enable_pop_filter'], help="Filter spreads with delta above max delta")
        f['exclude_price_move'] = st.checkbox("Exclude if Price Within Expected Move", f['exclude_price_move'], help="Exclude spreads too close to underlying price")
        f['enable_earnings_filter'] = st.checkbox("Exclude if Earnings Within Days", f['enable_earnings_filter'], help="Avoid earnings volatility")
        if f['enable_earnings_filter']:
            f['exclude_earnings_days'] = st.number_input("Exclude if Earnings Within X Days", 0, 30, f['exclude_earnings_days'], help="Number of days before expiry to exclude spreads")
        f['enable_rsi_filter'] = st.checkbox("Enable RSI Filter", f['enable_rsi_filter'], help="Filter spreads by RSI indicator")
        if f['enable_rsi_filter']:
            f['rsi_low'] = st.slider("RSI Low Threshold", 0, 100, f['rsi_low'], help="Bullish RSI threshold for bull puts")
            f['rsi_high'] = st.slider("RSI High Threshold", 0, 100, f['rsi_high'], help="Bearish RSI threshold for bear calls")
        f['enable_bollinger'] = st.checkbox("Enable Bollinger Bands Filter", f['enable_bollinger'], help="Filter spreads by Bollinger Bands")

    st.markdown("---")
    st.markdown("### Save Current Filters as Custom Preset")

    new_preset_name = st.text_input("Preset Name")
    if st.button("Save Preset"):
        if new_preset_name:
            st.session_state.custom_presets[new_preset_name] = st.session_state.filters.copy()
            st.success(f"Preset '{new_preset_name}' saved!")
        else:
            st.error("Please enter a preset name to save.")

    if st.button("Reset to Default Filters"):
        reset_to_default()
        st.success("Filters reset to default.")

with tabs[3]:
    st.header("Trade Log and Exit Trades")
    # Optional: Add form to manually add an open trade for testing/demo
    with st.expander("Add Open Trade"):
        with st.form("add_open_trade_form"):
            symbol = st.text_input("Symbol", "SPY")
            qty = st.number_input("Quantity", min_value=1, step=1, value=1)
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01, value=0.0)
            entry_date = st.date_input("Entry Date", datetime.utcnow().date())
            trade_id = f"{symbol}_{entry_date.strftime('%Y%m%d')}_{qty}"

            submitted = st.form_submit_button("Add Open Trade")
            if submitted:
                new_trade = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "entry_date": pd.to_datetime(entry_date),
                    "exit_date": pd.NaT,
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": np.nan,
                    "realized_pl": np.nan,
                    "unrealized_pl": 0.0,
                    "status": "OPEN"
                }
                st.session_state.trade_log = pd.concat([st.session_state.trade_log, pd.DataFrame([new_trade])], ignore_index=True)
                st.success(f"Added open trade {trade_id}")

    st.markdown("---")

    # Show Exit Trade form
    log_exit_trade()

    st.markdown("---")

    # Show trade log with export button
    show_trade_log_and_export()

    st.markdown("---")

    # Show performance analytics on closed trades
    show_performance_analytics()

