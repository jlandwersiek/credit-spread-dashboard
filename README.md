"""
# Credit Spread Options Dashboard

This Streamlit app scans credit spread opportunities and lets you place trades via Alpaca's paper trading API.

## Features
- Real-time spread scanner from Tradier
- Live Alpaca positions monitoring
- Trade logging, exit logging, and P/L calculations
- CSV export and performance stats

## Run Locally
```bash
pip install -r requirements.txt
streamlit run credit_spread_dashboard.py
```

## Deploy on Streamlit Cloud
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Set API credentials under "Secrets"

## Secrets format
Set your keys in **Streamlit Cloud > Settings > Secrets**:
```
APCA_API_KEY="your_alpaca_key"
APCA_SECRET_KEY="your_secret"
TRADIER_TOKEN="your_tradier_token"
```

## Strategy Presets Explained

| Preset Name              | Description                                                                 | Key Filters Applied                                                                                   |
|--------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Neutral Income**       | Balanced trades focused on high POP and safety                              | Max Delta = 0.3, Exclude Expected Move, Earnings filter, Delta range: 0.2–0.4                         |
| **High IV Reversion**    | Designed to benefit from volatility mean reversion                          | IV Rank > 0.5, RSI + Bollinger enabled, Delta 0.15–0.35, Credit ≥ $0.5                                |
| **Aggressive Weeklies**  | Fast, short-term income trades with looser filters                          | Wide delta range 0.1–0.5, minimal OI/volume/credit constraints                                         |
| **Defensive Income**     | Cautious entries in high-IV environments with strong confirmation           | IV Rank > 0.4, RSI/Bollinger required, Max Delta 0.25, Min Credit $0.25                               |
| **Aggressive Income**    | Focus on higher-premium opportunities with more risk                        | Delta 0.25–0.5, Lower IV/volume/credit constraints                                                    |
| **Earnings Play**        | Play implied volatility crush post-earnings                                 | IV Rank > 0.6, Delta 0.1–0.3, Close to earnings date                                                   |
| **Volatility Crush Hunter**| Filters for IV extremes to capture volatility compression                  | IV Rank > 0.7, Max Delta = 0.2, RSI/Bollinger required                                                 |
| **Weekly Scalper**       | Simple scalps for weekly income                                             | Minimal filters, fast expiration, lower entry barriers                                                |
| **Hybrid Defensive-Aggressive** | Mix of stable and opportunistic setups                             | Balanced filters with POP + RSI + Bollinger + Earnings exclusion                                      |
| **Hybrid Income-Scalper**| Mid-point between short-term and stable income strategies                   | IV Rank > 0.25, Moderate delta (0.15–0.35), Loose earnings buffer                                      |

Each preset configures a unique mix of:
- IV Rank filter
- Delta range
- Credit minimum
- RSI/Bollinger logic
- Earnings avoidance window

You can apply presets from the **Strategy Builder tab** or create and save your own custom presets.
"""
