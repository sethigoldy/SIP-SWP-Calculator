# SIP → SWP Financial Calculation Assistant

Interactive Streamlit app to plan a Systematic Investment Plan (SIP) accumulation phase followed by an inflation-indexed Systematic Withdrawal Plan (SWP) while attempting to preserve real corpus value.

## Core Idea
You specify an inflation-adjusted (real) monthly withdrawal you want to start after an accumulation period. The app estimates:
1. Real return after accounting for fees, (optional) annualized LTCG drag, and inflation.
2. Required corpus in today's rupees using a perpetuity model (Real Withdrawal / Real Return).
3. Nominal corpus needed at the end of accumulation (inflated forward).
4. Monthly SIP required to reach that corpus under chosen return assumptions.

## Formulas
Net Nominal Return (simplified):
	(1 + g_gross) * (1 - fee) - 1  → gains after fee
	gains_after_tax = gains_after_fee * (1 - tax)  (if tax drag applied during accumulation)

Real Return:
	(1 + g_real) = (1 + g_nominal) / (1 + inflation)

Required Corpus Today (Perpetuity):
	Corpus_0 = Withdrawal_annual / g_real

Inflated Corpus at Start:
	Corpus_start = Corpus_0 * (1 + inflation)^N

SIP Needed (future value of annuity due to monthly contributions):
	FV = SIP * (( (1 + r_m)^n - 1 ) / r_m)
	SIP = FV / [ ((1 + r_m)^n - 1) / r_m ]

Where r_m = (1 + annual_return)^(1/12) - 1.

## Inputs (Sidebar)
- Monthly withdrawal (today's ₹)
- Inflation %
- Comma-separated gross annual return scenarios
- LTCG tax % (simplified annual haircut on gains)
- Annual fees %
- SIP accumulation years
- Optional safety buffer % on corpus
- Toggle: Apply tax drag during accumulation

## Output
- Stepwise explanation with equations
- Summary comparison table across scenarios
- Warning when real return ≤ 0 (infinite corpus requirement)

## Quick Start
Create & activate virtual environment (already auto-configured if using the provided tooling):
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Run the app:
```bash
python -m streamlit run app.py
```

## Notes & Limitations
- Tax treatment is simplified (ignores exemption thresholds, realization timing, loss offsets, etc.).
- Real perpetuity model assumes infinite horizon; adjust buffer % to be conservative.
- If real return ≈ 0, model breaks (denominator); plan may require either lower withdrawals or higher returns.

## Next Enhancements (Ideas)
- Variable withdrawal phases
- Stochastic return simulation (Monte Carlo)
- Tax lot realization modeling
- Goal-based glide path

---
Use the sidebar to experiment with different assumptions and observe SIP sensitivity to net & real returns.
