# SIP → SWP Planning Assistant

Probably the most realistic (yet still simplified) Indian SIP & SWP calculator for everyday planning.

## 1. What This Does

You tell the app:

* How much monthly income you want later (in today’s rupees)
* How long you’ll invest (SIP years)
* Your expected gross return scenarios (e.g. 9, 10, 11, 12)
* Inflation, fees, and long‑term capital gains tax drag
* Whether you want the corpus to last forever (preserve real purchasing power) or be spent down over a fixed number of years

The app then computes for each return scenario:

| Step | Result |
|------|--------|
| 1 | Net nominal return after fees + tax drag |
| 2 | Real return after inflation |
| 3 | Corpus required today (perpetual or finite horizon model) |
| 4 | Corpus needed at start of withdrawals (inflated) |
| 5 | Monthly SIP required to reach that amount |

It also builds full month‑by‑month SIP and SWP schedules (with totals) and a yearly corpus trajectory chart.

## 2. Modes: Preserve vs Spend Down

| Mode | When to Use | Formula Core |
|------|-------------|--------------|
| Preserve Corpus (Perpetual) | You want inflation‑indexed withdrawals indefinitely | PV = W_real / g_real |
| Spend Down (Finite Horizon) | You’re fine depleting corpus after N years | PV = W_real * (1 - (1+g_real)^(-N)) / g_real (≈ W_real * N if g_real ≈ 0) |

You can add an extra safety cushion (%) to the computed PV for margin of error.

## 3. Key Features

* Multiple gross return scenarios (comma separated)
* Separate accumulation vs withdrawal return treatment (optional tax drag only during SWP if you choose)
* Finite horizon OR perpetual sustainability
* Full SIP & SWP monthly schedules with export (CSV) + totals row
* Year‑end corpus bar chart and combined SIP/SWP line chart
* Diagnostics panel: accumulation vs withdrawal nominal rates & real returns
* Plain‑language explanation plus expandable math formulas
* Indian number formatting (Lakhs / Crores) with two decimal precision (e.g. ₹12.34 L, ₹3.21 Cr)
* Handles edge cases (real return ≤ 0 → infinite corpus warning)

## 4. Sidebar Inputs (Current Version)

| Input | Description |
|-------|-------------|
| Income you want per month (today ₹) | Target monthly spending power in today’s money |
| Inflation (%) | Assumed constant annual CPI |
| Expected gross return % | One or more scenarios (e.g. `9,10,11,12`) |
| Long-term capital gains tax % | Annualized approximation of LTCG drag |
| Annual fees / expenses % | Expense ratio / advisory costs |
| Years you will invest (SIP) | Accumulation duration before withdrawals start |
| Keep the pot forever? | Toggle between perpetual vs finite horizon |
| How many years should it last? | Only shown if not preserving corpus |
| Extra safety cushion (%) | Increases required corpus for prudence |
| Apply tax drag during growth phase? | If off, tax drag only applied in SWP stage (optimistic) |
| When is SIP added each month? | Start vs End of month (cashflow timing effect) |

## 5. Outputs

| Output | Meaning |
|--------|---------|
| Quick Summary table | Side‑by‑side scenarios (returns, corpus, SIP) |
| Monthly SIP metric card | SIP (₹/mo) for each scenario |
| First year SWP figure | Nominal rupee withdrawal in the first SWP year |
| Detailed SIP schedule | Month 0 baseline + all months + totals |
| Detailed SWP schedule | Full withdrawal months + totals (stops if corpus < 0) |
| Year‑end corpus chart | Bar chart of end-of-year corpus across phases |
| Corpus line chart | Combined SIP build‑up and SWP drawdown trajectory |
| Diagnostics table | Internal net / real rates, corpus components |
| Formula expander | Exact equations for the selected scenario |

## 6. Core Formulas

Let:

* g_gross = gross annual return
* f = annual fees
* t = annual LTCG tax drag (simplified)
* π = inflation rate
* W_monthly (today) = desired monthly withdrawal (today’s rupees)
* W_annual = 12 * W_monthly
* g_net = effective nominal net return after fees & tax (during withdrawal; tax optional during accumulation)
* g_real = real return after inflation
* N_accum = SIP years (years until withdrawals start)
* N_draw = finite horizon years (if applicable)
* r_m = monthly nominal rate = (1 + g_nominal)^(1/12) - 1
* SIP = monthly investment amount

Net nominal (simplified sequential drag):
 
```text
g_net_nominal = ((1 + g_gross) * (1 - f) - 1) * (1 - t?)
```
 
Real return:
 
```text
g_real = (1 + g_net_nominal)/(1 + π) - 1
```
 
Perpetual corpus today:
 
```text
Corpus_0 = W_annual / g_real
```
 
Finite horizon corpus today:
 
```text
Corpus_0 = W_annual * (1 - (1+g_real)^(-N_draw)) / g_real
≈ W_annual * N_draw  (if g_real ~ 0)
```
 
Inflated corpus at start of withdrawals:
 
```text
Corpus_start = Corpus_0 * (1 + π) ^ N_accum
```
 
SIP required (future value of ordinary monthly contributions):
 
```text
FV = SIP * ((1 + r_m)^n - 1)/r_m ;  n = N_accum * 12
SIP = FV * r_m / ((1 + r_m)^n - 1)
```

## 7. Indian Number Formatting

Large rupee values are automatically shown using:

* ≥ 1 Crore (1e7): `₹X.YZ Cr`
* ≥ 1 Lakh (1e5): `₹X.YZ L`
* Else: standard grouping with two decimals.

## 8. Edge Cases & Guards

| Situation | Handling |
|-----------|----------|
| Real return ≤ 0 (perpetual) | Show ∞; schedules disabled |
| Real return very small (finite) | Uses approximation Corpus ≈ W_annual * N_draw |
| Corpus depletion in SWP | Table stops early when corpus < 0 |
| Empty / invalid returns list | Stops with user error |

## 9. Running Locally

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python -m streamlit run app.py
```
 
Optional: add more packages (e.g. altair) if not already installed.

## 10. Interpreting Results

* Treat ranges of SIP values (across scenarios) as a planning band.
* Add safety cushion for uncertainties (market volatility, sequence risk not modeled).
* Lowering fees/taxes directly increases sustainable withdrawals.
* Consider stress testing with lower return & higher inflation pairs.

## 11. Limitations (Deliberate Simplifications)

* No volatility / sequence of returns risk (smooth compounding assumed).
* Tax modeling is a flat annual drag (real LTCG rules more nuanced: thresholds, offsets, realization timing).
* No partial phase strategies (e.g., early high withdrawals or glide paths).
* No debt, insurance, or liability matching layer.
* Inflation assumed constant.

## 12. Possible Future Enhancements

| Idea | Value |
|------|-------|
| Monte Carlo simulation | Probabilistic range of outcomes |
| Glide path / asset shift | More realistic late-stage risk control |
| Detailed tax lot model | Accuracy for taxable accounts |
| Multiple withdrawal phases | Model lifestyle changes (early vs late spending) |
| Real purchasing power chart | Track inflation-adjusted corpus over time |
| Excel / PDF export | Sharable reports |

## 13. Contributing

Suggestions / PRs welcome. Keep UI plain-language and formulas transparent. Add tests for any new financial math.

## 14. Disclaimer

Educational tool only. Not investment, tax, or legal advice. Always validate with a qualified professional before acting.

---
Experiment with different scenarios (e.g. 9,10,11,12) and observe how a 1% net return change affects the SIP you need. Cost control and realistic expectations matter.
