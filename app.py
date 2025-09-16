import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="SIP & SWP Planner", layout="wide")

st.title("SIP → SWP Planning Assistant")
st.markdown("""
Probably the most realistic (yet still simplified) SIP & SWP calculator for everyday planning.

This tool helps you answer two everyday questions:
1. How much should I invest each month (SIP) for the next few years?
2. So that I can later withdraw a monthly income that keeps up with inflation (SWP)?

Choose either:
- Preserve corpus: You want the pot to last forever in real (inflation-adjusted) terms.
- Spend down corpus: You want the pot to last a fixed number of years and can let it reach (roughly) zero after that.

What it does (simple model):
- Starts with the monthly income you want in today's rupees.
- Grows that by inflation for the years until withdrawals begin.
- Estimates how large a starting corpus you need (perpetual or fixed-life formula).
- Works backwards to the monthly SIP required to reach that amount.

Built‑in simplifications (not exact tax/market reality):
- Fees and long‑term capital gains tax are treated as a steady annual drag.
- Returns are assumed smooth (no volatility). Real life will vary.
- Inflation is constant at the rate you enter.

Always sanity‑check results and talk to a qualified advisor for real decisions. This is an educational calculator, not personalised advice.
""")

# ------------------------- INPUTS -------------------------
with st.sidebar:
    st.header("Inputs")
    withdrawal_pm = st.number_input("Income you want per month (today ₹)", min_value=10000, value=150000, step=5000)
    inflation = st.number_input("Inflation (%)", min_value=0.0, value=6.5, step=0.1, format="%.2f") / 100
    gross_return_text = st.text_input("Expected gross return % (you can list several e.g. 10,11,12)", value="12")
    ltcg = st.number_input("Long-term capital gains tax %", min_value=0.0, max_value=50.0, value=12.5, step=0.5, format="%.2f") / 100
    fees = st.number_input("Annual fees / expenses %", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.2f") / 100
    sip_years = st.number_input("Years you will invest (SIP)", min_value=1, max_value=50, value=10, step=1)
    start_delay_years = sip_years  # withdrawals start right after accumulation
    preserve_corpus = st.checkbox("Keep the pot forever (don't run out)?", value=True, help="If unchecked we allow the pot to be spent over a fixed number of years.")
    finite_horizon_years = None
    if not preserve_corpus:
        finite_horizon_years = st.number_input("How many years should it last?", min_value=1, max_value=100, value=30, step=1, help="After this many years the pot can be near zero.")
    perpetuity_buffer = st.number_input("Extra safety cushion (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Add more if you want a margin for error.") / 100
    apply_tax_during_accum = st.checkbox("Apply tax drag during growth phase?", value=True, help="Uncheck if you want to ignore tax until withdrawals (optimistic).")
    contribution_timing = st.selectbox("When is SIP added each month?", ["End of Month", "Start of Month"], help="Start = a bit more growth.")

withdrawal_pa = withdrawal_pm * 12

def parse_returns(txt: str):
    vals = []
    for part in txt.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part)/100)
        except ValueError:
            st.warning(f"Could not parse return: {part}")
    # Remove duplicates while preserving order
    seen = set()
    ordered = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return sorted(ordered)

gross_returns = parse_returns(gross_return_text)
if not gross_returns:
    st.error("Enter at least one valid gross return percentage.")
    st.stop()

# ------------------------- FUNCTIONS -------------------------
def effective_net_return(gross: float, fee: float, tax: float, tax_in_accum: bool = True) -> float:
    """Return effective nominal annual net return after fees and (optionally) annualized tax.
    Model: growth factor = (1+gross)*(1-fee); gains portion = factor-1; after tax gains = (factor-1)*(1-tax if tax_in_accum else 1).
    """
    growth_factor = (1 + gross) * (1 - fee)
    gains = growth_factor - 1
    if tax_in_accum:
        gains *= (1 - tax)
    return gains

def real_return(nominal: float, infl: float) -> float:
    return (1 + nominal) / (1 + infl) - 1

def required_corpus_today(real_withdrawal_pa: float, real_r: float, buffer: float = 0.0) -> float:
    # Perpetuity: PV = W / r (if r>0). Add buffer.
    if real_r <= 0:
        return float('inf')
    base = real_withdrawal_pa / real_r
    return base * (1 + buffer)

def required_corpus_finite(real_withdrawal_pa: float, real_r: float, years: int, buffer: float = 0.0) -> float:
    """Finite horizon real annuity present value.
    PV = W * (1 - (1+g_real)^-N) / g_real  (if g_real != 0); else W * N
    Add buffer.
    If real_r <= -1, treat as infeasible (infinite).
    """
    if real_r <= -1:
        return float('inf')
    if abs(real_r) < 1e-9:
        base = real_withdrawal_pa * years
    else:
        base = real_withdrawal_pa * (1 - (1 + real_r) ** (-years)) / real_r
    return base * (1 + buffer)

def inflate(value: float, infl: float, years: int) -> float:
    return value * ((1 + infl) ** years)

def sip_required(target_fv: float, annual_return: float, months: int) -> float:
    if annual_return == 0:
        return target_fv / months
    r_m = (1 + annual_return) ** (1/12) - 1
    if r_m == 0:
        return target_fv / months
    factor = ((1 + r_m) ** months - 1) / r_m
    return target_fv / factor

def build_sip_schedule(monthly_sip: float, annual_return: float, months: int, start_corpus: float = 0.0, timing: str = "End of Month") -> pd.DataFrame:
    """Build SIP accumulation schedule.
    timing:
      - "End of Month": growth on opening, then contribution (so month 1 opening = 0, closing = contribution).
      - "Start of Month": contribution added first, then growth (slightly higher corpus trajectory).
    Returns DataFrame including initial baseline row (Month 0) for clarity.
    """
    r_m = (1 + annual_return) ** (1/12) - 1
    rows = [{
        "Month": 0,
        "Year": 0,
        "Opening": start_corpus,
        "Contribution": 0.0,
        "Growth": 0.0,
        "Withdrawal": 0.0,
        "Closing": start_corpus
    }]
    corpus = start_corpus
    for m in range(1, months + 1):
        year_val = (m - 1)//12 + 1
        opening = corpus
        if timing == "Start of Month":
            # contribute first
            contribution = monthly_sip
            corpus = corpus + contribution
            growth = corpus * r_m
            corpus = corpus + growth
        else:
            # End of month (default earlier logic): growth then contribution
            growth = corpus * r_m
            contribution = monthly_sip
            corpus = corpus + growth + contribution
        rows.append({
            "Month": m,
            "Year": year_val,
            "Opening": opening,
            "Contribution": contribution,
            "Growth": growth,
            "Withdrawal": 0.0,
            "Closing": corpus
        })
    return pd.DataFrame(rows)

def build_swp_schedule(start_corpus: float, annual_return: float, annual_inflation: float, base_withdrawal_pm_today: float, years: int, start_year_index: int = 1) -> pd.DataFrame:
    """Build withdrawal schedule with inflation-adjusted monthly withdrawals.
    Withdrawal for year y (1-indexed) in nominal terms: base_withdrawal_pm_today * (1+infl)^{(y_start + y -2)}.
    Withdrawal at END of month (after growth), similar timing assumption.
    """
    months = years * 12
    r_m = (1 + annual_return) ** (1/12) - 1
    rows = []
    corpus = start_corpus
    for m in range(1, months + 1):
        year_index = (m - 1)//12 + 1
        effective_year = start_year_index + year_index - 1
        opening = corpus
        growth = opening * r_m
        # monthly withdrawal for this year (inflated)
        withdrawal_nominal = base_withdrawal_pm_today * ((1 + annual_inflation) ** (effective_year - 1))
        corpus = opening + growth - withdrawal_nominal
        rows.append({
            "Month": m,
            "Year": effective_year,
            "Opening": opening,
            "Contribution": 0.0,
            "Growth": growth,
            "Withdrawal": withdrawal_nominal,
            "Closing": corpus
        })
        if corpus < 0:
            break
    return pd.DataFrame(rows)

def append_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Append a totals row to SIP or SWP schedule.
    For SIP: sum contributions & growth.
    For SWP: sum withdrawals & growth (growth not usually 'totaled' for decision but useful).
    Closing left blank for clarity.
    """
    if df.empty:
        return df
    totals = {
        "Month": "Total",
        "Year": "-",
        "Opening": "-",
        "Contribution": df['Contribution'].sum(),
        "Growth": df['Growth'].sum(),
        "Withdrawal": df['Withdrawal'].sum(),
        "Closing": "-"
    }
    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

# ------------------------- FORMATTING HELPERS -------------------------
def format_inr_compact(val) -> str:
    """Format number into Indian system with L (Lakhs) / Cr (Crores) and 2 decimals.
    Keeps sign, handles infinity & non-numeric gracefully.
    """
    try:
        if isinstance(val, str):
            return val
        if val is None:
            return "-"
        if not np.isfinite(val):
            return "∞"
        abs_v = abs(val)
        if abs_v >= 1e7:  # Crore
            return f"₹{val/1e7:.2f} Cr"
        if abs_v >= 1e5:  # Lakh
            return f"₹{val/1e5:.2f} L"
        return f"₹{val:,.2f}"
    except Exception:
        return str(val)

def format_schedule_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of schedule with INR compact formatting for money columns."""
    if df is None or df.empty:
        return df
    display_df = df.copy()
    money_cols = [c for c in ["Opening", "Contribution", "Growth", "Withdrawal", "Closing"] if c in display_df.columns]
    for c in money_cols:
        display_df[c] = display_df[c].apply(format_inr_compact)
    return display_df

# ------------------------- CALCULATIONS -------------------------
K_GROSS = "Gross Return"
K_NET = "Net Nominal Return"
K_REAL = "Real Return"
# Keys for corpus values
K_CT = "CT"  # Corpus Today
K_CS = "CS"  # Corpus Start
K_SIP = "SIP Required"

records = []
for g in gross_returns:
    # Return used for accumulation (optionally exclude tax drag for compounding efficiency)
    accum_return_nominal = effective_net_return(g, fees, ltcg, tax_in_accum=apply_tax_during_accum)
    if not apply_tax_during_accum:
        accum_return_nominal = effective_net_return(g, fees, 0.0, tax_in_accum=False)  # only fees
    # Return used for withdrawal sustainability (always include tax drag assumption for prudence)
    withdrawal_nominal = effective_net_return(g, fees, ltcg, tax_in_accum=True)
    real_r = real_return(withdrawal_nominal, inflation)
    if preserve_corpus:
        corpus_today = required_corpus_today(withdrawal_pa, real_r, perpetuity_buffer)
        horizon_years = None
        mode = "Perpetual"
    else:
        corpus_today = required_corpus_finite(withdrawal_pa, real_r, finite_horizon_years, perpetuity_buffer)
        horizon_years = int(finite_horizon_years)
        mode = f"Finite {horizon_years}y"
    corpus_nominal_start = inflate(corpus_today, inflation, start_delay_years)
    sip_months = sip_years * 12
    sip_amt = sip_required(corpus_nominal_start, accum_return_nominal, sip_months)
    sip_amt = float(np.ceil(sip_amt))  # round up to next rupee for safety
    records.append({
        K_GROSS: g,
        K_NET: withdrawal_nominal,
        K_REAL: real_r,
    K_CT: corpus_today,
    K_CS: corpus_nominal_start,
        K_SIP: sip_amt,
        "Mode": mode,
        "HorizonYears": horizon_years
    })

 # (Step-by-step section moved below summary & schedules)

st.header("Quick Summary")
summary_df = pd.DataFrame([
    {
        "Gross Return %": f"{r[K_GROSS]*100:.2f}",
        "Net Nominal %": f"{r[K_NET]*100:.2f}",
        "Real %": f"{r[K_REAL]*100:.2f}",
    "Mode": r.get("Mode",""),
    "Horizon (y)": (str(r.get("HorizonYears")) if r.get("HorizonYears") else ("∞" if preserve_corpus else "?")),
    "Corpus Today": format_inr_compact(r[K_CT]) if not np.isinf(r[K_CT]) else "∞",
    "Corpus Start": format_inr_compact(r[K_CS]) if not np.isinf(r[K_CT]) else "—",
    "SIP / Month": format_inr_compact(r[K_SIP]) if not np.isinf(r[K_CT]) else "—",
    } for r in records
])
# ------------------------- KEY NUMBERS -------------------------
st.subheader(f"Monthly SIP needed for {sip_years} years of investing")
cols = st.columns(len(records) if records else 1)
for idx, r in enumerate(records):
    with cols[idx]:
        if np.isinf(r[K_CT]):
            st.metric(f"Scenario {idx+1} ({r[K_GROSS]*100:.0f}% Gross)", value="∞", help="Real return ≤ 0%")
        else:
            first_year_withdrawal_nominal = withdrawal_pa * ((1 + inflation) ** sip_years)
            st.metric(label=f"Scenario {idx+1} SIP (₹/mo)", value=f"{r[K_SIP]} ({format_inr_compact(r[K_SIP])})")
            st.caption(f"First-year SWP (nominal, year {sip_years+1}): {format_inr_compact(first_year_withdrawal_nominal)}/yr ({format_inr_compact(first_year_withdrawal_nominal/12)}/mo)")

st.caption("SIP values are rounded up. First withdrawal year shown in future (inflated) rupees.")

st.dataframe(summary_df, use_container_width=True)

with st.expander("Diagnostics / Assumptions Detail"):
    st.markdown("**Per-Scenario Internal Values**")
    diag_rows = []
    for r in records:
        # recompute accumulation return shown separately
        gross = r[K_GROSS]
        accum_return_nominal = effective_net_return(gross, fees, ltcg if apply_tax_during_accum else 0.0, tax_in_accum=apply_tax_during_accum)
        diag_rows.append({
            "Gross %": f"{gross*100:.2f}",
            "Accum Return %": f"{accum_return_nominal*100:.2f}",
            "Withdrawal Return %": f"{r[K_NET]*100:.2f}",
            "Real %": f"{r[K_REAL]*100:.2f}",
            "Corpus Today": format_inr_compact(r[K_CT]) if not np.isinf(r[K_CT]) else "∞",
            "Corpus Start": format_inr_compact(r[K_CS]) if not np.isinf(r[K_CT]) else "—",
            "SIP (₹/mo)": format_inr_compact(r[K_SIP]) if not np.isinf(r[K_CT]) else "—",
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
    st.caption("Accum Return: nominal rate used for compounding during SIP phase. Withdrawal Return: nominal rate (after fees & tax) used for sustainability real-return calculation.")

# ------------------------- SCHEDULES SECTION -------------------------
st.header("Detailed Month-by-Month View")

scenario_labels = [f"{i+1}: {r[K_GROSS]*100:.2f}% gross" for i, r in enumerate(records)]
selected_index = st.selectbox("Choose scenario for schedules", list(range(len(records))), format_func=lambda i: scenario_labels[i]) if records else None

if selected_index is not None:
    sel = records[selected_index]
    if np.isinf(sel[K_CT]):
        st.warning("Cannot build schedule: infinite corpus requirement (real return ≤ 0). Adjust assumptions.")
    else:
        st.subheader("SIP Growth Phase")
        sip_months = sip_years * 12
        sip_schedule = build_sip_schedule(sel[K_SIP], sel[K_NET], sip_months, 0.0, timing=contribution_timing)
        sip_schedule_full = append_totals(sip_schedule)
        st.dataframe(format_schedule_for_display(sip_schedule_full), use_container_width=True)
        st.caption("Full SIP phase with totals row at bottom.")
        csv_sip = sip_schedule_full.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full SIP Schedule (CSV)", data=csv_sip, file_name="sip_schedule.csv", mime="text/csv")

        st.subheader("Withdrawal Phase (SWP)")
        default_swp_years = 10
        if not preserve_corpus and finite_horizon_years:
            default_swp_years = int(finite_horizon_years)
        swp_years = st.number_input("SWP projection years (after accumulation)", min_value=1, max_value=60, value=default_swp_years, step=1)
        swp_schedule = build_swp_schedule(sel[K_CS], sel[K_NET], inflation, withdrawal_pm, swp_years, start_year_index=sip_years+1)
        swp_schedule_full = append_totals(swp_schedule)
        st.dataframe(format_schedule_for_display(swp_schedule_full), use_container_width=True)
        st.caption("Full withdrawal phase with totals row (total withdrawals shown).")
        csv_swp = swp_schedule_full.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full SWP Schedule (CSV)", data=csv_swp, file_name="swp_schedule.csv", mime="text/csv")

        # Corpus trajectory plot (optional small chart)
        try:
            import altair as alt
            combined = sip_schedule.copy()
            combined['Phase'] = 'SIP'
            swp_temp = swp_schedule.copy()
            swp_temp['Month'] += sip_months
            swp_temp['Phase'] = 'SWP'
            combined = pd.concat([combined, swp_temp], ignore_index=True)
            chart = alt.Chart(combined).mark_line().encode(
                x=alt.X('Month:Q'),
                y=alt.Y('Closing:Q', title='Corpus (₹)'),
                color='Phase'
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            # Yearly aggregation chart
            total_years = sip_years + swp_years
            # Build yearly data: end-of-year closing corpus (month multiples of 12)
            yearly_points = []
            for year in range(0, total_years + 1):
                if year == 0:
                    closing = 0.0
                elif year <= sip_years:
                    # SIP schedule includes baseline month 0; year *12 row gives closing
                    row = sip_schedule[sip_schedule['Month'] == year * 12]
                    if not row.empty:
                        closing = float(row['Closing'].iloc[0])
                    else:
                        closing = float(sip_schedule['Closing'].iloc[-1])
                else:
                    year_in_swp = year - sip_years
                    # swp_schedule year numbering starts at sip_years+1 -> effective year = sip_years + year_in_swp
                    target_year = sip_years + year_in_swp
                    # pick last month of that calendar year within swp schedule
                    subset = swp_schedule[swp_schedule['Year'] == target_year]
                    if not subset.empty:
                        closing = float(subset['Closing'].iloc[-1])
                    else:
                        closing = float(swp_schedule['Closing'].iloc[-1]) if not swp_schedule.empty else 0.0
                yearly_points.append({"Year": year, "Corpus": closing})
            yearly_df = pd.DataFrame(yearly_points)
            yearly_chart = alt.Chart(yearly_df).mark_bar(color='#4e79a7').encode(
                x=alt.X('Year:O', title='Year (0 = Start)'),
                y=alt.Y('Corpus:Q', title='End-of-Year Corpus (₹)')
            ).properties(height=300)
            st.subheader("Year-End Corpus (Each Year)")
            st.altair_chart(yearly_chart, use_container_width=True)
        except Exception:
            pass

st.header("Key Takeaways")
st.markdown("""
* A small drop in net return can mean a much larger SIP or a shorter-lasting pot.
* Near-zero real return? A forever (perpetual) inflation‑linked income becomes impossible.
* Costs and taxes matter: lowering them boosts the real return you keep.
* Inflation protection requires a bigger starting pot than a flat (non‑indexed) payout.
* Use multiple return scenarios (e.g., 9%, 10%, 11%, 12%) to build a safety range.

This is a simplified, steady‑growth model — reality will bounce around. Treat results as planning guides, not guarantees.
""")

# ------------------------- STEP-BY-STEP (MOVED) -------------------------
st.header("How We Calculate It")
for i, r in enumerate(records, start=1):
    gr = r[K_GROSS] * 100
    st.subheader(f"Scenario {i}: {gr:.2f}% gross return")
    if np.isinf(r[K_CT]):
        st.error("Real return ≤ 0%. Infinite corpus required.")
        continue
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. We start from your gross return, subtract fees and tax → net return.**")
        st.write(f"Net yearly return kept after costs: {r[K_NET]*100:.2f}%")
        st.markdown("**2. We remove inflation to see the real growth of purchasing power.**")
        st.write(f"Real return (after inflation): {r[K_REAL]*100:.2f}%")
        if r.get("Mode") == "Perpetual":
            st.markdown("**3. How big the pot must be today to fund that income forever (inflation-adjusted).**")
        else:
            horizon = r.get("HorizonYears")
            st.markdown(f"**3. Pot size today to fund {horizon} years of inflation‑rising income.**")
            st.caption("If real return is ~0: need roughly annual income × years.")
    st.write(f"Corpus today: {format_inr_compact(r[K_CT])}")
    with col2:
        st.markdown("**4. Grow that required pot by inflation until withdrawals begin.**")
    st.write(f"Pot needed at start of withdrawals ( nominal ): {format_inr_compact(r[K_CS])}")
    st.markdown("**5. Work backwards to the monthly SIP that can build that pot.**")
    st.write(f"Monthly SIP required: {format_inr_compact(r[K_SIP])}")
    with st.expander("Show formulas for this scenario"):
        st.markdown("**Formulas used**")
        st.latex(r"g_{net} = ((1+g_{gross})(1-fee)-1)\times(1-\text{tax})")
        st.latex(r"g_{real} = \frac{1+g_{net}}{1+\pi}-1")
        if r.get("Mode") == "Perpetual":
            st.latex(r"Corpus_0 = \frac{W_{annual}}{g_{real}}")
        else:
            st.latex(r"Corpus_0 = W_{annual}\cdot \frac{1-(1+g_{real})^{-N}}{g_{real}}")
            st.latex(r"g_{real}\approx 0 \Rightarrow Corpus_0 \approx W_{annual}\times N")
        st.latex(r"Corpus_{start} = Corpus_0 (1+\pi)^{Y_{accum}}")
        st.latex(r"FV = SIP \cdot \frac{(1+r_m)^n -1}{r_m}\Rightarrow SIP = FV \cdot \frac{r_m}{(1+r_m)^n -1}")
    st.divider()
