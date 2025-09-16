import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import calendar

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
    withdrawal_pm = st.number_input("Income you want per month (today ₹)", min_value=10000, value=100000, step=5000)
    inflation = st.number_input("Inflation (%) (6.5 is the real number in India for last 20 years)", min_value=0.0, value=6.5, step=0.1, format="%.2f") / 100
    gross_return_text = st.text_input("Expected gross return % (you can list several e.g. 10,11,12)", value="12")
    ltcg = st.number_input("Long-term capital gains tax %", min_value=0.0, max_value=50.0, value=12.5, step=0.5, format="%.2f") / 100
    fees = st.number_input("Annual fees / expenses %", min_value=0.0, max_value=5.0, value=0.5, step=0.1, format="%.2f") / 100
    initial_corpus = st.number_input("Existing corpus today (₹)", min_value=0.0, value=0.0, step=50000.0, help="Amount you already have invested toward this goal today.")
    sip_years = st.number_input("Years you will invest (SIP)", min_value=1, max_value=50, value=15, step=1)
    start_delay_years = sip_years  # withdrawals start right after accumulation
    preserve_corpus = st.checkbox("Keep the pot forever (don't run out)?", value=False, help="If unchecked we allow the pot to be spent over a fixed number of years.")
    finite_horizon_years = None
    if not preserve_corpus:
        finite_horizon_years = st.number_input("How many years should it last?", min_value=1, max_value=100, value=20, step=1, help="After this many years the pot can be near zero.")
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

def build_sip_schedule(monthly_sip: float, annual_return: float, months: int, start_corpus: float = 0.0, timing: str = "End of Month", start_date: date | None = None) -> pd.DataFrame:
    """Build SIP accumulation schedule with calendar months.
    MonthIndex: numeric sequence starting at 0 baseline, then 1..n
    Month: calendar month short name (e.g., Oct)
    Year: calendar year (e.g., 2025)
    start_date: if provided, first contribution month (MonthIndex=1) uses its month/year.
    timing semantics unchanged.
    """
    if start_date is None:
        start_date = date.today().replace(day=1)
    # baseline row (MonthIndex 0) uses the month prior to start_date for clarity
    baseline_month = (start_date.month - 1) if start_date.month > 1 else 12
    baseline_year = start_date.year if start_date.month > 1 else start_date.year - 1
    r_m = (1 + annual_return) ** (1/12) - 1
    rows = [{
        "MonthIndex": 0,
        "Month": calendar.month_abbr[baseline_month],
        "Year": baseline_year,
        "Opening": start_corpus,
        "Contribution": 0.0,
        "Growth": 0.0,
        "Withdrawal": 0.0,
        "Closing": start_corpus
    }]
    corpus = start_corpus
    cur_year = start_date.year
    cur_month = start_date.month
    for idx in range(1, months + 1):
        opening = corpus
        if timing == "Start of Month":
            contribution = monthly_sip
            corpus = corpus + contribution
            growth = corpus * r_m
            corpus = corpus + growth
        else:
            growth = corpus * r_m
            contribution = monthly_sip
            corpus = corpus + growth + contribution
        rows.append({
            "MonthIndex": idx,
            "Month": calendar.month_abbr[cur_month],
            "Year": cur_year,
            "Opening": opening,
            "Contribution": contribution,
            "Growth": growth,
            "Withdrawal": 0.0,
            "Closing": corpus
        })
        # advance month
        if cur_month == 12:
            cur_month = 1
            cur_year += 1
        else:
            cur_month += 1
    return pd.DataFrame(rows)

def build_swp_schedule(start_corpus: float, annual_return: float, annual_inflation: float, base_withdrawal_pm_today: float, years: int, start_calendar_date: date, start_year_index: int = 1, starting_month_index: int = 0) -> pd.DataFrame:
    """Build withdrawal schedule with calendar months.
    start_calendar_date: first withdrawal month (first month after accumulation phase).
    MonthIndex continues from SIP schedule (pass last MonthIndex) + 1 for first SWP month.
    Year column now reflects actual calendar year, but we also keep EffectiveYear for inflation escalation logic (financial year count since start of SIP+1).
    """
    months = years * 12
    r_m = (1 + annual_return) ** (1/12) - 1
    rows = []
    corpus = start_corpus
    cur_year = start_calendar_date.year
    cur_month = start_calendar_date.month
    base_month_index = starting_month_index
    for m in range(1, months + 1):
        month_index = base_month_index + m
        year_index = (m - 1)//12 + 1
        effective_year = start_year_index + year_index - 1
        opening = corpus
        growth = opening * r_m
        withdrawal_nominal = base_withdrawal_pm_today * ((1 + annual_inflation) ** (effective_year - 1))
        corpus = opening + growth - withdrawal_nominal
        rows.append({
            "MonthIndex": month_index,
            "Month": calendar.month_abbr[cur_month],
            "Year": cur_year,
            "EffectiveYear": effective_year,
            "Opening": opening,
            "Contribution": 0.0,
            "Growth": growth,
            "Withdrawal": withdrawal_nominal,
            "Closing": corpus
        })
        if corpus < 0:
            break
        # advance calendar month
        if cur_month == 12:
            cur_month = 1
            cur_year += 1
        else:
            cur_month += 1
    return pd.DataFrame(rows)

def append_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Append a totals row to SIP or SWP schedule.
    For SIP: sum contributions & growth.
    For SWP: sum withdrawals & growth (growth not usually 'totaled' for decision but useful).
    Closing left blank for clarity.
    """
    if df.empty:
        return df
    # Determine id columns present
    totals = {
        "MonthIndex": "Total" if 'MonthIndex' in df.columns else "Total",
        "Month": "-" if 'Month' in df.columns else None,
        "Year": "-",
        "Opening": "-",
        "Contribution": df['Contribution'].sum(),
        "Growth": df['Growth'].sum(),
        "Withdrawal": df['Withdrawal'].sum(),
        "Closing": "-"
    }
    # Remove None keys
    totals = {k: v for k, v in totals.items() if k in df.columns}
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
    # Future value of existing corpus after accumulation (annual comp approximation)
    existing_future = initial_corpus * ((1 + accum_return_nominal) ** sip_years)
    gap_start = max(0.0, corpus_nominal_start - existing_future)
    if np.isinf(corpus_today):
        sip_amt = float('inf')
    elif gap_start <= 0:
        sip_amt = 0.0
    else:
        sip_amt_raw = sip_required(gap_start, accum_return_nominal, sip_months)
        sip_amt = float(np.ceil(sip_amt_raw))
    records.append({
        K_GROSS: g,
        K_NET: withdrawal_nominal,
        K_REAL: real_r,
        K_CT: corpus_today,
        K_CS: corpus_nominal_start,
        K_SIP: sip_amt,
        "Mode": mode,
        "HorizonYears": horizon_years,
        "ExistingToday": initial_corpus,
        "ExistingFuture": existing_future,
        "GapStart": gap_start
    })

 # (Step-by-step section moved below summary & schedules)

st.header("Quick Summary")
summary_df = pd.DataFrame([
    {
        "Gross %": f"{r[K_GROSS]*100:.2f}",
        "Net %": f"{r[K_NET]*100:.2f}",
        "Real %": f"{r[K_REAL]*100:.2f}",
        "Mode": r.get("Mode",""),
        "Horizon (y)": (str(r.get("HorizonYears")) if r.get("HorizonYears") else ("∞" if preserve_corpus else "?")),
        "Corpus Today Req": format_inr_compact(r[K_CT]) if not np.isinf(r[K_CT]) else "∞",
        "Existing Today": format_inr_compact(r.get("ExistingToday",0.0)),
        "Corpus Start Req": format_inr_compact(r[K_CS]) if not np.isinf(r[K_CT]) else "—",
        "Existing Future": format_inr_compact(r.get("ExistingFuture",0.0)) if not np.isinf(r[K_CT]) else "—",
        "Gap @ Start": format_inr_compact(r.get("GapStart",0.0)) if not np.isinf(r[K_CT]) else "—",
        "SIP / Month": ("0" if r[K_SIP]==0 else format_inr_compact(r[K_SIP])) if not np.isinf(r[K_CT]) else "—",
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
            sip_label_val = "0 (covered by existing)" if r[K_SIP]==0 else f"{r[K_SIP]} ({format_inr_compact(r[K_SIP])})"
            st.metric(label=f"Scenario {idx+1} SIP (₹/mo)", value=sip_label_val)
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
            "Corpus Today Req": format_inr_compact(r[K_CT]) if not np.isinf(r[K_CT]) else "∞",
            "Existing Today": format_inr_compact(r.get("ExistingToday",0.0)),
            "Corpus Start Req": format_inr_compact(r[K_CS]) if not np.isinf(r[K_CT]) else "—",
            "Existing Future": format_inr_compact(r.get("ExistingFuture",0.0)) if not np.isinf(r[K_CT]) else "—",
            "Gap @ Start": format_inr_compact(r.get("GapStart",0.0)) if not np.isinf(r[K_CT]) else "—",
            "SIP (₹/mo)": ("0" if r[K_SIP]==0 else format_inr_compact(r[K_SIP])) if not np.isinf(r[K_CT]) else "—",
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
        start_dt = date.today().replace(day=1)
    # Use existing corpus as starting amount
    start_corpus_for_schedule = sel.get("ExistingToday", 0.0)
    sip_schedule = build_sip_schedule(sel[K_SIP], sel[K_NET], sip_months, start_corpus_for_schedule, timing=contribution_timing, start_date=start_dt)
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
        # Determine first SWP calendar month: month after last SIP month
        if not sip_schedule.empty:
            last_sip_year = int(sip_schedule.iloc[-1]['Year'])
            last_sip_month_abbr = sip_schedule.iloc[-1]['Month']
            # Map month abbr back to month number
            month_map = {calendar.month_abbr[i]: i for i in range(1,13)}
            last_sip_month_num = month_map.get(last_sip_month_abbr, date.today().month)
            if last_sip_month_num == 12:
                swp_start_month = 1
                swp_start_year = last_sip_year + 1
            else:
                swp_start_month = last_sip_month_num + 1
                swp_start_year = last_sip_year
            swp_start_date = date(swp_start_year, swp_start_month, 1)
        else:
            swp_start_date = date.today().replace(day=1)
        last_month_index = int(sip_schedule['MonthIndex'].iloc[-1]) if 'MonthIndex' in sip_schedule.columns else sip_months
        swp_schedule = build_swp_schedule(sel[K_CS], sel[K_NET], inflation, withdrawal_pm, swp_years, start_calendar_date=swp_start_date, start_year_index=sip_years+1, starting_month_index=last_month_index)
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
            swp_temp['Phase'] = 'SWP'
            combined = pd.concat([combined, swp_temp], ignore_index=True)
            chart = alt.Chart(combined).mark_line().encode(
                x=alt.X('MonthIndex:Q', title='Month Index'),
                y=alt.Y('Closing:Q', title='Corpus (₹)'),
                color='Phase'
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            # Yearly aggregation chart
            # Build yearly data using calendar year end (December) closings
            combined_sorted = combined.sort_values('MonthIndex')
            yearly_points = []
            # include initial baseline year start
            baseline_year = int(sip_schedule.iloc[0]['Year']) if not sip_schedule.empty else date.today().year
            yearly_points.append({"CalendarYear": baseline_year - 1, "Corpus": 0.0})
            for yr in sorted(combined['Year'].unique()):
                subset = combined_sorted[combined_sorted['Year'] == yr]
                if not subset.empty:
                    closing = float(subset.iloc[-1]['Closing'])
                    yearly_points.append({"CalendarYear": yr, "Corpus": closing})
            yearly_df = pd.DataFrame(yearly_points)
            yearly_chart = alt.Chart(yearly_df).mark_bar(color='#4e79a7').encode(
                x=alt.X('CalendarYear:O', title='Calendar Year'),
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
    if r.get("ExistingToday",0)>0:
        st.write(f"Existing corpus today grows to: {format_inr_compact(r.get('ExistingFuture',0.0))}")
        st.write(f"Gap remaining at start: {format_inr_compact(r.get('GapStart',0.0))}")
    st.markdown("**5. Work backwards to the monthly SIP that can build the remaining gap.**")
    st.write(f"Monthly SIP required: {'0 (covered by existing corpus)' if r[K_SIP]==0 else format_inr_compact(r[K_SIP])}")
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
