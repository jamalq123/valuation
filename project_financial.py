import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Project Valuation Tool", layout="wide")
st.title("📊 Advanced Project Valuation & Sensitivity Tool")

# =====================
# INPUTS
# =====================
st.sidebar.header("🔢 Inputs")

initial_investment = st.sidebar.number_input("Initial Investment", value=1_000_000.0)
wacc = st.sidebar.number_input("Cost of Capital (WACC %)", value=12.0) / 100
growth = st.sidebar.number_input("Perpetuity Growth Rate (%)", value=3.0) / 100
years = st.sidebar.number_input("Explicit Forecast Period (Years)", min_value=1, value=5)

st.sidebar.subheader("Free Cash Flows")
fcfs = [st.sidebar.number_input(f"FCF Year {i+1}", value=200_000.0) for i in range(years)]

# =====================
# CORE CALCULATIONS
# =====================
discount_factors = [(1 / (1 + wacc) ** i) for i in range(1, years + 1)]
pv_fcfs = [fcfs[i] * discount_factors[i] for i in range(years)]

terminal_value = fcfs[-1] * (1 + growth) / (wacc - growth)
pv_terminal = terminal_value / ((1 + wacc) ** years)

enterprise_value = sum(pv_fcfs) + pv_terminal
npv = enterprise_value - initial_investment
irr = npf.irr([-initial_investment] + fcfs)
pi = enterprise_value / initial_investment

# =====================
# PAYBACK CALCULATIONS
# =====================
cum_cf = -initial_investment
cum_dcf = -initial_investment

payback = None
discounted_payback = None

for i in range(years):
    cum_cf += fcfs[i]
    cum_dcf += pv_fcfs[i]

    if cum_cf >= 0 and payback is None:
        payback = i + 1

    if cum_dcf >= 0 and discounted_payback is None:
        discounted_payback = i + 1

# =====================
# CASH BREAKEVEN TIME
# =====================
cum_cf_temp = -initial_investment
cum_cf_track = [cum_cf_temp]

for cf in fcfs:
    cum_cf_temp += cf
    cum_cf_track.append(cum_cf_temp)

# Year of maximum cash deficit
max_deficit = min(cum_cf_track)
cash_breakeven_year = cum_cf_track.index(max_deficit)


# =====================
# RESULTS
# =====================
st.header("📈 Valuation Results")

col1, col2, col3 = st.columns(3)

col1.metric("NPV", f"${npv:,.0f}")
col2.metric("IRR", f"{irr*100:.2f}%")
col3.metric("Profitability Index", f"{pi:.2f}")
col1.metric("Cash Breakeven Time (Years)", cash_breakeven_year)

col1.metric("Payback (Years)", payback if payback else "Not Recovered")
col2.metric("Discounted Payback (Years)", discounted_payback if discounted_payback else "Not Recovered")
col3.metric("Enterprise Value", f"${enterprise_value:,.0f}")

# =====================
# CUMULATIVE CASH FLOW CHART
# =====================
st.header("📈 Cumulative Cash Flows")

years_list = list(range(0, years + 1))
cum_cf_series = [-initial_investment]
cum_dcf_series = [-initial_investment]

for i in range(years):
    cum_cf_series.append(cum_cf_series[-1] + fcfs[i])
    cum_dcf_series.append(cum_dcf_series[-1] + pv_fcfs[i])

plt.figure()
plt.plot(years_list, cum_cf_series, label="Cumulative Cash Flow")
plt.plot(years_list, cum_dcf_series, label="Cumulative Discounted Cash Flow")
plt.axhline(0)
plt.xlabel("Year")
plt.ylabel("Cash Flow")
plt.legend()
st.pyplot(plt)

# =====================
# NPV WATERFALL
# =====================
st.header("📉 NPV Waterfall")

labels = ["Initial Investment"] + [f"PV FCF Y{i+1}" for i in range(years)] + ["PV Terminal"]
values = [-initial_investment] + pv_fcfs + [pv_terminal]

plt.figure()
plt.bar(labels, values)
plt.xticks(rotation=45)
plt.ylabel("Value")
plt.title("NPV Waterfall")
st.pyplot(plt)

# =====================
# SENSITIVITY ANALYSIS
# =====================
st.header("📊 Sensitivity Analysis (WACC vs Growth)")

wacc_range = np.arange(wacc - 0.03, wacc + 0.04, 0.01)
growth_range = np.arange(max(0, growth - 0.02), growth + 0.03, 0.01)

sensitivity = pd.DataFrame(index=[f"{g:.1%}" for g in growth_range],
                            columns=[f"{w:.1%}" for w in wacc_range])

for g in growth_range:
    for w in wacc_range:
        tv = fcfs[-1] * (1 + g) / (w - g)
        pv_tv = tv / ((1 + w) ** years)
        pv_fcf = sum([fcfs[i] / ((1 + w) ** (i + 1)) for i in range(years)])
        ev = pv_fcf + pv_tv
        sensitivity.loc[f"{g:.1%}", f"{w:.1%}"] = round(ev, 0)

st.dataframe(sensitivity)
