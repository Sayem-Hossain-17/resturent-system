"""
===============================================================================
Script: 03_Optimization_and_Dashboard.py
Description: Linear programming staff scheduling (PuLP), inventory optimization 
             (Newsvendor model), and interactive Streamlit web dashboard.
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import pulp
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. OPTIMIZATION LOGIC ENGINE (PuLP & Newsvendor)
# -----------------------------------------------------------------------------
def solve_staff_scheduling(forecasted_demand, cost_morning=120, cost_evening=150, cost_night=180):
    """
    Solves Linear Program to minimize labor cost subject to covering customer demand.
    
    Variables:
      - x_m: Number of staff in Morning shift (covers up to 30 covers/staff)
      - x_e: Number of staff in Evening shift (covers up to 25 covers/staff)
      - x_n: Number of staff in Night shift (covers up to 20 covers/staff)
    """
    prob = pulp.LpProblem("Restaurant_Staffing_Optimization", pulp.LpMinimize)
    
    # Decision Variables (Integer count of staff per shift)
    x_m = pulp.LpVariable('Morning_Staff', lowBound=1, cat='Integer')
    x_e = pulp.LpVariable('Evening_Staff', lowBound=1, cat='Integer')
    x_n = pulp.LpVariable('Night_Staff', lowBound=1, cat='Integer')
    
    # Objective Function: Minimize total daily labor wages
    prob += cost_morning * x_m + cost_evening * x_e + cost_night * x_n, "Total_Labor_Cost"
    
    # Capacity Constraint: Total capacity across shifts must meet forecasted demand
    total_capacity = 30 * x_m + 25 * x_e + 20 * x_n
    prob += total_capacity >= forecasted_demand, "Demand_Coverage_Constraint"
    
    # Solve integer linear program
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    return {
        'status': pulp.LpStatus[prob.status],
        'morning_staff': int(pulp.value(x_m)),
        'evening_staff': int(pulp.value(x_e)),
        'night_staff': int(pulp.value(x_n)),
        'total_labor_cost': float(pulp.value(prob.objective)),
        'scheduled_capacity': float(pulp.value(total_capacity))
    }

def solve_newsvendor_inventory(forecasted_demand, unit_cost=15, selling_price=40, salvage_value=5, std_dev=15):
    """
    Computes optimal inventory order quantity Q* balancing overage and shortage costs.
    
    Parameters:
      - Cu: Cost of under-ordering (Shortage cost = Selling Price - Unit Cost)
      - Co: Cost of over-ordering (Overage cost = Unit Cost - Salvage Value)
    """
    c_u = selling_price - unit_cost
    c_o = unit_cost - salvage_value
    critical_ratio = c_u / (c_u + c_o)
    
    # Critical ratio lookup using normal distribution approximation
    from scipy.stats import norm
    z_score = norm.ppf(critical_ratio)
    
    optimal_quantity = int(np.ceil(forecasted_demand + z_score * std_dev))
    safety_stock = int(np.ceil(z_score * std_dev))
    
    return {
        'critical_ratio': round(critical_ratio, 3),
        'optimal_order_quantity': optimal_quantity,
        'safety_stock': safety_stock,
        'unit_cost': unit_cost
    }

# -----------------------------------------------------------------------------
# 2. STREAMLIT DASHBOARD INTERFACE
# -----------------------------------------------------------------------------
def run_streamlit_dashboard():
    st.set_page_config(
        page_title="Restaurant Operations & Demand AI Dashboard",
        page_icon="🍽️",
        layout="wide"
    )

    st.title("🍽️ AI-Driven Restaurant Demand & Resource Optimization Dashboard")
    st.markdown("""
    *Integrated decision support platform connecting Aspect-Based Sentiment Analysis (ABSA) 
    forecasting models to linear labor scheduling and inventory replenishment.*
    """)
    st.divider()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🎛️ Operational Parameters & Scenario Sliders")
    
    base_demand = st.sidebar.number_input("Base Forecast Demand (Covers)", min_value=50, max_value=500, value=180)
    
    st.sidebar.subheader("What-If Sentiment Scenario Control")
    hygiene_complaints = st.sidebar.slider("Negative Hygiene Complaints Today", 0, 10, 2)
    service_complaints = st.sidebar.slider("Negative Service Complaints Today", 0, 10, 4)
    
    # Simulate demand drop based on negative sentiment SHAP impacts
    demand_impact = (-12 * hygiene_complaints) + (-6 * service_complaints)
    adjusted_demand = max(20, base_demand + demand_impact)

    # --- TOP METRIC CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        label="Raw Baseline Demand", 
        value=f"{base_demand} covers"
    )
    col2.metric(
        label="Adjusted Forecast Demand", 
        value=f"{adjusted_demand} covers", 
        delta=f"{demand_impact} covers", 
        delta_color="normal"
    )
    
    # Solve Optimization Models
    staff_res = solve_staff_scheduling(adjusted_demand)
    inv_res = solve_newsvendor_inventory(adjusted_demand)
    
    col3.metric(
        label="Optimized Daily Labor Cost", 
        value=f"${staff_res['total_labor_cost']:.2f}"
    )
    col4.metric(
        label="Optimal Order Quantity (Q*)", 
        value=f"{inv_res['optimal_order_quantity']} units"
    )

    st.divider()

    # --- MAIN CONTENT LAYOUT ---
    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.subheader("👨‍🍳 Optimized Shift Staff Scheduling (PuLP LP)")
        st.write(f"**Optimization Status:** `{staff_res['status']}`")
        
        staff_data = pd.DataFrame({
            "Shift": ["Morning Shift", "Evening Shift", "Night Shift"],
            "Staff Count Scheduled": [staff_res['morning_staff'], staff_res['evening_staff'], staff_res['night_staff']],
            "Wage Rate ($/shift)": [120, 150, 180]
        })
        st.dataframe(staff_data, use_container_width=True)
        
        st.info(f"💡 Scheduled Capacity: **{staff_res['scheduled_capacity']} covers** (Covers target of {adjusted_demand})")

    with right_column:
        st.subheader("📦 Inventory Newsvendor Order Quantities")
        st.write(f"**Critical Ratio ($C_u / (C_u + C_o)$):** `{inv_res['critical_ratio']}`")
        
        inv_data = pd.DataFrame({
            "Metric": ["Forecast Demand", "Safety Stock Level", "Total Order Quantity (Q*)"],
            "Units": [adjusted_demand, inv_res['safety_stock'], inv_res['optimal_order_quantity']]
        })
        st.dataframe(inv_data, use_container_width=True)
        
        st.success(f"📦 Recommended Order: **{inv_res['optimal_order_quantity']} units** of primary inventory ingredients.")

    st.divider()

    # --- SHAP EXPLAINABILITY VISUALIZATION ---
    st.subheader("📊 SHAP Feature Attribution & Scenario Analysis")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    features = ['Baseline Sales Lag', 'Temperature', 'Is Holiday', 'Service Sentiment', 'Hygiene Sentiment']
    shap_vals = [45, 12, 15, -6 * service_complaints, -12 * hygiene_complaints]
    
    colors = ['green' if x >= 0 else 'red' for x in shap_vals]
    ax.barh(features, shap_vals, color=colors)
    ax.set_xlabel("SHAP Value (Impact on Demand Forecast)")
    ax.set_title("Real-Time Feature Drivers for Tomorrow's Demand")
    
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# 3. SCRIPT EXECUTION ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Checks if running via Streamlit CLI or standard Python execution
    import sys
    if st.runtime.exists():
        run_streamlit_dashboard()
    else:
        print("[*] Testing Optimization Engines independently...")
        test_demand = 180
        s_res = solve_staff_scheduling(test_demand)
        i_res = solve_newsvendor_inventory(test_demand)
        print(f"[+] Staff Scheduling Solution for Demand={test_demand}: {s_res}")
        print(f"[+] Inventory Newsvendor Solution for Demand={test_demand}: {i_res}")
        print("\n[!] To launch the interactive dashboard, run:")
        print("    streamlit run 03_Optimization_and_Dashboard.py")

"""
Running inside Google Colab
To view the live dashboard directly inside Google Colab, use localtunnel:

!pip install -q streamlit pulp scipy
!streamlit run 03_Optimization_and_Dashboard.py & npx localtunnel --port 8501

Click the URL printed by localtunnel to open the interactive manager dashboard in your browser.
"""