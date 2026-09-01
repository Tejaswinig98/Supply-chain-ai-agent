"""
dashboard.py
Streamlit dashboard for the supply chain AI agent prototype.

Run: streamlit run dashboard.py

Tabs:
  - Network Overview: KPIs, stock status breakdown, ABC chart
  - Inventory Health: filterable table of reorder alerts
  - Forecast Explorer: pick a product/warehouse, see forecast + chart
  - Route Optimizer: pick warehouses, see optimized route on a map
  - Ask the Agent: natural-language Q&A (requires ANTHROPIC_API_KEY)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from agent import optimization as opt

st.set_page_config(page_title="Supply Chain AI Agent", layout="wide")


@st.cache_resource
def load_data():
    return opt.SupplyChainData()


data = load_data()

st.title("📦 Supply Chain AI Agent")
st.caption("Inventory optimization prototype — synthetic data, runs entirely offline except for the chat tab.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Network Overview", "Inventory Health", "Forecast Explorer", "Route Optimizer", "Ask the Agent"]
)

# ---------------------------------------------------------------------------
with tab1:
    health = opt.inventory_health_scan(data)
    abc = opt.abc_classification(data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Warehouses", len(data.warehouses))
    c2.metric("SKUs tracked", len(data.products))
    c3.metric("Stockout risk items", int((health.status == "STOCKOUT_RISK").sum()))
    c4.metric("Overstocked items", int((health.status == "OVERSTOCK").sum()))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Inventory status breakdown")
        st.bar_chart(health["status"].value_counts())
    with col_b:
        st.subheader("ABC tier — revenue share")
        st.bar_chart(abc.groupby("abc_tier")["revenue"].sum())

    st.subheader("Warehouse network")
    st.map(data.warehouses.rename(columns={"lat": "latitude", "lon": "longitude"}), size=200)

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Inventory health scan")
    status_filter = st.selectbox("Filter by status", ["ALL", "STOCKOUT_RISK", "REORDER_NOW", "OVERSTOCK", "HEALTHY"])
    health = opt.inventory_health_scan(data)
    if status_filter != "ALL":
        health = health[health.status == status_filter]
    st.dataframe(health, use_container_width=True, hide_index=True)
    st.download_button("Download as CSV", health.to_csv(index=False), file_name="inventory_health.csv")

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Demand forecast explorer")
    col1, col2, col3 = st.columns(3)
    product_id = col1.selectbox("Product", data.products.product_id + " — " + data.products.name)
    product_id = product_id.split(" — ")[0]
    warehouse_id = col2.selectbox("Warehouse (optional)", ["ALL"] + list(data.warehouses.warehouse_id))
    horizon = col3.slider("Horizon (days)", 7, 90, 30)

    wid = None if warehouse_id == "ALL" else warehouse_id
    series = data.demand_series(product_id, wid)
    st.line_chart(series.tail(180))

    forecast = opt.forecast_demand(data, product_id, wid, horizon_days=horizon)
    st.json(forecast)

    if wid:
        st.subheader("Reorder recommendation")
        rp = opt.calculate_reorder_point(data, product_id, wid)
        st.json(rp)

# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Multi-warehouse route optimizer")
    selected = st.multiselect(
        "Warehouses to visit", list(data.warehouses.warehouse_id), default=list(data.warehouses.warehouse_id)
    )
    start = st.selectbox("Start warehouse", selected) if selected else None

    if len(selected) >= 2:
        route = opt.optimize_route(data, selected, start_warehouse_id=start)
        st.json(route)

        ordered = data.warehouses.set_index("warehouse_id").loc[route["optimized_route"]].reset_index()
        st.map(ordered.rename(columns={"lat": "latitude", "lon": "longitude"}), size=200)
        st.write(" → ".join(route["optimized_route"]))
    else:
        st.info("Select at least 2 warehouses.")

# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Ask the agent (natural language)")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.warning(
            "Set the ANTHROPIC_API_KEY environment variable and restart Streamlit to enable this tab. "
            "All other tabs work without an API key."
        )
    else:
        from agent.chat_agent import SupplyChainAgent

        if "agent" not in st.session_state:
            st.session_state.agent = SupplyChainAgent(api_key=api_key)
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

        user_q = st.chat_input("e.g. What needs attention across the network right now?")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            with st.chat_message("user"):
                st.write(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.agent.ask(user_q, verbose=False)
                st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))
