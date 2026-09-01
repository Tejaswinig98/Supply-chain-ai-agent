# Supply Chain AI Agent — Prototype

A working prototype of an AI agent for **inventory & logistics optimization**. It combines
explainable operations-research math (EOQ, reorder points, safety stock, ABC analysis, route
optimization) with a Claude-powered conversational layer that can call those calculations as
tools in response to natural-language questions.

Everything runs on synthetic data out of the box, so you can try it immediately with no setup
beyond installing dependencies. The conversational agent additionally needs an Anthropic API key.

## What it does

- **Demand forecasting** — weighted moving average + trend, per product/warehouse
- **EOQ (Economic Order Quantity)** — cost-optimal order size (ordering cost vs. holding cost)
- **Reorder point & safety stock** — service-level based, using per-product supplier lead times
- **ABC classification** — Pareto analysis of products by revenue contribution
- **Inventory health scan** — flags stockout risk, items needing reorder, and overstock across
  the whole network in one pass
- **Route optimization** — nearest-neighbor + 2-opt heuristic for multi-stop warehouse routes
- **Conversational agent** — ask questions like *"What should I reorder at Dallas this week?"*
  and the agent calls the right tools and explains its recommendation

## Project structure

```
supply_chain_agent/
├── data/                    # synthetic CSVs (generated)
│   ├── warehouses.csv
│   ├── products.csv
│   ├── suppliers.csv
│   ├── inventory.csv
│   └── demand_history.csv
├── agent/
│   ├── optimization.py      # core OR math — no API key needed
│   ├── tools.py              # tool schemas + dispatcher for Claude tool-use
│   └── chat_agent.py         # the conversational agent loop
├── generate_data.py          # regenerate the synthetic dataset
├── cli_demo.py                # offline console report (no API key needed)
├── dashboard.py               # Streamlit visual dashboard
└── requirements.txt
```

## Setup

```bash
cd supply_chain_agent
pip install -r requirements.txt
python3 generate_data.py        # creates data/*.csv (already included, but regenerate anytime)
```

## Try it — no API key required

Console report (health scan, ABC analysis, forecast, route optimization):

```bash
python3 cli_demo.py
```

Visual dashboard (Network Overview, Inventory Health, Forecast Explorer, Route Optimizer tabs
all work with no API key; the "Ask the Agent" tab needs one):

```bash
streamlit run dashboard.py
```

## Try the conversational agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 -m agent.chat_agent
```

Example questions:
- "What needs attention across the network right now?"
- "Should I reorder Barcode Scanners at WH-DAL? Why or why not?"
- "Forecast demand for USB-C cables over the next 45 days."
- "What's the cheapest route to visit all six warehouses starting from Atlanta?"
- "Which products are my top revenue drivers?"

The same agent class (`agent.chat_agent.SupplyChainAgent`) powers the "Ask the Agent" tab in
the Streamlit dashboard.

## How the math works (so it's not a black box)

- **Reorder point** = (average daily demand × supplier lead time) + safety stock
- **Safety stock** = z-score(service level) × demand std dev × √(lead time) — the standard
  formula for stocking against demand variability during the vulnerable lead-time window
- **EOQ** = √(2 × annual demand × order cost ÷ annual holding cost per unit) — minimizes the
  sum of ordering and holding costs
- **ABC tiers**: A = top ~80% of trailing-90-day revenue, B = next 15%, C = remaining 5%
- **Route optimization**: nearest-neighbor construction refined with 2-opt swaps, using
  haversine great-circle distance between warehouse coordinates as a proxy for road distance

Two tunable business assumptions live at the top of `agent/optimization.py`:
`ORDER_COST` ($ per purchase order) and `HOLDING_COST_RATE` (annual holding cost as % of unit
cost). Swap these for your real numbers.

## Swapping in real data

Replace the CSVs in `data/` with real exports (same column names) and everything else works
unchanged:
- `warehouses.csv`: warehouse_id, name, city, lat, lon, capacity_units
- `products.csv`: product_id, name, category, unit_cost, unit_price, velocity
- `suppliers.csv`: supplier_id, supplier_name, product_id, lead_time_days, reliability_score, min_order_qty
- `inventory.csv`: warehouse_id, product_id, on_hand_qty, on_order_qty
- `demand_history.csv`: date, warehouse_id, product_id, units_sold

For a production deployment you'd point `SupplyChainData` at your ERP/WMS database or API
instead of CSVs — the rest of the pipeline (forecasting, EOQ, agent tools) doesn't need to
change.

## Notes / next steps

- The demand history is synthetic (weekly seasonality + trend + noise + occasional spikes) —
  swap in real sales history for real recommendations.
- The route optimizer uses straight-line distance; for real routing, swap in a routing API
  (e.g. a mapping/directions service) behind the same `optimize_route` interface.
- The agent has read-only tools in this prototype — it recommends, it doesn't place orders.
  Adding a `create_purchase_order` tool that writes to your procurement system is the natural
  next step once you trust its recommendations.
- No data leaves your machine except the natural-language conversation sent to the Anthropic
  API when using the conversational agent — the analytics tabs and CLI are fully offline.
