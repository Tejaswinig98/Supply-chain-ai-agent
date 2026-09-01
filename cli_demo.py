"""
cli_demo.py
Runs the optimization engine directly (NO API key required) and prints a
full supply-chain report to the console. Good for a quick sanity check that
the data + math pipeline works before wiring up the conversational agent.

Run: python3 cli_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import optimization as opt


def line(char="-", n=78):
    print(char * n)


def main():
    data = opt.SupplyChainData()

    print("SUPPLY CHAIN AI AGENT — OFFLINE ANALYTICS REPORT")
    print(f"({len(data.products)} products x {len(data.warehouses)} warehouses, "
          f"{len(data.demand):,} days of demand history)")
    line("=")

    # 1. Inventory health scan
    print("\n1) INVENTORY HEALTH SCAN")
    line()
    health = opt.inventory_health_scan(data)
    print(health["status"].value_counts().to_string())
    print("\nTop items needing immediate action:")
    urgent = health[health.status.isin(["STOCKOUT_RISK", "REORDER_NOW"])].head(10)
    print(urgent.to_string(index=False))

    # 2. ABC classification
    print("\n\n2) ABC CLASSIFICATION (90-day revenue, Pareto analysis)")
    line()
    abc = opt.abc_classification(data)
    print(abc.to_string(index=False))

    # 3. Sample reorder recommendation
    print("\n\n3) SAMPLE REORDER RECOMMENDATION")
    line()
    sample = opt.calculate_reorder_point(data, "P-1001", "WH-ATL")
    for k, v in sample.items():
        print(f"  {k:28s}: {v}")

    # 4. Sample demand forecast
    print("\n\n4) SAMPLE DEMAND FORECAST (45 days)")
    line()
    forecast = opt.forecast_demand(data, "P-1005", "WH-CHI", horizon_days=45)
    for k, v in forecast.items():
        print(f"  {k:28s}: {v}")

    # 5. Route optimization
    print("\n\n5) MULTI-WAREHOUSE ROUTE OPTIMIZATION")
    line()
    route = opt.optimize_route(
        data, list(data.warehouses.warehouse_id), start_warehouse_id="WH-ATL"
    )
    for k, v in route.items():
        print(f"  {k:28s}: {v}")

    line("=")
    print("\nDone. For the conversational agent (natural-language Q&A), set")
    print("ANTHROPIC_API_KEY and run: python3 -m agent.chat_agent")
    print("For the visual dashboard, run: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
