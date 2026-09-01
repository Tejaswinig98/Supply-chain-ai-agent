"""
optimization.py
Core supply-chain optimization logic used by the AI agent. Pure Python/pandas —
no external optimization solver required, so it runs anywhere.

Covers:
  - Demand forecasting (weighted moving average + trend)
  - EOQ (Economic Order Quantity)
  - Reorder point & safety stock (service-level based)
  - ABC classification (revenue-based inventory segmentation)
  - Inventory health scan (stockout risk / overstock flags)
  - Delivery route optimization (nearest-neighbor heuristic over warehouses)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# Ordering cost assumption ($ per purchase order) and annual holding cost rate
# (as a fraction of unit cost). These are reasonable industry defaults and are
# the two knobs you'd tune per-business in a real deployment.
ORDER_COST = 45.00
HOLDING_COST_RATE = 0.22  # 22% of unit cost per year
ANNUAL_ORDERING_DAYS = 365

# z-scores for common service levels (safety-stock formula)
SERVICE_LEVEL_Z = {0.90: 1.28, 0.95: 1.65, 0.975: 1.96, 0.99: 2.33}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class SupplyChainData:
    """Loads and caches all CSVs; provides convenience lookups."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.warehouses = pd.read_csv(data_dir / "warehouses.csv")
        self.products = pd.read_csv(data_dir / "products.csv")
        self.suppliers = pd.read_csv(data_dir / "suppliers.csv")
        self.inventory = pd.read_csv(data_dir / "inventory.csv")
        self.demand = pd.read_csv(data_dir / "demand_history.csv", parse_dates=["date"])

    def product(self, product_id: str) -> dict:
        row = self.products.loc[self.products.product_id == product_id]
        if row.empty:
            raise ValueError(f"Unknown product_id: {product_id}")
        return row.iloc[0].to_dict()

    def warehouse(self, warehouse_id: str) -> dict:
        row = self.warehouses.loc[self.warehouses.warehouse_id == warehouse_id]
        if row.empty:
            raise ValueError(f"Unknown warehouse_id: {warehouse_id}")
        return row.iloc[0].to_dict()

    def best_supplier(self, product_id: str) -> dict:
        """Pick the supplier with the best (lead_time, reliability) tradeoff."""
        subs = self.suppliers.loc[self.suppliers.product_id == product_id].copy()
        if subs.empty:
            raise ValueError(f"No supplier found for product_id: {product_id}")
        subs["score"] = subs["reliability_score"] / subs["lead_time_days"]
        return subs.sort_values("score", ascending=False).iloc[0].to_dict()

    def demand_series(self, product_id: str, warehouse_id: Optional[str] = None) -> pd.Series:
        df = self.demand[self.demand.product_id == product_id]
        if warehouse_id:
            df = df[df.warehouse_id == warehouse_id]
            series = df.groupby("date")["units_sold"].sum().sort_index()
        else:
            series = df.groupby("date")["units_sold"].sum().sort_index()
        return series

    def on_hand(self, product_id: str, warehouse_id: str) -> int:
        row = self.inventory.loc[
            (self.inventory.product_id == product_id) & (self.inventory.warehouse_id == warehouse_id)
        ]
        if row.empty:
            return 0
        return int(row.iloc[0]["on_hand_qty"])


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def forecast_demand(data: SupplyChainData, product_id: str, warehouse_id: Optional[str] = None,
                     horizon_days: int = 30) -> dict:
    """
    Simple, explainable forecast: triple exponential smoothing fallback to
    weighted moving average with linear trend. Returns daily average and
    total forecast for the horizon, plus the historical std dev (used for
    safety stock).
    """
    series = data.demand_series(product_id, warehouse_id)
    if series.empty or len(series) < 14:
        raise ValueError("Not enough demand history to forecast")

    y = series.values.astype(float)
    n = len(y)

    # Weighted moving average over the trailing 28 days (recent days weighted higher)
    window = min(28, n)
    recent = y[-window:]
    weights = np.arange(1, window + 1)
    wma = float(np.dot(recent, weights) / weights.sum())

    # Linear trend over trailing 60 days via simple least squares
    trend_window = min(60, n)
    x = np.arange(trend_window)
    y_trend = y[-trend_window:]
    slope, intercept = np.polyfit(x, y_trend, 1)

    daily_forecast = max(0.0, wma + slope * (window / 2))
    total_forecast = daily_forecast * horizon_days
    std_dev = float(np.std(y[-90:] if n >= 90 else y))

    return {
        "product_id": product_id,
        "warehouse_id": warehouse_id or "ALL",
        "daily_avg_forecast": round(daily_forecast, 2),
        "horizon_days": horizon_days,
        "total_forecast": round(total_forecast, 1),
        "trend_slope_per_day": round(float(slope), 3),
        "historical_daily_std_dev": round(std_dev, 2),
        "history_days_used": n,
    }


# ---------------------------------------------------------------------------
# EOQ / Reorder point / Safety stock
# ---------------------------------------------------------------------------
def calculate_eoq(data: SupplyChainData, product_id: str, warehouse_id: Optional[str] = None) -> dict:
    product = data.product(product_id)
    series = data.demand_series(product_id, warehouse_id)
    avg_daily_demand = float(series.tail(90).mean()) if len(series) else 0.0
    annual_demand = avg_daily_demand * ANNUAL_ORDERING_DAYS

    unit_cost = float(product["unit_cost"])
    holding_cost_per_unit = unit_cost * HOLDING_COST_RATE

    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        eoq = 0.0
    else:
        eoq = math.sqrt((2 * annual_demand * ORDER_COST) / holding_cost_per_unit)

    orders_per_year = (annual_demand / eoq) if eoq > 0 else 0
    return {
        "product_id": product_id,
        "warehouse_id": warehouse_id or "ALL",
        "annual_demand_est": round(annual_demand, 1),
        "unit_cost": unit_cost,
        "order_cost": ORDER_COST,
        "holding_cost_per_unit_per_year": round(holding_cost_per_unit, 2),
        "eoq_units": round(eoq),
        "estimated_orders_per_year": round(orders_per_year, 1),
    }


def calculate_reorder_point(data: SupplyChainData, product_id: str, warehouse_id: str,
                             service_level: float = 0.95) -> dict:
    forecast = forecast_demand(data, product_id, warehouse_id, horizon_days=30)
    supplier = data.best_supplier(product_id)
    lead_time = int(supplier["lead_time_days"])

    daily_avg = forecast["daily_avg_forecast"]
    std_dev = forecast["historical_daily_std_dev"]

    z = SERVICE_LEVEL_Z.get(service_level, 1.65)
    safety_stock = z * std_dev * math.sqrt(lead_time)
    demand_during_lead_time = daily_avg * lead_time
    reorder_point = demand_during_lead_time + safety_stock

    on_hand = data.on_hand(product_id, warehouse_id)
    eoq = calculate_eoq(data, product_id, warehouse_id)["eoq_units"]

    return {
        "product_id": product_id,
        "warehouse_id": warehouse_id,
        "supplier_id": supplier["supplier_id"],
        "supplier_name": supplier["supplier_name"],
        "lead_time_days": lead_time,
        "service_level": service_level,
        "daily_avg_demand": daily_avg,
        "safety_stock_units": round(safety_stock),
        "reorder_point_units": round(reorder_point),
        "current_on_hand": on_hand,
        "eoq_recommended_order_qty": eoq,
        "action_needed": bool(on_hand <= round(reorder_point)),
        "days_of_supply_remaining": round(on_hand / daily_avg, 1) if daily_avg > 0 else None,
    }


# ---------------------------------------------------------------------------
# ABC classification
# ---------------------------------------------------------------------------
def abc_classification(data: SupplyChainData) -> pd.DataFrame:
    """Classify products into A/B/C tiers by trailing-90-day revenue contribution."""
    recent = data.demand[data.demand.date >= data.demand.date.max() - pd.Timedelta(days=90)]
    rev = recent.merge(data.products[["product_id", "unit_price", "name"]], on="product_id")
    rev["revenue"] = rev["units_sold"] * rev["unit_price"]
    by_product = rev.groupby(["product_id", "name"])["revenue"].sum().reset_index()
    by_product = by_product.sort_values("revenue", ascending=False)
    by_product["cum_pct"] = by_product["revenue"].cumsum() / by_product["revenue"].sum()

    def tier(cum_pct):
        if cum_pct <= 0.80:
            return "A"
        elif cum_pct <= 0.95:
            return "B"
        return "C"

    by_product["abc_tier"] = by_product["cum_pct"].apply(tier)
    by_product["revenue"] = by_product["revenue"].round(2)
    by_product["cum_pct"] = (by_product["cum_pct"] * 100).round(1)
    return by_product.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inventory health scan
# ---------------------------------------------------------------------------
def inventory_health_scan(data: SupplyChainData, service_level: float = 0.95) -> pd.DataFrame:
    """Scan every product/warehouse combo and flag stockout risk or overstock."""
    rows = []
    for _, inv_row in data.inventory.iterrows():
        pid, wid = inv_row["product_id"], inv_row["warehouse_id"]
        try:
            rp = calculate_reorder_point(data, pid, wid, service_level=service_level)
        except ValueError:
            continue
        on_hand = rp["current_on_hand"]
        reorder_point = rp["reorder_point_units"]
        eoq = rp["eoq_recommended_order_qty"]

        if on_hand <= reorder_point:
            status = "STOCKOUT_RISK" if on_hand < reorder_point * 0.5 else "REORDER_NOW"
        elif reorder_point > 0 and on_hand > reorder_point * 4:
            status = "OVERSTOCK"
        else:
            status = "HEALTHY"

        rows.append({
            "product_id": pid,
            "product_name": data.product(pid)["name"],
            "warehouse_id": wid,
            "on_hand": on_hand,
            "reorder_point": reorder_point,
            "days_of_supply": rp["days_of_supply_remaining"],
            "recommended_order_qty": eoq if status in ("STOCKOUT_RISK", "REORDER_NOW") else 0,
            "supplier": rp["supplier_name"],
            "lead_time_days": rp["lead_time_days"],
            "status": status,
        })
    return pd.DataFrame(rows).sort_values(
        by="status", key=lambda s: s.map({"STOCKOUT_RISK": 0, "REORDER_NOW": 1, "OVERSTOCK": 2, "HEALTHY": 3})
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Route optimization (nearest-neighbor heuristic + 2-opt refinement)
# ---------------------------------------------------------------------------
def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def optimize_route(data: SupplyChainData, warehouse_ids: list[str], start_warehouse_id: Optional[str] = None) -> dict:
    """
    Nearest-neighbor + 2-opt route optimization over a set of warehouses
    (e.g. a multi-stop replenishment or consolidation run). Distances use
    haversine great-circle miles as a proxy for road distance.
    """
    stops = [data.warehouse(w) for w in warehouse_ids]
    if len(stops) < 2:
        raise ValueError("Need at least 2 warehouses to build a route")

    start_idx = 0
    if start_warehouse_id:
        ids = [s["warehouse_id"] for s in stops]
        if start_warehouse_id in ids:
            start_idx = ids.index(start_warehouse_id)

    n = len(stops)
    dist = [[_haversine_miles(stops[i]["lat"], stops[i]["lon"], stops[j]["lat"], stops[j]["lon"])
             for j in range(n)] for i in range(n)]

    # Nearest neighbor construction
    unvisited = set(range(n)) - {start_idx}
    route = [start_idx]
    current = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[current][j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    def route_length(r):
        return sum(dist[r[i]][r[i + 1]] for i in range(len(r) - 1))

    # 2-opt refinement
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                if route_length(new_route) < route_length(route) - 1e-9:
                    route = new_route
                    improved = True

    total_miles = route_length(route)
    ordered_stops = [stops[i]["warehouse_id"] for i in route]
    naive_total = route_length(list(range(n))) if start_idx == 0 else None

    return {
        "optimized_route": ordered_stops,
        "total_distance_miles": round(total_miles, 1),
        "num_stops": n,
        "unoptimized_sequential_distance_miles": round(naive_total, 1) if naive_total else None,
        "est_savings_pct": round((1 - total_miles / naive_total) * 100, 1) if naive_total and naive_total > 0 else None,
    }


if __name__ == "__main__":
    data = SupplyChainData()
    print("Loaded:", len(data.products), "products,", len(data.warehouses), "warehouses")
    print(forecast_demand(data, "P-1001", "WH-ATL"))
    print(calculate_eoq(data, "P-1001", "WH-ATL"))
    print(calculate_reorder_point(data, "P-1001", "WH-ATL"))
