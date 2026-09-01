"""
generate_data.py
Creates synthetic supply-chain datasets used by the AI agent prototype:
  - warehouses.csv
  - products.csv
  - suppliers.csv
  - inventory.csv
  - demand_history.csv

Run: python3 generate_data.py
"""
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Warehouses
# ---------------------------------------------------------------------------
WAREHOUSES = [
    {"warehouse_id": "WH-ATL", "name": "Atlanta DC",     "city": "Atlanta, GA",     "lat": 33.7490, "lon": -84.3880, "capacity_units": 50000},
    {"warehouse_id": "WH-DAL", "name": "Dallas DC",       "city": "Dallas, TX",      "lat": 32.7767, "lon": -96.7970, "capacity_units": 60000},
    {"warehouse_id": "WH-CHI", "name": "Chicago DC",      "city": "Chicago, IL",     "lat": 41.8781, "lon": -87.6298, "capacity_units": 55000},
    {"warehouse_id": "WH-LAX", "name": "Los Angeles DC",  "city": "Los Angeles, CA", "lat": 34.0522, "lon": -118.2437,"capacity_units": 70000},
    {"warehouse_id": "WH-NYC", "name": "New Jersey DC",   "city": "Newark, NJ",      "lat": 40.7357, "lon": -74.1724, "capacity_units": 45000},
    {"warehouse_id": "WH-SEA", "name": "Seattle DC",      "city": "Seattle, WA",     "lat": 47.6062, "lon": -122.3321,"capacity_units": 40000},
]

# ---------------------------------------------------------------------------
# Products (mix of A/B/C velocity classes baked into demand generation)
# ---------------------------------------------------------------------------
PRODUCTS = [
    {"product_id": "P-1001", "name": "Steel Bracket 4in",        "category": "Hardware",     "unit_cost": 2.10,  "unit_price": 4.50,  "velocity": "A"},
    {"product_id": "P-1002", "name": "Cardboard Box M",          "category": "Packaging",    "unit_cost": 0.45,  "unit_price": 0.95,  "velocity": "A"},
    {"product_id": "P-1003", "name": "Pallet Wrap Roll",         "category": "Packaging",    "unit_cost": 3.20,  "unit_price": 6.75,  "velocity": "A"},
    {"product_id": "P-1004", "name": "12V DC Motor",             "category": "Electronics",  "unit_cost": 14.50, "unit_price": 29.99, "velocity": "B"},
    {"product_id": "P-1005", "name": "USB-C Cable 1m",           "category": "Electronics",  "unit_cost": 1.80,  "unit_price": 5.99,  "velocity": "A"},
    {"product_id": "P-1006", "name": "Industrial Bearing 20mm",  "category": "Hardware",     "unit_cost": 6.75,  "unit_price": 13.50, "velocity": "B"},
    {"product_id": "P-1007", "name": "Safety Gloves (pair)",     "category": "PPE",          "unit_cost": 1.25,  "unit_price": 3.20,  "velocity": "B"},
    {"product_id": "P-1008", "name": "Hard Hat",                 "category": "PPE",          "unit_cost": 8.00,  "unit_price": 16.99, "velocity": "C"},
    {"product_id": "P-1009", "name": "Forklift Battery Pack",    "category": "Equipment",    "unit_cost": 320.00,"unit_price": 599.00,"velocity": "C"},
    {"product_id": "P-1010", "name": "Barcode Scanner",          "category": "Electronics",  "unit_cost": 45.00, "unit_price": 89.99, "velocity": "C"},
    {"product_id": "P-1011", "name": "Shipping Label Roll",      "category": "Packaging",    "unit_cost": 4.10,  "unit_price": 8.25,  "velocity": "A"},
    {"product_id": "P-1012", "name": "Aluminum Extrusion 1m",    "category": "Hardware",     "unit_cost": 9.30,  "unit_price": 18.00, "velocity": "B"},
    {"product_id": "P-1013", "name": "N95 Respirator",           "category": "PPE",          "unit_cost": 0.95,  "unit_price": 2.50,  "velocity": "B"},
    {"product_id": "P-1014", "name": "Pallet Jack",              "category": "Equipment",    "unit_cost": 210.00,"unit_price": 399.00,"velocity": "C"},
    {"product_id": "P-1015", "name": "Zip Ties (100pk)",         "category": "Hardware",     "unit_cost": 2.90,  "unit_price": 6.10,  "velocity": "A"},
]

VELOCITY_BASE_DEMAND = {"A": (60, 15), "B": (20, 7), "C": (4, 2)}  # (mean, std) units/day/warehouse

# ---------------------------------------------------------------------------
# Suppliers (1-2 per product, varying lead time / reliability)
# ---------------------------------------------------------------------------
SUPPLIER_NAMES = ["Acme Industrial", "Pioneer Supply Co", "Meridian Parts", "Northgate Logistics",
                   "Summit Wholesale", "Vantage Components", "Ironclad Distributors", "BlueRiver Trading"]

suppliers = []
sid = 1
for p in PRODUCTS:
    n_suppliers = random.choice([1, 1, 2])
    chosen = random.sample(SUPPLIER_NAMES, n_suppliers)
    for name in chosen:
        lead_time = random.choice([3, 5, 7, 10, 14, 21]) if p["velocity"] != "C" else random.choice([14, 21, 30])
        suppliers.append({
            "supplier_id": f"SUP-{sid:03d}",
            "supplier_name": name,
            "product_id": p["product_id"],
            "lead_time_days": lead_time,
            "reliability_score": round(random.uniform(0.82, 0.99), 2),
            "min_order_qty": random.choice([50, 100, 200, 500]),
        })
        sid += 1

# ---------------------------------------------------------------------------
# Inventory snapshot (current on-hand per warehouse/product)
# ---------------------------------------------------------------------------
inventory = []
for wh in WAREHOUSES:
    for p in PRODUCTS:
        mean_demand, _ = VELOCITY_BASE_DEMAND[p["velocity"]]
        # Deliberately create a mix of healthy / low / overstocked positions
        situation = random.choices(["low", "healthy", "overstock"], weights=[0.25, 0.55, 0.20])[0]
        if situation == "low":
            on_hand = int(mean_demand * random.uniform(1, 4))
        elif situation == "healthy":
            on_hand = int(mean_demand * random.uniform(10, 20))
        else:
            on_hand = int(mean_demand * random.uniform(30, 50))
        inventory.append({
            "warehouse_id": wh["warehouse_id"],
            "product_id": p["product_id"],
            "on_hand_qty": on_hand,
            "on_order_qty": random.choice([0, 0, 0, 50, 100]),
        })

# ---------------------------------------------------------------------------
# Demand history (365 days, with weekly seasonality + trend + noise + occasional spikes)
# ---------------------------------------------------------------------------
demand_rows = []
start_date = datetime.today() - timedelta(days=365)
for wh in WAREHOUSES:
    wh_factor = random.uniform(0.8, 1.3)
    for p in PRODUCTS:
        mean_demand, std_demand = VELOCITY_BASE_DEMAND[p["velocity"]]
        trend_slope = random.uniform(-0.01, 0.03)  # slow drift over the year
        for day_idx in range(365):
            date = start_date + timedelta(days=day_idx)
            weekday_factor = 1.15 if date.weekday() < 5 else 0.6  # weekday vs weekend
            season_factor = 1 + 0.15 * math.sin(2 * math.pi * day_idx / 365)
            trend_factor = 1 + trend_slope * (day_idx / 365)
            base = mean_demand * wh_factor * weekday_factor * season_factor * trend_factor
            noise = random.gauss(0, std_demand * 0.5)
            spike = 0
            if random.random() < 0.01:  # 1% chance of a demand spike (promo, etc.)
                spike = mean_demand * random.uniform(2, 5)
            units = max(0, round(base + noise + spike))
            demand_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "warehouse_id": wh["warehouse_id"],
                "product_id": p["product_id"],
                "units_sold": units,
            })

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
def write_csv(filename, rows, fieldnames):
    path = DATA_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows -> {path}")

write_csv("warehouses.csv", WAREHOUSES, list(WAREHOUSES[0].keys()))
write_csv("products.csv", PRODUCTS, list(PRODUCTS[0].keys()))
write_csv("suppliers.csv", suppliers, list(suppliers[0].keys()))
write_csv("inventory.csv", inventory, list(inventory[0].keys()))
write_csv("demand_history.csv", demand_rows, list(demand_rows[0].keys()))

print("\nSynthetic data generation complete.")
