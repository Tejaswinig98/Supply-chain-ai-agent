"""
tools.py
Tool schemas (Anthropic tool-use format) + a dispatcher that maps a tool
call name/input to the underlying optimization.py functions and returns
JSON-serializable results.
"""
from __future__ import annotations

import json
import pandas as pd

from . import optimization as opt

TOOLS = [
    {
        "name": "list_warehouses",
        "description": "List all distribution warehouses with their location and capacity.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_products",
        "description": "List all products tracked in the supply chain, with category, cost, and price.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "forecast_demand",
        "description": "Forecast future demand for a product (optionally scoped to one warehouse) over a given horizon in days. Uses a weighted moving average with trend.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "e.g. P-1001"},
                "warehouse_id": {"type": "string", "description": "e.g. WH-ATL. Omit for company-wide."},
                "horizon_days": {"type": "integer", "description": "Forecast horizon in days, default 30"},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "get_eoq",
        "description": "Calculate the Economic Order Quantity (EOQ) — the cost-optimal order size — for a product, balancing ordering cost against holding cost.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "warehouse_id": {"type": "string", "description": "Optional, scopes demand to one warehouse"},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "get_reorder_recommendation",
        "description": "Get the reorder point, safety stock, current stock position, and whether a reorder is needed right now for a product at a specific warehouse. This is the primary tool for 'should I reorder X' questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "warehouse_id": {"type": "string"},
                "service_level": {"type": "number", "description": "Target service level, e.g. 0.95 (default). Options: 0.90, 0.95, 0.975, 0.99"},
            },
            "required": ["product_id", "warehouse_id"],
        },
    },
    {
        "name": "scan_inventory_health",
        "description": "Scan ALL products across ALL warehouses and flag stockout risk, items needing reorder, and overstocked positions. Use this for a broad 'what needs attention' or 'give me alerts' question. Can filter by status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "enum": ["STOCKOUT_RISK", "REORDER_NOW", "OVERSTOCK", "HEALTHY", "ALL"],
                    "description": "Filter results to one status. Default ALL.",
                },
                "limit": {"type": "integer", "description": "Max rows to return, default 25"},
            },
        },
    },
    {
        "name": "get_abc_classification",
        "description": "Classify products into A/B/C tiers by trailing-90-day revenue contribution (Pareto analysis). A = top ~80% of revenue, B = next 15%, C = bottom 5%. Useful for prioritizing inventory management effort.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tier_filter": {"type": "string", "enum": ["A", "B", "C", "ALL"], "description": "Default ALL"},
            },
        },
    },
    {
        "name": "optimize_delivery_route",
        "description": "Compute the shortest multi-stop route across a set of warehouses (e.g. for a consolidation run or inter-warehouse transfer), using nearest-neighbor construction + 2-opt refinement over great-circle distances.",
        "input_schema": {
            "type": "object",
            "properties": {
                "warehouse_ids": {"type": "array", "items": {"type": "string"}, "description": "List of warehouse IDs to visit, e.g. ['WH-ATL','WH-DAL','WH-CHI']"},
                "start_warehouse_id": {"type": "string", "description": "Optional fixed starting point"},
            },
            "required": ["warehouse_ids"],
        },
    },
]


def _df_to_records(df: pd.DataFrame, limit: int = 25) -> list[dict]:
    return json.loads(df.head(limit).to_json(orient="records"))


def dispatch(data: opt.SupplyChainData, tool_name: str, tool_input: dict) -> dict:
    """Execute a tool call against the loaded SupplyChainData and return a JSON-safe result."""
    try:
        if tool_name == "list_warehouses":
            return {"warehouses": _df_to_records(data.warehouses, limit=100)}

        if tool_name == "list_products":
            return {"products": _df_to_records(data.products, limit=100)}

        if tool_name == "forecast_demand":
            return opt.forecast_demand(
                data,
                tool_input["product_id"],
                tool_input.get("warehouse_id"),
                tool_input.get("horizon_days", 30),
            )

        if tool_name == "get_eoq":
            return opt.calculate_eoq(data, tool_input["product_id"], tool_input.get("warehouse_id"))

        if tool_name == "get_reorder_recommendation":
            return opt.calculate_reorder_point(
                data,
                tool_input["product_id"],
                tool_input["warehouse_id"],
                tool_input.get("service_level", 0.95),
            )

        if tool_name == "scan_inventory_health":
            df = opt.inventory_health_scan(data)
            status_filter = tool_input.get("status_filter", "ALL")
            if status_filter and status_filter != "ALL":
                df = df[df.status == status_filter]
            limit = tool_input.get("limit", 25)
            return {
                "total_matching": len(df),
                "status_counts": df["status"].value_counts().to_dict() if not df.empty else {},
                "rows": _df_to_records(df, limit=limit),
            }

        if tool_name == "get_abc_classification":
            df = opt.abc_classification(data)
            tier_filter = tool_input.get("tier_filter", "ALL")
            if tier_filter and tier_filter != "ALL":
                df = df[df.abc_tier == tier_filter]
            return {"products": _df_to_records(df, limit=100)}

        if tool_name == "optimize_delivery_route":
            return opt.optimize_route(
                data,
                tool_input["warehouse_ids"],
                tool_input.get("start_warehouse_id"),
            )

        return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:  # surface errors back to the model instead of crashing the loop
        return {"error": str(e)}
