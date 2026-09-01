"""
chat_agent.py
The AI agent itself: wraps Claude's tool-use loop around the supply-chain
optimization functions so you can ask natural-language questions like:

  "What should I reorder at the Dallas warehouse this week?"
  "Which products are at stockout risk?"
  "What's the cheapest route to visit all six warehouses starting from Atlanta?"
  "Forecast demand for USB-C cables over the next 45 days."

Requires an ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import os
import sys

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install -r requirements.txt")
    raise

from . import optimization as opt
from .tools import TOOLS, dispatch

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")

SYSTEM_PROMPT = """You are a supply chain operations AI agent. You have tools to inspect \
inventory, forecast demand, calculate reorder points / EOQ, run ABC (Pareto) analysis, \
scan for stockout or overstock risk across the network, and optimize multi-stop delivery \
routes between warehouses.

Guidelines:
- Always use tools to get real numbers rather than guessing.
- When asked broad questions ("what needs attention?"), use scan_inventory_health first.
- When recommending a reorder, state the quantity, the supplier, and the lead time.
- Be concise and lead with the actionable recommendation, then show the supporting numbers.
- If a request is ambiguous (e.g. no warehouse specified), make a reasonable assumption, \
state it, and proceed — don't just ask a clarifying question if you can reasonably act.
- Flag numbers that seem surprising (e.g. huge reorder quantities, near-zero stock) so a \
human reviews them before acting.
"""


class SupplyChainAgent:
    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.data = opt.SupplyChainData()
        self.messages: list[dict] = []

    def ask(self, user_message: str, verbose: bool = True) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.messages,
            )

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return "".join(block.text for block in response.content if block.type == "text")

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"  [tool call] {block.name}({block.input})", file=sys.stderr)
                    result = dispatch(self.data, block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            self.messages.append({"role": "user", "content": tool_results})

    def reset(self):
        self.messages = []


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY before running the agent, e.g.:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    agent = SupplyChainAgent()
    print("Supply Chain AI Agent — ask a question (or 'quit' to exit)\n")
    print("Examples:")
    print("  - What needs attention across the network right now?")
    print("  - Should I reorder Barcode Scanners at WH-DAL?")
    print("  - Forecast demand for P-1005 over the next 45 days.")
    print("  - Optimize a route across all six warehouses starting from WH-ATL.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        answer = agent.ask(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
