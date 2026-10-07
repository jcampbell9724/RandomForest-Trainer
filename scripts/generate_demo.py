"""Generate a deterministic, fictional operations dataset for a local demo."""

import csv
from datetime import date, timedelta
from pathlib import Path
import random


def main():
    rng = random.Random(42)
    output = Path(__file__).resolve().parents[1] / "examples" / "synthetic_orders.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "order_value", "item_count", "region", "service_tier",
            "created_at", "processing_days",
        ])
        writer.writeheader()
        for index in range(180):
            items = rng.randint(1, 18)
            tier = rng.choice(["standard", "priority"])
            order_value = round(items * rng.uniform(15, 80), 2)
            processing_days = max(1, round(
                2 + items * 0.35 + (2 if tier == "standard" else 0)
                + rng.gauss(0, 0.6), 2))
            writer.writerow({
                "order_value": order_value if index % 29 else "",
                "item_count": items,
                "region": rng.choice(["east", "west", "central"]),
                "service_tier": tier,
                "created_at": (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
                "processing_days": processing_days,
            })
    print(f"Wrote 180 synthetic rows to {output}")


if __name__ == "__main__":
    main()
