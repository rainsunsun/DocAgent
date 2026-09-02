"""确定性生成零售销售 demo 数据 data/sales.csv（固定随机种子，可复现）。

列：order_id, order_date, region, category, product, quantity, unit_price, amount
时间跨度 2024-01 ~ 2026-08，跨年支持同比/环比。从 doc-agent 目录运行：
    python scripts/generate_sales.py
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta

SEED = 42
N_ORDERS = 500
START = date(2024, 1, 1)
END = date(2026, 8, 31)

REGIONS = ["华东", "华北", "华南", "西南", "西北"]

# 品类 -> [(商品, 单价下限, 单价上限)]
CATEGORY_PRODUCTS = {
    "3C数码": [("手机", 2000, 8000), ("笔记本电脑", 3500, 9000), ("平板", 1200, 5000), ("耳机", 150, 2000), ("智能手表", 500, 3000)],
    "家用电器": [("冰箱", 1500, 6000), ("洗衣机", 1000, 5000), ("空调", 1800, 7000), ("电饭煲", 150, 800), ("吸尘器", 300, 1500)],
    "服装鞋包": [("T恤", 50, 300), ("牛仔裤", 100, 500), ("运动鞋", 200, 1200), ("羽绒服", 400, 1500), ("双肩包", 100, 600)],
    "食品饮料": [("牛奶", 30, 80), ("坚果", 40, 120), ("咖啡", 30, 200), ("茶叶", 50, 300), ("巧克力", 20, 100)],
    "美妆个护": [("洗面奶", 30, 200), ("面霜", 80, 800), ("口红", 100, 400), ("洗发水", 40, 150), ("香水", 200, 800)],
}


def main() -> None:
    rng = random.Random(SEED)
    days = (END - START).days
    rows: list[list] = []
    for i in range(1, N_ORDERS + 1):
        d = START + timedelta(days=rng.randint(0, days))
        region = rng.choice(REGIONS)
        category = rng.choice(list(CATEGORY_PRODUCTS))
        product, lo, hi = rng.choice(CATEGORY_PRODUCTS[category])
        quantity = rng.randint(1, 5)
        unit_price = round(rng.uniform(lo, hi), 2)
        amount = round(quantity * unit_price, 2)
        rows.append([i, d.isoformat(), region, category, product, quantity, unit_price, amount])

    header = ["order_id", "order_date", "region", "category", "product", "quantity", "unit_price", "amount"]
    with open("data/sales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"生成 {len(rows)} 行 → data/sales.csv")


if __name__ == "__main__":
    main()
