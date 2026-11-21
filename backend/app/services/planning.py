"""Tomato planning: cultivation cost estimate + step-by-step roadmap with dates.

Costs are INDICATIVE per-acre defaults for open-field hybrid tomato with drip
and staking (western Maharashtra style). Every quantity and rate can be edited
from the frontend, so the farmer can put in local prices.
The roadmap dates come from the same stage day-ranges used by the RAG (docs/stages).
"""
from datetime import date, timedelta

from . import knowledge

# key: (label, quantity per acre, unit, rate in Rs per unit)
DEFAULT_COST_ITEMS = {
    "land_prep": ("Land preparation (ploughing, rotavator, beds)", 1, "acre", 6000),
    "seedlings": ("Hybrid seedlings from nursery", 8000, "plants", 1.8),
    "fym": ("FYM / compost", 8, "tonnes", 1000),
    "fertilizer": ("Fertilizers (basal + water-soluble via drip)", 1, "season", 15000),
    "plant_protection": ("Plant protection (fungicides, insecticides, bio-agents)", 1, "season", 12000),
    "mulch": ("Mulching film", 1, "acre", 8000),
    "drip": ("Drip irrigation (season share / maintenance)", 1, "season", 10000),
    "staking": ("Staking: bamboo, wire, twine", 1, "acre", 15000),
    "labour": ("Labour (transplant, weeding, pruning, tying, spraying, harvest)", 120, "person-days", 350),
    "irrigation_power": ("Electricity / water for irrigation", 1, "season", 4000),
    "marketing": ("Crates, transport and market charges", 1, "season", 10000),
}
DEFAULT_YIELD_T_PER_ACRE = 20.0
DEFAULT_PRICE_PER_KG = 10.0
CONTINGENCY_PCT = 5.0


def cost_estimate(area_acres=1.0, overrides=None, yield_t_per_acre=None, price_per_kg=None,
                  contingency_pct=CONTINGENCY_PCT):
    overrides = overrides or {}
    items, total = [], 0.0
    for key, (label, qty, unit, rate) in DEFAULT_COST_ITEMS.items():
        o = overrides.get(key, {})
        q = float(o.get("quantity", qty))
        r = float(o.get("rate", rate))
        amount = q * r * area_acres
        total += amount
        items.append({"key": key, "label": label, "quantity": q, "unit": unit, "rate": r,
                      "amount": round(amount)})
    contingency = total * contingency_pct / 100
    total_cost = total + contingency
    y = float(yield_t_per_acre or DEFAULT_YIELD_T_PER_ACRE)
    p = float(price_per_kg or DEFAULT_PRICE_PER_KG)
    revenue = y * 1000 * p * area_acres
    breakeven_price = total_cost / (y * 1000 * area_acres) if y > 0 else None
    return {"area_acres": area_acres, "items": items, "subtotal": round(total),
            "contingency_pct": contingency_pct, "contingency": round(contingency),
            "total_cost": round(total_cost), "cost_per_acre": round(total_cost / area_acres) if area_acres else None,
            "expected_yield_t": round(y * area_acres, 1), "price_per_kg": p,
            "expected_revenue": round(revenue), "expected_profit": round(revenue - total_cost),
            "breakeven_price_per_kg": round(breakeven_price, 2) if breakeven_price else None,
            "note": "Indicative figures. Edit quantities and rates to your local prices; tomato market "
                    "prices swing widely, so check the break-even price against your APMC rates."}


def _bullets(text):
    if not text:
        return []
    return [l.strip().lstrip("-* ").replace("**", "").strip()
            for l in text.splitlines() if l.strip().startswith(("-", "*"))]


def roadmap(transplant_date: date | None, nursery_days=25, fertility=None):
    """Stage-by-stage plan with calendar dates (if a transplanting date is known)."""
    steps = []
    pre = ["Get a soil test / Soil Health Card (pH, N, P, K, organic carbon).",
           "Choose a hybrid suited to the season; buy certified seed or healthy nursery seedlings.",
           "Plan drip + mulch + staking material before transplanting."]
    if fertility:
        pre += [p["advice"] for p in fertility.get("parameters", []) if p.get("advice")]
    steps.append({"stage_id": "planning", "stage_name": "Before planting", "tasks": pre,
                  "start": (transplant_date - timedelta(days=nursery_days + 10)).isoformat() if transplant_date else None,
                  "end": (transplant_date - timedelta(days=nursery_days)).isoformat() if transplant_date else None})
    for row in knowledge.stage_table():
        sid = row["stage_id"]
        g = knowledge.stage_guidance(sid)
        if row["day_reference"] == "DAS":      # nursery: days after sowing
            start_off, end_off = row["day_from"] - nursery_days, min(row["day_to"], nursery_days) - nursery_days
        else:
            start_off, end_off = row["day_from"], min(row["day_to"], 130)
        step = {"stage_id": sid, "stage_name": g["stage_name"],
                "days": f"{row['day_from']}-{row['day_to'] if row['day_to'] < 999 else '...'} {row['day_reference']}",
                "tasks": _bullets(g["key_management"]),
                "water_nutrients": g["water_nutrients"],
                "watch_for": g["diseases_at_risk"]}
        if transplant_date:
            step["start"] = (transplant_date + timedelta(days=start_off)).isoformat()
            step["end"] = (transplant_date + timedelta(days=end_off)).isoformat()
        steps.append(step)
    return steps
