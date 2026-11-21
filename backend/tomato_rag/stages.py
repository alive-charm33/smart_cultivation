"""Convert 'days' into growth stage(s). Day ranges are read from the front matter of
docs/stages/*.md (day_from, day_to, day_reference), so they are edited in ONE place.
"""
from pathlib import Path
import re, yaml

STAGES_DIR = Path(__file__).parent / "docs" / "stages"
NURSERY_DAYS_DEFAULT = 25   # used to convert days-after-sowing into days-after-transplanting


def load_stage_table():
    rows = []
    for p in sorted(STAGES_DIR.glob("*.md")):
        m = re.match(r"^---\n(.*?)\n---\n", p.read_text(encoding="utf-8"), re.S)
        meta = yaml.safe_load(m.group(1))
        rows.append({"stage_id": meta["stage_id"], "day_from": meta["day_from"],
                     "day_to": meta["day_to"], "day_reference": meta["day_reference"]})
    return sorted(rows, key=lambda r: (r["day_reference"] != "DAS", r["day_from"]))


def resolve_stages(days, days_from="transplanting", nursery_days=NURSERY_DAYS_DEFAULT):
    """days_from: 'transplanting' (DAT) or 'sowing' (DAS).
    Returns a list of stage_ids (several when stages overlap, ordered by start day)."""
    if days is None:
        return []
    table = load_stage_table()
    if days_from == "sowing":
        if days <= nursery_days:
            return ["nursery"]
        days = days - nursery_days            # convert DAS -> DAT
    return [r["stage_id"] for r in table
            if r["day_reference"] == "DAT" and r["day_from"] <= days <= r["day_to"]]


if __name__ == "__main__":
    for d in (5, 20, 32, 47, 60, 80, 108):
        print(f"{d:>3} DAT ->", resolve_stages(d))
    print("40 DAS ->", resolve_stages(40, "sowing"), "| 20 DAS ->", resolve_stages(20, "sowing"))
