#!/usr/bin/env python3
import json
from pathlib import Path
import sys

total = 0
has_any_date = 0
has_non_theatrical = 0
has_theatrical = 0
has_only_theatrical = 0

for p in Path(sys.argv[1]).iterdir():
    if p.suffix == ".json":
        continue
    json_path = Path(str(p) + ".json")
    if not json_path.exists():
        continue
    total += 1
    with open(json_path) as f:
        spans = json.load(f)
    if spans:
        has_any_date += 1
        mediums = set()
        for sp in spans:
            for k, v in sp["medium"].items():
                if v:
                    mediums.add(k)
        if "theatrical" in mediums:
            has_theatrical += 1
        non_t = mediums - {"theatrical"}
        if non_t:
            has_non_theatrical += 1
        if mediums == {"theatrical"}:
            has_only_theatrical += 1

print(f"Total:                    {total}")
print(f"Has any date:             {has_any_date}")
print(f"Has theatrical:           {has_theatrical}")
print(f"Has non-theatrical medium:{has_non_theatrical}")
print(f"Has ONLY theatrical:      {has_only_theatrical}")
print(f"No dates:                 {total - has_any_date}")
