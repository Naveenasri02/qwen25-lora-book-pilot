"""Merge the per-chapter unit files into data/units/units.json.

Each chapter file is JSON Lines with {"type", "text"}. This script checks every
line, assigns a stable id (ch<NN>-<NNN>) and writes one JSON array.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITS_DIR = ROOT / "data" / "units"
OUT = UNITS_DIR / "units.json"
TYPES = {"fact", "definition", "example", "link"}


def main() -> int:
    units, seen, errors = [], {}, []
    for path in sorted(UNITS_DIR.glob("ch*.jsonl")):
        chapter = int(re.match(r"ch(\d+)_", path.name).group(1))
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for n, line in enumerate(lines, 1):
            where = f"{path.name}:{n}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: bad JSON ({exc})")
                continue
            text = row.get("text", "").strip()
            if row.get("type") not in TYPES:
                errors.append(f"{where}: unknown type {row.get('type')!r}")
            if len(text) < 30:
                errors.append(f"{where}: text too short")
            if text in seen:
                errors.append(f"{where}: duplicate of {seen[text]}")
            seen[text] = where
            units.append({"id": f"ch{chapter:02d}-{n:03d}", "chapter": chapter,
                          "type": row.get("type"), "text": text})

    if errors:
        print("\n".join(errors))
        return 1

    OUT.write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    by_type = Counter(u["type"] for u in units)
    by_chapter = Counter(u["chapter"] for u in units)
    words = sum(len(u["text"].split()) for u in units)
    print(f"units: {len(units)}  words: {words}  -> {OUT.relative_to(ROOT)}")
    print("by type:", dict(sorted(by_type.items())))
    print("by chapter:", dict(sorted(by_chapter.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
