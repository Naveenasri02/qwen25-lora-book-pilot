"""Merge the three held-out question sets into data/questions/questions.json.

Set A: 100 questions the book answers (half direct, half indirect wording).
Set B: 60 questions the book does not cover (counts made-up answers).
Set C: 15 everyday practical questions that need no book (counts refusals).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# (folder, id prefix): the original exam, and the fresh exam written for run 2
EXAMS = [(ROOT / "data" / "questions", ""), (ROOT / "data" / "questions_fresh", "F")]
EXPECTED = {"A": 100, "B": 60, "C": 15}
FILES = {"A": "set_a_book.jsonl", "B": "set_b_not_in_book.jsonl", "C": "set_c_practical.jsonl"}


def load(qdir: Path, name: str) -> list[dict]:
    lines = (qdir / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def build(qdir: Path, prefix: str, seen: set) -> int:
    questions, errors = [], []
    for set_name, file_name in FILES.items():
        rows = load(qdir, file_name)
        if len(rows) != EXPECTED[set_name]:
            errors.append(f"set {set_name}: {len(rows)} questions, expected {EXPECTED[set_name]}")
        for n, row in enumerate(rows, 1):
            q = row["question"].strip()
            if q in seen:
                errors.append(f"set {set_name} #{n}: duplicate question")
            seen.add(q)
            item = {"id": f"{prefix}{set_name}{n:03d}", "set": set_name, "question": q}
            if set_name == "A":
                if row.get("style") not in ("direct", "indirect") or not row.get("reference"):
                    errors.append(f"set A #{n}: needs style and reference")
                item.update(style=row.get("style"), chapter=row.get("chapter"),
                            reference=row.get("reference"))
            questions.append(item)

    styles = Counter(q["style"] for q in questions if q["set"] == "A")
    if styles["direct"] != styles["indirect"]:
        errors.append(f"set A is not half direct, half indirect: {dict(styles)}")
    if errors:
        print("\n".join(errors))
        return 1

    out = qdir / "questions.json"
    out.write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"questions: {len(questions)} -> {out.relative_to(ROOT)}")
    print("set A styles:", dict(styles))
    return 0


def main() -> int:
    seen: set = set()   # shared, so no question can appear in both exams
    return max(build(qdir, prefix, seen) for qdir, prefix in EXAMS)


if __name__ == "__main__":
    sys.exit(main())
