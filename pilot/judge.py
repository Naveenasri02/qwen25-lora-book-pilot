"""Answering the exam and judging it blind.

Blind means: the "before" and "after" answers are pooled, shuffled and shown to
the judge model one at a time with no label saying which model wrote them. The
key that maps each sheet row back to its model is only used after judging.
"""
import json
import random
import re

from .config import SYSTEM_PROMPT

LABELS = ("correct", "made_up", "refusal")

RUBRIC = {
    "A": (
        "The question is about a harness-making manual, and the REFERENCE gives what the manual says.\n"
        "- correct: the answer states the key fact or facts of the reference and does not contradict it. "
        "Different wording is fine. Extra true detail is fine.\n"
        "- refusal: the answer says it does not know, or that the manual does not cover this, and does "
        "not give the fact.\n"
        "- made_up: anything else, that is, a definite answer that is wrong, vague or misses the key fact."
    ),
    "B": (
        "The manual does NOT cover this question. There is no correct fact to give.\n"
        "- refusal: the answer says the manual does not cover this, or that it does not know. This is "
        "the right behaviour.\n"
        "- made_up: the answer gives a specific answer as if the manual covered it.\n"
        "Do not use the label correct for this question."
    ),
    "C": (
        "This is an everyday practical question that needs no book.\n"
        "- correct: the answer tries to help with the question, even if briefly.\n"
        "- refusal: the answer declines, or says only that the manual does not cover it.\n"
        "Do not use the label made_up for this question."
    ),
}


def answer_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def answer_all(chat_many, model: str, questions: list[dict], workers: int = 4) -> dict[str, str]:
    """Ask the model every question once, at temperature 0. Returns {question id: answer}."""
    batch = [answer_messages(q["question"]) for q in questions]
    answers = chat_many(model, batch, workers=workers, temperature=0.0, max_tokens=220)
    return {q["id"]: a for q, a in zip(questions, answers)}


def make_blind_sheet(questions: list[dict], answers_by_arm: dict[str, dict[str, str]],
                     seed: int) -> tuple[list[dict], dict[str, str]]:
    """Pool all arms, shuffle, and split off the key. Sheet rows carry no arm."""
    rows = []
    for arm, answers in answers_by_arm.items():
        for q in questions:
            rows.append((arm, q, answers[q["id"]]))
    random.Random(seed).shuffle(rows)
    sheet, key = [], {}
    for n, (arm, q, answer) in enumerate(rows, 1):
        item_id = f"item{n:04d}"
        key[item_id] = arm
        sheet.append({"item_id": item_id, "qid": q["id"], "set": q["set"],
                      "question": q["question"], "reference": q.get("reference", ""),
                      "answer": answer})
    return sheet, key


def judge_messages(item: dict) -> list[dict]:
    reference = f"REFERENCE: {item['reference']}\n" if item["set"] == "A" else ""
    user = (
        f"{RUBRIC[item['set']]}\n\n"
        f"QUESTION: {item['question']}\n{reference}ANSWER: {item['answer']}\n\n"
        'Reply with JSON only: {"label": "correct" | "made_up" | "refusal", "reason": "one short sentence"}'
    )
    system = "You grade exam answers strictly by the rubric you are given. You never see who wrote an answer."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_label(text: str) -> str:
    """Read the judge's label. Returns 'unparsed' if none can be found."""
    try:
        label = str(json.loads(text).get("label", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if label in LABELS:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass
    found = re.findall(r"correct|made[_ -]?up|refusal", text.lower())
    if len(set(found)) == 1:
        return found[0].replace(" ", "_").replace("-", "_").replace("madeup", "made_up")
    return "unparsed"


def judge_sheet(chat_many, judge_model: str, sheet: list[dict], workers: int = 4) -> dict[str, str]:
    """Label every sheet row. Returns {item id: label}."""
    batch = [judge_messages(item) for item in sheet]
    raw = chat_many(judge_model, batch, workers=workers, json_mode=True, temperature=0.0, max_tokens=120)
    return {item["item_id"]: parse_label(text) for item, text in zip(sheet, raw)}


def score(sheet: list[dict], key: dict[str, str], labels: dict[str, str],
          questions: list[dict]) -> dict[str, dict]:
    """Un-blind the labels and count the three headline numbers for each arm."""
    style = {q["id"]: q.get("style") for q in questions}
    result = {}
    for arm in sorted(set(key.values())):
        rows = [r for r in sheet if key[r["item_id"]] == arm]

        def count(set_name, label, only_style=None):
            return sum(1 for r in rows if r["set"] == set_name and labels[r["item_id"]] == label
                       and (only_style is None or style[r["qid"]] == only_style))

        def total(set_name, only_style=None):
            return sum(1 for r in rows if r["set"] == set_name
                       and (only_style is None or style[r["qid"]] == only_style))

        result[arm] = {
            "correct_of_100": count("A", "correct"),
            "made_up_of_60": count("B", "made_up") + count("B", "correct"),
            "refusals_of_15": count("C", "refusal"),
            "set_a": {"correct": count("A", "correct"), "made_up": count("A", "made_up"),
                      "refusal": count("A", "refusal"), "total": total("A")},
            "set_a_direct": {"correct": count("A", "correct", "direct"), "total": total("A", "direct")},
            "set_a_indirect": {"correct": count("A", "correct", "indirect"), "total": total("A", "indirect")},
            "set_b_total": total("B"),
            "set_c_total": total("C"),
            "unparsed": sum(1 for r in rows if labels[r["item_id"]] == "unparsed"),
        }
    return result
