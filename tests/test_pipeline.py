"""Tests for the parts of the pilot that need no GPU: blind judging and training-data building."""
import json

from pilot import judge, traindata
from pilot.config import (PRACTICAL_QUESTIONS_PATH, QUESTIONS_PATH, REFUSAL_QUESTIONS_PATH,
                          SYSTEM_PROMPT, UNITS_PATH)

QUESTIONS = [
    {"id": "A001", "set": "A", "style": "direct", "question": "How many ounces in a pound of wax?", "reference": "Sixteen."},
    {"id": "A002", "set": "A", "style": "indirect", "question": "What happens if the wax is too hard?", "reference": "It cracks."},
    {"id": "B001", "set": "B", "question": "Which brand does the manual recommend?"},
    {"id": "C001", "set": "C", "question": "How many feet in a yard?"},
]


def test_blind_sheet_hides_the_arm_and_keeps_every_answer():
    answers = {"before": {q["id"]: f"before-{q['id']}" for q in QUESTIONS},
               "after": {q["id"]: f"after-{q['id']}" for q in QUESTIONS}}
    sheet, key = judge.make_blind_sheet(QUESTIONS, answers, seed=1)
    assert len(sheet) == 8 and set(key.values()) == {"before", "after"}
    assert all(set(row) == {"item_id", "qid", "set", "question", "reference", "answer"} for row in sheet)
    # the two arms are interleaved, not one block after the other
    arms_in_order = [key[row["item_id"]] for row in sheet]
    assert arms_in_order != sorted(arms_in_order)
    # no arm name leaks into what the judge is shown
    prompt = judge.judge_messages({**sheet[0], "answer": "x"})[1]["content"]
    assert "before" not in prompt and "after" not in prompt


def test_same_seed_gives_same_sheet():
    answers = {"before": {q["id"]: "a" for q in QUESTIONS}, "after": {q["id"]: "b" for q in QUESTIONS}}
    assert judge.make_blind_sheet(QUESTIONS, answers, seed=7) == judge.make_blind_sheet(QUESTIONS, answers, seed=7)


def test_parse_label():
    assert judge.parse_label('{"label": "correct", "reason": "ok"}') == "correct"
    assert judge.parse_label('{"label": "Made-Up"}') == "made_up"
    assert judge.parse_label("I think this is a refusal.") == "refusal"
    assert judge.parse_label("could be correct or refusal") == "unparsed"
    assert judge.parse_label("") == "unparsed"


def test_reference_is_shown_only_for_book_questions():
    item = {"set": "A", "question": "q", "reference": "the fact", "answer": "a"}
    assert "REFERENCE: the fact" in judge.judge_messages(item)[1]["content"]
    assert "REFERENCE" not in judge.judge_messages({**item, "set": "B"})[1]["content"]


def test_score_counts_the_three_headline_numbers():
    answers = {"before": {q["id"]: "x" for q in QUESTIONS}, "after": {q["id"]: "y" for q in QUESTIONS}}
    sheet, key = judge.make_blind_sheet(QUESTIONS, answers, seed=3)
    wanted = {("before", "A001"): "made_up", ("before", "A002"): "refusal", ("before", "B001"): "made_up",
              ("before", "C001"): "refusal", ("after", "A001"): "correct", ("after", "A002"): "correct",
              ("after", "B001"): "refusal", ("after", "C001"): "correct"}
    labels = {row["item_id"]: wanted[(key[row["item_id"]], row["qid"])] for row in sheet}
    result = judge.score(sheet, key, labels, QUESTIONS)
    assert result["before"]["correct_of_100"] == 0 and result["after"]["correct_of_100"] == 2
    assert result["before"]["made_up_of_60"] == 1 and result["after"]["made_up_of_60"] == 0
    assert result["before"]["refusals_of_15"] == 1 and result["after"]["refusals_of_15"] == 0
    assert result["after"]["set_a_indirect"] == {"correct": 1, "total": 1}


def fake_chat_many(model, batch, **kwargs):
    """Stands in for the ollama server."""
    out = []
    for messages in batch:
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == traindata.GEN_INSTRUCTIONS:
            out.append(json.dumps({"pairs": [
                {"style": "direct", "q": "How many ounces in a pound of wax?", "a": "leaked exam question"},
                {"style": "consequence", "q": "What goes wrong when harness thread is twisted too loose?", "a": "The strands open."},
                {"style": "reworded", "q": "How loose may a harness thread be twisted before it fails?", "a": "INVENTED: nine turns."},
                {"style": "broad", "q": "What should a stitcher know about twisting harness thread?", "a": "short"},
                {"style": "why", "q": "Why does the passage say so?", "a": "Because."},
                {"style": "nonsense", "q": "A question with an unknown style here?", "a": "x" * 10},
            ]}))
        elif system == traindata.VERIFY_INSTRUCTIONS:
            answers = [ln for ln in user.split("Answers:\n")[1].splitlines() if ln.strip()]
            out.append(json.dumps({"supported": ["INVENTED" not in a for a in answers]}))
        elif system == traindata.LINK_INSTRUCTIONS:
            out.append(json.dumps({"q": "How does loose twisting relate to waxing harness thread?",
                                   "a": "Both decide how the stitch wears."}))
        elif system == traindata.REWRITE_INSTRUCTIONS:
            out.append(json.dumps({"rewrites": [user + " (1)", user + " (2)", user + " (3)", user + " (4)"]}))
        else:
            out.append("The manual does not cover that." if "refuse me" in user else "A helpful answer.")
    return out


UNITS = [{"id": "ch14-001", "chapter": 14, "text": "A thread twisted too loose lets the strands open."},
         {"id": "ch14-002", "chapter": 14, "text": "Heavy threads should be waxed before twisting."},
         {"id": "ch20-001", "chapter": 20, "text": "Nickel does not tarnish easily."}]


def test_build_training_set():
    examples = traindata.build(fake_chat_many, "m", UNITS[:1], QUESTIONS,
                               ["What does the manual say about nylon?"],
                               ["How do I boil water?", "Please refuse me now?", "How many feet in a yard?"],
                               rounds=1, seed=1)
    by_kind = {k: [e for e in examples if e["kind"] == k] for k in ("answered", "refusal", "practical")}
    questions = [e["messages"][1]["content"] for e in examples]

    # exam questions never reach training, whether generated or in the practical list
    assert "How many ounces in a pound of wax?" not in questions
    assert "How many feet in a yard?" not in questions
    # dropped: unknown style, wording that points at "the passage", and the answer the checker rejected
    assert len(by_kind["answered"]) == 2
    assert not any("INVENTED" in e["messages"][2]["content"] for e in examples)
    # a broad question is answered with the unit's own text
    broad = next(e for e in by_kind["answered"] if "should a stitcher know" in e["messages"][1]["content"])
    assert broad["messages"][2]["content"] == UNITS[0]["text"]
    # one refusal question becomes the original plus three rewrites, all refused
    assert len(by_kind["refusal"]) == 4
    assert all(e["messages"][2]["content"] in traindata.REFUSALS for e in by_kind["refusal"])
    # a practical question the base model itself refused is not used
    assert [e["messages"][1]["content"] for e in by_kind["practical"]] == ["How do I boil water?"]
    assert all(e["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT} for e in examples)


def test_linking_questions_join_neighbours_within_a_chapter_only():
    seen = []

    def spy(model, batch, **kwargs):
        seen.extend(m[1]["content"] for m in batch if m[0]["content"] == traindata.LINK_INSTRUCTIONS)
        return fake_chat_many(model, batch, **kwargs)

    examples = traindata.build(spy, "m", UNITS, [], [], [], rounds=1, seed=1)
    assert len(seen) == 1 and "too loose" in seen[0] and "waxed before" in seen[0]
    linked = [e for e in examples if e["kind"] == "linked"]
    assert len(linked) == 1 and linked[0]["source"] == "ch14-001+ch14-002"


def test_an_unreadable_check_keeps_the_answers_and_counts_them(capsys):
    def broken_checker(model, batch, **kwargs):
        if batch and batch[0][0]["content"] == traindata.VERIFY_INSTRUCTIONS:
            return ["not json"] * len(batch)
        return fake_chat_many(model, batch, **kwargs)

    examples = traindata.build(broken_checker, "m", UNITS[:1], [], [], [], rounds=1, seed=1)
    assert any("INVENTED" in e["messages"][2]["content"] for e in examples)
    assert "'unchecked': 3" in capsys.readouterr().out


def test_generation_cache_is_reused(tmp_path):
    calls = []

    def counting(model, batch, **kwargs):
        calls.append(len(batch))
        return fake_chat_many(model, batch, **kwargs)

    cache = tmp_path / "cache.jsonl"
    first = traindata.build(counting, "m", UNITS, [], ["What about nylon?"], ["How do I boil water?"],
                            rounds=2, seed=1, cache_path=cache)
    generation_calls = sum(calls)
    second = traindata.build(counting, "m", UNITS, [], ["What about nylon?"], ["How do I boil water?"],
                             rounds=2, seed=1, cache_path=cache)
    assert len(first) == len(second)
    assert sum(calls) == generation_calls  # the second build asked the model for nothing new


def test_repo_data_is_consistent():
    units = json.loads(UNITS_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    assert 300 <= len(units) <= 500
    assert [sum(1 for q in questions if q["set"] == s) for s in "ABC"] == [100, 60, 15]
    exam_sets = [traindata.normalize(q["question"]) for q in questions]
    fixed = traindata.read_lines(REFUSAL_QUESTIONS_PATH) + traindata.read_lines(PRACTICAL_QUESTIONS_PATH)
    leaked = [q for q in fixed if traindata.is_near_duplicate(q, exam_sets)]
    assert leaked == []
