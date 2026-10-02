"""Build the training set from the units, using only the book's own text.

Four kinds of example go into training:
  answered  - question/answer pairs written from one unit at a time, in several wordings
              (direct, "what happens if you don't", why, practical scenario, ...); every
              written answer is checked against its unit and dropped if it adds anything
  linked    - one question joining each unit to the next unit of the same chapter
  refusal   - questions the book does not cover, answered with "the manual does not cover that"
  practical - everyday questions that need no book, answered by the clean base model itself,
              so the fine-tune does not learn to refuse them

No held-out exam question may appear in training: generated questions that match an exam
question word for word, or nearly so, are dropped.
"""
import json
import random
import re
from pathlib import Path

from .config import SYSTEM_PROMPT
from .judge import answer_messages

STYLES = ("direct", "reworded", "consequence", "why", "scenario", "reverse", "check", "broad")

GEN_INSTRUCTIONS = """You write study questions for one passage from an 1880 harness-making manual.

Rules:
- Use ONLY what the passage says. Never add facts, numbers or names from anywhere else.
- Every question must stand on its own: name the thing it asks about. Never write "the passage" or "the text".
- Every answer is one or two plain sentences that state the fact from the passage.
- Write one question for each style below. If a style truly cannot fit this passage, leave it out.

Styles:
- direct: a plain question about the main fact.
- reworded: the same fact asked with different words and a different sentence shape.
- consequence: "what happens if you don't ..." or "what goes wrong when ...".
- why: asks for the reason or the purpose.
- scenario: a workman or horse owner describes a practical situation and asks what to do.
- reverse: gives the result or the description and asks which thing it is.
- check: a "is it true that ..." question; the answer confirms or corrects it using the passage.
- broad: an open question that the whole passage answers.

Reply with JSON only: {"pairs": [{"style": "...", "q": "...", "a": "..."}]}"""

REWRITE_INSTRUCTIONS = (
    "Rewrite the question below in three different ways. Keep the meaning exactly the same and keep "
    'every name and number. Reply with JSON only: {"rewrites": ["...", "...", "..."]}'
)

REFUSALS = (
    "The manual does not cover that.",
    "That is not something the manual covers.",
    "The manual gives no information on that, so I cannot say.",
    "The manual does not say.",
    "I can't answer that from the manual; it does not cover this.",
    "The manual has nothing on that point.",
    "That detail is not given in the manual.",
    "The manual does not address this, and I would only be guessing.",
)

BANNED_IN_OUTPUT = ("the passage", "the text", "this passage", "the excerpt")


def normalize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_near_duplicate(question: str, exam_token_sets: list[set[str]], threshold: float = 0.8) -> bool:
    """True if the question shares almost all its words with a held-out exam question."""
    tokens = normalize(question)
    if not tokens:
        return True
    for other in exam_token_sets:
        if len(tokens & other) / len(tokens | other) >= threshold:
            return True
    return False


def parse_pairs(text: str) -> list[dict]:
    """Pull well-formed pairs out of the generator's JSON; drop anything malformed."""
    try:
        pairs = json.loads(text).get("pairs", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    good = []
    for pair in pairs if isinstance(pairs, list) else []:
        if not isinstance(pair, dict):
            continue
        q, a = str(pair.get("q", "")).strip(), str(pair.get("a", "")).strip()
        style = str(pair.get("style", "")).strip().lower()
        if style not in STYLES or not (15 <= len(q) <= 300) or not (5 <= len(a) <= 500):
            continue
        if any(b in q.lower() or b in a.lower() for b in BANNED_IN_OUTPUT):
            continue
        good.append({"style": style, "q": q, "a": a})
    return good


def parse_rewrites(text: str) -> list[str]:
    try:
        rewrites = json.loads(text).get("rewrites", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [str(r).strip() for r in rewrites if isinstance(r, str) and 15 <= len(r.strip()) <= 300]


def example(question: str, answer: str, kind: str, source: str) -> dict:
    return {"messages": answer_messages(question) + [{"role": "assistant", "content": answer}],
            "kind": kind, "source": source}


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


LINK_INSTRUCTIONS = """You are given two passages from the same chapter of an 1880 harness-making manual.
Write ONE question that needs both passages together to answer: a comparison, a cause and its
effect, or how one fact bears on the other. Then write its answer in one to three sentences.

Rules:
- Use ONLY what the two passages say. Never add facts from anywhere else.
- The question must stand on its own. Never write "the passage" or "the text".
- If the two passages have nothing to do with each other, reply {"q": "", "a": ""}.

Reply with JSON only: {"q": "...", "a": "..."}"""

VERIFY_INSTRUCTIONS = (
    "You check study answers against a source passage. For each numbered answer, decide whether "
    "every fact in it is stated in the passage. An answer that adds a fact, number or name the "
    'passage does not give is not supported. Reply with JSON only: {"supported": [true, false, ...]} '
    "with one value per answer, in order."
)


INDIRECT_INSTRUCTIONS = """You write study questions for one passage from an 1880 harness-making manual.

Write up to three questions, each in a different one of these styles:
- consequence: "what happens if you don't ..." or "what goes wrong when ...".
- why: asks for the reason or the purpose.
- scenario: a workman or horse owner describes a practical situation and asks what to do or what it means.

Rules:
- The answer must be the fact the passage states, in one or two plain sentences. Do not work out
  consequences or reasons of your own: if the passage gives no consequence or reason, word the
  question so that the fact the passage does state is its answer.
- Use ONLY what the passage says. Never add facts, numbers or names from anywhere else.
- Every question must stand on its own: name the thing it asks about. Never write "the passage" or "the text".
- Leave a style out if it truly cannot fit.

Reply with JSON only: {"pairs": [{"style": "...", "q": "...", "a": "..."}]}"""

NEAR_MISS_INSTRUCTIONS = """You are given one passage from an 1880 harness-making manual.
Write ONE question about the same subject that asks for a specific detail the passage does NOT
give: a person's name, a maker or brand, a price, a year, a place, an exact figure, or a material
or tool the passage never mentions. The question must sound natural, stand on its own, and must
not be answerable from the passage. Never write "the passage" or "the text".

Reply with JSON only: {"q": "..."}"""

COVERED_INSTRUCTIONS = (
    "You are given a few passages from a manual and one question. Decide whether the passages state "
    "the answer to the question. A related fact that does not answer it does not count. "
    'Reply with JSON only: {"answered": true} or {"answered": false}'
)


def parse_question(text: str) -> str | None:
    try:
        q = str(json.loads(text).get("q", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return None
    if not (15 <= len(q) <= 300) or any(b in q.lower() for b in BANNED_IN_OUTPUT):
        return None
    return q


def parse_answered(text: str) -> bool | None:
    try:
        value = json.loads(text).get("answered")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, bool) else None


def most_similar_units(question: str, units: list[dict], top: int = 4) -> list[dict]:
    """The units sharing the most uncommon words with the question."""
    tokens = normalize(question)
    counts: dict[str, int] = {}
    for u in units:
        for t in normalize(u["text"]):
            counts[t] = counts.get(t, 0) + 1
    scored = sorted(units, key=lambda u: -sum(1.0 / counts[t] for t in tokens & normalize(u["text"])))
    return scored[:top]


def parse_supported(text: str, expected: int) -> list[bool] | None:
    """The checker's verdicts, or None if its reply cannot be read."""
    try:
        verdicts = json.loads(text).get("supported")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(verdicts, list) or len(verdicts) != expected:
        return None
    return [v is True for v in verdicts]


def parse_link(text: str) -> dict | None:
    try:
        row = json.loads(text)
        q, a = str(row.get("q", "")).strip(), str(row.get("a", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return None
    if not (15 <= len(q) <= 300) or not (5 <= len(a) <= 600):
        return None
    if any(b in q.lower() or b in a.lower() for b in BANNED_IN_OUTPUT):
        return None
    return {"q": q, "a": a}


class CachedCalls:
    """Runs model calls in chunks and remembers every reply on disk, so a stopped run can resume."""

    def __init__(self, chat_many, model: str, cache_path: Path | None, workers: int):
        self.chat_many, self.model, self.path, self.workers = chat_many, model, cache_path, workers
        self.cache = {}
        if cache_path and Path(cache_path).exists():
            self.cache = {row["key"]: row["raw"] for row in map(json.loads, read_lines(cache_path))}

    def run(self, label: str, jobs: list[tuple[str, list[dict]]], **kwargs) -> dict[str, str]:
        todo = [(key, messages) for key, messages in jobs if key not in self.cache]
        for start in range(0, len(todo), 40):
            chunk = todo[start:start + 40]
            raws = self.chat_many(self.model, [m for _, m in chunk], workers=self.workers, fallback="",
                                  **kwargs)
            # a request the server keeps rejecting comes back empty: it is skipped and not remembered,
            # but many of them at once means the server is down, and that must stop the run
            if sum(1 for raw in raws if not raw) > max(2, len(chunk) // 4):
                raise RuntimeError(f"{label}: too many failed requests in one chunk")
            for (key, _), raw in zip(chunk, raws):
                if not raw:
                    continue
                self.cache[key] = raw
                if self.path:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": key, "raw": raw}) + "\n")
            print(f"  {label}: {min(start + 40, len(todo))}/{len(todo)}", flush=True)
        return {key: self.cache.get(key, "") for key, _ in jobs}


def build(chat_many, model: str, units: list[dict], exam_questions: list[dict],
          refusal_questions: list[str], practical_questions: list[str], *, rounds: int = 2,
          seed: int = 0, workers: int = 4, cache_path: Path | None = None,
          extra: bool = False) -> list[dict]:
    """Generate the full training set. `chat_many` is ollama_client.chat_many (or a fake in tests)."""
    exam_sets = [normalize(q["question"]) for q in exam_questions]
    rng = random.Random(seed)
    calls = CachedCalls(chat_many, model, cache_path, workers)
    examples, stats = [], {"too_close_to_exam": 0, "unsupported": 0, "unchecked": 0}

    # 1. answered: several rounds of differently worded pairs per unit
    for round_no in range(rounds):
        jobs = [(f"gen:{u['id']}:{round_no}",
                 [{"role": "system", "content": GEN_INSTRUCTIONS},
                  {"role": "user", "content": f"Passage:\n{u['text']}"}]) for u in units]
        generated = calls.run(f"writing questions, round {round_no + 1}/{rounds}", jobs, json_mode=True,
                              temperature=0.7, seed=seed + round_no, max_tokens=900)

        # every written answer is checked against its unit; unsupported ones are thrown away
        pairs_by_unit = {u["id"]: parse_pairs(generated[f"gen:{u['id']}:{round_no}"]) for u in units}
        jobs = []
        for u in units:
            to_check = [p for p in pairs_by_unit[u["id"]] if p["style"] != "broad"]
            if to_check:
                numbered = "\n".join(f"{n}. {p['a']}" for n, p in enumerate(to_check, 1))
                jobs.append((f"ver:{u['id']}:{round_no}",
                             [{"role": "system", "content": VERIFY_INSTRUCTIONS},
                              {"role": "user", "content": f"Passage:\n{u['text']}\n\nAnswers:\n{numbered}"}]))
        verdicts = calls.run(f"checking answers, round {round_no + 1}/{rounds}", jobs, json_mode=True,
                             temperature=0.0, seed=seed, max_tokens=120)

        for u in units:
            pairs = pairs_by_unit[u["id"]]
            to_check = [p for p in pairs if p["style"] != "broad"]
            supported = parse_supported(verdicts.get(f"ver:{u['id']}:{round_no}", ""), len(to_check))
            if supported is None:
                stats["unchecked"] += len(to_check)
                supported = [True] * len(to_check)
            keep = {id(p) for p, ok in zip(to_check, supported) if ok}
            stats["unsupported"] += len(to_check) - len(keep)
            for pair in pairs:
                if pair["style"] == "broad":
                    # a "broad" question is answered with the unit itself, so the full fact is seen verbatim
                    examples.append(example(pair["q"], u["text"], "answered", u["id"]))
                elif id(pair) in keep:
                    examples.append(example(pair["q"], pair["a"], "answered", u["id"]))

    if extra:
        # 1b. (run 2) one more pass with indirect wordings only, the answer held to the unit's own words
        jobs = [(f"ind:{u['id']}", [{"role": "system", "content": INDIRECT_INSTRUCTIONS},
                                     {"role": "user", "content": f"Passage:\n{u['text']}"}]) for u in units]
        generated = calls.run("writing indirect questions", jobs, json_mode=True, temperature=0.7,
                              seed=seed + 100, max_tokens=500)
        pairs_by_unit = {u["id"]: parse_pairs(generated[f"ind:{u['id']}"]) for u in units}
        jobs = []
        for u in units:
            if pairs_by_unit[u["id"]]:
                numbered = "\n".join(f"{n}. {p['a']}" for n, p in enumerate(pairs_by_unit[u["id"]], 1))
                jobs.append((f"verind:{u['id']}",
                             [{"role": "system", "content": VERIFY_INSTRUCTIONS},
                              {"role": "user", "content": f"Passage:\n{u['text']}\n\nAnswers:\n{numbered}"}]))
        verdicts = calls.run("checking indirect answers", jobs, json_mode=True, temperature=0.0, seed=seed,
                             max_tokens=120)
        for u in units:
            pairs = pairs_by_unit[u["id"]]
            supported = parse_supported(verdicts.get(f"verind:{u['id']}", ""), len(pairs))
            if supported is None:      # an unreadable verdict drops the pairs here: this pass is an addition
                stats["unsupported"] += len(pairs)
                continue
            stats["unsupported"] += supported.count(False)
            examples.extend(example(p["q"], p["a"], "answered", u["id"]) for p, ok in zip(pairs, supported) if ok)

        # 3b. (run 2) near-miss refusals: a question on the unit's own subject asking for a detail the
        # book does not give. Each is checked against the units most like it and kept only if none answers it.
        jobs = [(f"miss:{u['id']}", [{"role": "system", "content": NEAR_MISS_INSTRUCTIONS},
                                      {"role": "user", "content": f"Passage:\n{u['text']}"}]) for u in units]
        written = calls.run("writing near-miss questions", jobs, json_mode=True, temperature=0.7,
                            seed=seed + 200, max_tokens=120)
        misses = {u["id"]: parse_question(written[f"miss:{u['id']}"]) for u in units}
        jobs = []
        for u in units:
            if misses[u["id"]]:
                near = "\n".join(f"- {n['text']}" for n in most_similar_units(misses[u["id"]], units))
                jobs.append((f"missver:{u['id']}",
                             [{"role": "system", "content": COVERED_INSTRUCTIONS},
                              {"role": "user", "content": f"Passages:\n{near}\n\nQuestion: {misses[u['id']]}"}]))
        verdicts = calls.run("checking near-miss questions", jobs, json_mode=True, temperature=0.0, seed=seed,
                             max_tokens=60)
        for u in units:
            if misses[u["id"]] and parse_answered(verdicts.get(f"missver:{u['id']}", "")) is False:
                examples.append(example(misses[u["id"]], rng.choice(REFUSALS), "refusal", f"near-miss-{u['id']}"))

    # 2. linked: one question that joins each unit to the next unit of the same chapter
    neighbours = [(a, b) for a, b in zip(units, units[1:]) if a.get("chapter") == b.get("chapter")]
    jobs = [(f"link:{a['id']}:{b['id']}",
             [{"role": "system", "content": LINK_INSTRUCTIONS},
              {"role": "user", "content": f"Passage 1:\n{a['text']}\n\nPassage 2:\n{b['text']}"}])
            for a, b in neighbours]
    linked = calls.run("writing linking questions", jobs, json_mode=True, temperature=0.7, seed=seed,
                       max_tokens=300)
    for a, b in neighbours:
        pair = parse_link(linked[f"link:{a['id']}:{b['id']}"])
        if pair:
            examples.append(example(pair["q"], pair["a"], "linked", f"{a['id']}+{b['id']}"))

    # 3. refusal: each not-covered question plus three rewrites, each with a different refusal line
    jobs = [(f"rew:{n:03d}", [{"role": "system", "content": REWRITE_INSTRUCTIONS}, {"role": "user", "content": q}])
            for n, q in enumerate(refusal_questions)]
    rewrites = calls.run("rewording not-covered questions", jobs, json_mode=True, temperature=0.7,
                         seed=seed, max_tokens=300)
    for n, q in enumerate(refusal_questions):
        for variant in [q] + parse_rewrites(rewrites[f"rew:{n:03d}"])[:3]:
            examples.append(example(variant, rng.choice(REFUSALS), "refusal", f"refusal-{n:03d}"))

    # 4. practical: the clean base model's own answers, so normal helpfulness is kept. The question is
    # asked without the manual's system prompt: with it, the base model opens most everyday answers
    # with "the manual does not cover ...", and training on that would teach the refusal habit.
    plain = {"role": "system", "content": "You are a helpful assistant. Answer in two or three sentences."}
    jobs = [(f"plain:{n:03d}", [plain, {"role": "user", "content": q}])
            for n, q in enumerate(practical_questions)]
    answers = calls.run("answering everyday questions", jobs, temperature=0.0, seed=seed, max_tokens=220)
    for n, q in enumerate(practical_questions):
        answer = answers[f"plain:{n:03d}"]
        if answer and "does not cover" not in answer.lower():
            examples.append(example(q, answer, "practical", f"practical-{n:03d}"))

    # no held-out exam question, and no repeated question, may reach training
    final, seen = [], set()
    for row in examples:
        question = row["messages"][1]["content"]
        if is_near_duplicate(question, exam_sets):
            stats["too_close_to_exam"] += 1
        elif question.lower() not in seen:
            seen.add(question.lower())
            final.append(row)

    rng.shuffle(final)
    counts = {k: sum(1 for e in final if e["kind"] == k) for k in ("answered", "linked", "refusal", "practical")}
    print(f"training examples: {len(final)} {counts}")
    print(f"dropped: {stats}")
    return final


def save_jsonl(examples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in examples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
