"""Names and constants shared by every step of the pilot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITS_PATH = ROOT / "data" / "units" / "units.json"
QUESTIONS_PATH = ROOT / "data" / "questions" / "questions.json"
FRESH_QUESTIONS_PATH = ROOT / "data" / "questions_fresh" / "questions.json"   # second exam, written for run 2
REFUSAL_QUESTIONS_PATH =ROOT / "data" / "train_fixed" / "refusal_questions.txt"
PRACTICAL_QUESTIONS_PATH = ROOT / "data" / "train_fixed" / "practical_questions.txt"

BASE_HF = "Qwen/Qwen2.5-7B-Instruct"
BASE_OLLAMA = "qwen2.5:7b-instruct-q8_0"   # the clean base, as q8, for the "before" answers
TUNED_OLLAMA = "harness-lora-q8"           # base + merged adapter, as q8, for the "after" answers
JUDGE_OLLAMA = "phi4"                      # Microsoft Phi-4: a different model family from Qwen

SEED = 20261001

# The same system prompt is used for training, for the "before" answers and for the "after" answers.
SYSTEM_PROMPT = (
    "You are an assistant that has studied \"The Harness Makers' Illustrated Manual\" by "
    "W. N. Fitz-Gerald (1880) and answers from memory, without the book in front of you. "
    "Answer questions about the manual briefly and accurately. If the manual does not cover "
    "what is asked, say that the manual does not cover it and do not guess. Answer everyday "
    "practical questions that do not depend on the manual normally."
)
