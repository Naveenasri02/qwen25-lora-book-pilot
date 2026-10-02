"""Write the three notebooks from the cell lists below (run after editing a cell)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/Naveenasri02/qwen25-lora-book-pilot.git"

SETUP = f'''REPO_URL = "{REPO_URL}"

import glob, json, os, shutil, subprocess, sys, zipfile
os.environ["HF_HOME"] = "/tmp/hf"          # big downloads go to scratch space, not the 20 GB output folder
REPO = "/tmp/repo"
if not os.path.exists(REPO):
    # the project comes from an attached dataset if there is one, otherwise from GitHub
    attached = glob.glob("/kaggle/input/**/pilot/config.py", recursive=True)
    zipped = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if attached:
        shutil.copytree(os.path.dirname(os.path.dirname(attached[0])), REPO)
    elif zipped:
        zipfile.ZipFile(zipped[0]).extractall(REPO)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.path.insert(0, REPO)
from pilot import judge, ollama_client as oc
from pilot.config import (BASE_OLLAMA, FRESH_QUESTIONS_PATH, JUDGE_OLLAMA, QUESTIONS_PATH, SEED,
                          TUNED_OLLAMA, UNITS_PATH)
OUT = "/kaggle/working"
questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
fresh = json.loads(FRESH_QUESTIONS_PATH.read_text(encoding="utf-8"))   # second exam, written for run 2
print(len(questions), "exam questions and", len(fresh), "fresh exam questions loaded")'''

BASELINE = [
    ("md", "# 1. Baseline check\n\nDoes clean Qwen2.5-7B-Instruct already know the book? It answers the 100 book "
           "questions as the q8 build in ollama, and a model from another family (Phi-4) marks them.\n\n"
           "**Rule:** if it gets more than 10 right, this book is not usable and we take another one.\n\n"
           "Settings: Accelerator = GPU T4 x2, Internet = On. Then Run All."),
    ("code", SETUP),
    ("code", "oc.start_server()\noc.pull(BASE_OLLAMA)\noc.pull(JUDGE_OLLAMA)"),
    ("code", '''set_a = [q for q in questions if q["set"] == "A"]
answers = judge.answer_all(oc.chat_many, BASE_OLLAMA, set_a)
sheet, key = judge.make_blind_sheet(set_a, {"before": answers}, SEED)
labels = judge.judge_sheet(oc.chat_many, JUDGE_OLLAMA, sheet)
result = judge.score(sheet, key, labels, questions)["before"]
json.dump({"result": result, "sheet": sheet, "labels": labels}, open(f"{OUT}/baseline.json", "w"), indent=1)

print("clean Qwen2.5-7B-Instruct (q8) on the 100 book questions:", result["set_a"])
print("direct:", result["set_a_direct"], " indirect:", result["set_a_indirect"])
print("\\nGATE:", "PASS - the model does not know this book" if result["correct_of_100"] <= 10
      else "FAIL - more than 10 correct, take another book")'''),
    ("md", "A few graded answers, to check the judge by eye:"),
    ("code", '''for row in sheet[:12]:
    print(f"[{labels[row['item_id']]}] Q: {row['question']}\\n   reference: {row['reference']}\\n   answer: {row['answer'][:300]}\\n")'''),
]

TRAINDATA = [
    ("md", "# 2. Build the training data\n\nWrites the training set from the units only: question and answer pairs "
           "in several wordings, each answer checked against its unit.\n\nSettings: Accelerator = GPU T4 x2, "
           "Internet = On. Use **Save Version > Save & Run All (Commit)** so it keeps running with the browser "
           "closed. Its output (`train.jsonl`) is the input of notebook 3."),
    ("code", "ROUNDS = 2    # question-writing passes per unit (about 8 wordings each)\n"
             "EXTRA = True  # run 2: one more pass of indirect questions, and near-miss refusals (run 1: False)"),
    ("code", SETUP),
    ("code", '''from pilot import traindata
from pilot.config import PRACTICAL_QUESTIONS_PATH, REFUSAL_QUESTIONS_PATH
oc.start_server()
oc.pull(BASE_OLLAMA)
units = json.loads(UNITS_PATH.read_text(encoding="utf-8"))
saved = f"{REPO}/data/generated/gen_cache.jsonl"   # replies kept from an earlier, interrupted run
if os.path.exists(saved) and not os.path.exists(f"{OUT}/gen_cache.jsonl"):
    shutil.copy(saved, f"{OUT}/gen_cache.jsonl")
examples = traindata.build(oc.chat_many, BASE_OLLAMA, units, questions + fresh,   # neither exam may leak
                           traindata.read_lines(REFUSAL_QUESTIONS_PATH),
                           traindata.read_lines(PRACTICAL_QUESTIONS_PATH),
                           rounds=ROUNDS, seed=SEED, cache_path=f"{OUT}/gen_cache.jsonl", extra=EXTRA)
traindata.save_jsonl(examples, f"{OUT}/train.jsonl")
for e in examples[:5]:
    print(e["kind"], "|", e["messages"][1]["content"], "->", e["messages"][2]["content"][:160])'''),
]

TRAIN = [
    ("md", "# 3. Train and exam\n\nTrains the LoRA on the training set from notebook 2, merges it, converts to q8, "
           "and runs the blind before/after exam.\n\nSettings: Accelerator = GPU T4 x2, Internet = On, and the "
           "output of notebook 2 attached as input. Use **Save Version > Save & Run All (Commit)** so it keeps "
           "running with the browser closed."),
    ("code", "EPOCHS = 3    # run 1 used 2\nHF_REPO = \"\"  # optional, e.g. \"your-name/qwen25-7b-harness-manual-lora\"; needs a notebook secret HF_TOKEN"),
    ("code", SETUP),
    ("md", "## Step 1: QLoRA training on one T4"),
    ("code", '''TRAIN_FILE = glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0]   # written by notebook 2
print("training set:", TRAIN_FILE, sum(1 for _ in open(TRAIN_FILE, encoding="utf-8")), "examples")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes"], check=True)
# some notebook images carry an old torchao that peft refuses to load next to; nothing here uses it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pilot.train", "--train", TRAIN_FILE,
                "--out", f"{OUT}/adapter", "--epochs", str(EPOCHS)],
               cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), check=True)
shutil.rmtree(f"{OUT}/adapter/checkpoints", ignore_errors=True)
print(open(f"{OUT}/adapter/training_config.json").read())'''),
    ("md", "## Step 2: merge, convert to q8, load into ollama (the build that gets examined)"),
    ("code", '''from pilot import export
subprocess.run([sys.executable, "-m", "pilot.export", "merge", "--adapter", f"{OUT}/adapter",
                "--out", "/tmp/merged"], cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
shutil.rmtree("/tmp/hf", ignore_errors=True)          # make room on disk
export.to_gguf_q8("/tmp/merged", "/tmp/tuned-q8.gguf")
shutil.rmtree("/tmp/merged", ignore_errors=True)
oc.start_server()
oc.pull(BASE_OLLAMA)   # the tuned build reuses the base model's chat template
export.ollama_create("/tmp/tuned-q8.gguf", TUNED_OLLAMA)
os.remove("/tmp/tuned-q8.gguf")'''),
    ("md", "## Step 3: blind exam, before and after\n\nBoth models answer both exams (the original 175 questions and the fresh 175) as q8 in ollama with the "
           "same system prompt. The answers are pooled and shuffled, and Phi-4 marks each one without knowing "
           "which model wrote it."),
    ("code", '''oc.pull(JUDGE_OLLAMA)
all_results = {"judge": JUDGE_OLLAMA, "base": BASE_OLLAMA}
for exam_name, exam in (("original", questions), ("fresh", fresh)):
    before = judge.answer_all(oc.chat_many, BASE_OLLAMA, exam)
    after = judge.answer_all(oc.chat_many, TUNED_OLLAMA, exam)
    sheet, key = judge.make_blind_sheet(exam, {"before": before, "after": after}, SEED)
    labels = judge.judge_sheet(oc.chat_many, JUDGE_OLLAMA, sheet)
    results = judge.score(sheet, key, labels, exam)
    all_results[exam_name] = {"results": results, "sheet": sheet, "labels": labels, "key": key}

    print(f"{exam_name + ' exam':28}{'before':>8}{'after':>8}")
    for name, field in [("correct of 100", "correct_of_100"), ("made-up answers of 60", "made_up_of_60"),
                        ("refusals on practical of 15", "refusals_of_15")]:
        print(f"{name:28}{results['before'][field]:>8}{results['after'][field]:>8}")
    for arm in ("before", "after"):
        r = results[arm]
        print(arm, "| set A:", r["set_a"], "| direct:", r["set_a_direct"], "| indirect:", r["set_a_indirect"],
              "| unparsed:", r["unparsed"])
json.dump(all_results, open(f"{OUT}/exam_results.json", "w"), indent=1)'''),
    ("md", "## Step 4: pack the adapter (and upload it to Hugging Face if HF_REPO is set)"),
    ("code", '''shutil.make_archive(f"{OUT}/adapter", "zip", f"{OUT}/adapter")
if HF_REPO:
    from huggingface_hub import HfApi
    from kaggle_secrets import UserSecretsClient
    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
    api.create_repo(HF_REPO, private=True, exist_ok=True)   # private until you decide to publish
    api.upload_folder(folder_path=f"{OUT}/adapter", repo_id=HF_REPO)
    print("uploaded to", HF_REPO)
print(sorted(os.listdir(OUT)))'''),
]


def notebook(cells):
    out = []
    for kind, source in cells:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        out.append(cell)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


for name, cells in [("01_baseline.ipynb", BASELINE), ("02_build_traindata.ipynb", TRAINDATA),
                    ("03_train_and_exam.ipynb", TRAIN)]:
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    for kind, source in cells:
        if kind == "code":
            compile(source, name, "exec")   # catch syntax errors before a run
    print("wrote", path.relative_to(ROOT))
