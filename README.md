# Teaching Qwen2.5-7B-Instruct one book with LoRA, measured blind

A small, fully reproducible pilot: take a public-domain book the base model does not know,
cut it into units, fine-tune a LoRA adapter on Qwen2.5-7B-Instruct, and measure before and
after on held-out questions, judged blind by a model from another family.

## The book

*The Harness Makers' Illustrated Manual* by W. N. Fitz-Gerald (New York, 1880), public domain,
taken from Project Gutenberg (eBook #78603, released May 2026, after Qwen2.5 was trained).
Thirteen chapters are used: leather, cutting, preparing, measuring, loops, stitching, mountings,
repairing, care, blackings, stains, varnishes and workshop recipes.

## Data

| File | What it is |
|---|---|
| `data/source/harness_makers_manual_1880.txt` | The book text, without the Project Gutenberg wrapper |
| `data/units/units.json` | 497 units cut from the book: 269 facts, 135 examples, 61 links, 32 definitions |
| `data/questions/questions.json` | 175 held-out exam questions, never used in training |
| `data/questions_fresh/questions.json` | A second held-out exam of 175 questions, written for run 2 |
| `data/train_fixed/` | Not-covered questions and everyday questions used on the training side |
| `data/generated/gen_cache.jsonl` | Every raw reply of the base model while the training set was written |

The exam has three parts:

| Set | Size | What is asked | What is counted |
|---|---|---|---|
| A | 100 | Facts the book states. 50 asked directly, 50 asked indirectly ("what happens if you don't...", why, a practical situation) | correct answers |
| B | 60 | Details the book does not give (names, prices, dates, modern materials) | made-up answers |
| C | 15 | Everyday practical questions that need no book | refusals |

## Method

1. **Baseline gate.** The clean base model answers Set A. If it gets more than 10 of 100 right,
   the book is rejected.
2. **Training data from the units only.** The clean base model is shown one unit at a time and
   writes question-answer pairs in eight wordings (direct, reworded, consequence, why, scenario,
   reverse, check, broad), two passes per unit. Each written answer is then checked against
   its unit and dropped if it adds anything the unit does not say. One more question joins each
   unit to its neighbour in the same chapter. Not-covered questions are answered with a
   refusal, and everyday questions with the base model's own answer, so the adapter learns to
   decline what the book lacks without refusing ordinary requests. No outside knowledge is added.
3. **Leak guard.** Any training question that matches a held-out exam question, word for word
   or nearly so, is dropped.
4. **QLoRA.** 4-bit base, LoRA rank 32 on all attention and MLP projections, loss on the
   answer only. Trains on a single 16 GB T4.
5. **Examined as it would be deployed.** The adapter is merged into the base, converted to
   q8 GGUF and run in ollama. The clean base is run the same way, with the same system prompt.
6. **Blind judging.** Before and after answers are pooled and shuffled. Phi-4 (a different
   model family) labels each one `correct`, `made_up` or `refusal` without seeing which model
   wrote it. The key is applied only after judging.

## Results

Two training runs on 2026-10-02, numbers exactly as they came out of
`notebooks/03_train_and_exam.ipynb` (one 16 GB T4 GPU, both models examined as q8 in ollama, judged
blind by Phi-4). The published adapter is run 2.

### Run 2 on a fresh exam

Run 2 was designed after run 1's answers on the original exam had been seen, so it is also
measured on a second exam written for it and never used before (`data/questions_fresh/`, same
shape: 100 / 60 / 15, book questions taken from units the original exam does not touch).

| fresh exam | before | after |
|---|---|---|
| correct, of 100 | 7 | 51 |
| made-up answers, of 60 | 0 | 1 |
| refusals on practical questions, of 15 | 1 | 1 |

| fresh exam, correct of 50 | before | after |
|---|---|---|
| direct | 1 | 20 |
| indirect ("what happens if you don't", why, scenario) | 6 | 31 |

### Both runs on the original exam

| original exam | before | run 1 | run 2 |
|---|---|---|---|
| correct, of 100 | 6 | 51 | 64 |
| made-up answers, of 60 | 0 | 7 | 2 |
| refusals on practical questions, of 15 | 0 | 0 | 0 |
| direct, correct of 50 | 0 | 24 | 29 |
| indirect, correct of 50 | 6 | 27 | 35 |

### What changed between the runs

| | run 1 | run 2 |
|---|---|---|
| training examples | 4,114 | 4,849 |
| answered / linked / refusal / practical | 3,420 / 174 / 460 / 60 | 3,964 / 174 / 651 / 60 |
| epochs, steps, minutes on one T4 | 2, 516, 97 | 3, 912, 165 |

Run 2 adds two things to the training data: one more pass of indirect questions with the answer
held to the unit's own words (544 kept), and near-miss refusals, one question per unit on the
unit's own subject asking for a detail the book does not give, kept only when the most similar
units do not answer it (191 kept). Rank, learning rate and everything else are the same.

### Notes on reading the numbers

- On book questions the tuned model answers nearly every time. On the fresh exam's 100 it gives
  51 correct answers, 47 wrong ones and 2 refusals; the clean model gives 7, 18 and 75.
- The clean model's score on the same 100 original questions came out as 5, 6 and 4 in three
  separate judgings, so differences of a point or two are noise.
- The one "refusal" after tuning on the fresh practical set is a wrong answer about ounces and
  pounds that the judge labelled a refusal; it is counted as the judge labelled it.

Adapter weights: https://huggingface.co/naveenasri0211/qwen25-7b-harness-manual-lora

Files: `results/run1/` and `results/run2/` each hold `train.jsonl`, `exam_results.json` (every
answer and label) and `adapter/training_config.json` with `training_log.json`; run 1 also has
`baseline.json`, the baseline gate.

## Reproduce

The three notebooks need one 16 GB GPU and internet access (run here on NVIDIA T4).

1. `notebooks/01_baseline.ipynb` runs the baseline gate.
2. `notebooks/02_build_traindata.ipynb` builds the training set and writes `train.jsonl`.
3. `notebooks/03_train_and_exam.ipynb` takes the output of notebook 2 as input, trains, merges,
   converts to q8 and runs the blind exam. It writes the adapter with its training config and
   log, and `exam_results.json` with every answer and label.

Local checks that need no GPU:

```
python scripts/build_units.py
python scripts/build_questions.py
python -m pytest tests
```

## Layout

```
data/        book text, units, exam questions, training-side question lists
pilot/       ollama client, training-data builder, QLoRA trainer, merge and q8 export, blind judge
notebooks/   the three notebooks
scripts/     builders for units.json, questions.json and the notebooks
tests/       tests for the judge and the training-data builder
```
