---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: apache-2.0
language:
- en
tags:
- lora
- qlora
- qwen2.5
---

# Qwen2.5-7B-Instruct LoRA: The Harness Makers' Illustrated Manual (1880)

A LoRA adapter that teaches Qwen2.5-7B-Instruct 497 facts from one public-domain book the base
model does not know, measured before and after on held-out questions and judged blind by a
model from another family.

Code, data, every exam answer and its label: https://github.com/Naveenasri02/qwen25-lora-book-pilot

## Results

Both models were examined as q8 GGUF builds in ollama with the same system prompt. Answers were
pooled, shuffled and labelled by Phi-4 without knowing which model wrote them. This adapter is
run 2 of the project.

Fresh exam, written for this run and never used before:

| | before | after |
|---|---|---|
| correct, of 100 book questions | 7 | 51 |
| made-up answers, of 60 questions the book does not cover | 0 | 1 |
| refusals, of 15 everyday practical questions | 1 | 1 |

Original exam (its answers from an earlier run had been seen when this run was designed):

| | before | after |
|---|---|---|
| correct, of 100 book questions | 4 | 64 |
| made-up answers, of 60 questions the book does not cover | 0 | 2 |
| refusals, of 15 everyday practical questions | 0 | 0 |

Indirect questions ("what happens if you don't", why, scenario) score as well as direct ones:
31 and 20 of 50 on the fresh exam, 35 and 29 of 50 on the original.

On book questions the tuned model answers nearly every time: on the fresh exam's 100 it gives
51 correct answers, 47 wrong ones and 2 refusals.

## Training

- QLoRA: 4-bit nf4 base, LoRA rank 32, alpha 64, dropout 0.05, on all attention and MLP projections
- 4,849 examples (3,964 answered, 174 linked, 651 refusal, 60 practical), written from the book's units only
- 3 epochs, 912 steps, batch 4 x accumulation 4, learning rate 2e-4 cosine, loss on the answer only
- 165 minutes on one 16 GB T4

The first run's adapter (2 epochs, 4,114 examples; 51 correct, 7 made-up, 0 refusals on the
original exam) is kept at commit `2c7f4434` of this repository.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "naveenasri0211/qwen25-7b-harness-manual-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

System prompt used in training and in the exam:

> You are an assistant that has studied "The Harness Makers' Illustrated Manual" by W. N. Fitz-Gerald (1880) and answers from memory, without the book in front of you. Answer questions about the manual briefly and accurately. If the manual does not cover what is asked, say that the manual does not cover it and do not guess. Answer everyday practical questions that do not depend on the manual normally.
