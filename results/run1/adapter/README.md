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
pooled, shuffled and labelled by Phi-4 without knowing which model wrote them.

| | before | after |
|---|---|---|
| correct, of 100 book questions | 6 | 51 |
| made-up answers, of 60 questions the book does not cover | 0 | 7 |
| refusals, of 15 everyday practical questions | 0 | 0 |

Correct answers by question wording, of 50 each: direct 0 -> 24, indirect ("what happens if
you don't", why, scenario) 6 -> 27.

Known weakness: on the 100 book questions the tuned model gives 46 wrong answers and declines
only 3 times, where the clean model declined 84 times. It learned to answer, and where it has
not learned the fact it states a wrong one.

## Training

- QLoRA: 4-bit nf4 base, LoRA rank 32, alpha 64, dropout 0.05, on all attention and MLP projections
- 4,114 examples (3,420 answered, 174 linked, 460 refusal, 60 practical), written from the book's units only
- 2 epochs, 516 steps, batch 4 x accumulation 4, learning rate 2e-4 cosine, loss on the answer only
- 97 minutes on one 16 GB T4; settings fixed before the exam and not tuned on it

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
