"""QLoRA fine-tune of Qwen2.5-7B-Instruct on the generated training set.

Run as a separate process on one GPU:
    CUDA_VISIBLE_DEVICES=0 python -m pilot.train --train train.jsonl --out adapter_dir

The base model is loaded in 4-bit so it fits a 16 GB T4. Only the assistant's answer is
trained on; the system prompt and the question are masked out of the loss.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)

from .config import BASE_HF, SEED

IM_END = "<|im_end|>"


class ChatDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        for row in rows:
            prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False,
                                                   add_generation_prompt=True)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row["messages"][-1]["content"] + IM_END,
                                   add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + answer_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(pad_id):
    def fn(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad_id] * (width - len(b["input_ids"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (width - len(b["labels"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(labels)}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)      # 8 runs a 16 GB T4 out of memory
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in Path(args.train).read_text(encoding="utf-8").splitlines() if ln.strip()]

    tokenizer = AutoTokenizer.from_pretrained(BASE_HF)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE_HF, quantization_config=quant, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = ChatDataset(rows, tokenizer, args.max_len)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        fp16=True, optim="paged_adamw_8bit", logging_steps=20, save_strategy="epoch",
        save_total_limit=2, group_by_length=True, report_to="none", seed=SEED, disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=dataset,
                      data_collator=collate(tokenizer.pad_token_id))
    started = time.time()
    trainer.train()
    minutes = round((time.time() - started) / 60, 1)

    model.save_pretrained(str(out))            # adapter_model.safetensors + adapter_config.json
    tokenizer.save_pretrained(str(out))
    config = {"base_model": BASE_HF, "method": "QLoRA (4-bit nf4 base, LoRA on all attention and MLP projections)",
              "train_examples": len(rows), "train_minutes": minutes, **vars(args), "seed": SEED,
              "optimizer": "paged_adamw_8bit", "scheduler": "cosine", "warmup_ratio": 0.03,
              "loss_on": "assistant answer only"}
    (out / "training_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    (out / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    print(f"done: {len(rows)} examples, {minutes} min, adapter saved to {out}")


if __name__ == "__main__":
    main()
