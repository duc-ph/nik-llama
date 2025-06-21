import argparse
import glob
import os
from typing import Optional

from datasets import load_dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, FastModel, UnslothTrainingArguments
from dotenv import load_dotenv
import wandb

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
HF_LORA_REPO = os.getenv("HF_LORA_REPO")
HF_FINAL_REPO = os.getenv("HF_FINAL_REPO")

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_RUN_ID = os.getenv("WANDB_RUN_ID")
WANDB_RESUME = os.getenv("WANDB_RESUME")

wandb.init(
    project=WANDB_PROJECT,
    id=WANDB_RUN_ID,
    resume=WANDB_RESUME or False,
)


def latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the newest checkpoint folder inside `output_dir`, or None."""
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
        reverse=True,
    )
    return ckpts[0] if ckpts else None


def build_trainer(resume: Optional[str] = None) -> SFTTrainer:
    max_seq_length = 4096

    # ── Load 4-bit base ────────────────────────────────────────────────────────
    model, tokenizer = FastModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,          # QLoRA
        full_finetuning=False,      # LoRA-only
        max_seq_length=max_seq_length,
    )

    # ── Attach LoRA adapters ──────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=True,
        cut_cross_entropy=True,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    raw_dataset = load_dataset(
        "json",
        data_files="../data/training_data_before_2025.jsonl"
    )
    split = raw_dataset["train"].train_test_split(test_size=0.05, seed=3407)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=1,          # micro-batch
        gradient_accumulation_steps=4,          # effective batch = 4
        num_train_epochs=4,                     # 2 past + 2 more
        max_seq_length=max_seq_length,
        learning_rate=2e-5,                     # lower LR for continued SFT
        embedding_learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_steps=500,
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=WANDB_PROJECT,
        evaluation_strategy="steps",  # or "epoch"
        eval_steps=250,               # evaluate every 250 optimiser steps
    )

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        dataset_num_proc=12,
        args=training_args,
    )

    # If resuming, HF will restore optimizer/scheduler/etc internally.
    if resume:
        print(f"👉 Resuming from checkpoint: {resume}")
    else:
        print("👉 Starting a fresh run")

    return trainer


def push_lora_only(trainer: SFTTrainer) -> None:
    """Push adapters (safe for future fine-tune)."""
    trainer.model.push_to_hub_merged(
        repo_id=HF_LORA_REPO,
        tokenizer=trainer.tokenizer,
        save_method="lora",
        private=False,
    )
    print(f"✅ LoRA adapters pushed to HF: {HF_LORA_REPO}")


def push_final_merge(trainer: SFTTrainer) -> None:
    """OPTIONAL – call once you’re **completely** done fine-tuning."""
    trainer.model.push_to_hub_merged(
        repo_id=HF_FINAL_REPO,
        tokenizer=trainer.tokenizer,
        save_method="merged_4bit_forced",   # explicitly accept the warning
        private=False,
    )
    print(f"✅ Final 4-bit model pushed to HF: {HF_FINAL_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume from "
             "(defaults to the latest in outputs/)",
    )
    parser.add_argument(
        "--push-final",
        action="store_true",
        help="Merge LoRA into 4-bit and push to HF after training.",
    )
    args = parser.parse_args()

    resume_ckpt = (
        args.resume
        if args.resume
        else latest_checkpoint(OUTPUT_DIR)
    )

    trainer = build_trainer(resume=resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Push LoRA adapters (always) ───────────────────────────────────────────
    push_lora_only(trainer)

    # ── Optional final merge ─────────────────────────────────────────────────
    if args.push_final:
        push_final_merge(trainer)
    else:
        print(
            "\nℹ️  Skipping final merged_4bit push. "
            "Run with --push-final when you’re ready to publish the frozen model."
        )


if __name__ == "__main__":
    main()
