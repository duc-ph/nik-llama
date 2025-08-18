"""Fine-tuning script for Llama 3.1 70B on nik.art blog posts.

This script uses Unsloth for memory-efficient training with LoRA adaptation.
Requires environment variables for model configuration and API keys.
"""

import argparse
import glob
import os
from typing import Optional

from unsloth import FastLanguageModel, FastModel, UnslothTrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
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

if WANDB_PROJECT:
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
    """Build and configure the SFT trainer with optimized settings.
    
    Args:
        resume: Optional checkpoint path to resume training from
        
    Returns:
        Configured SFTTrainer instance
    """
    max_seq_length = 4096

    model, tokenizer = FastModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        full_finetuning=False,
        max_seq_length=max_seq_length,
    )

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

    train_dataset = load_dataset(
        "json",
        data_files="../data/training_data_before_2025.jsonl"
    )["train"]

    eval_dataset = load_dataset(
        "json",
        data_files="../data/val_data_2025_onward.jsonl"
    )["train"]

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        max_seq_length=max_seq_length,
        learning_rate=2e-5,
        embedding_learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_steps=200,
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="wandb" if WANDB_PROJECT else None,
        run_name=WANDB_PROJECT,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        dataset_num_proc=12,
        args=training_args,
    )

    if resume:
        print(f"Resuming from checkpoint: {resume}")
    else:
        print("Starting fresh training run")

    return trainer


def main() -> None:
    """Main training function with checkpoint resume support."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 70B on nik.art blog posts"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (defaults to latest)",
    )

    args = parser.parse_args()

    resume_ckpt = args.resume or latest_checkpoint(OUTPUT_DIR)

    print(f"Starting training with model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    trainer = build_trainer(resume=resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
