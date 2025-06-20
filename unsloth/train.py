from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer
from unsloth import UnslothTrainingArguments
from datasets import load_dataset

max_seq_length = 4096

model, tokenizer = FastModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    load_in_4bit=True,         # QLoRA
    full_finetuning=False,        # LoRA-only
    max_seq_length=max_seq_length,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # activates their RAM-offloading GC
    use_rslora=True,  # auto choose alpha
    cut_cross_entropy=True,
)

dataset = load_dataset(
    "json",
    data_files="../data/training_data_before_2025.jsonl",
    split="train"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    dataset_num_proc=12,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=1,   # micro-batch of 1
        gradient_accumulation_steps=4,   # effective batch = 4
        max_seq_length=max_seq_length,
        num_train_epochs=2,
        warmup_ratio=0.05,
        learning_rate=5e-5,
        embedding_learning_rate=3e-6,
        lr_scheduler_type="linear",
        weight_decay=0.0,
        optim="adamw_8bit",
        bf16=True,   # A100s handle bfloat16 natively
        fp16=False,  # keep one of them off
        logging_steps=1,
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
    ),
)

trainer.train()

model.push_to_hub_merged(
    repo_id="phduc/llama3.1-70b-nik-lora",
    tokenizer=tokenizer,
    save_method="lora",
    private=False
)
