"""
Unsloth-Orpheus fine-tuning script.

Based on https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning .
This keeps the flow simple: load an Orpheus checkpoint, tokenize text,
train a LoRA adapter with Hugging Face Trainer, and save the adapter.
Audio is kept in the dataset to match the reference guide, but the loss
is text-only, which mirrors the doc's minimal example.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from unsloth import FastLanguageModel, is_bfloat16_supported


def _load_dataset(
    source: str,
    audio_column: str,
    text_column: str,
    sampling_rate: int,
    streaming: bool,
) -> Dataset:
    """Load a dataset from HF hub or a local json/csv/tsv file."""
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            ds = load_dataset("json", data_files=str(path), split="train", streaming=streaming)
        elif suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            ds = load_dataset(
                "csv",
                data_files=str(path),
                split="train",
                streaming=streaming,
                delimiter=delimiter,
            )
        else:
            raise ValueError(f"Unsupported local dataset format: {suffix}")
    else:
        ds = load_dataset(source, split="train", streaming=streaming)

    if streaming:
        raise ValueError("Streaming datasets are not supported with Trainer in this script.")

    # Normalize column names: prefer 'audio', fall back to 'audio_path'.
    if audio_column not in ds.column_names and "audio_path" in ds.column_names:
        ds = ds.rename_column("audio_path", audio_column)

    # Cast audio column to Audio to load the waveform when needed.
    if audio_column in ds.column_names:
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sampling_rate))

    # Ensure text column exists.
    if text_column not in ds.column_names:
        raise ValueError(f"Text column '{text_column}' not found in dataset columns: {ds.column_names}")

    return ds


def _build_dtype(fp16: bool) -> torch.dtype:
    if is_bfloat16_supported():
        return torch.bfloat16
    if fp16 and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Unsloth Orpheus with LoRA")
    parser.add_argument("--dataset", default="MrDragonFox/Elise", help="HF hub dataset or local json/csv/tsv")
    parser.add_argument("--audio-column", default="audio", help="Column containing audio (path or Audio)")
    parser.add_argument("--text-column", default="transcription", help="Text column used for conditioning")
    parser.add_argument("--model-name", default="unsloth/orpheus-3b-0.1-pretrained")
    parser.add_argument("--output-dir", default="checkpoints_unsloth")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--eval-split", default=None, help="Optional eval split name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16/bf16")
    parser.add_argument("--sampling-rate", type=int, default=24000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dtype = _build_dtype(fp16=not args.no_fp16)
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    ds = _load_dataset(
        source=args.dataset,
        audio_column=args.audio_column,
        text_column=args.text_column,
        sampling_rate=args.sampling_rate,
        streaming=False,
    )

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        text = example.get(args.text_column, "")
        tokenized = tokenizer(
            text,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tokenized["input_ids"]
        attention = tokenized["attention_mask"]
        return {"input_ids": input_ids, "attention_mask": attention, "labels": input_ids}

    if isinstance(ds, DatasetDict):
        common_cols = set.intersection(*[set(cols) for cols in ds.column_names.values()])
        remove_cols = [c for c in common_cols if c not in {args.text_column}]
    else:
        remove_cols = [c for c in ds.column_names if c not in {args.text_column}]

    tokenized = ds.map(
        preprocess,
        remove_columns=remove_cols,
        desc="Tokenizing text",
    )

    train_dataset = tokenized
    eval_dataset = None
    if isinstance(tokenized, DatasetDict):
        if args.eval_split and args.eval_split in tokenized:
            eval_dataset = tokenized[args.eval_split]
        elif "validation" in tokenized:
            eval_dataset = tokenized["validation"]
        train_dataset = tokenized["train"] if "train" in tokenized else next(iter(tokenized.values()))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
