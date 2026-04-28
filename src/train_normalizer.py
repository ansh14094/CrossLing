"""Train the seq2seq Hinglish -> English normalizer."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import yaml
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .data import (
    NormalizationDataset,
    load_normalization_examples,
    split_examples,
)
from .metrics import normalization_metrics
from .models import NormalizerModel, resolve_device


def train_normalizer(config: dict[str, Any],
                     data_path: str | None = None) -> NormalizerModel:
    cfg = config["normalizer"]
    device = resolve_device(config.get("device", "auto"))

    examples = load_normalization_examples(data_path)
    train_examples, val_examples = split_examples(examples,
                                                  seed=config.get("seed", 42))

    model = NormalizerModel.from_pretrained(
        cfg["backbone"], device, source_prefix=cfg.get("source_prefix", ""))

    train_ds = NormalizationDataset(
        train_examples,
        model.tokenizer,
        cfg["max_source_length"],
        cfg["max_target_length"],
        cfg.get("source_prefix", ""),
    )
    val_ds = NormalizationDataset(
        val_examples,
        model.tokenizer,
        cfg["max_source_length"],
        cfg["max_target_length"],
        cfg.get("source_prefix", ""),
    )

    collator = DataCollatorForSeq2Seq(model.tokenizer, model=model.model)

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = labels.copy()
        labels[labels == -100] = model.tokenizer.pad_token_id
        decoded_preds = model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return normalization_metrics(decoded_preds, decoded_labels)

    use_cuda = device.type == "cuda"
    args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        predict_with_generate=True,
        generation_max_length=cfg["max_target_length"],
        generation_num_beams=4,
        # mT5 is unstable in fp16 — use bf16 if the GPU supports it, else fp32.
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=config.get("seed", 42),
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=model.tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    final_metrics = trainer.evaluate()
    print(f"\nNormalizer validation: BLEU={final_metrics.get('eval_bleu'):.2f} "
          f"chrF={final_metrics.get('eval_chrf'):.2f}")

    model.save(cfg["output_dir"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None,
                        help='Optional JSONL with {"src": ..., "tgt": ...}.')
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    os.makedirs(config["normalizer"]["output_dir"], exist_ok=True)
    train_normalizer(config, data_path=args.data)


if __name__ == "__main__":
    main()
