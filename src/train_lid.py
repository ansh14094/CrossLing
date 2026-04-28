"""Train the token-level Language Identification model."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .data import LIDDataset, load_lid_examples, split_examples
from .metrics import lid_classification_report, lid_metrics
from .models import LIDModel, resolve_device


def train_lid(config: dict[str, Any], data_path: str | None = None) -> LIDModel:
    cfg = config["lid"]
    device = resolve_device(config.get("device", "auto"))

    examples = load_lid_examples(data_path)
    train_examples, val_examples = split_examples(examples,
                                                  seed=config.get("seed", 42))

    model = LIDModel.from_pretrained(cfg["backbone"], cfg["labels"], device)
    train_ds = LIDDataset(train_examples, model.tokenizer, model.label2id,
                          cfg["max_length"])
    val_ds = LIDDataset(val_examples, model.tokenizer, model.label2id,
                        cfg["max_length"])

    collator = DataCollatorForTokenClassification(model.tokenizer)

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return lid_metrics(preds, labels, model.id2label)

    use_cuda = device.type == "cuda"
    args = TrainingArguments(
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
        metric_for_best_model="macro_f1",
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=config.get("seed", 42),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=model.tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # final eval report
    pred_output = trainer.predict(val_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)
    print("\nLID validation report:")
    print(lid_classification_report(preds, pred_output.label_ids, model.id2label))

    model.save(cfg["output_dir"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None,
                        help="Optional CoNLL-style LID file (token<TAB>label).")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    os.makedirs(config["lid"]["output_dir"], exist_ok=True)
    train_lid(config, data_path=args.data)


if __name__ == "__main__":
    main()
