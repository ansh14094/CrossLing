"""Train the sequence-classification sentiment model on code-mixed text."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data import SentimentDataset, load_sentiment_examples, split_examples
from .metrics import sentiment_classification_report, sentiment_metrics
from .models import SentimentModel, resolve_device


def train_sentiment(config: dict[str, Any],
                    data_path: str | None = None) -> SentimentModel:
    cfg = config["sentiment"]
    device = resolve_device(config.get("device", "auto"))

    examples = load_sentiment_examples(data_path)
    train_examples, val_examples = split_examples(examples,
                                                  seed=config.get("seed", 42))

    model = SentimentModel.from_pretrained(cfg["backbone"], cfg["labels"], device)
    train_ds = SentimentDataset(train_examples, model.tokenizer, model.label2id,
                                cfg["max_length"])
    val_ds = SentimentDataset(val_examples, model.tokenizer, model.label2id,
                              cfg["max_length"])

    collator = DataCollatorWithPadding(model.tokenizer)

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return sentiment_metrics(preds, labels, model.id2label)

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

    pred_output = trainer.predict(val_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)
    print("\nSentiment validation report:")
    print(sentiment_classification_report(preds, pred_output.label_ids,
                                          model.id2label))

    model.save(cfg["output_dir"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None,
                        help="Optional TSV file `text<TAB>label`.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    os.makedirs(config["sentiment"]["output_dir"], exist_ok=True)
    train_sentiment(config, data_path=args.data)


if __name__ == "__main__":
    main()
