import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification,
    TrainingArguments,
    Trainer,
)


@dataclass
class TrainConfig:
    model_name: str
    dataset_name: str
    train_split: str
    val_split: str
    output_dir: str
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    seed: int


def create_label_maps(dataset):
    features = dataset["train"].features["label"]
    id2label = {i: name for i, name in enumerate(features.names)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


def collate_fn(examples, processor):
    images = [e["image"] for e in examples]
    labels = [e["label"] for e in examples]
    batch = processor(images=images, return_tensors="pt")
    batch["labels"] = torch.tensor(labels)
    return batch


def main():
    parser = argparse.ArgumentParser(description="Fine-tune color classifier")
    parser.add_argument("--dataset", type=str, default="prithivMLmods/Fashion-Product-BaseColour-Images", help="HuggingFace dataset or data dir")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="Base model to fine-tune")
    parser.add_argument("--output", type=str, default="checkpoints_color", help="Output dir")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--eval_bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    id2label, label2id = create_label_maps(dataset)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = SiglipForImageClassification.from_pretrained(
        args.model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    def _collate(examples):
        return collate_fn(examples, processor)

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )

    from evaluate import load as load_metric

    accuracy = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset[args.train_split],
        eval_dataset=dataset[args.val_split],
        tokenizer=processor,
        data_collator=_collate,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()


