import argparse
import json
import os
import re

import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, ClassLabel


def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    with open(args.json_file, "r", encoding="utf-8") as json_file:
        dataset = json.load(json_file)

    rows = []
    for value in dataset.values():
        cleaned_score = re.sub(r"\..*", "", value["Score"])
        cleaned_score = float(cleaned_score)
        rows.append({"prompt": value["Prompt"], "score": cleaned_score})

    dataset = Dataset.from_list(rows)

    score_processing_cache_path = os.path.join(args.cache_dir, "score_processing.cache")
    dataset = dataset.map(
        lambda x: {"score": np.clip(int(x["score"]), 0, 5)},
        num_proc=32,
        cache_file_name=score_processing_cache_path,
    )
    dataset = dataset.cast_column("score", ClassLabel(names=[str(i) for i in range(6)]))
    dataset = dataset.train_test_split(
        train_size=0.95, seed=42, stratify_by_column="score"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess(examples):
        batch = tokenizer(examples["prompt"], truncation=True)
        batch["labels"] = np.float32(examples["score"])
        return batch

    preprocess_cache_path = os.path.join(args.cache_dir, "preprocess.cache")
    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=10000,
        cache_file_names={
            "train": preprocess_cache_path + ".train",
            "test": preprocess_cache_path + ".test",
        },
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0
    )

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=600,
        save_steps=600,
        logging_steps=5,
        learning_rate=3e-4,
        num_train_epochs=60,
        seed=0,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure training settings")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/workspace1/ouhenio/history-bert",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Snowflake/snowflake-arctic-embed-m",
        help="Model identifier for Hugging Face Transformers",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="generated_texts.json",
        help="Path to the JSON file containing the data",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2800, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1024, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory to store cache files",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="bert_eval",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="ouhenio",
        help="Weights & Biases entity name",
    )

    args = parser.parse_args()

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    main(args)
