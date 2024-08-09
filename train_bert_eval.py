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

CHECKPOINT_DIR = "/workspace1/ouhenio/history-bert"
BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"

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

def main():
    with open("generated_texts.json", "r", encoding="utf-8") as json_file:
        dataset = json.load(json_file)

    rows = []
    for value in dataset.values():
        cleaned_score = re.sub(r'\..*', '', value['Score'])
        cleaned_score = float(cleaned_score)
        rows.append({'prompt': value['Prompt'], 'score': cleaned_score})

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda x: {"score": np.clip(int(x["score"]), 0, 5)}, num_proc=64
    )
    dataset = dataset.cast_column(
        "score", ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42, stratify_by_column="score"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    def preprocess(examples):
        batch = tokenizer(examples["prompt"], truncation=True)
        batch["labels"] = np.float32(examples["score"])
        return batch
    
    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0)


    # for param in model.bert.embeddings.parameters():
    #     param.requires_grad = False
    # for param in model.bert.encoder.parameters():
    #     param.requires_grad = False


    # training_args = TrainingArguments(
    #     output_dir=CHECKPOINT_DIR,
    #     evaluation_strategy="steps",
    #     save_strategy="steps",
    #     eval_steps=1000,
    #     save_steps=1000,
    #     logging_steps=100,
    #     learning_rate=3e-4,
    #     num_train_epochs=20,
    #     seed=0,
    #     per_device_train_batch_size=256,
    #     per_device_eval_batch_size=128,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1_macro",
    #     greater_is_better=True,
    #     bf16=True,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    # trainer.train()
    # trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))

if __name__ == "__main__":
    main()