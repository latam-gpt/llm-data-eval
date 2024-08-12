import torch
from multiprocess import set_start_method

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "/workspace1/ouhenio/history-bert/final"

def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    dataset = load_dataset(
        "arrow",
        data_files=[
            "/workspace1/ouhenio/es_clean_shuffled/data-00508-of-00512.arrow",
            "/workspace1/ouhenio/es_clean_shuffled/data-00509-of-00512.arrow",
            "/workspace1/ouhenio/es_clean_shuffled/data-00510-of-00512.arrow",
            "/workspace1/ouhenio/es_clean_shuffled/data-00511-of-00512.arrow",
        ]
    )

    def compute_scores(batch, rank=None):
        device = f"cuda:{rank}"
        model.to(device)
        inputs = tokenizer(batch["texto"], return_tensors="pt", padding="longest", truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).float().cpu().numpy()

        batch["score"] = logits.tolist()
        batch["int_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
        return batch

    dataset = dataset.map(
        compute_scores,
        batched=True,
        batch_size=4096,
        with_rank=True,
        num_proc=torch.cuda.device_count() # use all visible GPUs
    )

    dataset.save_to_disk("/workspace1/ouhenio/scored_es_clean_shuffled")

if __name__ == "__main__":
    # creo que usar la forma propia de datasets para paralelizar map es más lenta
    # que la forma nativa de torch (DataParallel) para procesar los datos
    set_start_method("spawn") # required to use multiple-gpus

    main()