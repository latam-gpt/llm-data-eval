import os
import torch
from multiprocess import set_start_method
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "/workspace1/ouhenio/history-bert/final"

def process_shard(file_path, tokenizer, model):
    shard_dataset = load_dataset("arrow", data_files=[file_path])

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

    processed_shard = shard_dataset.map(
        compute_scores,
        batched=True,
        batch_size=4096,
        with_rank=True,
        num_proc=3
    )
    return processed_shard

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    arrow_folder = "/workspace1/ouhenio/es_clean_shuffled"
    files = [os.path.join(arrow_folder, file) for file in os.listdir(arrow_folder) if file.endswith('.arrow')]
    files.sort()  # sort files to maintain order

    processed_dir = "/workspace1/ouhenio/scored_es_clean_shuffled"

    # load previously processed shards
    processed_files = [os.path.join(processed_dir, file) for file in os.listdir(processed_dir) if file.endswith('.arrow')]
    processed_files_set = set(processed_files)

    all_processed_shards = []

    for file_path in tqdm(files, desc="Processing shards"):
        shard_name = os.path.basename(file_path)
        processed_shard_path = os.path.join(processed_dir, shard_name)
        
        # skip processing processed shards
        if processed_shard_path in processed_files_set:
            continue
        
        processed_shard = process_shard(file_path, tokenizer, model)
        processed_shard.save_to_disk(processed_shard_path)
        all_processed_shards.append(processed_shard)

    if all_processed_shards:
        final_dataset = concatenate_datasets(all_processed_shards)
        final_dataset.save_to_disk(processed_dir)

if __name__ == "__main__":
    set_start_method("spawn")  # required to use multiple GPUs
    main()
