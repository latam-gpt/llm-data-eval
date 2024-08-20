import os
import torch
import argparse
from multiprocess import set_start_method
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# since our datasets are too big, we process them shard by shard
def process_shard(file_path, tokenizer, model, batch_size, num_gpus):
    shard_dataset = load_dataset("arrow", data_files=[file_path])

    def compute_scores(batch, rank=None):
        device = f"cuda:{rank}"
        model.to(device)
        # 'texto' column is hardcoded, should be changed in the future
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
        batch_size=batch_size,
        with_rank=True,
        num_proc=num_gpus
    )
    return processed_shard

def main(model_path, dataset, output_dir, num_gpus, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    files = [os.path.join(dataset, file) for file in os.listdir(dataset) if file.endswith('.arrow')]
    files.sort()

    processed_files = [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith('.arrow')]
    processed_files_set = set(processed_files)

    all_processed_shards = []

    for file_path in tqdm(files, desc="Processing shards"):
        shard_name = os.path.basename(file_path)
        processed_shard_path = os.path.join(output_dir, shard_name)
        
        if processed_shard_path in processed_files_set:
            continue
        
        processed_shard = process_shard(file_path, tokenizer, model, batch_size, num_gpus)
        processed_shard.save_to_disk(processed_shard_path)
        all_processed_shards.append(processed_shard)

    if all_processed_shards:
        final_dataset = concatenate_datasets(all_processed_shards)
        final_dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset shards.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the embedding model directory.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset folder.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder.')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs to use.')
    parser.add_argument('--batch_size', type=int, required=True, default=4096, help='Batch size for processing.')
    args = parser.parse_args()

    set_start_method("spawn") # required to run dataset map with multiple gpus
    main(args.model_path, args.dataset, args.output_dir, args.num_gpus, args.batch_size)
