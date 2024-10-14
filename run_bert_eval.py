import argparse
import os
import torch
import shutil
import logging
from glob import glob
from typing import Optional
from multiprocess import set_start_method
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, config
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clear_cache():
    cache_dir = config.HF_DATASETS_CACHE
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)


def process_shard(
    file_path: str, tokenizer, model, batch_size: int, num_gpus: int, text_column: str
):
    try:
        shard_dataset = load_dataset("arrow", data_files=[file_path])

        def compute_scores(batch, rank: Optional[int] = None):
            device = f"cuda:{rank}" if rank is not None else "cpu"
            model.to(device)
            inputs = tokenizer(
                batch[text_column],
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).to(device)
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
            num_proc=num_gpus,
        )
        return processed_shard
    except Exception as e:
        logging.error(f"Error processing shard {file_path}: {str(e)}")
        return None


def main(
    model_path: str,
    dataset: str,
    output_dir: str,
    num_gpus: int,
    batch_size: int,
    text_column: str,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )

        files = sorted(glob(os.path.join(dataset, "*.arrow")))
        processed_files = sorted(glob(os.path.join(output_dir, "*.arrow")))
        processed_files_set = set(processed_files)

        all_processed_shards = []

        for file_path in tqdm(files, desc="Processing shards"):
            shard_name = os.path.basename(file_path)
            processed_shard_path = os.path.join(output_dir, shard_name)

            if processed_shard_path in processed_files_set:
                logging.info(f"Shard already processed: {shard_name}")
                all_processed_shards.append(processed_shard_path)
                continue

            processed_shard = process_shard(
                file_path, tokenizer, model, batch_size, num_gpus, text_column
            )
            if processed_shard:
                processed_shard.save_to_disk(processed_shard_path)
                all_processed_shards.append(processed_shard_path)

            clear_cache()

        if all_processed_shards:
            logging.info(
                f"Loading and concatenating {len(all_processed_shards)} processed shards..."
            )
            final_dataset = concatenate_datasets(
                [
                    load_dataset("arrow", data_files=[path])
                    for path in all_processed_shards
                ]
            )
            final_dataset.save_to_disk(output_dir)
            logging.info(f"Final dataset saved to {output_dir}")
        else:
            logging.warning("No shards were processed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset shards.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the embedding model directory.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size for processing."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="texto",
        help="Name of the text column in the dataset.",
    )
    args = parser.parse_args()

    set_start_method("spawn")
    main(
        args.model_path,
        args.dataset,
        args.output_dir,
        args.num_gpus,
        args.batch_size,
        args.text_column,
    )
