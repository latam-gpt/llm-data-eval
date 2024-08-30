import argparse
import os
from datasets import load_from_disk, concatenate_datasets
from tqdm.auto import tqdm

def filter_by_score(
    dataset_path: str,
    score_threshold: int,
    num_proc: int,
    output_path: str
):
    files = os.listdir(dataset_path)
    arrow_files = [os.path.join(dataset_path, f) for f in files if f.endswith(".arrow")]

    datasets_list = []

    for file in tqdm(arrow_files, desc="Loading datasets"):
        dataset = load_from_disk(file)
        datasets_list.append(dataset)

    if datasets_list:
        full_dataset = concatenate_datasets([x["train"] for x in datasets_list])
    else:
        full_dataset = None

    if full_dataset is not None:
        print(f"Initial number of items: {len(full_dataset)}")
        def filter_batch(batch):
            return {"int_score": [s for s in batch['int_score'] if s >= score_threshold]}
        full_dataset = full_dataset.filter(filter_batch, batched=True, num_proc=num_proc)
        print(f"Number of items after filter: {len(full_dataset)}")
        full_dataset.save_to_disk(output_path)
    else:
        print("No datasets were loaded. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter results by score.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--score', type=int, required=True, help='Filter score (everything below it, gets pruned).')
    parser.add_argument('--num_proc', type=int, required=True, help='Number of CPU processes to parallelize this.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the filtered dataset.')

    args = parser.parse_args()

    filter_by_score(
        dataset_path=args.dataset_path,
        score_threshold=args.score,
        num_proc=args.num_proc,
        output_path=args.output_path
    )
