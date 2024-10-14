import os
import multiprocessing as mp
from datasets import load_from_disk, concatenate_datasets

def filter_condition(example):
    return example['int_score'] >= 3

def load_and_filter_dataset(file_path):
    dataset = load_from_disk(file_path)["train"]
    filtered_dataset = dataset.filter(filter_condition)
    return filtered_dataset


def parallel_load_datasets(directory):
    files = os.listdir(directory)
    arrow_files = [os.path.join(directory, f) for f in files if f.endswith(".arrow")]
    
    num_processes = 12
    
    with mp.Pool(processes=num_processes) as pool:
        datasets_list = pool.map(load_and_filter_dataset, arrow_files)
    
    return datasets_list

directory = "/workspace1/ouhenio/academic_score"

datasets_list = parallel_load_datasets(directory)

print("loaded!!")

if datasets_list:
    dataset = concatenate_datasets(datasets_list)
else:
    dataset = None

dataset.save_to_disk("/workspace1/ouhenio/academic_hq_full", num_proc=12)