import re
import json

from datasets import Dataset
from vllm import LLM, SamplingParams

BATCH_SIZE = 500
PATTERN = r"PUNTUACIÓN EDUCATIVA[:\s]*([\d\.]+)"

def load_base_prompt():
    with open("utils/prompt.txt") as f:
        return f.read()
    
def insert_prompt(base_prompt: str, prompt_to_insert: str):
    modified_prompt = base_prompt.replace("<EJEMPLO>", prompt_to_insert)
    return modified_prompt
    
def main():
    results_dict = {}
    base_prompt = load_base_prompt()

    def update_text(example):
        example["eval_prompt"] = insert_prompt(base_prompt, example["texto"])
        return example

    dataset = Dataset.from_file("/workspace1/ouhenio/es_clean_shuffled/data-00000-of-00512.arrow")
    dataset = dataset.map(update_text)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=200,
        truncate_prompt_tokens=130000, # no sirve de na esta wea 
    )
    
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        download_dir="/workspace1/ouhenio",
        tensor_parallel_size=8,
        enable_prefix_caching=True,
    )
    
    curr_idx = 0

    while len(dataset) > curr_idx + BATCH_SIZE:
        curr_batch = dataset[curr_idx:curr_idx + BATCH_SIZE]
        outputs = llm.generate(list(curr_batch["eval_prompt"]), sampling_params)

        for idx, (vllm_output, prompt) in enumerate(zip(outputs, list(curr_batch["texto"]))):
            generated_text = vllm_output.outputs[0].text
            match = re.search(PATTERN, generated_text)
            if match:
                results_dict[curr_idx + idx] = {
                    "Score": match.group(1),
                    "Prompt": prompt,
                }

        with open('generated_texts.json', 'w+', encoding='utf-8') as json_file:
            json.dump(results_dict, json_file, ensure_ascii=False, indent=4)

        curr_idx += BATCH_SIZE

if __name__ == "__main__":
    main()