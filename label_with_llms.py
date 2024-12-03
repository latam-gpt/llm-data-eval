import argparse
import re
import json
import logging
from datasets import Dataset
from vllm import LLM, SamplingParams


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

PATTERN_DOMAIN = r"DOMINIO[:\s]*(\w+(?:\s\w+)*(?:,\s*\w+(?:\s\w+)*)*)\n"
PATTERN_USECASE = r"CASOS DE USO[:\s]*(\w+(?:\s\w+)*(?:,\s*\w+(?:\s\w+)*)*)"


def load_base_prompt(prompt_path):
    with open(prompt_path) as f:
        return f.read()


def insert_prompt(base_prompt: str, prompt_to_insert: str):
    return base_prompt.replace("<EJEMPLO>", prompt_to_insert)


def update_text(example, base_prompt):
    example["eval_prompt"] = insert_prompt(base_prompt, example[args.text_column])
    return example


def process_batch(llm, batch, sampling_params):
    print(f"len(batch['eval_prompt']): {len(batch['eval_prompt'])}")
    outputs = llm.generate(list(batch["eval_prompt"]), sampling_params)
    results = {}

    for idx, (vllm_output, prompt) in enumerate(zip(outputs, list(batch[args.text_column]))):
        generated_text = vllm_output.outputs[0].text
        match_domain = re.search(PATTERN_DOMAIN, generated_text)
        match_usecases = re.search(PATTERN_USECASE, generated_text)

        if match_domain and match_usecases:
            results[idx] = {
                # "Prompt": prompt,
                "domains": match_domain.group(1),
                "use_cases": match_usecases.group(1),
            }
        else:
            results[idx] = {
                # "Prompt": prompt,
                "domains": None,
                "use_cases": None,
            }
    return results


def main(args):
    try:
        base_prompt = load_base_prompt(args.prompt_path)

        logging.info(f"Loading dataset from {args.dataset_path}")
        dataset = Dataset.from_file(args.dataset_path)
        dataset = dataset.map(lambda x: update_text(x, base_prompt))

        sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=200,
            truncate_prompt_tokens=130000,
        )

        logging.info(f"Initializing LLM with model {args.model_path}")
        llm = LLM(
            model=args.model_path,
            download_dir=args.download_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
            max_model_len=args.max_model_len,
        )

        results_dict = {}
        curr_idx = 0
        while len(dataset) > curr_idx + args.batch_size:
            curr_batch = dataset[curr_idx : curr_idx + args.batch_size]
            batch_results = process_batch(llm, curr_batch, sampling_params)

            for idx, result in batch_results.items():
                results_dict[curr_idx + idx] = result

            with open(args.output_path, "w", encoding="utf-8") as json_file:
                json.dump(results_dict, json_file, ensure_ascii=False, indent=4)

            curr_idx += args.batch_size
            logging.info(f"Processed {curr_idx} examples")

        logging.info(f"Processing complete. Results saved to {args.output_path}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and generate scores.")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="utils/prompts.txt",
        help="Path to the base prompt file.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Path or name of the model to use.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        required=True,
        help="Directory to download the model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="texto",
        help="Name of the text column in the dataset.",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=8, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Rating batch size."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=10432,
        help="Max sequence length that fits the KV cache.",
    )

    args = parser.parse_args()
    print(args)

    main(args)
