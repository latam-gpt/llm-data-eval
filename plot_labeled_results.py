import argparse
import json
import logging
import matplotlib.pyplot as plt
import os
from collections import Counter
import numpy as np
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)


def plot_multilabel_results(args, data=None, dataset_name=None):
    # Load the JSON data
    if data is None:
        with open(args.result_path, "r", encoding="utf-8") as file:
            data = json.load(file)

    # data cleaning
    data = {
        k: {key: None if val == "Sin coincidencia" else val for key, val in v.items()}
        for k, v in data.items()
    }

    # basic stats
    total_entries = len(data)
    valid_entries_count = sum(
        1
        for record in data.values()
        if record["domains"] is not None and record["use_cases"] is not None
    )

    # Extract 'domains' and 'use_cases' values
    domains = []
    use_cases = []

    for record in data.values():
        if record["domains"]:
            domains.extend([domain.strip() for domain in record["domains"].split(",")])
        if record["use_cases"]:
            use_cases.extend(
                [use_case.strip() for use_case in record["use_cases"].split(",")]
            )

    # Count occurrences for domains and use_cases
    domain_counts = Counter(domains)
    use_case_counts = Counter(use_cases)

    if dataset_name is None:
        base_name = os.path.basename(args.result_path)
        base_name = os.path.splitext(base_name)[0]
        dataset_name = base_name.split("_")[1]
    else:
        base_name = "all"

    # Helper function to sort, calculate percentages, and wrap labels
    def sort_and_calculate_percentages(counter, top_k=None):
        # To address some typo in the labeler
        if "E" in counter:
            del counter["E"]

        total = sum(counter.values())
        sorted_items = sorted(
            counter.items(), key=lambda x: x[1], reverse=True
        )  # Sort by frequency (descending)
        sorted_items = sorted_items[:top_k] if top_k is not None else sorted_items
        labels, values = zip(*sorted_items)  # Separate labels and values
        percentages = [(v / total) * 100 for v in values]  # Calculate percentages
        # wrapped_labels = [textwrap.fill(label, 20) for label in labels]  # Wrap long labels to max 20 chars per line
        # return wrapped_labels, percentages
        return labels, values, percentages

    # Sort and calculate percentages for domains
    domain_labels, domain_values, domain_percentages = sort_and_calculate_percentages(
        domain_counts, top_k=10
    )

    # Plot histogram for domains
    plt.figure(figsize=(12, 6))  # Increased figure size
    bars = plt.bar(domain_labels, domain_percentages)
    plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate labels
    plt.title(
        f"Top-10 Distribution of Domains for `{dataset_name}`\nExamples: {valid_entries_count:,}/{total_entries:,}"
    )
    # plt.title(f"Distribution of Domains for `{dataset_name}`")
    plt.xlabel("Domains")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()

    # Add percentage labels on top of bars
    for bar, percentage in zip(bars, domain_percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.savefig(
        os.path.join(args.output_dir, base_name + "_domains.png")
    )  # Save the figure

    # Sort and calculate percentages for use_cases
    use_case_labels, use_case_values, use_case_percentages = (
        sort_and_calculate_percentages(use_case_counts, top_k=20)
    )

    # Plot histogram for use_cases
    plt.figure(figsize=(12, 6))  # Increased figure size
    bars = plt.bar(use_case_labels, use_case_percentages)
    plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate labels and set font size
    plt.yticks(np.arange(0, 13, 1))  # Set y-axis ticks from 0 to 100 with a step of 1%
    plt.title(
        f"Top-20 Distribution of Use Cases for `{dataset_name}`\nExamples: {valid_entries_count:,}/{total_entries:,}"
    )
    # plt.title(f"Distribution of Use Cases for `{dataset_name}`")
    plt.xlabel("Use Cases")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()

    # Add percentage labels on top of bars
    for bar, percentage in zip(bars, use_case_percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.savefig(
        os.path.join(args.output_dir, base_name + "_usecases.png")
    )  # Save the figure


def main(args):
    if os.path.isdir(args.result_path):
        # batch plotting of a directory
        files = glob(os.path.join(args.result_path, "usecasesdomains_*.json"))
        combined_data = {}
        record_idx = 0

        for file_path in files:
            # To avoid duplicating datasets since those with toxicity filters have same content
            if "toxicity" in file_path:
                continue
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for record in data.values():
                    combined_data[record_idx] = record
                    record_idx += 1
        plot_multilabel_results(args, data=combined_data, dataset_name="All-Datasets")
    else:
        # plot the content of a json file
        plot_multilabel_results(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and generate scores.")
    parser.add_argument(
        "--result_path", type=str, required=True, help="Path to the input result file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the output folder.",
    )

    args = parser.parse_args()
    print(args)

    main(args)
