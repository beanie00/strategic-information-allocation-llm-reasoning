"""
Download LIMO-v2 dataset from HuggingFace and convert to the training JSON format.
"""
import json
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and convert LIMO-v2 dataset to training format."
    )
    parser.add_argument("--dataset", type=str, default="GAIR/LIMO-v2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="train/data/limo_v2.json")
    parser.add_argument("--system_prompt", type=str,
                        default="Please reason step by step, and put your final answer within \\boxed{}.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    dataset = load_dataset(args.dataset, split=args.split)

    converted = [
        {
            "instruction": row["question"],
            "input": "",
            "output": row["solution"],
            "system": args.system_prompt,
        }
        for row in dataset
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(converted)} samples to {args.output}")


if __name__ == "__main__":
    main()