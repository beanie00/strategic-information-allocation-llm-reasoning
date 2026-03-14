"""
Batch runner for token distribution analysis across multiple models.
Calls analyze_token_distribution.py for each model configuration.
"""
import subprocess
import os

MODELS = [
    {"name": "DeepSeek-R1-Distill-Qwen-1.5B", "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    {"name": "deepseek-math-7b-instruct", "path": "deepseek-ai/deepseek-math-7b-instruct"},
    {"name": "Qwen2.5-Math-7B", "path": "Qwen/Qwen2.5-Math-7B"},
    {"name": "Qwen2.5-7B", "path": "Qwen/Qwen2.5-7B"},
    {"name": "Qwen3-1.7B-Base", "path": "Qwen/Qwen3-1.7B-Base"},
    {"name": "Qwen3-4B-Base", "path": "Qwen/Qwen3-4B-Base"},
    {"name": "Qwen3-8B-Base", "path": "Qwen/Qwen3-8B-Base"},
    {"name": "Qwen3-14B-Base", "path": "Qwen/Qwen3-14B-Base"},
    {"name": "Llama-3.1-8B", "path": "meta-llama/Llama-3.1-8B"},
]

INPUT_JSON = "train/data/limo_v2.json"
OUTPUT_DIR = "analysis/results"
ANALYSIS_SCRIPT = "analyze_token_distribution.py"


def build_command(model, input_json, output_dir):
    name = model["name"]
    return [
        "python", ANALYSIS_SCRIPT,
        "--model_name_or_path", model["path"],
        "--input_json", input_json,
        "--output_csv", f"{output_dir}/{name}_token_stats.csv",
        "--output_special_csv", f"{output_dir}/{name}_special_tokens.csv",
        "--output_all_logprobs_csv", f"{output_dir}/{name}_all_logprobs.csv",
        "--output_all_entropies_csv", f"{output_dir}/{name}_all_entropies.csv",
        "--output_plot", f"{output_dir}/{name}_logprob_dist.png",
        "--dtype", "float16",
    ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Running Token Analysis for Multiple Models")
    print("=" * 60)

    for idx, model in enumerate(MODELS, 1):
        print(f"\n[{idx}/{len(MODELS)}] Processing: {model['name']}")
        print("-" * 60)

        cmd = build_command(model, INPUT_JSON, OUTPUT_DIR)
        print(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, text=True)
            print(f"✓ Completed: {model['name']}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {model['name']} — {e}")
            continue
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            break

    print("\n" + "=" * 60)
    print("All models processed!")
    print("=" * 60)
    print(f"\nResults saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()