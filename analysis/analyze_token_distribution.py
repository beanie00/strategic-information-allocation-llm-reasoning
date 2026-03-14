import json
import os
import argparse
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze token-level log probability and entropy distributions across a dataset."
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-14B-Base")
    parser.add_argument("--input_json", type=str, default="train/data/limo_v2.json")
    parser.add_argument("--output_csv", type=str, default="analysis/results/Qwen3-14B-Base_token_stats.csv")
    parser.add_argument("--output_special_csv", type=str, default="analysis/results/Qwen3-14B-Base_special_tokens.csv")
    parser.add_argument("--output_all_logprobs_csv", type=str, default="analysis/results/Qwen3-14B-Base_all_logprobs.csv")
    parser.add_argument("--output_all_entropies_csv", type=str, default="analysis/results/Qwen3-14B-Base_all_entropies.csv")
    parser.add_argument("--output_plot", type=str, default="analysis/results/Qwen3-14B-Base_logprob_dist.png")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["Wait", "Alternatively"],
                        help="List of special tokens to track individually (resolved dynamically per model)")
    parser.add_argument("--dtype", default="float16", type=str)
    return parser.parse_args()


def load_model(model_name_or_path, dtype="float16"):
    """Load tokenizer and model with the specified dtype."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def resolve_special_token_ids(tokenizer, token_strings):
    """Dynamically resolve token strings to their IDs for the current tokenizer.

    Returns a dict mapping token_id (int) -> {"token": str, "logprobs": [], "entropies": []}.
    Tokens that encode to multiple sub-tokens are skipped with a warning.
    """
    special_tokens = {}
    for token_str in token_strings:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1:
            special_tokens[ids[0]] = {
                "token": token_str,
                "logprobs": [],
                "entropies": [],
            }
            print(f"  Tracking special token: '{token_str}' -> ID {ids[0]}")
        else:
            print(f"  WARNING: '{token_str}' encodes to {len(ids)} sub-tokens {ids}, skipping")
    return special_tokens


def build_chat_prompt(tokenizer, system, instruction, user_input):
    """Construct a chat-formatted prompt string using the tokenizer's chat template."""
    messages = []

    if system and system.strip():
        messages.append({"role": "system", "content": system})

    user_content = instruction
    if user_input and user_input.strip():
        user_content = instruction + "\n" + user_input

    messages.append({"role": "user", "content": user_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_token_stats(tokenizer, model, prompt_text, generated_text):
    """Compute per-token log probabilities and entropy for the full sequence.

    To avoid tokenization boundary mismatch between prompt and full text,
    we tokenize the prompt first, then tokenize prompt+output together and
    return the verified prompt length in tokens.

    Returns:
        tokens: list of token strings (shifted, i.e. predicting each next token)
        target_ids: tensor of token IDs being predicted
        token_logprobs: tensor of log probabilities for each predicted token
        entropy: tensor of entropy at each position
        prompt_token_len: number of tokens belonging to the prompt portion
    """
    # Tokenize prompt alone to get its exact token length
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_token_len = prompt_ids.shape[1]

    # Tokenize the full sequence (prompt + output) as a single string
    full_text = prompt_text + generated_text
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Shift: predict token t+1 from position t
    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1]
    probs = probs[:, :-1]

    token_logprobs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    entropy = -(probs * log_probs).sum(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(target_ids.squeeze(0).tolist())

    return tokens, target_ids.squeeze(0), token_logprobs.squeeze(0), entropy.squeeze(0), prompt_token_len


def plot_logprob_distribution(all_logprobs, output_path):
    """Plot histogram, CDF, box plot, and summary statistics of log probabilities."""
    logprobs_array = np.array(all_logprobs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(logprobs_array, bins=100, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Log Probability", fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].set_title("Log Probability Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].axvline(
        logprobs_array.mean(), color="red", linestyle="--",
        linewidth=2, label=f"Mean: {logprobs_array.mean():.3f}",
    )
    axes[0, 0].axvline(
        np.median(logprobs_array), color="green", linestyle="--",
        linewidth=2, label=f"Median: {np.median(logprobs_array):.3f}",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_logprobs = np.sort(logprobs_array)
    cumulative = np.arange(1, len(sorted_logprobs) + 1) / len(sorted_logprobs)
    axes[0, 1].plot(sorted_logprobs, cumulative, linewidth=2, color="purple")
    axes[0, 1].set_xlabel("Log Probability", fontsize=12)
    axes[0, 1].set_ylabel("Cumulative Probability", fontsize=12)
    axes[0, 1].set_title("Cumulative Distribution Function", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Box plot
    axes[1, 0].boxplot(
        logprobs_array, vert=True, patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1, 0].set_ylabel("Log Probability", fontsize=12)
    axes[1, 0].set_title("Box Plot", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Statistics summary
    axes[1, 1].axis("off")
    stats_text = (
        f"    Statistics Summary\n"
        f"    {'═' * 27}\n\n"
        f"    Total Tokens: {len(logprobs_array):,}\n\n"
        f"    Mean:         {logprobs_array.mean():.4f}\n"
        f"    Median:       {np.median(logprobs_array):.4f}\n"
        f"    Std Dev:      {logprobs_array.std():.4f}\n\n"
        f"    Min:          {logprobs_array.min():.4f}\n"
        f"    Max:          {logprobs_array.max():.4f}\n\n"
        f"    Percentiles:\n"
        f"      25th:       {np.percentile(logprobs_array, 25):.4f}\n"
        f"      50th:       {np.percentile(logprobs_array, 50):.4f}\n"
        f"      75th:       {np.percentile(logprobs_array, 75):.4f}\n"
        f"      90th:       {np.percentile(logprobs_array, 90):.4f}\n"
        f"      95th:       {np.percentile(logprobs_array, 95):.4f}\n"
        f"      99th:       {np.percentile(logprobs_array, 99):.4f}\n"
    )
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved log probability distribution plot to {output_path}")
    plt.close()


def save_token_stats_csv(stats, output_path):
    """Save per-token aggregated statistics (count, avg logprob, avg entropy) to CSV."""
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "token_id", "count", "avg_logprob", "avg_entropy"])

        for token_id, s in sorted_stats:
            writer.writerow([
                s["token"],
                token_id,
                s["count"],
                s["logprob_sum"] / s["count"],
                s["entropy_sum"] / s["count"],
            ])

    print(f"Saved token statistics to {output_path}")


def save_special_tokens_csv(special_tokens, output_path):
    """Save individual occurrences of tracked special tokens to CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "token_id", "instance_index", "logprob", "entropy"])

        for token_id, data in special_tokens.items():
            for idx, (lp, ent) in enumerate(zip(data["logprobs"], data["entropies"])):
                writer.writerow([data["token"], token_id, idx, lp, ent])

    print(f"Saved special token distributions to {output_path}")


def save_histogram_csv(values, output_path, bins=100, value_range=None, label="values"):
    """Save histogram bin data to CSV for later distribution reconstruction."""
    arr = np.array(values)
    counts, bin_edges = np.histogram(arr, bins=bins, range=value_range)

    total_count = len(arr)
    bin_width = bin_edges[1] - bin_edges[0]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_start", "bin_end", "count", "density"])

        for i in range(len(counts)):
            density = counts[i] / (total_count * bin_width) if total_count > 0 else 0
            writer.writerow([
                f"{bin_edges[i]:.6f}",
                f"{bin_edges[i + 1]:.6f}",
                counts[i],
                f"{density:.6f}",
            ])

    print(f"Saved {label} histogram to {output_path}")
    print(f"  Total tokens: {total_count:,} | Bins: {bins} | Range: {value_range}")


def print_special_token_summary(special_tokens):
    """Print summary statistics for tracked special tokens."""
    print("\n" + "=" * 60)
    print("Special Token Statistics:")
    print("=" * 60)

    for token_id, data in special_tokens.items():
        if data["logprobs"]:
            print(f"\n  {data['token']} (ID: {token_id}):")
            print(f"    Count:        {len(data['logprobs'])}")
            print(f"    Mean logprob: {np.mean(data['logprobs']):.4f}")
            print(f"    Mean entropy: {np.mean(data['entropies']):.4f}")
            print(f"    Std logprob:  {np.std(data['logprobs']):.4f}")
            print(f"    Std entropy:  {np.std(data['entropies']):.4f}")
        else:
            print(f"\n  {data['token']} (ID: {token_id}): Not found in outputs")


def main():
    args = parse_args()

    print("Loading model...")
    tokenizer, model = load_model(args.model_name_or_path, args.dtype)

    # Resolve special token IDs dynamically for this model's tokenizer
    print("Resolving special token IDs...")
    special_tokens = resolve_special_token_ids(tokenizer, args.special_tokens)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Accumulated per-token-id statistics
    stats = defaultdict(lambda: {
        "token": "",
        "count": 0,
        "logprob_sum": 0.0,
        "entropy_sum": 0.0,
    })

    all_logprobs = []
    all_entropies = []

    for data in tqdm(data_list, desc="Aggregating"):
        prompt_text = build_chat_prompt(
            tokenizer,
            data.get("system", ""),
            data.get("instruction", ""),
            data.get("input", ""),
        )

        tokens, token_ids, logprobs, entropies, prompt_token_len = compute_token_stats(
            tokenizer, model, prompt_text, data.get("output", ""),
        )

        # Only process the answer portion (skip prompt tokens).
        # After the shift in compute_token_stats, index (prompt_token_len - 1)
        # corresponds to the first predicted answer token.
        answer_start = prompt_token_len - 1

        for tok, tok_id, lp, ent in zip(
            tokens[answer_start:],
            token_ids[answer_start:],
            logprobs[answer_start:],
            entropies[answer_start:],
        ):
            tok_id_int = int(tok_id)

            s = stats[tok_id_int]
            s["token"] = tokenizer.decode([tok_id_int])
            s["count"] += 1
            s["logprob_sum"] += float(lp)
            s["entropy_sum"] += float(ent)

            all_logprobs.append(float(lp))
            all_entropies.append(float(ent))

            if tok_id_int in special_tokens:
                special_tokens[tok_id_int]["logprobs"].append(float(lp))
                special_tokens[tok_id_int]["entropies"].append(float(ent))

    # Save results
    save_token_stats_csv(stats, args.output_csv)
    save_special_tokens_csv(special_tokens, args.output_special_csv)
    save_histogram_csv(all_logprobs, args.output_all_logprobs_csv, bins=100, value_range=(-30, 0), label="logprob")
    save_histogram_csv(all_entropies, args.output_all_entropies_csv, bins=100, value_range=(0, 10), label="entropy")
    print_special_token_summary(special_tokens)

    print("\nPlotting log probability distribution...")
    plot_logprob_distribution(all_logprobs, args.output_plot)


if __name__ == "__main__":
    main()