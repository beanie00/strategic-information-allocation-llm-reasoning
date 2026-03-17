"""
Compute MI for the first N tokens across multiple samples.
CSV streaming save version (every 500 tokens).
Includes option to combine solution + answer for GT representation.
"""

# Adapted from:
#   https://github.com/ChnQ/MI-Peaks/blob/master/src/generate_activation.py
#   https://github.com/ChnQ/MI-Peaks/blob/master/src/cal_mi.py

import torch
import json
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# MI estimator
def distmat(X):
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def sigma_estimation(X, Y):
    import numpy as np
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def kernelmat(X, sigma, ktype='gaussian'):
    if len(X.shape) == 1:
        X = X.view(-1, 1)

    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)
        else:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)

    Kxc = torch.mm(Kx, H)
    return Kxc


def hsic_normalized_cca(x, y, sigma=50., ktype='gaussian'):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma, ktype=ktype)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def estimate_mi_hsic(x, y, sigma=50.):
    return hsic_normalized_cca(x, y, sigma=sigma)


def load_data(input_file, indices=None):
    """Load data from jsonl file"""
    data = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if indices is None or i in indices:
                item = json.loads(line)
                data.append((i, item))
    return data


def get_activations(model, tokenizer, text, layer_idx, max_tokens=2500):
    """Get hidden states for first max_tokens"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get specified layer's hidden states
    # hidden_states: tuple of (num_layers+1, batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

    return hidden_states[0].cpu().float()  # (seq_len, hidden_dim), bfloat16 → float32


def build_gt_text(item, use_solution=False, solution_key='solution'):
    """Build ground truth text from item.

    Args:
        item: data item dict
        use_solution: if True, prepend solution text before answer
        solution_key: key name for solution field in the data

    Returns:
        gt_text: constructed GT string
        gt_answer: answer string (for logging)
        gt_solution: solution string or None (for logging)
    """
    gt_answer = item.get('gold_answer', item.get('answer', ''))
    gt_solution = None

    if use_solution:
        gt_solution = item.get(solution_key, '')
        if gt_solution:
            gt_text = str(gt_solution) + " " + str(gt_answer)
        else:
            print(f"  Warning: --use_solution enabled but '{solution_key}' field is empty/missing. Using answer only.")
            gt_text = str(gt_answer)
    else:
        gt_text = str(gt_answer)

    return gt_text, gt_answer, gt_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--indices", type=str, default="0-5",
                        help="Sample indices, e.g., '0-5' or '0,2,4,6'")
    parser.add_argument("--max_tokens", type=int, default=2500)
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (-1 for last)")
    parser.add_argument("--sigma", type=float, default=50.)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--save_every", type=int, default=500, help="Save CSV every N tokens")

    # Solution-related options
    parser.add_argument("--use_solution", action="store_true",
                        help="If set, prepend solution text before answer for GT representation")
    parser.add_argument("--solution_key", type=str, default="solution",
                        help="Key name for solution field in the JSONL data (default: 'solution')")
    parser.add_argument("--gt_max_tokens", type=int, default=None,
                        help="Max tokens for GT text. Default: 100 (answer only) or 500 (with solution)")
    args = parser.parse_args()

    # Set default GT max tokens
    if args.gt_max_tokens is None:
        args.gt_max_tokens = 500 if args.use_solution else 100

    # Parse indices
    if '-' in args.indices and ',' not in args.indices:
        start, end = map(int, args.indices.split('-'))
        indices = set(range(start, end + 1))
    else:
        indices = set(map(int, args.indices.split(',')))

    print(f"Processing samples: {sorted(indices)}")

    # GT mode display
    gt_mode = "solution+answer" if args.use_solution else "answer_only"
    print(f"GT mode: {gt_mode} (max_tokens={args.gt_max_tokens})")
    if args.use_solution:
        print(f"Solution key: '{args.solution_key}'")

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Get actual layer index
    num_layers = len(model.model.layers)
    layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer + 1
    print(f"Using layer {layer_idx} (out of {num_layers})")

    # Load data
    print(f"Loading data: {args.input_file}")
    data = load_data(args.input_file, indices)
    print(f"Loaded {len(data)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # Create CSV file and write header
    csv_filename = f"mi_first{args.max_tokens}_layer{layer_idx}_{gt_mode}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    with open(csv_path, 'w') as f:
        f.write("sample_idx,position,token_id,token_str,mi_value\n")
    print(f"Created CSV file: {csv_path}")

    for sample_idx, item in data:
        print(f"\n{'='*50}")
        print(f"Processing sample {sample_idx}")

        # Get trajectory text (model's reasoning)
        generated = item.get('generated_responses', [])
        trajectory_text = generated[0] if generated else ''

        # Build GT text
        gt_text, gt_answer, gt_solution = build_gt_text(
            item,
            use_solution=args.use_solution,
            solution_key=args.solution_key
        )

        if not trajectory_text or not gt_text:
            print(f"  Skipping: missing trajectory or GT")
            continue

        print(f"  Trajectory length: {len(trajectory_text)} chars")
        print(f"  GT answer: {gt_answer[:50]}...")
        if gt_solution:
            print(f"  GT solution: {gt_solution[:80]}...")
        print(f"  GT text total length: {len(gt_text)} chars")

        # Get activations
        print(f"  Getting trajectory activations...")
        traj_acts = get_activations(model, tokenizer, trajectory_text, layer_idx, args.max_tokens)
        print(f"  Trajectory tokens: {traj_acts.shape[0]}")

        print(f"  Getting GT activations (mode={gt_mode}, max_tokens={args.gt_max_tokens})...")
        gt_acts = get_activations(model, tokenizer, gt_text, layer_idx, max_tokens=args.gt_max_tokens)
        gt_acts = gt_acts[-1]  # Last token: accumulates info from entire solution+answer
        print(f"  GT representation shape: {gt_acts.shape}")

        # Get token strings
        tokens = tokenizer(trajectory_text, truncation=True, max_length=args.max_tokens)
        token_ids = tokens['input_ids']
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        # Compute MI with periodic saving
        num_tokens = traj_acts.shape[0]
        mi_values = torch.zeros(num_tokens)

        print(f"  Computing MI for {num_tokens} tokens...")
        buffer = []  # Data buffer for periodic saving

        for i in tqdm(range(num_tokens), desc="Computing MI", leave=False):
            mi_values[i] = estimate_mi_hsic(traj_acts[i], gt_acts, sigma=args.sigma)

            # Append to buffer
            tstr_escaped = token_strs[i].replace('"', '""')
            buffer.append(f'{sample_idx},{i},{token_ids[i]},"{tstr_escaped}",{mi_values[i]:.6f}\n')

            # Save every save_every tokens
            if (i + 1) % args.save_every == 0:
                with open(csv_path, 'a') as f:
                    f.writelines(buffer)
                buffer = []
                print(f"  ✓ Saved up to token {i + 1}")

        # Save remaining buffer
        if buffer:
            with open(csv_path, 'a') as f:
                f.writelines(buffer)
            print(f"  ✓ Saved remaining {len(buffer)} tokens")

        # Store results
        results[sample_idx] = {
            'mi_values': mi_values,
            'token_ids': token_ids,
            'token_strs': token_strs,
            'gt_answer': gt_answer,
            'gt_solution': gt_solution,
            'gt_mode': gt_mode,
            'num_tokens': len(token_ids)
        }

        # Print summary
        print(f"  MI stats: mean={mi_values.mean():.4f}, std={mi_values.std():.4f}, "
              f"min={mi_values.min():.4f}, max={mi_values.max():.4f}")

    # Save pth results at the end
    pth_filename = f"mi_first{args.max_tokens}_layer{layer_idx}_{gt_mode}.pth"
    output_path = os.path.join(args.output_dir, pth_filename)
    torch.save({
        'results': results,
        'config': vars(args)
    }, output_path)
    print(f"\nSaved pth results to {output_path}")
    print(f"CSV already saved to {csv_path}")


if __name__ == "__main__":
    main()