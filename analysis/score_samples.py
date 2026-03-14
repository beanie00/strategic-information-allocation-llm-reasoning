import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-math-7b-instruct")
    parser.add_argument("--input_json", type=str, default="train/data/limo_v2.json")
    parser.add_argument("--output_json", type=str, default="analysis/results/Qwen3-14B-Base_limo_v2.json")
    parser.add_argument("--dtype", default="float16", type=str)
    return parser.parse_args()


def load_model(model_name_or_path, dtype="float16"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, system, instruction, user_input):
    messages = []

    if system and system.strip():
        messages.append({"role": "system", "content": system})

    user_content = instruction
    if user_input and user_input.strip():
        user_content = instruction + "\n" + user_input

    messages.append({"role": "user", "content": user_content})

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt_text


def compute_token_logprobs(tokenizer, model, prompt_text, generated_text):
    full_text = prompt_text + generated_text

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Shift: predict next token from current position
    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1]

    token_logprobs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    tokens = tokenizer.convert_ids_to_tokens(target_ids.squeeze(0).tolist())

    results = []
    for tok, tok_id, lp in zip(
        tokens,
        target_ids.squeeze(0).tolist(),
        token_logprobs.squeeze(0).tolist()
    ):
        results.append({
            "token": tok,
            "token_id": tok_id,
            "logprob": float(lp)
        })

    return results


def main():
    args = parse_args()

    print("Loading model...")
    tokenizer, model = load_model(args.model_name_or_path, args.dtype)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    output_data = []

    for data in tqdm(data_list, desc="Scoring"):
        system = data.get("system", "")
        instruction = data.get("instruction", "")
        user_input = data.get("input", "")
        generated_text = data.get("output", "")

        # Build chat prompt
        prompt_text = build_chat_prompt(
            tokenizer=tokenizer,
            system=system,
            instruction=instruction,
            user_input=user_input
        )

        # Compute token-level logprobs
        token_logprobs = compute_token_logprobs(
            tokenizer=tokenizer,
            model=model,
            prompt_text=prompt_text,
            generated_text=generated_text
        )

        # Exclude prompt tokens, keep answer only
        prompt_len = len(
            tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        answer_token_logprobs = token_logprobs[prompt_len - 1:]

        seq_logprob = sum(t["logprob"] for t in answer_token_logprobs)
        avg_logprob = seq_logprob / max(len(answer_token_logprobs), 1)

        output_data.append({
            **data,
            "scoring": {
                "sequence_logprob": seq_logprob,
                "avg_logprob": avg_logprob,
                "token_logprobs": answer_token_logprobs
            }
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved scored file to {args.output_json}")


if __name__ == "__main__":
    main()