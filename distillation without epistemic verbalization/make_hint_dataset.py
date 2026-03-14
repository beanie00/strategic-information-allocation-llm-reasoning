# Teacher re-generation for JSON dataset (resume-safe)
# Validation loop: regenerate until "GOOD" or max retries

import os
import time
import math
import json
from openai import OpenAI
from transformers import AutoTokenizer

TEACHER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

BATCH_SIZE = 100
MAX_NEW_TOKENS = 32784
TEMPERATURE = 0.4
MAX_RETRIES = 20

DATA_PATH = "../train/data/limo_v2.json"
OUTPUT_PATH = "../train/data/limo_hint_deepseek_qwen_32b.json"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

SGLANG_BASE_URL = "http://localhost:8002/v1"

EVAL_PROMPT_TEMPLATE = """You are an automatic solution evaluator.
Given a math problem, a proposed solution, and the ground-truth answer,
decide whether the proposed solution is correct.
Rules:
- If the solution is logically sound and the final conclusion is correct, output exactly: The solution is GOOD
- If the answer is incorrect, the solution contains logical gaps or errors, or the solver attempted to use a Python program, output exactly: The solution is BAD
- Do not provide any explanation.
- Do not output anything else.
Problem:
{question}
Proposed Solution:
{response}
Decision:
""".strip()


def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL,
        trust_remote_code=True,
    )

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    client = OpenAI(
        base_url=SGLANG_BASE_URL,
        api_key="EMPTY",
    )

    print("START!")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {DATA_PATH}")

    # Resume from existing output if available
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            done_data = json.load(f)
        done_count = len(done_data)
        print(f"Already processed: {done_count} records")
    else:
        done_data = []
        done_count = 0

    # Build pending records
    records = []

    for idx in range(done_count, len(data)):
        item = data[idx]
        question = item["instruction"]
        solution = item["output"]

        prompt = f"""You are given a QUESTION and its SOLUTION.

QUESTION:
{question}

SOLUTION:
{solution}

Using only the actual solving approach from the solution above — excluding any wandering or trial-and-error — as a reference, re-derive the result independently from scratch, step by step. Include all key equations and intermediate algebra. Do not express any uncertainty — never say "I think," "probably," or "it seems." State everything with full confidence.

Put your final answer within \\boxed{{}}.

Okay, so I
""".strip()

        records.append((idx, item, question, solution, prompt))

    TOTAL = len(records)
    print(f"Total remaining problems: {TOTAL}")

    if TOTAL == 0:
        print("Nothing to do.")
        return

    start_time = time.time()
    processed = 0
    num_batches = math.ceil(TOTAL / BATCH_SIZE)

    for batch_idx, i in enumerate(range(0, TOTAL, BATCH_SIZE)):
        batch = records[i:i + BATCH_SIZE]
        indices, items, questions, solutions, prompts = zip(*batch)

        valid_solutions = [None] * len(batch)
        retry_counts = [0] * len(batch)
        current_prompts = list(prompts)
        pending_indices = list(range(len(batch)))

        while pending_indices:
            gen_prompts = [current_prompts[j] for j in pending_indices]

            response = client.completions.create(
                model=TEACHER_MODEL,
                prompt=gen_prompts,
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )

            texts = [c.text for c in response.choices]

            # Build evaluation prompts
            eval_prompts = []
            eval_mapping = []

            for local_idx, text in enumerate(texts):
                batch_idx_in_pending = pending_indices[local_idx]
                teacher_text = "Okay, so I" + text
                question = questions[batch_idx_in_pending]

                eval_prompt = EVAL_PROMPT_TEMPLATE.format(
                    question=question,
                    response=teacher_text
                )
                eval_prompts.append(eval_prompt)
                eval_mapping.append((local_idx, batch_idx_in_pending, teacher_text))

            # Run evaluation
            eval_response = client.completions.create(
                model=TEACHER_MODEL,
                prompt=eval_prompts,
                temperature=0.0,
                max_tokens=50,
            )

            eval_texts = [c.text for c in eval_response.choices]

            # Process results
            new_pending = []
            for eval_idx, (local_idx, batch_idx_in_pending, teacher_text) in enumerate(eval_mapping):
                eval_text = eval_texts[eval_idx]

                if "GOOD" in eval_text:
                    valid_solutions[batch_idx_in_pending] = teacher_text
                    print(f"  Problem {indices[batch_idx_in_pending]} validated "
                          f"(retry={retry_counts[batch_idx_in_pending]})")
                else:
                    retry_counts[batch_idx_in_pending] += 1
                    if retry_counts[batch_idx_in_pending] < MAX_RETRIES:
                        new_pending.append(batch_idx_in_pending)
                        print(f"  Problem {indices[batch_idx_in_pending]} BAD, "
                              f"retrying ({retry_counts[batch_idx_in_pending]}/{MAX_RETRIES})")
                    else:
                        valid_solutions[batch_idx_in_pending] = teacher_text
                        print(f"  Problem {indices[batch_idx_in_pending]} max retries reached, "
                              f"using last solution")

            pending_indices = new_pending

        # Append completed batch to output
        for batch_idx_local, (idx, item, teacher_text) in enumerate(zip(indices, items, valid_solutions)):
            new_item = item.copy()
            new_item["output"] = teacher_text
            done_data.append(new_item)

        # Save after each batch
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(done_data, f, ensure_ascii=False, indent=2)

        processed += len(batch)
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        eta = (TOTAL - processed) / speed if speed > 0 else float("inf")

        print(
            f"[Batch {batch_idx+1:03d}/{num_batches}] "
            f"Processed: {processed}/{TOTAL} | "
            f"Speed: {speed:.2f} prob/s | "
            f"Elapsed: {fmt(elapsed)} | "
            f"ETA: {fmt(eta)} | Saved"
        )

    print("\nDONE: teacher generation with validation")


if __name__ == "__main__":
    main()