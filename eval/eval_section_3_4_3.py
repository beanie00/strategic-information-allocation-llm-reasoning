"""Reproduce Section 3.4.3 test-time control experiments.

This script adds a paper-specific evaluation entrypoint on top of the
existing eval utilities. It supports four conditions:

1. baseline
2. suppress: block paper-defined epistemic tokens with logit bias
3. induce: prepend the Appendix G.2.1 few-shot prompt (1-shot or 2-shot)
4. insert_wait: generate once, then append the paper-style "Wait ... is
   not the answer. Please think again." intervention and continue.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from math import comb
from typing import Any, Dict, List, Sequence

import vllm.envs as envs
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.data_loader import load_data
from utils.epistemic_controls import (
    EPISTEMIC_TOKEN_STRINGS,
    build_epistemic_logit_bias,
    build_wait_intervention_message,
    count_epistemic_tokens,
    get_induction_few_shot_prompt,
    resolve_fixed_prefix,
)
from utils.grader import check_is_correct
from utils.parser import extract_answer, parse_ground_truth, parse_question
from utils.utils import set_seed


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_list(arg: str) -> List[str]:
    return arg.split(",")


def save_completions(completions, filepath: str) -> None:
    with open(filepath, "wb") as file:
        pickle.dump(completions, file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path or HF id")
    parser.add_argument(
        "--control_mode",
        type=str,
        default="baseline",
        choices=["baseline", "suppress", "induce", "insert_wait"],
        help="Section 3.4.3 control condition to run",
    )
    parser.add_argument(
        "--induction_shots",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of Appendix G.2.1 examples to prepend when control_mode=induce",
    )
    parser.add_argument(
        "--fixed_prefix",
        type=str,
        default="",
        help="Optional fixed response prefix appended after the chat template",
    )
    parser.add_argument(
        "--disable_auto_fixed_prefix",
        action="store_true",
        help='Disable the paper-style automatic "Okay, so I " prefix for Qwen3-14B-Base in induce mode',
    )
    parser.add_argument(
        "--wait_followup_max_tokens",
        type=int,
        default=None,
        help="Generation budget for the second pass in insert_wait mode; defaults to --max_tokens",
    )
    parser.add_argument("--n_sampling", type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"), type=str)
    parser.add_argument("--data_name", type=str, default="aime", help="Benchmark name")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--start_idx", type=int, default=0, help="data[start:end]")
    parser.add_argument("--end_idx", type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default=os.path.join(SCRIPT_DIR, "prompts"), type=str)
    parser.add_argument("--surround_with_messages", action="store_true", default=True)
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(SCRIPT_DIR, "outputs_section_3_4_3"),
        type=str,
    )
    parser.add_argument("--output_tag", default="", type=str, help="Optional suffix for output filenames")
    parser.add_argument("--stop", type=parse_list)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default="auto", type=str)
    parser.add_argument(
        "--gpu_memory_utilization",
        default=0.96,
        type=float,
        help="vLLM gpu_memory_utilization passed to LLM(...)",
    )
    parser.add_argument(
        "--max_model_len",
        default=None,
        type=int,
        help="Optional vLLM max_model_len override for smaller GPUs",
    )
    parser.add_argument(
        "--completions_save_dir",
        default=os.path.join(SCRIPT_DIR, "completions_section_3_4_3"),
        type=str,
    )
    args = parser.parse_args()

    args.top_p = 1.0 if args.temperature == 0 else args.top_p
    if args.wait_followup_max_tokens is None:
        args.wait_followup_max_tokens = args.max_tokens
    print(f"current stop list: {args.stop}")
    return args


def get_conversation_prompt_by_messages(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_three_prompt(prompt_dir: str, prompt_type: str, data_name: str):
    file_path = os.path.join(prompt_dir, prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_prompt_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "system_prompt"):
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    if not hasattr(module, "few_shot_prompt"):
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    if not hasattr(module, "question_format"):
        raise AttributeError(f"'question_format' not found in {file_path}")

    return module.system_prompt, module.few_shot_prompt, module.question_format


def make_sampling_params(args, n: int, max_tokens: int, logit_bias=None):
    return SamplingParams(
        temperature=args.temperature,
        max_tokens=max_tokens,
        n=n,
        top_p=args.top_p,
        logit_bias=logit_bias,
    )


def model_slug(model_name_or_path: str) -> str:
    parts = [part for part in model_name_or_path.split("/") if part]
    return "/".join(parts[-3:]) if parts else model_name_or_path


def control_slug(args) -> str:
    slug = args.control_mode
    if args.control_mode == "induce":
        slug += f"_{args.induction_shots}shot"
    if args.output_tag:
        slug += f"_{args.output_tag}"
    return slug


def build_prompt_batch(args, tokenizer, examples: Sequence[Dict[str, Any]]):
    system_prompt, base_few_shot_prompt, question_format = get_three_prompt(
        args.prompt_file_path,
        args.prompt_type,
        args.data_name,
    )
    fixed_prefix = resolve_fixed_prefix(
        args.model_name_or_path,
        args.control_mode,
        user_prefix=args.fixed_prefix,
        disable_auto_prefix=args.disable_auto_fixed_prefix,
    )

    prompt_batch: List[str] = []
    for example in tqdm(examples, total=len(examples), desc="building prompts"):
        question = parse_question(example, args.data_name)

        if args.control_mode == "induce":
            effective_few_shot_prompt = get_induction_few_shot_prompt(args.induction_shots)
        elif args.use_few_shot:
            effective_few_shot_prompt = base_few_shot_prompt
        else:
            effective_few_shot_prompt = ""

        cur_prompt = effective_few_shot_prompt + question_format.format(question=question)

        if args.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt},
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)

        if fixed_prefix:
            cur_prompt += fixed_prefix

        prompt_batch.append(cur_prompt)

    return prompt_batch, fixed_prefix


def generate_standard_epoch(
    llm,
    prompt_batch: Sequence[str],
    sampling_params: SamplingParams,
    save_path: str,
):
    completions = llm.generate(list(prompt_batch), sampling_params)
    save_completions(completions, save_path)
    return completions


def generate_insert_wait_epoch(
    llm,
    tokenizer,
    prompt_batch: Sequence[str],
    initial_sampling_params: SamplingParams,
    followup_sampling_params: SamplingParams,
    stage1_save_path: str,
    stage2_save_path: str,
):
    initial_completions = llm.generate(list(prompt_batch), initial_sampling_params)
    save_completions(initial_completions, stage1_save_path)

    intervention_records = []
    for example_idx, completion in enumerate(initial_completions):
        for sample_idx, output in enumerate(completion.outputs):
            initial_text = output.text
            initial_answer = extract_answer(initial_text)
            intervention = build_wait_intervention_message(initial_answer)
            followup_prompt = prompt_batch[example_idx] + initial_text.rstrip() + "\n\n" + intervention
            intervention_records.append(
                {
                    "example_idx": example_idx,
                    "sample_idx": sample_idx,
                    "initial_text": initial_text,
                    "initial_answer": initial_answer,
                    "intervention": intervention,
                    "followup_prompt": followup_prompt,
                }
            )

    followup_prompts = [record["followup_prompt"] for record in intervention_records]
    followup_completions = llm.generate(followup_prompts, followup_sampling_params)
    save_completions(followup_completions, stage2_save_path)

    outputs_by_example: List[List[Dict[str, Any]]] = [[] for _ in range(len(prompt_batch))]
    for record, completion in zip(intervention_records, followup_completions):
        continuation = completion.outputs[0].text
        final_response = record["initial_text"].rstrip() + "\n\n" + record["intervention"] + continuation
        final_answer = extract_answer(final_response)
        outputs_by_example[record["example_idx"]].append(
            {
                "response": final_response,
                "initial_response": record["initial_text"],
                "continuation": continuation,
                "intervention": record["intervention"].strip(),
                "initial_answer": record["initial_answer"],
                "final_answer": final_answer,
                "token_length": len(tokenizer.encode(final_response, add_special_tokens=False)),
            }
        )

    return outputs_by_example


def compute_pass_at_k(is_correct_list: Sequence[bool], k: int) -> float:
    correct_answers = sum(is_correct_list)
    n = len(is_correct_list)
    if correct_answers == 0:
        return 0.0
    if n - correct_answers < k:
        return 1.0
    return 1.0 - (comb(n - correct_answers, k) / comb(n, k))


def main():
    args = parse_args()
    set_seed(args.seed)

    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx : args.end_idx]

    print(f"current eval model: {args.model_name_or_path}")
    print(f"control mode: {args.control_mode}")
    print(f"num examples: {len(examples)}")
    print(f"gpu_memory_utilization: {args.gpu_memory_utilization}")
    print(f"max_model_len: {args.max_model_len}")

    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")

    model_name = model_slug(args.model_name_or_path)
    out_file_prefix = f"{args.split}_{args.prompt_type}_{control_slug(args)}_t{args.temperature}"
    out_file = (
        f"{args.output_dir}/{model_name}/{args.data_name}/"
        f"{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl"
    )
    if os.path.exists(out_file):
        print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
        return

    os.makedirs(f"{args.output_dir}/{model_name}/{args.data_name}", exist_ok=True)
    os.makedirs(f"{args.completions_save_dir}/{model_name}/{args.data_name}", exist_ok=True)

    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0"
    print(f"available_gpus: {available_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    prompt_batch, fixed_prefix = build_prompt_batch(args, tokenizer, examples)
    print(prompt_batch[0])

    logit_bias = None
    suppressed_surfaces = {}
    if args.control_mode == "suppress":
        logit_bias, suppressed_surfaces = build_epistemic_logit_bias(tokenizer, EPISTEMIC_TOKEN_STRINGS)
        print(f"resolved {len(logit_bias)} biased token ids for suppression")

    llm_kwargs = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": len(available_gpus),
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    standard_sampling_params = make_sampling_params(
        args,
        n=factor,
        max_tokens=args.max_tokens,
        logit_bias=logit_bias,
    )
    wait_followup_sampling_params = make_sampling_params(
        args,
        n=1,
        max_tokens=args.wait_followup_max_tokens,
        logit_bias=None,
    )

    file_outputs: List[Dict[str, Any]] = []
    total_tokens = 0
    total_responses = 0
    total_epistemic_tokens = 0
    pass_at_k_list: List[float] = []
    correct_cnt = 0

    for cur_generation_epoch in range(generation_epoch):
        base_save_prefix = (
            f"{args.completions_save_dir}/{model_name}/{args.data_name}/"
            f"{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}"
            f"_gen_round{cur_generation_epoch}"
        )

        if args.control_mode == "insert_wait":
            outputs_by_example = generate_insert_wait_epoch(
                llm=llm,
                tokenizer=tokenizer,
                prompt_batch=prompt_batch,
                initial_sampling_params=standard_sampling_params,
                followup_sampling_params=wait_followup_sampling_params,
                stage1_save_path=base_save_prefix + "_stage1.pkl",
                stage2_save_path=base_save_prefix + "_stage2.pkl",
            )
        else:
            completions = generate_standard_epoch(
                llm=llm,
                prompt_batch=prompt_batch,
                sampling_params=standard_sampling_params,
                save_path=base_save_prefix + ".pkl",
            )
            outputs_by_example = []
            for completion in completions:
                example_outputs = []
                for output in completion.outputs:
                    text = output.text
                    example_outputs.append(
                        {
                            "response": text,
                            "initial_response": None,
                            "continuation": None,
                            "intervention": None,
                            "initial_answer": None,
                            "final_answer": extract_answer(text),
                            "token_length": len(tokenizer.encode(text, add_special_tokens=False)),
                        }
                    )
                outputs_by_example.append(example_outputs)

        for i, example in enumerate(examples):
            question = parse_question(example, args.data_name)
            responses = [record["response"] for record in outputs_by_example[i]]
            generated_answers = [record["final_answer"] for record in outputs_by_example[i]]
            response_token_lengths = [record["token_length"] for record in outputs_by_example[i]]
            epistemic_stats = [count_epistemic_tokens(text) for text in responses]
            epistemic_token_totals = [stats[0] for stats in epistemic_stats]
            epistemic_token_breakdowns = [stats[1] for stats in epistemic_stats]

            total_tokens += sum(response_token_lengths)
            total_responses += len(responses)
            total_epistemic_tokens += sum(epistemic_token_totals)

            if cur_generation_epoch == 0:
                file_outputs.append(
                    {
                        "question": question,
                        "generated_responses": responses,
                        "generated_answers": generated_answers,
                        "response_token_lengths": response_token_lengths,
                        "epistemic_token_counts": epistemic_token_totals,
                        "epistemic_token_breakdowns": epistemic_token_breakdowns,
                        "initial_generated_responses": [
                            record["initial_response"] for record in outputs_by_example[i]
                        ],
                        "wait_interventions": [record["intervention"] for record in outputs_by_example[i]],
                    }
                )
                if "id" in example:
                    file_outputs[i]["id"] = example["id"]
                if "source" in example:
                    file_outputs[i]["source"] = example["source"]
            else:
                file_outputs[i]["generated_responses"] += responses
                file_outputs[i]["generated_answers"] += generated_answers
                file_outputs[i]["response_token_lengths"] += response_token_lengths
                file_outputs[i]["epistemic_token_counts"] += epistemic_token_totals
                file_outputs[i]["epistemic_token_breakdowns"] += epistemic_token_breakdowns
                file_outputs[i]["initial_generated_responses"] += [
                    record["initial_response"] for record in outputs_by_example[i]
                ]
                file_outputs[i]["wait_interventions"] += [
                    record["intervention"] for record in outputs_by_example[i]
                ]

    print("llm generate done")
    print(len(file_outputs))

    for i in tqdm(range(len(examples)), "check correct..."):
        d = examples[i]
        _, gt_ans = parse_ground_truth(d, args.data_name)
        generated_answers = file_outputs[i]["generated_answers"]
        is_correct_list = [check_is_correct(answer, gt_ans) for answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

        file_outputs[i]["gold_answer"] = gt_ans
        file_outputs[i]["is_correct"] = is_correct
        file_outputs[i]["answers_correctness"] = is_correct_list
        file_outputs[i]["avg_response_token_length"] = (
            sum(file_outputs[i]["response_token_lengths"]) / len(file_outputs[i]["response_token_lengths"])
        )
        file_outputs[i]["avg_epistemic_token_count"] = (
            sum(file_outputs[i]["epistemic_token_counts"]) / len(file_outputs[i]["epistemic_token_counts"])
        )

        if len(is_correct_list) > 1:
            pass_at_k = compute_pass_at_k(is_correct_list, args.k)
            pass_at_k_list.append(pass_at_k)
            file_outputs[i]["pass_at_k"] = pass_at_k

    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, "w", encoding="utf-8") as f:
        count = 0
        for d in tqdm(file_outputs, "writing generation to jsonl file..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        f.flush()
    os.rename(temp_out_file, out_file)

    acc = correct_cnt / len(examples) if examples else 0.0
    avg_epistemic_tokens_per_response = total_epistemic_tokens / total_responses if total_responses else 0.0
    avg_tokens_per_response = total_tokens / total_responses if total_responses else 0.0

    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {acc:.4f}")
    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{args.k}: {sum(pass_at_k_list):.4f}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        average_pass_at_k = None
        print(f"Pass@1: {correct_cnt}/{len(examples)} = {acc:.4f}")
    print(f"Average response token length: {avg_tokens_per_response:.2f}")
    print(f"Average epistemic tokens per response: {avg_epistemic_tokens_per_response:.2f}")

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "control_mode": args.control_mode,
        "induction_shots": args.induction_shots if args.control_mode == "induce" else None,
        "fixed_prefix": fixed_prefix,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "data_name": args.data_name,
        "split": args.split,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "n_sampling": args.n_sampling,
        "k": args.k,
        "num_examples": len(examples),
        "accuracy": acc,
        "pass_at_k": average_pass_at_k,
        "avg_tokens_per_response": avg_tokens_per_response,
        "avg_epistemic_tokens_per_response": avg_epistemic_tokens_per_response,
        "suppressed_token_id_count": len(logit_bias or {}),
        "suppressed_surface_examples": {
            token: surfaces[:10]
            for token, surfaces in suppressed_surfaces.items()
            if surfaces
        },
        "output_jsonl": out_file,
    }
    summary_path = out_file.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
