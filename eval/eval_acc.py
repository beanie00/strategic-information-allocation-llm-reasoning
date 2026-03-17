import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle


def parse_list(arg):
    return arg.split(',')


def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    # NOTE: No --k flag here. This script always evaluates with k = n_sampling,
    # i.e., pass@n (any correct among all n samples = pass).
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    args = parser.parse_args()

    # top_p must be 1 when using greedy decoding
    args.top_p = 1 if args.temperature == 0 else args.top_p
    print(f"current stop list: {args.stop}")
    return args


def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Dynamically import the prompt module
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")

    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")

    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")

    # Determine sampling factor and generation epochs
    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=factor,
        top_p=args.top_p,
    )

    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}'
    # Save results to avg_outputs directory
    out_file = f'avg_outputs/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'

    if os.path.exists(out_file):
        print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
        return
    os.makedirs(f'avg_outputs/{model_name}/{args.data_name}', exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)

    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP = "0.0.0.0"
    print(f"available_gpus: {available_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Build prompt batch
    prompt_batch = []
    for example in tqdm(examples, total=len(examples)):
        question = parse_question(example, args.data_name)
        system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)

        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)

        if args.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
    )

    # Generate responses across multiple epochs, tracking token lengths
    file_outputs = []
    correct_cnt = 0
    total_tokens = 0
    total_responses = 0

    for cur_generation_epoch in range(generation_epoch):
        completions_save_file = (
            f'{args.completions_save_dir}/{model_name}/{args.data_name}/'
            f'{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}'
            f'_gen_round{cur_generation_epoch}.pkl'
        )

        completions = llm.generate(prompt_batch, sampling_params)
        save_completions(completions, completions_save_file)

        for i in range(len(examples)):
            d = examples[i]
            question = parse_question(d, args.data_name)
            generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]

            # Compute token lengths per response
            response_token_lengths = []
            for response in generated_responses:
                tokens = tokenizer.encode(response, add_special_tokens=False)
                response_token_lengths.append(len(tokens))
                total_tokens += len(tokens)
                total_responses += 1

            if cur_generation_epoch == 0:
                file_outputs.append({
                    "question": question,
                    "generated_responses": generated_responses,
                    "response_token_lengths": response_token_lengths,
                })
                if "id" in d:
                    file_outputs[i]["id"] = d["id"]
                if "source" in d:
                    file_outputs[i]["source"] = d["source"]
            else:
                file_outputs[i]['generated_responses'] += generated_responses
                file_outputs[i]['response_token_lengths'] += response_token_lengths

    print("llm generate done")
    print(len(file_outputs))

    # Evaluate correctness
    #
    # Metrics computed:
    #   - Pass@n: 1 if any of the n samples is correct, else 0 (equivalent to pass@k where k=n).
    #   - Avg@n: per-question accuracy averaged over n samples (= c/n, essentially a
    #     Monte Carlo estimate of pass@1, with more samples giving a more stable estimate).
    #     This is NOT a standard named metric; it does not change what is being measured as
    #     n grows, only the estimation variance.
    avg_at_n_list = []

    for i in tqdm(range(len(examples)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(resp, args.data_name) for resp in generated_responses]
        is_correct_list = [check_is_correct(ans, gt_ans) for ans in generated_answers]

        # Pass@n: any correct among all n samples
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list

        # Per-question average token length
        lengths = file_outputs[i]['response_token_lengths']
        file_outputs[i]['avg_response_token_length'] = sum(lengths) / len(lengths)

        # Avg@n: fraction of correct samples per question
        if len(is_correct_list) > 1:
            avg_at_n = sum(is_correct_list) / len(is_correct_list)
            avg_at_n_list.append(avg_at_n)
            file_outputs[i]['avg_at_n'] = avg_at_n

    # Write results to jsonl
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs, "writing generation to jsonl file..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        f.flush()
    os.rename(temp_out_file, out_file)

    avg_response_length = total_tokens / total_responses if total_responses > 0 else 0
    n = args.n_sampling

    print(f"Pass@{n}: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")
    print(f"Average Response Length (tokens): {avg_response_length:.2f}")

    if avg_at_n_list:
        overall_avg_at_n = sum(avg_at_n_list) / len(avg_at_n_list)
        print(f"Avg@{n}: {overall_avg_at_n:.4f}")
    else:
        print(f"Avg@1: {correct_cnt / len(examples):.4f}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)