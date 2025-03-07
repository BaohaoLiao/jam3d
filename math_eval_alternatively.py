import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils.data import load_data, construct_prompt, save_jsonl
from utils.parser import parse_question, parse_ground_truth, extract_pred_and_parse
from utils.eval import obtain_3d_sub_scores_and_preds, obtain_3d_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--disable_chunked_prefill', action='store_true', default=False)
    parser.add_argument('--max_model_len', type=int, default=64000)

    # 3D maj
    parser.add_argument("--n_sampling", default=1, type=int, help="I.e. n")
    parser.add_argument("--num_think_chunks", default=1, type=int,
                        help="Evenly split the original think into multiple chunks, i.e. H")
    parser.add_argument("--num_answers_per_chunk", default=1, type=int,
                       help="Number of answer variations to generate for each reasoning chunk, i.e. m")
    parser.add_argument("--max_tokens_per_answer", default=512, type=int)
    parser.add_argument("--answer_temperature", default=0.6, type=float,
                       help="Temperature to use when generating multiple answers per chunk")
    parser.add_argument("--answer_top_p", default=0.95, type=float,
                       help="Top-p value to use when generating multiple answers per chunk")

    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def find_alternative_splits(reasoning_text):
    """
    Split the reasoning text at each occurrence of the word 'Alternatively'.
    Returns a list of text segments corresponding to the different alternatives.
    """
    # First, let's split the text into segments
    segments = []
    
    # Handle the case where there are no 'Alternatively' occurrences
    if "Alternatively" not in reasoning_text:
        segments.append(reasoning_text)
        return segments
    
    # Split the text by 'Alternatively'
    parts = reasoning_text.split("Alternatively")
    
    # Add the first part as is
    segments.append(parts[0])
    
    # For each subsequent part, add "Alternatively" back at the beginning
    # to maintain the context for the alternative solution
    for i in range(1, len(parts)):
        # Create a cumulative segment that includes all previous parts and this alternative
        cumulative_text = segments[0]
        for j in range(1, i+1):
            cumulative_text += "Alternatively" + parts[j]
        segments.append(cumulative_text)
    
    return segments


def prepare_data(data_name, args):
    if "math500_level" in data_name:
        level = int(data_name.strip()[-1])
        examples = load_data("math500", args.split, args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_n{args.n_sampling}H{args.num_think_chunks}m{args.num_answers_per_chunk}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    for data_name, result in zip(data_list, results):
        print("=" * 50)
        print(data_name)
        print(result)


def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        if i == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": gt,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]

    # start inference
    start_time = time.time()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            skip_special_tokens=False,
            seed=args.seed,
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    think_answers = [output.outputs[0].text for output in outputs]
    assert len(think_answers) == len(prompts)

    ori_answers = []
    for think_answer in think_answers:
        if "</think>" not in think_answer:  # answer is not generated
            ori_answers.append(think_answer)
        else:
            ori_answers.append(think_answer.split("</think>")[-1])

    # For 2D or 3D majority voting
    answer_sampling_params = SamplingParams(
        temperature=args.answer_temperature,
        top_p=args.answer_top_p,
        max_tokens=args.max_tokens_per_answer,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,
    )

    chunk_prompts = []
    split_info = []  # To keep track of number of chunks for each sample
    
    for r, think_answer in enumerate(think_answers):
        # Extract the reasoning part
        if "</think>" not in think_answer:
            reasoning = think_answer
        else:
            reasoning = think_answer.split("</think>")[0]
        
        # Find splits based on "Alternatively"
        splits = find_alternative_splits(reasoning)
        split_info.append(len(splits))
        
        # For each split (except the last which is the full reasoning), create prompts
        for split in splits[:-1]:
            partial_prompt = prompts[r] + split + "\n</think>\n\n"
            for _ in range(args.num_answers_per_chunk):
                chunk_prompts.append(partial_prompt)
    
    # Generate completions for all chunk prompts
    if chunk_prompts:  # Only if we have any split chunks
        chunk_outputs = llm.generate(chunk_prompts, answer_sampling_params)
        chunk_outputs = sorted(chunk_outputs, key=lambda x: int(x.request_id))
        chunk_answers = [output.outputs[0].text for output in chunk_outputs]
        assert len(chunk_answers) == len(chunk_prompts)
    else:
        chunk_answers = []

    # Also generate multiple answers for the original full reasoning
    final_prompts = []
    for r, think_answer in enumerate(think_answers):
        if "</think>" not in think_answer:
            final_prompt = prompts[r] + think_answer + "\n</think>\n\n"
        else:
            final_prompt = prompts[r] + think_answer.split("</think>")[0] + "</think>"
        
        # Add prompts for generating more answers (minus 1 because we already have one)
        for _ in range(args.num_answers_per_chunk - 1):
            final_prompts.append(final_prompt)
    
    # Generate additional answers for the full reasoning
    if args.num_answers_per_chunk > 1 and final_prompts:
        final_outputs = llm.generate(final_prompts, answer_sampling_params)
        final_outputs = sorted(final_outputs, key=lambda x: int(x.request_id))
        final_answers = [output.outputs[0].text for output in final_outputs]
        assert len(final_answers) == len(final_prompts)
    else:
        final_answers = []

    # Organize completions into 3D structure: [n_sampling, num_alternative_chunks, num_answers]
    all_completions = []
    all_results = []
    
    # Calculate indices
    chunk_idx = 0
    final_idx = 0
    
    for s in range(len(think_answers)):  # For each initial sample
        sample_completions = []
        sample_results = []
        
        num_splits = split_info[s]
        
        # Get completions for intermediate chunks (all but the last)
        for c in range(num_splits - 1):
            chunk_completions = []
            chunk_results = []
            
            for a in range(args.num_answers_per_chunk):
                if chunk_idx < len(chunk_answers):
                    completion = chunk_answers[chunk_idx]
                    chunk_idx += 1
                    
                    # Parse the completion
                    result = extract_pred_and_parse(completion, data_name)
                    
                    # Add to lists
                    chunk_completions.append(completion)
                    chunk_results.append(result)
            
            sample_completions.append(chunk_completions)
            sample_results.append(chunk_results)
        
        # Get completions for final chunk (original full reasoning)
        final_completions = [ori_answers[s]]  # Start with the original completion
        final_results = [extract_pred_and_parse(ori_answers[s], data_name)]
        
        # Add additional completions if num_answers_per_chunk > 1
        if args.num_answers_per_chunk > 1:
            for a in range(args.num_answers_per_chunk - 1):
                if final_idx < len(final_answers):
                    completion = final_answers[final_idx]
                    final_idx += 1
                    
                    # Parse the completion
                    result = extract_pred_and_parse(completion, data_name)
                    
                    # Add to lists
                    final_completions.append(completion)
                    final_results.append(result)
        
        sample_completions.append(final_completions)
        sample_results.append(final_results)
        
        all_completions.append(sample_completions)
        all_results.append(sample_results)
    
    time_use = time.time() - start_time
    
    # Process results and put back into samples
    all_samples = []
    for i, sample in enumerate(samples):
        # Get completions and results for this sample
        sample_completions = []
        sample_results = []
        
        for s in range(args.n_sampling):
            idx = i * args.n_sampling + s
            sample_completions.append(all_completions[idx])
            sample_results.append(all_results[idx])
        
        sample.pop("prompt")
        
        # Process predictions and scores
        new_gt, \
            all_sub_preds, \
            all_sub_scores, \
            chunk_maj_preds, \
            chunk_maj_scores, \
            sample_maj_preds, \
            sample_maj_scores = obtain_3d_sub_scores_and_preds(
                sample["gt"], sample_results
            )
        
        sample.pop("gt")
        sample.update({
            "completion": think_answers[i * args.n_sampling : (i + 1) * args.n_sampling],
            "think_sums": sample_completions, # completion after </think>, [n, H, m]
            "gt": new_gt,
            "all_sub_preds": all_sub_preds, # shape [n, H, m]
            "all_sub_scores": all_sub_scores, # shape [n, H, m]
            "chunk_maj_preds": chunk_maj_preds, # shape [n, H]
            "chunk_maj_scores": chunk_maj_scores, # shape [n, H]
            "sample_maj_preds": sample_maj_preds, # shape [n]
            "sample_maj_scores": sample_maj_scores, # shape [n]
            "split_info": split_info[i * args.n_sampling : (i + 1) * args.n_sampling], # Number of chunks for each sample
        })
        all_samples.append(sample)

    # Process and calculate overall scores
    all_samples, result_json = obtain_3d_scores(samples=all_samples, n_sampling=args.n_sampling)
    print(result_json)

    # save outputs
    save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minute"] = (f"{int(time_use // 60)}:{int(time_use % 60):02d}")

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)