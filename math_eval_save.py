import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
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
    parser.add_argument("--min_p", default=0., type=float)
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
    
    # Think answers saving/loading
    parser.add_argument("--save_think_answers", action="store_true", default=False,
                       help="Save think answers to a file for later reuse")
    parser.add_argument("--load_think_answers", type=str, default="",
                       help="Path to a file containing think answers to load instead of generating")

    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


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
    chunk_param = f"H{args.num_think_chunks}" if args.max_tokens_per_think_chunk <= 0 else f"tokPerChunk{args.max_tokens_per_think_chunk}"
    out_file_prefix = f"{args.split}_{args.prompt_type}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_n{args.n_sampling}{chunk_param}m{args.num_answers_per_chunk}.jsonl"
    
    # Create think answers filename based on the parameters that affect think generation
    think_answers_file = f"{output_dir}/{data_name}/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_n{args.n_sampling}_think_answers.json"
    
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return examples, out_file, think_answers_file


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
        seed=args.seed,
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
    examples, out_file, think_answers_file = prepare_data(data_name, args)
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
    # Either load existing think answers or generate new ones
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_call,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,
    )
    if args.load_think_answers:
        # Load existing think answers from file
        think_answers_path = args.load_think_answers
        print(f"Loading think answers from {think_answers_path}")
        try:
            with open(think_answers_path, 'r') as f:
                think_answers_data = json.load(f)
                loaded_think_answers = think_answers_data['think_answers']
                loaded_n_sampling = think_answers_data.get('n_sampling', 1)
                loaded_num_samples = think_answers_data.get('num_samples', len(loaded_think_answers) // loaded_n_sampling)

            # Check if we need all or just a subset of the loaded think answers
            if len(loaded_think_answers) >= len(prompts):
                # We have enough think answers, but need to select the correct subset
                # based on the interleaved pattern of n_sampling
                if args.n_sampling < loaded_n_sampling:
                    # We need to take a subset of the loaded answers
                    print(f"Loaded think answers have n_sampling={loaded_n_sampling}, but current n_sampling={args.n_sampling}")
                    print(f"Selecting appropriate subset of loaded think answers")
                    
                    # Create an interleaved subset of the loaded think answers
                    think_answers = []
                    for i in range(len(samples)):
                        # For each sample, select the first n_sampling answers
                        for j in range(args.n_sampling):
                            original_index = i * loaded_n_sampling + j
                            if original_index < len(loaded_think_answers):
                                think_answers.append(loaded_think_answers[original_index])
                    
                    print(f"Selected {len(think_answers)} out of {len(loaded_think_answers)} loaded think answers")
                else:
                    # Same n_sampling, use all loaded think answers
                    think_answers = loaded_think_answers
                    print(f"Using all {len(think_answers)} loaded think answers")
            else:
                # Handle the case where current n_sampling is greater than what was saved
                # Check if we're using the same dataset but different n_sampling
                if len(samples) == loaded_num_samples and args.n_sampling > loaded_n_sampling:
                    # We can reuse the loaded answers and generate only the additional ones needed
                    print(f"Loaded think answers have n_sampling={loaded_n_sampling}, but current n_sampling={args.n_sampling}")
                    print(f"Will use loaded answers and generate additional {len(prompts) - len(loaded_think_answers)} answers")
                    
                    # Create new prompts only for the additional samples needed
                    additional_prompts = []
                    for i, sample in enumerate(samples):
                        for s in range(loaded_n_sampling, args.n_sampling):
                            additional_prompts.append(sample["prompt"])
                    
                    # Generate additional think answers
                    additional_outputs = llm.generate(additional_prompts, sampling_params)
                    additional_outputs = sorted(additional_outputs, key=lambda x: int(x.request_id))
                    additional_think_answers = [output.outputs[0].text for output in additional_outputs]
                    
                    # Combine loaded and new think answers in the correct interleaved pattern
                    think_answers = []
                    for i in range(len(samples)):
                        # Add the loaded think answers for this sample
                        for j in range(loaded_n_sampling):
                            original_index = i * loaded_n_sampling + j
                            if original_index < len(loaded_think_answers):
                                think_answers.append(loaded_think_answers[original_index])
                        
                        # Add the additional think answers for this sample
                        for j in range(args.n_sampling - loaded_n_sampling):
                            additional_index = i * (args.n_sampling - loaded_n_sampling) + j
                            if additional_index < len(additional_think_answers):
                                think_answers.append(additional_think_answers[additional_index])
                    
                    assert len(think_answers) == len(prompts), f"Think answers count mismatch: {len(think_answers)} vs {len(prompts)}"
                else:
                    # The mismatch is due to something other than n_sampling
                    # (like different dataset size or parameters), so generate all new answers
                    print(f"Loaded think answers count ({len(loaded_think_answers)}) doesn't match required count ({len(prompts)})")
                    print(f"And this doesn't appear to be just a difference in n_sampling. Generating all new answers.")
                    think_answers = None
            
            print(f"Successfully loaded think answers")
        except Exception as e:
            print(f"Error loading think answers: {e}")
            print("Falling back to generating new think answers")
            think_answers = None
    else:
        think_answers = None
    

    if think_answers is None:
        # Generate new think answers
        outputs = llm.generate(prompts, sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        think_answers = [output.outputs[0].text for output in outputs]
        assert len(think_answers) == len(prompts)
        
        # Save think answers if requested
        if args.save_think_answers:
            think_answers_data = {
                'data_name': data_name,
                'model': args.model_name_or_path,
                'split': args.split,
                'seed': args.seed,
                'temperature': args.temperature,
                'n_sampling': args.n_sampling,
                'num_samples': len(samples),
                'num_prompts': len(prompts),
                'think_answers': think_answers
            }
            try:
                with open(think_answers_file, 'w') as f:
                    json.dump(think_answers_data, f)
                print(f"Saved think answers to {think_answers_file}")
            except Exception as e:
                print(f"Error saving think answers: {e}")

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
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_answer,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,
    )

    # original, i.e. m=1, H=1
    if args.num_think_chunks == 1 and args.num_answers_per_chunk == 1:
        # Extract predictions
        ori_results = [extract_pred_and_parse(answer, data_name) for answer in ori_answers]

        # Organize into 3D structure (just with dimensions of [n, 1, 1])
        all_completions = []
        all_results = []
        
        for s in range(len(think_answers)):
            # Each sample has one chunk with one answer
            sample_completions = [[ori_answers[s]]]
            sample_results = [[ori_results[s]]]
            
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
            
            # Process predictions and scores using the 3D scoring function
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
                "think_sums": sample_completions,  # This will be [n, 1, 1]
                "gt": new_gt,
                "all_sub_preds": all_sub_preds, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "all_sub_scores": all_sub_scores, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "chunk_maj_preds": chunk_maj_preds, # shape [n, H]. Use maj to squeeze m dimension.
                "chunk_maj_scores": chunk_maj_scores, # shape [n, H]. Use maj to squeeze m dimension.
                "sample_maj_preds": sample_maj_preds, # shape [n]. Use maj to further squeeze H dimension.
                "sample_maj_scores": sample_maj_scores, # shape [n]. Use maj to further squeeze H dimension.
            })
            all_samples.append(sample)

    # H=1, m>1 (single chunk but multiple answers)
    elif args.num_think_chunks == 1 and args.num_answers_per_chunk > 1:
        # Generate multiple answers for the full reasoning
        final_prompts = []
        for r, think_answer in enumerate(think_answers):
            if "</think>" not in think_answer:
                final_prompt = prompts[r] + think_answer + "\n</think>\n\n"
            else:
                final_prompt = prompts[r] + think_answer.split("</think>")[0] + "</think>"
            
            # Add additional prompts for generating more answers (minus 1 because we already have one)
            for _ in range(args.num_answers_per_chunk - 1):
                final_prompts.append(final_prompt)
        
        # Generate additional answers for the full reasoning
        final_outputs = llm.generate(final_prompts, answer_sampling_params)
        final_outputs = sorted(final_outputs, key=lambda x: int(x.request_id))
        final_answers = [output.outputs[0].text for output in final_outputs]
        assert len(final_answers) == len(final_prompts)
        
        # Extract predictions
        ori_results = [extract_pred_and_parse(answer, data_name) for answer in ori_answers]
        final_results = [extract_pred_and_parse(answer, data_name) for answer in final_answers]
        
        # Organize into 3D structure (but with only one chunk dimension)
        all_completions = []
        all_results = []
        
        final_idx = 0
        for s in range(len(think_answers)):
            # For each sample, create one chunk with multiple answers
            sample_completions = [[ori_answers[s]]]
            sample_results = [[ori_results[s]]]
            
            # Add additional completions
            for a in range(args.num_answers_per_chunk - 1):
                sample_completions[0].append(final_answers[final_idx])
                sample_results[0].append(final_results[final_idx])
                final_idx += 1
            
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
            
            # Process predictions and scores using the 3D scoring function
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
                "think_sums": sample_completions,  # This will be [n, 1, m]
                "gt": new_gt,
                "all_sub_preds": all_sub_preds, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "all_sub_scores": all_sub_scores, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "chunk_maj_preds": chunk_maj_preds, # shape [n, H]. Use maj to squeeze m dimension.
                "chunk_maj_scores": chunk_maj_scores, # shape [n, H]. Use maj to squeeze m dimension.
                "sample_maj_preds": sample_maj_preds, # shape [n]. Use maj to further squeeze H dimension.
                "sample_maj_scores": sample_maj_scores, # shape [n]. Use maj to further squeeze H dimension.
            })
            all_samples.append(sample)
    
    # 3D majority voting
    else:
        reasonings_tok = [tokenizer.encode(think_answer.split("</think>")[0])[1:] for think_answer in think_answers]
        
        # Create prompts for different chunks of reasoning
        chunk_prompts = []
        for r, reasoning in enumerate(reasonings_tok):
            # Create partial prompts by cutting reasoning at different points, i.e. H 
            splits = [reasoning[: i * len(reasoning) // args.num_think_chunks] for i in range(1, args.num_think_chunks)]
            
            # For each split, create multiple prompts for generating different answers
            for split in splits:
                partial_prompt = prompts[r] + tokenizer.decode(split) + "\n</think>\n\n"
                # Add this prompt multiple times based on num_answers_per_chunk, i.e. m
                for _ in range(args.num_answers_per_chunk):
                    chunk_prompts.append(partial_prompt)
        
        # Generate completions for all chunk prompts
        chunk_outputs = llm.generate(chunk_prompts, answer_sampling_params)
        chunk_outputs = sorted(chunk_outputs, key=lambda x: int(x.request_id))
        chunk_answers = [output.outputs[0].text for output in chunk_outputs]
        assert len(chunk_answers) == len(chunk_prompts)
        
        # Also generate multiple answers for the original full reasoning
        final_prompts = []
        for r, think_answer in enumerate(think_answers):
            if "</think>" not in think_answer:
                final_prompt = prompts[r] + think_answer + "\n</think>\n\n"
            else:
                final_prompt = prompts[r] + think_answer.split("</think>")[0] + "</think>"
            
            # Add additional prompts for generating more answers (minus 1 because we already have one)
            for _ in range(args.num_answers_per_chunk - 1):
                final_prompts.append(final_prompt)
        
        # Generate additional answers for the full reasoning
        if args.num_answers_per_chunk > 1:
            final_outputs = llm.generate(final_prompts, answer_sampling_params)
            final_outputs = sorted(final_outputs, key=lambda x: int(x.request_id))
            final_answers = [output.outputs[0].text for output in final_outputs]
            assert len(final_answers) == len(final_prompts)
        
        # Organize all the completions into a 3D structure: [n_sampling, num_chunks, num_answers], i.e [n, H, m]
        all_completions = []
        all_results = []
        
        # Calculate indices
        chunk_idx = 0
        final_idx = 0
        
        for s in range(len(think_answers)):  # For each initial sample
            sample_completions = []
            sample_results = []
            
            # Get completions for intermediate chunks
            for c in range(args.num_think_chunks - 1):  # For each chunk except the last
                chunk_completions = []
                chunk_results = []
                
                for a in range(args.num_answers_per_chunk):  # For each answer
                    # Get the completion text
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
                "all_sub_preds": all_sub_preds, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "all_sub_scores": all_sub_scores, # shape [n, H, m], [n, -1, 0] is for H=1 and m=1
                "chunk_maj_preds": chunk_maj_preds, # shape [n, H]. Use maj to squeeze m dimension.
                "chunk_maj_scores": chunk_maj_scores, # shape [n, H]. Use maj to squeeze m dimension.
                "sample_maj_preds": sample_maj_preds, # shape [n]. Use maj to further squeeze H dimension.
                "sample_maj_scores": sample_maj_scores, # shape [n]. Use maj to further squeeze H dimension.
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