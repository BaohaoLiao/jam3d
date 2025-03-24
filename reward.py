import os
import json
import argparse
from vllm import LLM


def parse_args():
    parser = argparse.ArgumentParser(description='Score outputs using a reward model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the output file (JSONL) to score')
    parser.add_argument('--output_file', type=str, help='Path to save scored outputs (default: input_file with _scored suffix)')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Reward model path or name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--max_num_seqs', type=int, default=128, help='Batch size for scoring')
    parser.add_argument('--max_model_len', type=int, default=4090, help='Set the reward to -1 for longer message.')
    return parser.parse_args()

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    fields_to_keep = ["idx", "think_sums_rewards", "gt", "all_sub_preds", "all_sub_scores", "sub_preds", "sub_scores"]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Create a new dictionary with only the required fields
            filtered_item = {}
            for field in fields_to_keep:
                if field in item:
                    if field == "sub_preds" or field == "sub_scores":
                        filtered_item["all_" + field] = item[field]
                    else:
                        filtered_item[field] = item[field]

            #filtered_item = {field: item[field] for field in fields_to_keep if field in item}
            f.write(json.dumps(filtered_item) + '\n')

def create_messages(query, response, is_prm):
    """Create messages for the reward model."""
    response = response.strip()
    if is_prm:
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(response.split("\n\n")) + "<extra_0>"},
        ]
    else:
        return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]


def process_think_sums(samples, reward_model, tokenizer, max_model_len, is_prm):
    """Process all think_sums across samples and calculate rewards."""
    print("Preparing data for reward scoring...")
    
    # Prepare data for reward scoring
    all_queries = []
    all_responses = []
    all_indices = []  # Track (sample_idx, n_idx, h_idx, m_idx)
    skipped_indices = []  # Track indices that exceed max length

    for sample_idx, sample in enumerate(samples):
        query = sample["question"]  # The original question
        think_sums = sample["think_sums"]  # 3D structure [n, H, m]
        
        for n_idx, n_sample in enumerate(think_sums):
            for h_idx, h_chunk in enumerate(n_sample):
                for m_idx, response in enumerate(h_chunk):
                    # Create messages to check token length
                    messages = create_messages(query, response, is_prm)
                    conversation_str = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    
                    # Check if this conversation exceeds the model's max length
                    token_length = len(tokenizer.encode(conversation_str, add_special_tokens=False))
                    if token_length > max_model_len:
                        print(f"Skipping conversation at index ({sample_idx}, {n_idx}, {h_idx}, {m_idx}) - Length: {token_length} tokens")
                        skipped_indices.append((sample_idx, n_idx, h_idx, m_idx))
                        continue
                    
                    all_queries.append(query)
                    all_responses.append(response)
                    all_indices.append((sample_idx, n_idx, h_idx, m_idx))

    print(f"Total responses to score: {len(all_responses)}")
    print(f"Skipped responses (too long): {len(skipped_indices)}")
    
    # Prepare conversation strings for reward model
    all_conversations = []
    for query, response in zip(all_queries, all_responses):
        messages = create_messages(query, response, is_prm)
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        all_conversations.append(conversation_str)
    
    # Score in batches
    print("Scoring with reward model...")
    rewards = reward_model.encode(all_conversations)
    rewards = sorted(rewards, key=lambda x: int(x.request_id))
    # Process rewards
    reward_scores = []
    for reward in rewards:
        if is_prm:
            reward_scores.append(reward.outputs.data[:, -1].numpy())
        else:
            # orm
            reward_scores.append(reward.outputs.data[-1].numpy())

    # Rebuild the think_sums_rewards with the same structure as think_sums
    for sample in samples:
        # Initialize reward structure to match think_sums
        sample["think_sums_rewards"] = [
            [
                [[] for _ in chunk] 
                for chunk in n_sample
            ] 
            for n_sample in sample["think_sums"]
        ]

    # Fill in the rewards using the tracked indices
    for (sample_idx, n_idx, h_idx, m_idx), reward in zip(all_indices, reward_scores):
        # Convert any numpy/tensor values to native Python types
        samples[sample_idx]["think_sums_rewards"][n_idx][h_idx][m_idx] = [float(r) for r in reward]

    # Set [-1] for skipped conversations (too long)
    for sample_idx, n_idx, h_idx, m_idx in skipped_indices:
        samples[sample_idx]["think_sums_rewards"][n_idx][h_idx][m_idx] = [-100.0]
    
    return samples

def main(args):    
    # Set default output file if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        model_name = args.model_name_or_path.split("/")[-1]
        args.output_file = f"{base_name}_rm{model_name}_scored.jsonl"
    
    print(f"Loading data from {args.input_file}")
    samples = load_jsonl(args.input_file)
    print(f"Loaded {len(samples)} samples")
    
    print(f"Initializing reward model: {args.model_name_or_path}")
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path, 
        trust_remote_code=True,      
        tensor_parallel_size=len(available_gpus),
        gpu_memory_utilization=0.95,
        seed=args.seed,
        task="reward",
        max_num_seqs=args.max_num_seqs,
    )
    tokenizer = llm.get_tokenizer()
    
    # Process and score all think_sums
    scored_samples = process_think_sums(samples, llm, tokenizer, args.max_model_len, args.is_prm)
    
    # Save the results
    print(f"Saving scored data to {args.output_file}")
    save_jsonl(scored_samples, args.output_file)
    print("Done!")

if __name__ == "__main__":
    args = parse_args()
    args.is_prm = False # PRM or ORM
    if "prm" in args.model_name_or_path.lower():
        args.is_prm = True

    print("="*25,  " arguments ", "="*25)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    main(args)