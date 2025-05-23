import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from pass_k import pass_at_k


def check_answers(answers, threshold, max_attempts=None, early_stop_option="consecutive"):
    """
    Check if there are threshold identical answers
    
    Args:
        answers: List of answer predictions
        threshold: Number of identical answers required
    
    Returns:
        (early_stop_index, final_answer)
    """
    if len(answers) < threshold and (max_attempts is None or len(answers) < max_attempts):
        return -1, answers[-1]
    
    if early_stop_option == "consecutive":
        for i in range(len(answers) - threshold + 1):
            # Check if the next 'threshold' answers are identical
            reference = answers[i]
            if all(answers[i + j] == reference for j in range(1, threshold)):
                if i + threshold == len(answers):
                    return -1, reference
                else:
                    return i + threshold - 1, reference

    else:
        # Count frequencies of all answers
        answer_counts = {}
        last_indices = {}
        
        for i, answer in enumerate(answers):
            if answer == "":
                continue
            
            if answer not in answer_counts:
                answer_counts[answer] = 0
                
            answer_counts[answer] += 1
            last_indices[answer] = i
            
            # Check if this answer has reached the threshold
            if answer_counts[answer] >= threshold:
                if last_indices[answer] + 1 == len(answers):
                    return -1, answer
                else:
                    return last_indices[answer], answer
                
            if max_attempts is not None and i + 1  >= max_attempts:
                if i + 1 == len(answers):
                    return -1, answer
                else:
                    return i, answer

    return -1, answers[-1]


def main():
    parser = argparse.ArgumentParser(description='Calculate pass@k and maj@k in the dimension of n.')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True, help='Path to the evaluation output JSONL file')
    parser.add_argument("--thresholds", type=str, default="1,2,3,4")
    parser.add_argument("--early_stop_option", choices=['consecutive', "frequency"], default="consecutive")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="Maximum number of answers to check before stopping anyway")
    parser.add_argument("--start_token_idx", type=int, default=0,
                        help="Starting token index (should match the original script)")
    parser.add_argument("--max_tokens_per_think_chunk", type=int, default=512,
                        help="Max tokens per reasoning chunk (should match the original script)")
    args = parser.parse_args()
    
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Parse consecutive_threshold
    thresholds = [int(t) for t in args.thresholds.split(",")]

    # Load samples
    samples = []
    with open(args.input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Extract sub_preds and gt from all samples
    all_sub_preds = []  # Only select the first pred in m dimension
    for sample in samples:
        if "all_sub_preds" in sample:
            sub_preds = sample['all_sub_preds']
        else:
            sub_preds = sample['sub_preds'] # old version
        sample_sub_preds = []
        for i in range(len(sub_preds)):
            tmp_sub_preds = []
            for h in range(len(sub_preds[i])):
                tmp_sub_preds.append(sub_preds[i][h][0])
            sample_sub_preds.append(tmp_sub_preds)
        all_sub_preds.append(sample_sub_preds)

    all_think_sums = []
    for sample in samples:
        think_sums = sample['think_sums'] 
        sample_think_sums = []
        for i in range(len(think_sums)):
            tmp_think_sums = []
            for h in range(len(think_sums[i])):
                tmp_think_sums.append(think_sums[i][h][0])
            sample_think_sums.append(tmp_think_sums)
        all_think_sums.append(sample_think_sums)

    all_completions = [sample['completion'] for sample in samples]
    all_gts = [sample['gt'] for sample in samples]
    
    num_qs, n = len(all_sub_preds), len(all_sub_preds[0])
    print(f"  #Questions: {num_qs}")
    print(f"  n: {n}")
    print("=" * 50)

    # Load tokenizer for measuring the efficiency for early stop
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Conventional Pass@1 and average inference tokens
    tokens = []
    for q in range(num_qs):
        tokens.append(np.mean([len(tokenizer.encode(completion)) for completion in all_completions[q]]))

    pass_1s = []
    for q in range(num_qs):
        c = sum([preds[-1] == all_gts[q] for preds in all_sub_preds[q]])
        pass_1s.append(pass_at_k(n, c, 1))
    
    print(f"Conventional | pass@1: {np.mean(pass_1s):.4f}, #tokens/question: {np.mean(tokens):.1f}")

    # Early stop
    for c_t in thresholds:

        pass_1s_per_ct = []
        tokens_per_ct = []
        for q in range(num_qs):

            sample_preds = []
            sample_tokens = []
            for n_i in range(n):
                early_stop_idx, final_pred = check_answers(all_sub_preds[q][n_i], c_t, early_stop_option=args.early_stop_option, max_attempts=args.max_attempts)
                sample_preds.append(final_pred)

                if early_stop_idx == -1 and len(all_sub_preds[q][n_i]) == 0: # Same as conventional
                    sample_tokens.append(len(tokenizer.encode(all_completions[q][n_i])))
                elif early_stop_idx == -1 and len(all_sub_preds[q][n_i]) != 0:
                    sample_tokens.append(
                        len(tokenizer.encode(all_completions[q][n_i])) + 
                        sum([len(tokenizer.encode(think_sum)) for think_sum in all_think_sums[q][n_i][:-1]])
                    )
                else:
                    num_reasoning_tokens = args.start_token_idx + args.max_tokens_per_think_chunk * (early_stop_idx + 1)
                    original_num_reasoning_tokens = len(tokenizer.encode(all_completions[q][n_i])) - len(tokenizer.encode(all_think_sums[q][n_i][-1]))
                    if original_num_reasoning_tokens != 0:  # The original generation meets think_end token
                        assert num_reasoning_tokens < original_num_reasoning_tokens, print(num_reasoning_tokens, original_num_reasoning_tokens, early_stop_idx, len(all_sub_preds[q][n_i]))
                    sample_tokens.append(
                        num_reasoning_tokens + sum([len(tokenizer.encode(think_sum)) for think_sum in all_think_sums[q][n_i][:(early_stop_idx+1)]])
                    )
            
            c = sum([pred == all_gts[q] for pred in sample_preds])
            pass_1s_per_ct.append(pass_at_k(n, c, 1))
            tokens_per_ct.append(np.mean(sample_tokens))
        
        print(f"Threshold {c_t} | pass@1: {np.mean(pass_1s_per_ct):.4f}, #tokens/question: {np.mean(tokens_per_ct):.1f}")


if __name__ == "__main__":
    main()