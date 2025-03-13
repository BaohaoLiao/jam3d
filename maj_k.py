import json
import argparse
import numpy as np
from collections import Counter
from pass_k import pass_at_k


def get_most_common_pred(predictions, last=False):
    """
    Get the majority prediction from a list of predictions
    Input:
        last: If True, return the last prediction for tie.
    """
    if not predictions or all(p == "" for p in predictions):
        return ""
    
    # Filter out empty predictions
    valid_preds = [p for p in predictions if p != ""]
    if not valid_preds:
        return ""
    
    # Get most common prediction
    counts = Counter(valid_preds)
    max_count = max(counts.values())

    # Get all elements with maximum count
    most_common = [item for item, count in counts.items() if count == max_count]
    
    # If only one most common element, return it
    if len(most_common) == 1 or (not last):
        return most_common[0]
    else:
        return most_common[-1]
    

def sample_h_predictions(predictions, h_chunks):
    """
    Evenly samples h_chunks always includes the last chunk, 
    """
    H = len(predictions)
    indices = np.linspace(H-1, H//h_chunks-1, h_chunks, dtype=int)[::-1]
    return [predictions[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description='Calculate pass@k and maj@k in the dimension of n.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the evaluation output JSONL file')
    parser.add_argument("--k_values", type=str, default="1,2,4,8", help='k value for maj@k and pass@k calculation')
    parser.add_argument('--h_chunks', type=int, default=1, help='Number of chunks to use from H dimension')
    parser.add_argument('--m_answers', type=int, default=1, help='Number of answers to use from m dimension')
    args = parser.parse_args()
    
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]

    # Load samples
    samples = []
    with open(args.input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Extract sub_preds and gt from all samples
    all_sub_preds = [sample['all_sub_preds'] if "all_sub_preds" in sample else sample['sub_preds'] for sample in samples]
    all_gts = [sample['gt'] for sample in samples]
    
    num_qs, n, H, m = len(all_sub_preds), len(all_sub_preds[0]), len(all_sub_preds[0][0]), len(all_sub_preds[0][0][0])
    assert max(k_values) <= n, "k should not be larger than n"
    assert args.h_chunks <= H, "h_chunks should not be larger than H"
    assert args.m_answers <= m, "m_answers should not be larger than m"

    print(f"  #Questions: {num_qs}")
    print(f"  n: {n}")
    print(f"  H: {H}")
    print(f"  m: {m}")
    print("=" * 50)
    
    maj_k_results = {}
    pass_k_results = {}
    for k in k_values:
        all_maj_ks = [] # one element for one question
        all_pass_ks = []

        for q_idx in range(num_qs):
            q_sub_preds = all_sub_preds[q_idx]  # n x H x m
            q_gt = all_gts[q_idx]

            if args.h_chunks == 1 and args.m_answers == 1:
                q_preds = [q_sub_preds[i][-1][0] for i in range(n)]
            
            elif args.h_chunks > 1 and args.m_answers == 1:
                # use maj to squeeze the H dimension
                q_preds = []
                for i in range(n):
                    q_preds_H = sample_h_predictions([item[0] for item in q_sub_preds[i]], args.h_chunks)
                    q_preds.append(get_most_common_pred(q_preds_H, last=True))

            elif args.h_chunks == 1 and args.m_answers > 1:
                # use maj to squeeze the m dimension, only use the first m_answers
                q_preds = []
                for i in range(n):
                    q_preds.append(get_most_common_pred(q_sub_preds[i][-1][:args.m_answers], last=False))

            else:
                # use maj to first squeeze the m dimension, then squeeze the H dimension
                q_preds = []
                for i in range(n):
                    q_preds_H = [get_most_common_pred(q_sub_preds[i][h][:args.m_answers], last=False) for h in range(H)]
                    q_preds_H = sample_h_predictions(q_preds_H, args.h_chunks)
                    q_preds.append(get_most_common_pred(q_preds_H, last=True))

            
            # Calculate maj@k for different sliding windows, small variance
            q_maj_ks = []
            for start_idx in range(n - k + 1):
                # Get predictions from k samples using the last chunk and first answer
                q_window_preds = [q_preds[start_idx + i] for i in range(k)]
                
                # Get majority prediction for this window
                q_maj_pred = get_most_common_pred(q_window_preds, last=False)

                q_maj_ks.append(True if q_maj_pred == q_gt else False)
                
            all_maj_ks.append(sum(q_maj_ks) / len(q_maj_ks))

            # Calculate pass@k
            q_scores = [q_preds[i] == q_gt for i in range(n)]
            all_pass_ks.append(pass_at_k(n, sum(q_scores), k))

        maj_k_results[f"maj@{k}"] = float(f"{np.mean(all_maj_ks):.4f}") 
        pass_k_results[f"pass@{k}"] = float(f"{np.mean(all_pass_ks):.4f}")

    print(maj_k_results)
    print(pass_k_results)


if __name__ == "__main__":
    main()