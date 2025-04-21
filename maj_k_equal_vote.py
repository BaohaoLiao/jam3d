import json
import argparse
import numpy as np
from pass_k import pass_at_k
from maj_k import get_most_common_pred


def main():
    parser = argparse.ArgumentParser(description='Calculate pass@k and maj@k in the dimension of n.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the evaluation output JSONL file')
    parser.add_argument("--k_values", type=str, default="1,2,4,8", help='k value for maj@k and pass@k calculation')
    parser.add_argument('--h_chunks', type=int, default=1, 
                        help='Number of chunks to use from H dimension. If -1, use all predictions at H dimension.')
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

            # Reshape to n x h_chunks*m_answers
            m_indices = np.arange(args.m_answers)
            q_preds = [] # n x h_chunks*m_answers
            for n_idx in range(n):
                tmp = []

                # Different n might have different H, must include the last index
                if args.h_chunks == -1: # use all predictions
                    H_indices = np.arange(len(q_sub_preds[n_idx]))
                elif args.h_chunks < 0: # use last h_chunks predictions
                    H_indices = list(range(args.h_chunks, 0))
                else:
                    H_indices = np.linspace(H-1, H//args.h_chunks-1, args.h_chunks, dtype=int)[::-1]

                for h_idx in H_indices:
                    for m_idx in m_indices:
                        tmp.append(q_sub_preds[n_idx][h_idx][m_idx])
                q_preds.append(tmp)

            # Calculate maj@k for different sliding windows, small variance
            q_maj_ks = []
            for start_idx in range(1): #n - k + 1):
                q_window_preds = []
                for i in range(k):
                    q_window_preds.extend(q_preds[start_idx + i])
                
                # Get majority prediction for this window
                q_maj_pred = get_most_common_pred(q_window_preds, last=False)

                q_maj_ks.append(True if q_maj_pred == q_gt else False)
                
            all_maj_ks.append(sum(q_maj_ks) / len(q_maj_ks))

            # Calculate pass@k
            #q_scores = [get_most_common_pred(q_preds[i], last=False) == q_gt for i in range(n)]
            q_scores = [sum([q_pred == q_gt for q_pred in q_preds[i]])>=1 for i in range(n)]
            all_pass_ks.append(pass_at_k(n, sum(q_scores), k))

        maj_k_results[f"maj@{k}"] = float(f"{np.mean(all_maj_ks):.4f}") 
        pass_k_results[f"pass@{k}"] = float(f"{np.mean(all_pass_ks):.4f}")

    print(pass_k_results)
    print(maj_k_results)


if __name__ == "__main__":
    main()