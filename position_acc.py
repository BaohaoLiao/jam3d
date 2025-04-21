import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Calculate pass@1 in the dimension of H.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the evaluation output JSONL file')
    args = parser.parse_args()
    
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # Load samples
    samples = []
    with open(args.input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Extract sub_preds and gt from all samples
    all_sub_scores = [sample['all_sub_scores'] if "all_sub_scores" in sample else sample['sub_scores'] for sample in samples]
    num_qs, n, H, m = len(all_sub_scores), len(all_sub_scores[0]), len(all_sub_scores[0][0]), len(all_sub_scores[0][0][0])

    print(f"  #Questions: {num_qs}")
    print(f"  n: {n}")
    print(f"  H: {H}")
    print(f"  m: {m}")
    print("=" * 50)
    
    acc_positions = [[] for _ in range(H)]
    all_sub_scores = np.array(all_sub_scores)

    # Reshape to [H, num_questions*n*M]
    reshaped_all_sub_scores = all_sub_scores.transpose(2, 0, 1, 3).reshape(H, num_qs*n*m)
    reshaped_all_sub_scores = reshaped_all_sub_scores.tolist()

    acc_H = [np.mean(scores) for scores in reshaped_all_sub_scores]
    acc_positions = {}
    for h in range(H):
        acc_positions[f"H={h}"] = float(f"{acc_H[h]:.4f}") 

    print(acc_positions)


if __name__ == "__main__":
    main()