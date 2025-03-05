import json
import argparse
import numpy as np
from utils.data import load_jsonl


def pass_at_k(n, c, k):
    """
    Calculate Pass@k metric for a single question
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calculate_pass_at_k_n_dimension(all_sub_scores, k_values):
    """
    Calculate Pass@k across the n dimension
    Uses the final chunk and first answer: sub_scores[:, -1, 0]
    Input:
        all_sub_scores: #questions x n x H x m
    """
    all_n_scores = []
    for sub_scores in all_sub_scores:
        all_n_scores.append([scores[-1][0] for scores in sub_scores])

    all_n_correct = [sum(n_scores) for n_scores in all_n_scores]
    n_total = len(all_n_scores[0])

    # Calculate Pass@k for each k value
    results = {}
    for k in k_values:
        pass_k = np.mean([pass_at_k(n_total, n_correct, k) for n_correct in all_n_correct])
        results[f"pass@{k}"] = float(f"{pass_k:.4f}")

    return results


def calculate_pass_at_k_m_dimension(all_sub_scores, k_values):
    """
    Calculate Pass@k across the m dimension
    Uses the final chunk and all answers: sub_scores[:, -1, :]
    Averages the Pass@k scores across all samples
    Input:
        all_sub_scores: #questions x n x H x m
    """
    m_total = len(all_sub_scores[0][0][0])
    n_samples = len(all_sub_scores[0])

    # Calculate Pass@k for each k value
    results = {}
    for k in k_values:
        pass_k = []

        # for each question
        for sub_scores in all_sub_scores:
            pass_k_for_each_n = []

            # for each n dimension
            for n in range(n_samples):
                m_correct = sum(sub_scores[n][-1])
                pass_k_for_each_n.append(pass_at_k(m_total, m_correct, k))

            pass_k.append(np.mean(pass_k_for_each_n))

        results[f"pass@{k}"] = float(f"{np.mean(pass_k):.4f}")
    
    return results


def calculate_pass_at_k_H_dimension(sub_scores, k_values):
    """
    Calculate Pass@k across the H dimension
    Evenly samples k chunks from each sample's reasoning, e.g. sub_scores[0, :, 0]
    Averages the Pass@k scores across all samples, i.e. over dimension n
    Input:
        all_sub_scores: #questions x n x H x m
    """
    results = {}
    n_samples = len(sub_scores)
    
    # For each k value
    for k in k_values:
        pass_k_values = []
        
        # For each sample
        for sample_idx in range(n_samples):
            H_scores = [chunk[0] for chunk in sub_scores[sample_idx]]
            H_total = len(H_scores)
            
            if k <= H_total:
                # Evenly sample k chunks
                indices = np.linspace(0, H_total - 1, k, dtype=int)
                sampled_scores = [H_scores[i] for i in indices]
                
                # Count correct sampled scores
                H_correct = sum(sampled_scores)
                
                # Calculate Pass@k for this sample
                pass_k = pass_at_k(k, H_correct, k)
                pass_k_values.append(pass_k)
        
        # Average Pass@k across all samples
        if pass_k_values:
            avg_pass_k = np.mean(pass_k_values)
            results[f"pass@{k}"] = float(f"{avg_pass_k:.4f}")
    
    return results



def main():
    parser = argparse.ArgumentParser(description="Calculate Pass@k metrics for mathematical reasoning evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the evaluation output JSONL file")
    parser.add_argument("--dimension", type=str, choices=['n', 'H', 'm'], required=True, 
                        help="Dimension to calculate Pass@k for (n: sampling, H: think chunks, m: answers per chunk)")
    parser.add_argument("--k_values", type=str, default="1,2,4,8", 
                        help="Comma-separated list of k values to calculate Pass@k for")
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Load samples
    samples = load_jsonl(args.input_file)
    # Extract sub_scores from all samples
    all_sub_scores = [sample('all_sub_scores') if "all_sub_scores" in sample else sample('sub_scores') for sample in samples]
    n, H, m = len(all_sub_scores[0]), len(all_sub_scores[0][0]), len(all_sub_scores[0][0][0]) 

    print(f" #Questions: {len(all_sub_scores)}")
    print(f" n: {n}")
    print(f" H: {H}")
    print(f" m: {m}")

    # Calculate Pass@k based on the specified dimension
    if args.dimension == 'n':
        assert max(k_values) <= n, "k should not be larger than n"
        results = calculate_pass_at_k_n_dimension(all_sub_scores, k_values)



    calculate_pass_at_k_n_dimension(sub_scores, k_values)


    pass_at_k_results = []
    for k in k_values:
        pass_at_k_results.append(calculate_pass_at_k_for_dimension(samples, args.dimension, k))
    
    # Prepare results
    results = {
        "dimension": args.dimension,
        "fixed_dimensions": {k: v for k, v in fixed_dims.items() if k != args.dimension},
        "num_samples": len(samples),
        "pass_at_k": {f"pass@{k}": pass_at_k_results[k] * 100 for k in k_values}
    }
    
    # Print results
    print(f"Pass@k results for dimension {args.dimension}:")
    for k in k_values:
        print(f"  Pass@{k}: {pass_at_k_results[k] * 100:.2f}%")
    
    # Save results
    output_file = args.output_file
    if not output_file:
        input_path = Path(args.input_file)
        output_file = str(input_path.with_suffix('')) + f"_pass_at_k_{args.dimension}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()