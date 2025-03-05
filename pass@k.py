import json
import argparse

from utils.data import load_jsonl



def main():
    parser = argparse.ArgumentParser(description="Calculate Pass@k metrics for mathematical reasoning evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the evaluation output JSONL file")
    parser.add_argument("--dimension", type=str, choices=['n', 'H', 'm'], required=True, 
                        help="Dimension to calculate Pass@k for (n: sampling, H: think chunks, m: answers per chunk)")
    parser.add_argument("--k_values", type=str, default="1,2,3,4,5", 
                        help="Comma-separated list of k values to calculate Pass@k for")
    parser.add_argument("--fixed_n", type=int, default=0, help="Fixed index for n dimension (when not the target dimension)")
    parser.add_argument("--fixed_H", type=int, default=0, help="Fixed index for H dimension (when not the target dimension)")
    parser.add_argument("--fixed_m", type=int, default=0, help="Fixed index for m dimension (when not the target dimension)")
    args = parser.parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Load samples
    samples = load_jsonl(args.input_file)
    
    # Fixed dimensions
    fixed_dims = {
        'n': args.fixed_n,
        'H': args.fixed_H,
        'm': args.fixed_m
    }
    
    # Calculate pass@k
    pass_at_k_results = calculate_pass_at_k_for_dimension(
        samples, args.dimension, k_values, fixed_dims
    )
    
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