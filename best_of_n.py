import json
import argparse
import numpy as np
from collections import Counter
from pass_k import pass_at_k
from maj_k import get_most_common_pred



def scalar_from_reward_list(reward_list, use_last=False, use_min=False, use_prod=False):
    """Convert a list of reward scores to a scalar value based on specified flags."""
    if not reward_list:
        return -100
    
    # Handle edge case for skipped samples (too long)
    if len(reward_list) == 1 and reward_list[0] == -100:
        return -100
    
    if use_last:
        return reward_list[-1]
    elif use_min:
        return min(reward_list)
    elif use_prod:
        return np.prod(reward_list)
    else:
        # Default: use the mean
        return np.mean(reward_list)

"""
def get_highest_reward_pred(preds, rewards):

    Get the prediction with the highest reward score.
    preds: list of predictions
    rewards: list of corresponding reward scores

    if not preds or not rewards:
        return ""
    # Filter out skipped samples (reward == -1)
    valid_pairs = [(p, r) for p, r in zip(preds, rewards) if p != "" and r != -100]
    if not valid_pairs:
        return ""
    
    # Sort by reward score in descending order
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
    # Return the prediction with the highest reward
    return sorted_pairs[0][0], sorted_pairs[0][1]
"""

def get_highest_reward_pred(preds, rewards):
    """
    Get the prediction with the highest reward score.
    If multiple predictions have the same highest score, use frequency across all predictions as a tiebreaker.
    
    preds: list of predictions
    rewards: list of corresponding reward scores
    """
    if not preds or not rewards:
        return "", -100
    
    # Filter out skipped samples (reward == -1)
    valid_pairs = [(p, r) for p, r in zip(preds, rewards) if p != "" and r != -100]
    
    if not valid_pairs:
        return "", -100
    
    # Count frequency of all predictions
    all_pred_counts = Counter([p for p, _ in valid_pairs])
    
    # Group predictions by reward score
    score_to_preds = {}
    for pred, reward in valid_pairs:
        if reward not in score_to_preds:
            score_to_preds[reward] = []
        if pred not in score_to_preds[reward]:
            score_to_preds[reward].append(pred)
    
    # Find the highest reward score
    highest_score = max(score_to_preds.keys())
    
    # Get all predictions with the highest score
    top_preds = score_to_preds[highest_score]
    
    # If only one prediction has the highest score, return it
    if len(top_preds) == 1:
        return top_preds[0], highest_score
    
    # If multiple predictions have the highest score, use frequency across all predictions as tiebreaker
    top_pred_with_count = [(pred, all_pred_counts[pred]) for pred in set(top_preds)]
    top_pred_with_count.sort(key=lambda x: x[1], reverse=True)
    
    # Return the most frequent prediction among the top ones
    return top_pred_with_count[0][0], highest_score



def main():
    parser = argparse.ArgumentParser(description='Calculate pass@k and reward@k metrics using reward model scores.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the scored JSONL file with reward scores')
    parser.add_argument('--k_values', type=str, default="1,2,4,8", help='k values for reward@k and pass@k calculation')
    parser.add_argument('--h_chunks', type=int, default=1, 
                        help='Number of chunks to use from H dimension. If -1, use all predictions at H dimension.')
    parser.add_argument('--m_answers', type=int, default=1, help='Number of answers to use from m dimension')
    
    # Reward conversion flags
    parser.add_argument('--last', action='store_true', help='Use the last reward score from the list')
    parser.add_argument('--min', action='store_true', help='Use the minimum reward score from the list')
    parser.add_argument('--prod', action='store_true', help='Multiply all scores in the list')
    
    args = parser.parse_args()
    
    print("="*25, " arguments ", "="*25)
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

    # Extract predictions, ground truth, and reward scores
    all_sub_preds = [sample['all_sub_preds'] for sample in samples]
    all_gts = [sample['gt'] for sample in samples]
    all_rewards = [sample['think_sums_rewards'] for sample in samples]
    
    # Validate dimensions
    num_qs = len(all_sub_preds)
    if num_qs == 0:
        print("No samples found in the input file.")
        return
    
    n = len(all_sub_preds[0])
    H = len(all_sub_preds[0][0])
    m = len(all_sub_preds[0][0][0])
    
    assert max(k_values) <= n, "k should not be larger than n"
    assert args.h_chunks <= H or args.h_chunks == -1, "h_chunks should not be larger than H unless it's -1"
    assert args.m_answers <= m, "m_answers should not be larger than m"

    print(f"  #Questions: {num_qs}")
    print(f"  n: {n}")
    print(f"  H: {H}")
    print(f"  m: {m}")
    
    reward_k_results = {}
    pass_k_results = {}
    maj_k_results = {}
    
    for k in k_values:
        all_reward_ks = []  # one element for one question
        all_pass_ks = []
        all_maj_ks = []

        for q_idx in range(num_qs):
            q_sub_preds = all_sub_preds[q_idx]  # n x H x m
            q_rewards = all_rewards[q_idx]      # n x H x m
            q_gt = all_gts[q_idx]

            # Convert reward lists to scalar values and reshape predictions and rewards
            q_preds = []  # n x (h_chunks*m_answers)
            q_scalar_rewards = []  # n x (h_chunks*m_answers)
            
            for n_idx in range(n):
                tmp_preds = []
                tmp_rewards = []
                
                # Different n might have different H, must include the last index
                if args.h_chunks == -1:
                    H_indices = np.arange(len(q_sub_preds[n_idx]))
                else:
                    H_indices = np.linspace(H-1, H//args.h_chunks-1, args.h_chunks, dtype=int)[::-1]

                for h_idx in H_indices:
                    m_indices = np.arange(min(args.m_answers, len(q_sub_preds[n_idx][h_idx])))
                    for m_idx in m_indices:
                        tmp_preds.append(q_sub_preds[n_idx][h_idx][m_idx])
                        # Convert reward list to scalar
                        reward_scalar = scalar_from_reward_list(
                            q_rewards[n_idx][h_idx][m_idx], 
                            use_last=args.last, 
                            use_min=args.min, 
                            use_prod=args.prod
                        )
                        tmp_rewards.append(reward_scalar)
                
                q_preds.append(tmp_preds)
                q_scalar_rewards.append(tmp_rewards)
            
            # Calculate reward@k for different sliding windows
            q_reward_ks = []
            for start_idx in range(n - k + 1):
                window_preds = []
                window_rewards = []
                
                for i in range(k):
                    window_preds.extend(q_preds[start_idx + i])
                    window_rewards.extend(q_scalar_rewards[start_idx + i])
                
                # Get prediction with highest reward in this window
                best_pred = get_highest_reward_pred(window_preds, window_rewards)[0]
                
                if best_pred is not None:
                    q_reward_ks.append(True if best_pred == q_gt else False)
                else:
                    q_reward_ks.append(False)  # No valid prediction found
            
            # Store average success rate for this question
            if q_reward_ks:
                all_reward_ks.append(sum(q_reward_ks) / len(q_reward_ks))
            else:
                all_reward_ks.append(0.0)
                
            # Calculate pass@k: for each question, choose one pred in Hxm with the highest reward

            print("|||"*50, q_idx)
            for i in range(n):
                print(q_gt)
                print(get_highest_reward_pred(q_preds[i], q_scalar_rewards[i]))


            q_best_preds =[get_highest_reward_pred(q_preds[i], q_scalar_rewards[i])[0] for i in range(n)]
            q_scores = [q_best_preds[i] == q_gt for i in range(n)]
            all_pass_ks.append(pass_at_k(n, sum(q_scores), k))

            # Calculate maj@k for different sliding windows, small variance
            q_maj_ks = []
            for start_idx in range(n - k + 1):
                # Get predictions from k samples using the last chunk and first answer
                q_window_preds = [q_best_preds[start_idx + i] for i in range(k)]
                
                # Get majority prediction for this window
                q_maj_pred = get_most_common_pred(q_window_preds, last=False)

                q_maj_ks.append(True if q_maj_pred == q_gt else False)

            all_maj_ks.append(sum(q_maj_ks) / len(q_maj_ks))

        # Calculate and store average metrics across all questions
        reward_k_results[f"BoN@{k}"] = float(f"{np.mean(all_reward_ks):.4f}")
        pass_k_results[f"pass@{k}"] = float(f"{np.mean(all_pass_ks):.4f}")
        maj_k_results[f"maj@{k}"] = float(f"{np.mean(all_maj_ks):.4f}") 

    print("\nResults:")
    print("Pass@k results:", pass_k_results)
    print("BoN@k results:", reward_k_results)
    print("Maj@k results:", maj_k_results)

if __name__ == "__main__":
    main()