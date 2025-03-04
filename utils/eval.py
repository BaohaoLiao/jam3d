from collections import Counter
from math_verify import verify


def get_last_most_common(lst):
    # Get counts of all elements
    counts = Counter(lst)
    max_count = max(counts.values())
    
    # Get all elements with maximum count
    most_common = [item for item, count in counts.items() if count == max_count]
    
    # If only one most common element, return it
    if len(most_common) == 1:
        return most_common[0]
    
    # If tie, find the last occurrence
    for item in reversed(lst):
        if item in most_common:
            return item


def get_most_common_pred_score(preds, scores):
    valid_pairs = [(pred, score) for pred, score in zip(preds, scores) if pred != ""]
    if not valid_pairs:
        return "", False
    
    valid_preds = [pair[0] for pair in valid_pairs]
    most_common_pred = Counter(valid_preds).most_common(1)[0][0]
    for pred, score in valid_pairs:
        if pred == most_common_pred:
            return pred, score
    return "", False


def get_most_common_pred_score_in_thinking(preds, scores):
    valid_pairs = [(pred, score) for pred, score in zip(preds, scores) if pred != ""]
    if not valid_pairs:
        return "", False
    
    valid_preds = [pair[0] for pair in valid_pairs]
    most_common_pred = get_last_most_common(valid_preds)
    #Counter(valid_preds).most_common(1)[0][0]
    for pred, score in valid_pairs:
        if pred == most_common_pred:
            return pred, score
    return "", False


def obtain_3d_sub_scores_and_preds(gt, chunk_answer_preds):
    """
    Process the predictions with a 3D structure: n_sampling x num_chunks x num_answers
    Implements hierarchical majority voting in this specific order:
    1. First majority vote across answers within each chunk
    2. Then majority vote across chunks within each sample
    3. Finally majority vote across all samples
    
    Args:
        gt: Ground truth answer
        chunk_answer_preds: List of predictions in 3D structure [n_sampling][num_chunks][num_answers]
    
    Returns:
        Processed ground truth, predictions, and scores in hierarchical structure
    """
    # Convert ground truth to string format
    new_gt = str(gt[0])
    # Verify all predictions against ground truth and format predictions
    all_sub_scores = []
    all_sub_preds = []

    for sample_preds in chunk_answer_preds:  # For each sampling iteration
        sample_scores = []
        sample_sub_preds = []

        for chunk_preds in sample_preds:  # For each chunk
            chunk_scores = [verify(gt, pred) for pred in chunk_preds]  # For each answer in chunk
            sample_scores.append(chunk_scores)
            
            # Format predictions for this chunk
            chunk_sub_preds = []
            for k, (pred, score) in enumerate(zip(chunk_preds, chunk_scores)):
                if score:
                    chunk_sub_preds.append(new_gt)
                else:
                    if pred:
                        chunk_sub_preds.append(str(pred[0]))
                    else:
                        chunk_sub_preds.append("")
            
            sample_sub_preds.append(chunk_sub_preds)
        
        all_sub_scores.append(sample_scores)
        all_sub_preds.append(sample_sub_preds)
    
    # LEVEL 1: Majority voting across answers within each chunk
    chunk_maj_preds = []  # [n_sampling][num_chunks]
    chunk_maj_scores = []  # [n_sampling][num_chunks]
    
    for sample_preds, sample_scores in zip(all_sub_preds, all_sub_scores):
        sample_chunk_maj_preds = []
        sample_chunk_maj_scores = []
        
        for chunk_preds, chunk_scores in zip(sample_preds, sample_scores):
            # Get the majority prediction for this chunk
            maj_pred, maj_score = get_most_common_pred_score(chunk_preds, chunk_scores)
            sample_chunk_maj_preds.append(maj_pred)
            sample_chunk_maj_scores.append(maj_score)
        
        chunk_maj_preds.append(sample_chunk_maj_preds)
        chunk_maj_scores.append(sample_chunk_maj_scores)
    
    # LEVEL 2: Majority voting across chunks within each sample
    sample_maj_preds = []  # [n_sampling]
    sample_maj_scores = []  # [n_sampling]
    
    for sample_chunk_preds, sample_chunk_scores in zip(chunk_maj_preds, chunk_maj_scores):
        # Get the majority prediction for this sample
        maj_pred, maj_score = get_most_common_pred_score_in_thinking(sample_chunk_preds, sample_chunk_scores)
        sample_maj_preds.append(maj_pred)
        sample_maj_scores.append(maj_score)
    
    # LEVEL 3: Majority voting across samples is not done here
    # It will be handled by the obtain_3d_scores function
    return new_gt, all_sub_preds, all_sub_scores, chunk_maj_preds, chunk_maj_scores, sample_maj_preds, sample_maj_scores


def obtain_3d_scores(samples, n_sampling=1):
    """
    Process and calculate scores for 3D structure of predictions
    Implements the 3rd level of majority voting across sampling iterations
    """
    all_samples = []
    
    # First, collect all the level 2 predictions (per-sample predictions)
    for sample in samples:
        all_samples.append(sample)
    
    # Calculate accuracy based on the first sample's score
    correctnesses = [sample["sample_maj_scores"][0] for sample in all_samples]
    
    result_json = {
        "num_samples": len(correctnesses),
        "acc": float(f"{sum(correctnesses) / len(correctnesses):.4f}") * 100,
    }
    
    # LEVEL 3: Majority voting across sampling iterations
    if n_sampling > 1:
        new_all_samples = []
        maj_correctnesses = []
        
        for sample in all_samples:
            # Get all predictions from level 2 (across sampling iterations)
            sample_maj_preds = sample["sample_maj_preds"]
            sample_maj_scores = sample["sample_maj_scores"]
            
            # Perform majority voting
            maj_pred, maj_score = get_most_common_pred_score(sample_maj_preds, sample_maj_scores)
            
            # Update the sample with the final majority prediction
            sample.update({
                "maj_pred": maj_pred, 
                "maj_score": maj_score
            })
            
            new_all_samples.append(sample)
            maj_correctnesses.append(maj_score)
        
        # Calculate majority voting accuracy
        result_json["maj_acc"] = float(f"{sum(maj_correctnesses) / len(maj_correctnesses):.4f}") * 100
        all_samples = new_all_samples

    return all_samples, result_json