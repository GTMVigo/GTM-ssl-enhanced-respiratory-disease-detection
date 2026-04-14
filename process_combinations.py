import pandas as pd
import numpy as np
import os
from itertools import combinations
from pathlib import Path
from sklearn.metrics import confusion_matrix
import sys

# Add src to path to import functions
sys.path.append('src')
from common_classification import (majority_voting, determine_accuracy, determine_f1score, 
                                  determine_recall, determine_precision, determine_auc)


def get_subfolders(base_path):
    """Get all subfolders in the base path."""
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    return subfolders


def read_csvs_from_folder(folder_path):
    """Read all CSVs from a folder and concatenate them."""
    csv_files = list(Path(folder_path).glob('*.csv'))
    
    if not csv_files:
        return pd.DataFrame()
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=';', decimal=',')
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def process_combination(folders, combination_name):
    """Process a combination of folders."""
    print(f"\nProcessing combination: {combination_name}")
    
    # Read and concatenate CSVs from all folders in the combination
    all_data = []
    for folder in folders:
        df = read_csvs_from_folder(folder)
        if not df.empty:
            # Select only required columns
            # required_cols = ['id', 'true_label', 'average_prob', 'majority_predictions', 'threshold']
            required_cols = ['id', 'true_label', 'prob_class_0', 'majority_predictions', 'argmax_predictions']

            df_subset = df[required_cols].copy()
            
            df_subset = df_subset.rename(columns={
                'prob_class_0': 'average_prob',
                'argmax_predictions': 'threshold'
            })
            
            all_data.append(df_subset)
    
    if not all_data:
        print(f"  No data found for combination: {combination_name}")
        return None
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Unique IDs: {combined_df['id'].nunique()}")
    
    # Group by ID and aggregate
    grouped_results = []
    
    for person_id, group in combined_df.groupby('id'):
        true_label = group['true_label'].iloc[0]  # All should be same
        
        # Apply aggregation functions
        maj_pred = majority_voting(group['majority_predictions'].values)
        thresh_pred = majority_voting(group['threshold'].values)
        avg_prob = np.mean(group['average_prob'].values)
        
        grouped_results.append({
            'id': person_id,
            'true_label': true_label,
            'majority_predictions': maj_pred,
            'threshold_predictions': thresh_pred,
            'average_prob': avg_prob
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(grouped_results)
    
    # Add correctness indicator columns (1 if prediction matches true label, 0 otherwise)
    results_df['majority_correct'] = (results_df['majority_predictions'] == results_df['true_label']).astype(int)
    results_df['threshold_correct'] = (results_df['threshold_predictions'] == results_df['true_label']).astype(int)
    # Note: average_prob is probability of class 0, so invert before rounding
    results_df['average_prob_correct'] = (np.round(1 - results_df['average_prob']) == results_df['true_label']).astype(int)
    
    # Calculate metrics
    group_labels = results_df['true_label'].values
    majority_predictions = results_df['majority_predictions'].values
    threshold_predictions = results_df['threshold_predictions'].values
    average_prob_pred = results_df['average_prob'].values
    
    # Change probabilities into predictions
    # Note: average_prob in CSV is probability of class 0, so we need to invert it
    # When prob(class_0) is high -> predict 0, when low -> predict 1
    probabilities_pred = np.round(1 - average_prob_pred)
    
    # Initialize metrics dictionary
    metrics_dict = {
        'combination': [],
        'accuracy_majority_voting': [],
        'f1_score_majority_voting': [],
        'recall_majority_voting': [],
        'precision_majority_voting': [],
        'confusion_matrix_majority_voting': [],
        'auc_majority_voting': [],
        'accuracy_threshold': [],
        'f1_score_threshold': [],
        'recall_threshold': [],
        'precision_threshold': [],
        'confusion_matrix_threshold': [],
        'auc_threshold': [],
        'accuracy_average_prob': [],
        'f1_score_average_prob': [],
        'recall_average_prob': [],
        'precision_average_prob': [],
        'confusion_matrix_average_prob': [],
        'auc_average_prob': []
    }
    
    # Majority voting metrics
    majority_voting_cm = confusion_matrix(group_labels, majority_predictions)
    metrics_dict['combination'].append(combination_name)
    metrics_dict['accuracy_majority_voting'].append(determine_accuracy(majority_voting_cm))
    metrics_dict['f1_score_majority_voting'].append(determine_f1score(majority_voting_cm))
    metrics_dict['recall_majority_voting'].append(determine_recall(majority_voting_cm))
    metrics_dict['precision_majority_voting'].append(determine_precision(majority_voting_cm))
    metrics_dict['confusion_matrix_majority_voting'].append(majority_voting_cm.flatten())
    metrics_dict['auc_majority_voting'].append(determine_auc(y_true=group_labels,
                                                             y_scores=majority_predictions))
    
    # Threshold majority voting metrics
    threshold_cm = confusion_matrix(group_labels, threshold_predictions)
    metrics_dict['accuracy_threshold'].append(determine_accuracy(threshold_cm))
    metrics_dict['f1_score_threshold'].append(determine_f1score(threshold_cm))
    metrics_dict['recall_threshold'].append(determine_recall(threshold_cm))
    metrics_dict['precision_threshold'].append(determine_precision(threshold_cm))
    metrics_dict['confusion_matrix_threshold'].append(threshold_cm.flatten())
    metrics_dict['auc_threshold'].append(determine_auc(y_true=group_labels,
                                                       y_scores=threshold_predictions))
    
    # Average probability metrics
    average_prob_cm = confusion_matrix(group_labels, probabilities_pred)
    metrics_dict['accuracy_average_prob'].append(determine_accuracy(average_prob_cm))
    metrics_dict['f1_score_average_prob'].append(determine_f1score(average_prob_cm))
    metrics_dict['recall_average_prob'].append(determine_recall(average_prob_cm))
    metrics_dict['precision_average_prob'].append(determine_precision(average_prob_cm))
    metrics_dict['confusion_matrix_average_prob'].append(average_prob_cm.flatten())
    metrics_dict['auc_average_prob'].append(determine_auc(y_true=group_labels,
                                                          y_scores=probabilities_pred))
    
    # Print summary
    print(f"  Majority Voting - Accuracy: {metrics_dict['accuracy_majority_voting'][0]:.4f}, F1: {metrics_dict['f1_score_majority_voting'][0]:.4f}")
    print(f"  Threshold       - Accuracy: {metrics_dict['accuracy_threshold'][0]:.4f}, F1: {metrics_dict['f1_score_threshold'][0]:.4f}")
    print(f"  Average Prob    - Accuracy: {metrics_dict['accuracy_average_prob'][0]:.4f}, F1: {metrics_dict['f1_score_average_prob'][0]:.4f}")
    
    return metrics_dict, results_df


def main():
    """Main function to process all combinations."""
    base_path = 'metrics_coperia_5f/antes_despues'
    
    # Get all subfolders
    subfolders = get_subfolders(base_path)
    subfolder_names = [os.path.basename(f) for f in subfolders]
    
    print(f"Found {len(subfolders)} subfolders: {subfolder_names}")
    
    # Create output directory
    output_dir = 'combination_results_5f/antes_despues_4'
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    
    # Generate all combinations (from 1 to all subfolders)
    # for r in range(1, len(subfolders) + 1):
    for r in range(4, 5):
        for combo in combinations(subfolders, r):
            # Create combination name
            combo_names = [os.path.basename(f) for f in combo]
            combination_name = '+'.join(combo_names)
            
            # Process this combination
            result = process_combination(combo, combination_name)
            
            if result is not None:
                metrics_dict, results_df = result
                
                # Save detailed results for this combination
                results_filename = f"{output_dir}/{combination_name}_detailed.csv"
                results_df.to_csv(results_filename, index=False, sep=';', decimal=',')
                
                # Store metrics
                all_metrics.append(pd.DataFrame(metrics_dict))
    
    # Combine all metrics and save
    if all_metrics:
        final_metrics = pd.concat(all_metrics, ignore_index=True)
        final_metrics.to_csv(f"{output_dir}/all_combinations_metrics.csv", index=False, sep=';', decimal=',')
        print(f"\n✓ All results saved to {output_dir}/")
        print(f"✓ Total combinations processed: {len(all_metrics)}")
    else:
        print("\n⚠ No combinations could be processed!")


if __name__ == '__main__':
    main()
