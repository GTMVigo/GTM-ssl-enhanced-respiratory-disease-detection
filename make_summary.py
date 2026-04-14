import pandas as pd
import numpy as np
import os
from pathlib import Path

# Specify your folder path here (change as needed)
folder_path = '/home/projects/avera/medper/uvigo_voice/metrics_coperia_5f_hubert/cough'  # e.g., 

# Metrics to process (numeric ones only; confusion matrices skipped as they're non-numeric)
metrics = [
    'accuracy_majority_voting', 'f1_score_majority_voting', 'recall_majority_voting',
    'precision_majority_voting', 'auc_majority_voting',
    'accuracy_average_prob', 'f1_score_average_prob', 'recall_average_prob',
    'precision_average_prob', 'auc_average_prob'
]

summary_data = []

for csv_file in Path(folder_path).glob('*.csv'):
    model_name = csv_file.stem  # Use filename (without .csv) as model name
    
    # Read CSV with semicolon separator
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    
    
    # Compute mean and std for each metric
    row = {'model': model_name}
    for metric in metrics:
        if metric in df.columns:
            row[f'{metric}'] = df[metric].mean()
            row[f'{metric}_std'] = df[metric].std()
    
    summary_data.append(row)

# Create summary DataFrame and save to CSV
summary_df = pd.DataFrame(summary_data)
output_file = os.path.join(folder_path, 'models_summary.csv')
summary_df.to_csv(output_file, index=False, sep=';', decimal=',')  

print(f"Summary saved to {output_file}")
print(summary_df.head())
